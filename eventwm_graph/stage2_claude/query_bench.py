"""Query-ranking benchmark (diagnostic, development split): do Stage 2 predictions rank the future-observation
consequences of alternative Query Outcomes from the same Root?

  prepare   : replay every development Root once (the original legal DAG replay of stage2_segment.data, with the
              window list replaced by the comparison outcomes of query_audit.outcomes_at) and cache, per distinct
              outcome, the executed trajectory and the observed increment (labels only).
  predict   : run one frozen checkpoint through the public LegalVoxelPredictor on the cached Roots and store
              per-outcome predicted / actual gains, confusions and calibration histograms.
  heuristic : deterministic legal baselines from the same legal inputs (no learning, no privileged data).
  report    : metrics, scene bootstrap, JSON + Markdown (query_report.py).

Recorded future observations are used only to build labels. Predictions use the Root's legal partial graph, the
legal range frames up to the decision time and the legal track history, exactly as the original evaluator.
"""
import argparse
import gzip
import json
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from eventwm_graph.stage2_repair.diagnose import OUT, source_rows, member
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_repair.full_evaluate import labels
from eventwm_graph.stage2_segment import data as segdata
from .query_audit import outcomes_at, HORIZONS

DEST = OUT.parent/'claude_stage2'/'query_bench'
SETS = ('full',)+HORIZONS
RB = 20


def yaw_diff(a, b):
    d = np.asarray(a)-np.asarray(b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


# ------------------------------------------------------------------ prepare
def prepare_root(row):
    path = DEST/'cache'/(row['root_id']+'.pkl.gz')
    if path.exists():
        return row['root_id'], 'cached'
    legal, queries, outcomes, nodes, children = segdata.load(row)
    t0 = legal['partial_graph']['timestamp']
    wins = {}
    for h in SETS:
        for qid, end, b, dur, term in outcomes_at(legal, queries, outcomes, nodes, h):
            o = next(o for o in outcomes if o['query_id'] == qid)
            w = wins.setdefault(end, dict(start_node=None, end_node=end, node_ids=o['node_ids'][:b], query_id=qid,
                                          query_aliases=[], start_step=0, end_step=b, termination=term, sets=[]))
            w['sets'].append([str(h), qid, term])
    selected = list(wins.values())
    original = segdata.windows
    segdata.windows = lambda *a, **k: [dict(w, query_aliases=[]) for w in selected]
    try:
        graph = None; out = {}
        for ex in segdata.replay(row, 'mixed'):
            if 'audit' in ex:
                continue
            graph = ex['graph']
            target = ex['target']
            tk = np.asarray(list(target), np.int32).reshape(-1, 2)
            ts = np.asarray([v[0] for v in target.values()], np.uint8)
            p = ex['trajectory']
            dist = float(np.linalg.norm(np.diff(p[:, [1, 3]], axis=0), axis=1).sum()) if len(p) > 1 else 0.
            rot = float(yaw_diff(p[1:, 4], p[:-1, 4]).sum()) if len(p) > 1 else 0.
            out[ex['end_node']] = dict(end_node=ex['end_node'], trajectory=p, target_keys=tk, target_states=ts,
                                       sets=wins[ex['end_node']]['sets'], duration=float(p[-1, 0]-p[0, 0]), distance=dist,
                                       rotation=rot, known_terminal_differences=ex['known_terminal_differences'])
    finally:
        segdata.windows = original
    assert set(out) == set(wins), 'replay did not emit every comparison outcome'
    hist_yaw = np.asarray([h['robot_pose'][3] for h in legal['history']], np.float64)
    for w in out.values():
        yaws = w['trajectory'][1:, 4]
        far = np.array([yaw_diff(y, hist_yaw).min() for y in yaws]) if len(yaws) else np.zeros(1)
        w['unseen_yaw_max'] = float(far.max()); w['unseen_yaw_frac'] = float((far > np.pi/4).mean())
    value = dict(root_id=row['root_id'], scene_id=row['scene_id'], graph=graph, outcomes=list(out.values()),
                 history_frames=len(legal['history']), history_seconds=float(t0-legal['history'][0]['timestamp']), t0=t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_bytes(gzip.compress(pickle.dumps(value), 3)); tmp.replace(path)
    return row['root_id'], len(out)


def prepare(workers=6):
    rows, _ = source_rows('dev')
    tick = time.time()
    with get_context('spawn').Pool(workers) as pool:
        done = pool.map(prepare_root, rows, chunksize=4)
    print('prepared', len(done), 'Roots in %.0fs' % (time.time()-tick), flush=True)


def load_cache(root_id):
    return pickle.loads(gzip.decompress((DEST/'cache'/(root_id+'.pkl.gz')).read_bytes()))


# ------------------------------------------------------------------ predict
MODEL = None
UNK_BANDS = ('floor_unk', 'low_unk', 'mid_unk')


def _init(checkpoint, device, lanes):
    global MODEL
    import torch
    from .api import LegalVoxelPredictor
    torch.set_num_threads(1)
    MODEL = LegalVoxelPredictor(checkpoint, device=device, lanes=lanes)


def packet_for(row, sensor_directory, sid=None):
    with np.load(OUT/sensor_directory/(row['root_id']+'.npz')) as z:
        s = {k: z[k] for k in z.files}
    take = ancestor_frames(s['parents'], s['history_count'], sid)
    legal = member(row, 'legal_root.json.gz'); nodes = member(row, 'future_observations.json.gz')
    tracks = [o['tracks'] for o in legal['history']]+[n['observation']['tracks'] for n in nodes]
    packet = dict(ranges=s['depth'][take], origins=s['origins'][take], rotations=s['rotations'][take], timestamps=s['timestamps'][take],
                  sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
    return packet, [tracks[j] for j in take]


def never_observed(ctx, keys):
    """Candidate cells whose columns hold no observed voxel in the floor/low/mid bands of the legal map."""
    from .features import MAP_CHANNELS
    idx = [MAP_CHANNELS.index(n) for n in UNK_BANDS]
    a = keys[:, 0].astype(np.int64)-int(ctx.lo[0]); b = keys[:, 1].astype(np.int64)-int(ctx.lo[1])
    ok = (a >= 0) & (b >= 0) & (a < ctx.shape[0]) & (b < ctx.shape[1])
    unk = np.ones(len(keys), bool)
    if ok.any():
        ch = ctx.map_channels[idx][:, a[ok], b[ok]].float().cpu().numpy()
        unk[ok] = (ch >= .999).all(0)
    return unk


def predict_root(args):
    row, sensor_directory, chunk = args
    cache = load_cache(row['root_id'])
    packet, tracks = packet_for(row, sensor_directory)
    outs = cache['outcomes']; recs = []
    hist = np.zeros((3, RB, 2)); brier = np.zeros(2)
    MODEL.clear_cache()
    for i in range(0, len(outs), chunk):
        part = outs[i:i+chunk]
        deltas = MODEL.predict_segments(partial_graph=cache['graph'], trajectories=[o['trajectory'] for o in part],
                                        range_observations=packet, track_history=tracks, materialize=False)
        ctx = next(reversed(MODEL.map_cache.values()))
        for o, d in zip(part, deltas):
            keys = d['candidate_cells']; p = d['probabilities'].astype(np.float64); hard = d['states']
            target = {tuple(k): (int(s), 0.) for k, s in zip(o['target_keys'], o['target_states'])}
            y, _, missing = labels(keys, target)
            y = y.astype(np.int64)
            cm = np.bincount(3*y+hard, minlength=9).reshape(3, 3)
            unk = never_observed(ctx, keys)
            cm_unk = np.bincount(3*y[unk]+hard[unk], minlength=9).reshape(3, 3)
            r = p[:, 1]+p[:, 2]
            rev = (y > 0).astype(np.float64)
            onehot = np.eye(3)[y]
            for c, (score, truth) in enumerate(((r, rev), (p[:, 1], (y == 1).astype(np.float64)), (p[:, 2], (y == 2).astype(np.float64)))):
                bins = np.minimum((score*RB).astype(np.int64), RB-1)
                np.add.at(hist[c], (bins, truth.astype(np.int64)), 1)
            brier += [((r-rev)**2).sum(), ((p-onehot)**2).sum()]
            # per-outcome 20 (revelation) x 10 (BLOCKED-given-revelation) histogram of the candidate cells: lets a binned
            # recalibration fitted on TRAIN validation be applied to the expected gains afterwards
            bb = np.where(r > 0, p[:, 2]/np.maximum(r, 1e-12), 0.)
            hrb = np.bincount(np.minimum((r*20).astype(np.int64), 19)*10+np.minimum((bb*10).astype(np.int64), 9), minlength=200)
            recs.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], end_node=o['end_node'], sets=o['sets'],
                             duration=o['duration'], distance=o['distance'], rotation=o['rotation'],
                             unseen_yaw_max=o['unseen_yaw_max'], unseen_yaw_frac=o['unseen_yaw_frac'],
                             history_seconds=cache['history_seconds'], candidates=len(keys), out_of_domain=len(missing),
                             actual_free=int((y == 1).sum()+sum(1 for s, _ in missing if s == 1)),
                             actual_blocked=int((y == 2).sum()+sum(1 for s, _ in missing if s == 2)),
                             exp_free=float(p[:, 1].sum()), exp_blocked=float(p[:, 2].sum()),
                             hard_free=int((hard == 1).sum()), hard_blocked=int((hard == 2).sum()),
                             cm=cm.tolist(), cm_never_observed=cm_unk.tolist(), never_observed_cells=int(unk.sum()),
                             brier_rev=float(((r-rev)**2).sum()), brier_multi=float(((p-onehot)**2).sum()),
                             hist_rb=hrb.tolist()))
    return recs, hist, brier


def predict(checkpoint, name, sensor_directory='full_range_branches_256', device='cuda:0', workers=2, lanes=3, chunk=8, limit=0):
    rows, _ = source_rows('dev')
    rows = [r for r in rows if (DEST/'cache'/(r['root_id']+'.pkl.gz')).exists()]
    if limit:
        rows = rows[:limit]
    tick = time.time(); recs = []; hist = np.zeros((3, RB, 2)); brier = np.zeros(2)
    with get_context('spawn').Pool(workers, initializer=_init, initargs=(checkpoint, device, lanes)) as pool:
        for k, (r, h, b) in enumerate(pool.imap_unordered(predict_root, [(row, sensor_directory, chunk) for row in rows], chunksize=2)):
            recs.extend(r); hist += h; brier += b
            if (k+1) % 200 == 0:
                print('%d Roots %.0fs' % (k+1, time.time()-tick), flush=True)
    out = dict(name=name, checkpoint=str(checkpoint), sensor_directory=sensor_directory, roots=len(rows), records=recs,
               calibration=dict(bins=RB, revelation=hist[0].tolist(), free=hist[1].tolist(), blocked=hist[2].tolist(), brier_sums=brier.tolist()),
               seconds=time.time()-tick)
    (DEST/'predictions').mkdir(parents=True, exist_ok=True)
    (DEST/'predictions'/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(out).encode()))
    print('wrote', name, len(recs), 'outcomes from', len(rows), 'Roots in %.0fs' % (time.time()-tick), flush=True)


# ------------------------------------------------------------------ deterministic legal heuristics
def heuristic_root(args):
    """Per outcome: unknown candidate cells whose center lies in a future camera frustum (<= 10 m) of some pose
    (frustum count, occlusion ignored), whose 63 probes have no KNOWN blocker from some pose (legal visibility
    count, unknown space treated as passable) and that are fully visible through known free space (known-visible)."""
    row, sensor_directory, chunk = args
    import torch
    from .features import build_inputs, crop_box, QUERY_CHANNELS, MAP_CHANNELS
    cache = load_cache(row['root_id'])
    packet, tracks = packet_for(row, sensor_directory)
    prepared = MODEL.prepare_graph(cache['graph'])
    nm = len(MAP_CHANNELS); q = {n: nm+i for i, n in enumerate(QUERY_CHANNELS)}
    MODEL.clear_cache(); recs = []
    outs = cache['outcomes']
    for i in range(0, len(outs), chunk):
        part = outs[i:i+chunk]; paths = [o['trajectory'] for o in part]
        ctx, _ = MODEL._context(prepared, cache['graph'], packet, tracks, paths)
        boxes = [crop_box(p, prepared.resolution) for p in paths]
        x, mask, fr = build_inputs(ctx, paths, [(lo, hi-lo) for lo, hi in boxes])
        m = mask.bool()
        for o, k in zip(part, range(len(part))):
            mk = m[k]
            recs.append(dict(root_id=row['root_id'], end_node=o['end_node'],
                             frustum=int((x[k, q['center_frustum_poses']][mk] > 0).sum()),
                             legal_visible=int((x[k, q['probe_possible']][mk] > 0).sum()),
                             known_visible=int((x[k, q['probe_known']][mk] > 0).sum()),
                             candidates=int(mk.sum())))
    return recs


def heuristic(checkpoint, sensor_directory='full_range_branches_256', device='cuda:0', workers=2, chunk=8, limit=0):
    rows, _ = source_rows('dev')
    rows = [r for r in rows if (DEST/'cache'/(r['root_id']+'.pkl.gz')).exists()]
    if limit:
        rows = rows[:limit]
    tick = time.time(); recs = []
    with get_context('spawn').Pool(workers, initializer=_init, initargs=(checkpoint, device, 1)) as pool:
        for r in pool.imap_unordered(heuristic_root, [(row, sensor_directory, chunk) for row in rows], chunksize=2):
            recs.extend(r)
    (DEST/'predictions').mkdir(parents=True, exist_ok=True)
    (DEST/'predictions'/'heuristics.json.gz').write_bytes(gzip.compress(json.dumps(dict(
        geometry_from=str(checkpoint), sensor_directory=sensor_directory, records=recs)).encode()))
    print('wrote heuristics', len(recs), 'outcomes in %.0fs' % (time.time()-tick), flush=True)


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    sub = pa.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--workers', type=int, default=6)
    b = sub.add_parser('predict'); b.add_argument('--checkpoint', required=True); b.add_argument('--name', required=True)
    b.add_argument('--sensor-directory', default='full_range_branches_256'); b.add_argument('--device', default='cuda:0')
    b.add_argument('--workers', type=int, default=2); b.add_argument('--lanes', type=int, default=3); b.add_argument('--limit', type=int, default=0)
    c = sub.add_parser('heuristic'); c.add_argument('--checkpoint', required=True)
    c.add_argument('--sensor-directory', default='full_range_branches_256'); c.add_argument('--device', default='cuda:0')
    c.add_argument('--workers', type=int, default=2); c.add_argument('--limit', type=int, default=0)
    args = vars(pa.parse_args()); cmd = args.pop('cmd')
    dict(prepare=prepare, predict=predict, heuristic=heuristic)[cmd](**args)
