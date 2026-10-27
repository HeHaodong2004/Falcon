"""Integrated Layer 2 validation on the frozen development split (genuine inference, recorded trajectories).

Per development Root (sharded across processes):

1. Root decision state (legal history, legal graph, 256 px range frames); candidates = every distinct
   recorded Query execution from that Root; one ``Layer2WorldModel.predict`` call (K = all).
2. Stage 2 preservation: each candidate's hard states, candidate domain, expected counts and the
   label-aware confusion (labels from the recorded future, evaluation only) equal the standalone v17_a
   query-benchmark records; pooled label-aware IoU over all recorded full executions.
3. Stage 1 preservation: the shared bank equals a standalone ``HumanWorldModel`` call (bit-identical).
4. Shared decision state: a rebuilt state (new object, same stream) and one call per candidate give
   bit-identical banks and per-candidate increments (on every ``--consistency-every``-th Root).
5. Alignment: t=0 bank positions of currently visible tracks vs their legal positions; candidate
   robot poses on the 0.048 s clock vs the recorded poses; predicted cells vs trajectory distance.
6. Legality: the hypothetical future graph is rejected as a decision-state input; inputs unmutated.
7. Later decision on a recorded branch (incremental Stage 1 stream): bank == fresh-stream standalone;
   continuation candidates predicted and scored (diagnostic IoU).

    python scripts/layer2_run.py eventwm_graph.layer2.validate run NAME --shard I --shards N [--limit L]
    python scripts/layer2_run.py eventwm_graph.layer2.validate summarize NAME
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .devdata import dev_rows, DevRoot, reference_records, confusion
from .state import DecisionState, LegalityError

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'output/layer2/validation'


def _bank_equal(a, b):
    return (a.positions.shape == b.positions.shape and np.array_equal(a.positions, b.positions) and np.array_equal(a.present, b.present)
            and a.track_keys == b.track_keys and np.array_equal(a.count_log_probability, b.count_log_probability)
            and np.array_equal(a.existence_probability, b.existence_probability) and np.array_equal(a.knots, b.knots))


def _delta_equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ('candidate_cells', 'probabilities', 'states', 'heights'))


def visible_track_error(state, bank):
    """Distance (m) at t=0 between each currently visible track slot and its legal position."""
    from eventwm_graph.stage1_hwm.legal_state import positives
    now = {str(t['track_id']): np.asarray(t['position'], np.float64)[[0, 2]] for t in positives(state.frames[-1])}
    err = []
    for slot, key in enumerate(bank.track_keys):
        if key.startswith('track:') and key[6:] in now:
            p = bank.positions[:, 0, slot]
            err.append(float(np.max(np.linalg.norm(p-now[key[6:]], axis=-1))))
    return err


def humans_in_blocked(state, bank):
    """Fraction of (world, visible-track) positions at t=0 inside known BLOCKED cells (diagnostic)."""
    res = state.resolution
    blocked = {tuple(c['cell']) for c in state.partial_graph['cells'] if c['state'] == 2}
    n = hit = 0
    for slot, key in enumerate(bank.track_keys):
        if key.startswith('track:') and bank.present[0, slot]:
            for p in bank.positions[:, 0, slot]:
                n += 1; hit += (int(np.floor(p[0]/res)), int(np.floor(p[1]/res))) in blocked
    return hit, n


def robot_alignment(out):
    """Max |interaction robot pose - recorded pose| at recorded times on the 0.048 s clock."""
    from .model import robot_steps
    worst = 0.
    for c in out.candidates:
        p = c.trajectory
        robots, lengths, rel, _ = robot_steps([p], out.timestamp, .048, 62)
        t = out.timestamp+rel[:lengths[0]]
        for i, ti in enumerate(t):
            j = np.flatnonzero(np.abs(p[:, 0]-ti) < 1e-9)
            if len(j):
                worst = max(worst, float(np.max(np.abs(robots[0, i, :3]-p[j[0], 1:4]))))
    return worst


def cell_distance(out, res):
    """Max over candidates of the min distance from a predicted (FREE/BLOCKED) cell centre to the trajectory (m)."""
    worst = 0.
    for c in out.candidates:
        keys = c.map['candidate_cells'][c.map['states'] > 0]
        if not len(keys):
            continue
        centres = (keys.astype(np.float64)+.5)*res
        xy = c.trajectory[:, [1, 3]]
        d = np.sqrt(((centres[:, None]-xy[None])**2).sum(-1)).min(1)
        worst = max(worst, float(d.max()))
    return worst


def validate_root(l2, s1ref, row, ref, consistency):
    r = DevRoot(row)
    rec = dict(root_id=r.root_id, scene_id=r.scene_id, frames=len(r.legal['history']))
    state = r.decision_state()
    graph_before = copy.deepcopy(state.partial_graph)
    cands = r.root_candidates()
    out = l2.predict(state, [c[2] for c in cands])
    prof = l2.last_profile
    rec['profile'] = dict(k=len(cands), total_ms=prof['total_ms'], stage1_ms=prof['stage1_ms'], stage2_ms=prof['stage2_ms'],
                          interaction_ms=prof['interaction_ms'])
    # ---- Stage 2 preservation (against the standalone v17_a records) and labels
    s2 = dict(outcomes=0, compared=0, states_equal=0, cm_equal=0, exp_equal=0, cm=np.zeros((3, 3), np.int64).tolist(), windows=[])
    total = np.zeros((3, 3), np.int64)
    for (end, qid, path), cp in zip(cands, out.candidates):
        target = r.target(end)
        cm, missing = confusion(cp.map, target)
        total += cm
        s2['outcomes'] += 1
        sm = cp.summary()
        s2['windows'].append(dict(end=end, cm=cm.tolist(), missing=len(missing), duration=cp.duration,
                                  predicted_free=sm['predicted_free'], predicted_blocked=sm['predicted_blocked'],
                                  exp_gain=sm['expected_free_uncalibrated']+sm['expected_blocked_uncalibrated'],
                                  actual_gain=int(sum(1 for v in target.values() if v[0] in (1, 2))),
                                  p_social=sm['p_social'], p_collision=sm['p_collision'], human_truncated=sm['human_horizon_truncated']))
        x = ref.get((r.root_id, end))
        if x is not None:
            s2['compared'] += 1
            s2['states_equal'] += int(x['hard_free'] == sm['predicted_free'] and x['hard_blocked'] == sm['predicted_blocked']
                                      and x['candidates'] == sm['candidate_cells'])
            s2['cm_equal'] += int(x['cm'] == cm.tolist())
            s2['exp_equal'] += int(abs(x['exp_free']-sm['expected_free_uncalibrated']) <= 1e-9*max(1., x['exp_free'])
                                   and abs(x['exp_blocked']-sm['expected_blocked_uncalibrated']) <= 1e-9*max(1., x['exp_blocked']))
    s2['cm'] = total.tolist()
    outside = l2.maps.outside_default_extent(state, [c[2] for c in cands]) if cands else []
    s2['outside_default_extent'] = len(outside)
    if outside:     # canonical semantics: every candidate == a cold single-candidate Stage 2 call
        ok = 0
        for (end, qid, path), cp in zip(cands, out.candidates):
            l2.maps.model.clear_cache()
            raw = l2.maps.model.predict_segments(partial_graph=state.partial_graph, trajectories=[path], range_observations=state.range_packet,
                                                 track_history=state.track_history, materialize=False)[0]
            ok += int(_delta_equal(raw, cp.map))
        s2['equal_to_cold_single'] = ok
        l2.maps.model.clear_cache()
    rec['stage2'] = s2
    # ---- Stage 1 preservation (standalone model, fresh stream)
    alone = s1ref.predict(list(state.frames[:-1]), state.frames[-1], state.partial_graph, stream='ref:'+r.root_id)
    s1ref.streams.pop('ref:'+r.root_id, None)
    rec['stage1'] = dict(identical=bool(_bank_equal(alone, out.humans)), slots=int(out.humans.positions.shape[2]),
                         visible_track_error=visible_track_error(state, out.humans), humans_in_blocked=humans_in_blocked(state, out.humans),
                         minimum_count=int(out.humans.minimum_count))
    # ---- alignment, legality
    rec['alignment'] = dict(robot_pose_error=robot_alignment(out), max_predicted_cell_distance=cell_distance(out, state.resolution),
                            bank_clock_error=abs(float(out.humans.timestamp)-state.timestamp),
                            map_clock_error=max([abs(float(c.map['base_timestamp'])-state.timestamp) for c in out.candidates] or [0.]))
    legality = dict(graph_unmutated=bool(graph_before == state.partial_graph))
    if out.candidates and out.candidates[0].future_graph is not None:
        fg = out.candidates[0].future_graph
        legality['future_graph_flagged'] = bool(fg.get('hypothetical'))
        try:
            DecisionState(state.frames, fg, state.range_packet)
            legality['future_graph_rejected'] = False
        except LegalityError:
            legality['future_graph_rejected'] = True
    rec['legality'] = legality
    # ---- shared decision-state consistency: rebuilt state object, one call per candidate
    if consistency and cands:
        state2 = r.decision_state()
        ok_bank = ok_maps = True
        for (end, qid, path), cp in zip(cands, out.candidates):
            single = l2.predict(state2, [path])
            ok_bank &= _bank_equal(single.humans, out.humans)
            ok_maps &= _delta_equal(single.candidates[0].map, cp.map)
        rec['consistency'] = dict(bank_identical=bool(ok_bank), maps_identical=bool(ok_maps), k=len(cands))
    # ---- later decision on a recorded branch (incremental stream)
    longest = max(((o['node_ids'], q['query_id']) for q, o in zip(r.queries, r.outcomes)), key=lambda x: (len(x[0]), x[1]))[0]
    if len(longest) >= 6:
        mid = longest[len(longest)//2-1]
        sb = r.decision_state(mid)                     # same stream as the Root decision -> incremental ingest
        cont = r.continuation_candidates(mid)
        ob = l2.predict(sb, [c[2] for c in cont])
        fresh = s1ref.predict(list(sb.frames[:-1]), sb.frames[-1], sb.partial_graph, stream='ref:'+r.root_id)
        s1ref.streams.pop('ref:'+r.root_id, None)
        bcm = np.zeros((3, 3), np.int64)
        for (end, qid, path), cp in zip(cont, ob.candidates):
            bcm += confusion(cp.map, r.target(end, start=mid))[0]
        rec['branch'] = dict(node=mid, frames=len(sb.frames), k=len(cont), bank_equals_fresh_stream=bool(_bank_equal(fresh, ob.humans)),
                             cm=bcm.tolist(), total_ms=l2.last_profile['total_ms'], stage1_ms=l2.last_profile['stage1_ms'],
                             visible_track_error=visible_track_error(sb, ob.humans))
    return rec


def run(name, shard=0, shards=1, limit=0, device='cuda:0', consistency_every=10):
    import torch
    from .model import Layer2WorldModel
    from eventwm_graph.stage1_hwm.api import HumanWorldModel
    dest = DEST/name
    dest.mkdir(parents=True, exist_ok=True)
    rows = dev_rows()
    if limit:
        rows = rows[:limit]
    mine = [(i, r) for i, r in enumerate(rows) if i % shards == shard]
    l2 = Layer2WorldModel.from_release(map_device=device)
    s1ref = HumanWorldModel(str(ROOT/l2.release['stage1']['checkpoint']), 'cpu')
    ref = reference_records()
    path = dest/('shard_%02d_of_%02d.jsonl' % (shard, shards))
    done = set()
    if path.exists():
        done = {json.loads(line)['root_id'] for line in path.read_text().splitlines() if line.strip()}
    tick = time.time()
    with path.open('a') as f:
        for n, (i, row) in enumerate(mine):
            if row['root_id'] in done:
                continue
            crashes = dest/('shard_%02d_of_%02d.cuda_faults.jsonl' % (shard, shards))
            prior = [json.loads(x) for x in crashes.read_text().splitlines()] if crashes.exists() else []
            try:
                rec = validate_root(l2, s1ref, row, ref, consistency=(i % consistency_every == 0))
            except Exception as e:      # recorded, never silently skipped
                if 'CUDA error' in repr(e) and sum(c['root_id'] == row['root_id'] for c in prior) < 1:
                    with crashes.open('a') as cf:
                        cf.write(json.dumps(dict(root_id=row['root_id'], index=i, error=repr(e), time=time.time(),
                                                 roots_done_in_process=n))+'\n')
                    print('CUDA fault on %s; exiting for restart' % row['root_id'], flush=True)
                    os._exit(3)
                rec = dict(root_id=row['root_id'], error=repr(e), trace=traceback.format_exc())
            rec['index'] = i
            f.write(json.dumps(rec)+'\n'); f.flush()
            if (n+1) % 50 == 0:
                print('shard %d: %d/%d Roots %.0fs' % (shard, n+1, len(mine), time.time()-tick), flush=True)
    meta = dict(name=name, shard=shard, shards=shards, roots=len(mine), seconds=time.time()-tick, device=device,
                gpu=torch.cuda.get_device_name(torch.device(device)), max_reserved_gb=torch.cuda.max_memory_reserved(torch.device(device))/2**30,
                stage1_sha256=l2.humans.checkpoint_sha256, stage2_sha256=l2.maps.checkpoint_sha256, load_average=os.getloadavg())
    (dest/('shard_%02d_of_%02d.meta.json' % (shard, shards))).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def _iou(cm):
    cm = np.asarray(cm, np.float64)
    out = {}
    for c, n in ((1, 'free'), (2, 'blocked')):
        tp = cm[c, c]; fp = cm[:, c].sum()-tp; fn = cm[c, :].sum()-tp
        out[n] = tp/(tp+fp+fn) if tp+fp+fn else float('nan')
    out['label_aware'] = (out['free']+out['blocked'])/2
    return out


def _equal_weight_iou(recs):
    """Original Stage 2 reduction (stage2_repair.full_evaluate.summarize): mean raw confusion per Root, then per
    scene, then rates. Applied here to the Root full-execution windows (not the 47,128 mixed evaluation windows)."""
    from scripts.stage2_diagnostic.common import extended_rates
    by_scene = {}
    for r in recs:
        wins = [np.asarray(w['cm'], np.float64) for w in r['stage2']['windows']]
        if wins:
            by_scene.setdefault(r['scene_id'], []).append(np.mean(wins, 0))
    if not by_scene:
        return None
    rates = extended_rates(np.mean([np.mean(v, 0) for v in by_scene.values()], 0))
    return {k: v for k, v in rates.items() if 'iou' in k}


def pairwise(recs):
    """Root-macro pairwise ranking accuracy of predicted (uncalibrated) gain vs actual gain over recorded executions."""
    acc = []
    for r in recs:
        w = r['stage2']['windows']
        ok = n = 0
        for i in range(len(w)):
            for j in range(i+1, len(w)):
                if w[i]['actual_gain'] == w[j]['actual_gain']:
                    continue
                n += 1
                ok += (w[i]['exp_gain']-w[j]['exp_gain'])*(w[i]['actual_gain']-w[j]['actual_gain']) > 0
        if n:
            acc.append(ok/n)
    return float(np.mean(acc)) if acc else None, len(acc)


def summarize(name):
    dest = DEST/name
    recs = []
    for p in sorted(dest.glob('shard_*.jsonl')):
        recs += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    errors = [r for r in recs if 'error' in r]
    ok = [r for r in recs if 'error' not in r]
    s2 = {k: sum(r['stage2'][k] for r in ok) for k in ('outcomes', 'compared', 'states_equal', 'cm_equal', 'exp_equal')}
    inside = [r for r in ok if not r['stage2'].get('outside_default_extent')]
    outside = [r for r in ok if r['stage2'].get('outside_default_extent')]
    s2['roots_all_candidates_inside_default_map_extent'] = dict(
        roots=len(inside), **{k: sum(r['stage2'][k] for r in inside) for k in ('compared', 'states_equal', 'cm_equal', 'exp_equal')})
    s2['roots_with_candidates_outside_default_map_extent'] = dict(
        roots=len(outside), candidates_outside=sum(r['stage2']['outside_default_extent'] for r in outside),
        candidates=sum(r['stage2']['outcomes'] for r in outside), equal_to_cold_single_call=sum(r['stage2']['equal_to_cold_single'] for r in outside),
        **{k: sum(r['stage2'][k] for r in outside) for k in ('compared', 'states_equal', 'cm_equal', 'exp_equal')})
    faults = []
    for p in sorted(dest.glob('shard_*.cuda_faults.jsonl')):
        faults += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    pooled = np.sum([r['stage2']['cm'] for r in ok], 0)
    vt = [e for r in ok for e in r['stage1']['visible_track_error']]
    hb = np.sum([r['stage1']['humans_in_blocked'] for r in ok], 0)
    cons = [r['consistency'] for r in ok if 'consistency' in r]
    br = [r['branch'] for r in ok if 'branch' in r]
    pct = lambda x, q: float(np.percentile(x, q)) if len(x) else None
    prof = lambda key, rs: dict(p50=pct([r[key] for r in rs], 50), p95=pct([r[key] for r in rs], 95))
    summary = dict(
        name=name, roots=len(recs), errors=len(errors), error_examples=[e['error'] for e in errors[:5]],
        cuda_faults=dict(count=len(faults), roots=[f['root_id'] for f in faults],
                         retried_ok=sum(any(r['root_id'] == f['root_id'] for r in ok) for f in faults)),
        scenes=len({r['scene_id'] for r in ok}),
        stage2=dict(**s2, pooled_iou_full_executions=_iou(pooled), equal_weight_iou_full_executions=_equal_weight_iou(ok),
                    pairwise_ranking_full_executions=pairwise(ok)),
        stage1=dict(identical_to_standalone=sum(r['stage1']['identical'] for r in ok), of=len(ok),
                    visible_track_t0_error_m=dict(n=len(vt), max=max(vt) if vt else None, p99=pct(vt, 99)),
                    visible_humans_in_known_blocked_cells=dict(hits=int(hb[0]), positions=int(hb[1])),
                    minimum_count_respected='asserted inside HumanWorlds construction (every world)'),
        alignment=dict(robot_pose_error_max=max(r['alignment']['robot_pose_error'] for r in ok),
                       predicted_cell_distance_max_m=max(r['alignment']['max_predicted_cell_distance'] for r in ok),
                       bank_clock_error_max=max(r['alignment']['bank_clock_error'] for r in ok),
                       map_clock_error_max=max(r['alignment']['map_clock_error'] for r in ok)),
        legality=dict(graph_unmutated=sum(r['legality']['graph_unmutated'] for r in ok),
                      future_graph_flagged=sum(r['legality'].get('future_graph_flagged', False) for r in ok),
                      future_graph_rejected=sum(r['legality'].get('future_graph_rejected', False) for r in ok),
                      roots_with_future_graph=sum('future_graph_rejected' in r['legality'] for r in ok)),
        consistency=dict(roots=len(cons), bank_identical=sum(c['bank_identical'] for c in cons), maps_identical=sum(c['maps_identical'] for c in cons),
                         candidates=sum(c['k'] for c in cons)),
        branch=dict(decisions=len(br), bank_equals_fresh_stream=sum(b['bank_equals_fresh_stream'] for b in br),
                    candidates=sum(b['k'] for b in br), pooled_iou=_iou(np.sum([b['cm'] for b in br], 0)) if br else None,
                    visible_track_t0_error_max=max([e for b in br for e in b['visible_track_error']] or [0.]),
                    incremental_total_ms=prof('total_ms', br), incremental_stage1_ms=prof('stage1_ms', br)),
        validation_run_profile_note='validation calls run beside other validation shards; latency numbers come from benchmark.py',
        root_decision_total_ms=prof('total_ms', [r['profile'] for r in ok]))
    (dest/'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run'); a.add_argument('name'); a.add_argument('--shard', type=int, default=0); a.add_argument('--shards', type=int, default=1)
    a.add_argument('--limit', type=int, default=0); a.add_argument('--device', default='cuda:0'); a.add_argument('--consistency-every', type=int, default=10)
    b = sub.add_parser('summarize'); b.add_argument('name')
    x = p.parse_args()
    if x.cmd == 'run':
        run(x.name, x.shard, x.shards, x.limit, x.device, x.consistency_every)
    else:
        summarize(x.name)
