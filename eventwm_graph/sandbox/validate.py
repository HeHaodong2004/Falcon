"""Sandbox fidelity: imagined executions vs the same Queries executed for real (PRIVILEGED VALIDATION ONLY).

    python scripts/sandbox_run.py eventwm_graph.sandbox.validate fidelity --out DIR --episodes dev:0,unseen:2 \
        --decisions 6 [--planner orca --dynamics native --config k10]

Along a default-policy trunk in querygen's exact branch simulator (``EpisodeRuntime``: the Env.step mirror with the
recorded human tape, bit-identical to the unchanged habitat.Env), at every decision state each candidate Query of the
frozen generator is

* executed for real from that state (a branch; its true poses, outcome and clearance are labels), and
* imagined by the sandbox with three nominal human futures: the shared Stage 1 bank (``mean``: what the sandbox
  uses), constant-velocity tracks (``cv``) and the true future of the tracked people (``true``: privileged, isolates
  the frozen-map / kinematics part of the error from human prediction).

The Layer 2 legal stream (frames, graph, range frames) is built along the trunk only, from the rows the runtime's
legal sensor published.  Labels never enter the sandbox.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def _match_true(legal, tape, index, steps, step_s):
    """Planar true paths of the tracked people (nearest tape human within 0.6 m at t) -- privileged."""
    from eventwm_graph.querygen.control import human_estimates
    hp = tape.positions[index:min(len(tape), index+steps+1)]
    if len(hp) < steps+1:
        hp = np.concatenate([hp, np.repeat(hp[-1:], steps+1-len(hp), 0)])
    ids, heights, paths = [], [], []
    by_id = {t['track_id']: t for t in legal.tracks}
    for h in human_estimates(legal.tracks, legal.time):
        p = np.array([h.position[0], -h.position[1]])
        if not hp.shape[1]:
            break
        d = np.hypot(hp[0, :, 0]-p[0], hp[0, :, 2]-p[1])
        k = int(np.argmin(d))
        if d[k] > 0.6:
            continue
        ids.append(h.track_id); heights.append(float(by_id[h.track_id]['position'][1])); paths.append(hp[:, k][:, [0, 2]])
    return ids, heights, paths


def _compare(real_poses, imag):
    n = min(len(real_poses), len(imag.poses))
    R = np.asarray(real_poses[:n]); I = imag.poses[:n]
    dev = np.hypot(R[:, 0]-I[:, 0], R[:, 2]-I[:, 2])
    yaw = np.abs(np.arctan2(np.sin(R[:, 3]-I[:, 3]), np.cos(R[:, 3]-I[:, 3])))
    end = float(np.hypot(real_poses[-1][0]-imag.poses[-1, 0], real_poses[-1][2]-imag.poses[-1, 2]))
    return dict(common=int(n), mean_dev=float(dev.mean()), max_dev=float(dev.max()), final_dev=float(dev[-1]),
                end_dev=end, mean_yaw_dev=float(yaw.mean()), len_real=len(real_poses)-1, len_imag=imag.steps)


def fidelity_episode(sb, spec, config='k10', planner='orca', dynamics='native', decisions=6, interval=16):
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES, STEP_SECONDS
    from .episode import generator, default_query
    from .context import LegalStream
    from .fastrange import FastRangeSensor
    from .core import Candidate
    from .imagine import NominalHumans
    dyn = FAMILIES[dynamics]
    gen = generator(config, dynamics)
    rt = EpisodeRuntime(spec, planner, dyn, strict=False, velocity_action=not dyn.native_actions)
    stream = LegalStream(spec.key)
    fast = FastRangeSensor(rt.task, rt.sensor.config)
    capture = [True]
    observe = rt.sensor.observe

    def observed(ts):
        row = observe(ts)
        if capture[0]:
            stream.observe(row, fast.measure(timestamp=row['timestamp']))      # branch simulator: tape clock
        return row
    rt._load(rt.root)                       # robot at the reset, humans at tape index 0 (the tape may have been recorded)
    root_row = rt.root.sensor[3]            # the reset observation row the runtime's legal state was built from
    stream.observe(root_row, fast.measure(timestamp=root_row['timestamp']))
    rt.sensor.observe = observed
    q = default_query()
    st = rt.root
    out = []
    for d in range(int(decisions)):
        capture[0] = True
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else q, horizon=interval, cont=cont)
        if st.terminal:
            break
        if abs(stream.timestamp-st.legal.time) > 1e-7:
            raise RuntimeError('trunk stream and branch state differ in time')
        state = stream.decision_state(dict(key=spec.key, index=st.index))
        queries, _ = gen.propose(st.legal, st.execution)
        cands = [Candidate(None, planner, dynamics, execution=st.execution, level=st.dyn_state, query_id='continue')
                 if qq.kind == 'continue' else Candidate(qq, planner, dynamics, level=st.dyn_state, query_id=qq.label)
                 for qq in queries]
        steps = [sb.config.steps_for(c) for c in cands]
        # real executions (labels)
        capture[0] = False
        real = []
        for c, h in zip(cands, steps):
            t0 = time.perf_counter()
            child = rt.run(st, c.query, horizon=h, cont=c.query is None) if h > 0 else st
            poses = [np.asarray(p, np.float64) for p in child.legal.poses[len(st.legal.poses)-1:]]
            real.append(dict(poses=poses, outcome=child.outcome, status=child.execution.status if child.execution else None,
                             min_rect=float(child.min_rect_dist), static_contacts=int(child.static_contacts),
                             real_s=time.perf_counter()-t0))
        # imagined: Stage 1 nominal (the sandbox itself, with Stage 2 and risk) + cv + true humans
        t0 = time.perf_counter()
        res = sb.evaluate(state, st.legal, cands)
        sandbox_s = time.perf_counter()-t0
        H = max(steps, default=0)
        cv = sb.imagine_all(st.legal, cands, NominalHumans(st.legal.tracks, st.legal.time, H, bank=None))
        ids, hs, paths = _match_true(st.legal, rt.tape, st.index, H, STEP_SECONDS)
        tr = sb.imagine_all(st.legal, cands, NominalHumans.from_paths(ids, hs, paths, st.legal.time, H, mode='true'))
        rows = []
        for k, c in enumerate(cands):
            o = res.candidates[k]
            row = dict(label=c.query.label if c.query is not None else 'continue', kind=(c.query.kind if c.query else 'continue'),
                       real_outcome=real[k]['outcome'], real_status=real[k]['status'], real_min_rect=real[k]['min_rect'],
                       real_static_contacts=real[k]['static_contacts'], real_s=real[k]['real_s'],
                       imag_ended_by=o.rollout.ended_by, imag_status=o.rollout.status,
                       p_collision=float(o.humans['p_collision']), p_social=float(o.humans['p_social']),
                       min_q10=(None if o.humans['minimum_quantiles'][0] is None else float(o.humans['minimum_quantiles'][0])))
            for mode, roll in (('mean', o.rollout), ('cv', cv[k]), ('true', tr[k])):
                row[mode] = _compare(real[k]['poses'], roll)
            g = np.asarray(st.legal.goal)
            rp = real[k]['poses']
            row['real_progress'] = float(math.hypot(g[0]-rp[0][0], g[2]-rp[0][2])-math.hypot(g[0]-rp[-1][0], g[2]-rp[-1][2]))
            row['imag_progress'] = float(o.rollout.progress)
            rows.append(row)
        out.append(dict(decision=d, index=st.index, tracks=len(st.legal.tracks), sandbox_s=sandbox_s, candidates=rows))
        print(spec.key, 'decision', d, 'tracks', len(st.legal.tracks), 'cands', len(rows),
              'mean dev (mean/cv/true) %.3f/%.3f/%.3f m' % tuple(np.mean([r[m]['mean_dev'] for r in rows]) for m in ('mean', 'cv', 'true')),
              flush=True)
    rt.close()
    return dict(key=spec.key, config=config, planner=planner, dynamics=dynamics, decisions=out)


def summarize(records):
    rows = [r for rec in records for d in rec['decisions'] for r in d['candidates']]
    out = dict(candidates=len(rows), decisions=sum(len(rec['decisions']) for rec in records))
    for m in ('mean', 'cv', 'true'):
        out[m] = {k: dict(p50=float(np.median([r[m][k] for r in rows])), mean=float(np.mean([r[m][k] for r in rows])),
                          p90=float(np.percentile([r[m][k] for r in rows], 90)))
                  for k in ('mean_dev', 'final_dev', 'end_dev', 'mean_yaw_dev')}
        out[m]['same_length'] = float(np.mean([r[m]['len_real'] == r[m]['len_imag'] for r in rows]))
    rp = np.array([r['real_progress'] for r in rows]); ip = np.array([r['imag_progress'] for r in rows])
    out['progress_corr'] = float(np.corrcoef(rp, ip)[0, 1]) if rp.std() > 0 and ip.std() > 0 else None
    # within-decision ranking: does the imagined best-progress candidate match the real best?
    top, taus = [], []
    for rec in records:
        for d in rec['decisions']:
            c = d['candidates']
            if len(c) < 2:
                continue
            ri = np.array([x['real_progress'] for x in c]); ii = np.array([x['imag_progress'] for x in c])
            top.append(float(ri[int(np.argmax(ii))] >= ri.max()-0.05))
            from scipy.stats import kendalltau
            t = kendalltau(ri, ii).correlation
            if t == t:
                taus.append(float(t))
    out['top1_progress_agreement'] = float(np.mean(top)) if top else None
    out['kendall_tau_progress_mean'] = float(np.mean(taus)) if taus else None
    coll = [r for r in rows if r['real_outcome'] in ('human_collision', 'body_overlap')]
    out['real_collisions'] = len(coll)
    out['p_collision_on_real_collisions'] = float(np.mean([r['p_collision'] for r in coll])) if coll else None
    out['p_collision_on_others'] = float(np.mean([r['p_collision'] for r in rows if r not in coll])) if rows else None
    return out


def ranges_episode(spec, steps=64, every=1):
    """Fast vs native range frames along a default-policy episode in the unchanged habitat.Env: per-frame mask
    equality, ulp-level agreement, every native human hit re-cast natively (privileged checks, validation only)."""
    from .episode import RealEpisode, default_query
    from .fastrange import compare
    ep = RealEpisode(spec, 'orca', 'native', range_sensor='fast')
    fast = ep.ranges
    hid = set().union(*fast.native.boundary.human_ids) if fast.native.boundary.human_ids else set()
    rows = []

    def check(e, row=None, rf=None):
        if e.task.steps % every:
            return
        t0 = time.perf_counter(); f = fast.measure(); tf = time.perf_counter()-t0
        t0 = time.perf_counter(); ref = fast.native.measure(); tn = time.perf_counter()-t0
        c = compare(f['range_image'], ref['range_image'])
        origin, rotation = fast.native.boundary.camera.camera_pose()
        dirs = fast.native.local@rotation.T
        _, fl = fast.cast(origin, dirs)
        nat = fast.native.batch.nearest_hits(origin, dirs, np.full(len(dirs), 10.), fast.ignored)
        hum = np.isin(nat['object_ids'], list(hid)) & nat['valid'] if hid else np.zeros(len(dirs), bool)
        c.update(fast_ms=1000*tf, native_ms=1000*tn, native_rays=int(f['fast']['native_rays']), human_hits=int(hum.sum()),
                 human_unflagged=int((hum & ((fl & 2) == 0)).sum()), step=int(e.task.steps))
        rows.append(c)
    check(ep)
    q = default_query()
    while ep.task.steps < steps and not ep.over:
        r = ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16, on_step=check)
        if r['steps'] == 0:
            break
    meta = dict(triangles=fast.index.triangles, local_frame=fast.index.local)
    ep.close()
    return dict(key=spec.key, frames=rows, **meta)


def summarize_ranges(recs):
    rows = [r for rec in recs for r in rec['frames']]
    return dict(episodes=len(recs), frames=len(rows), pixels=int(sum(r['pixels'] for r in rows)),
                mask_equal_frames=int(sum(r['mask_equal'] for r in rows)), mask_mismatch_pixels=int(sum(r['mask_mismatch'] for r in rows)),
                bit_equal_mean=float(np.mean([r['bit_equal'] for r in rows])), ulp_le2_mean=float(np.mean([r['ulp_le2'] for r in rows])),
                max_abs_m=float(max(r['max_abs'] for r in rows)), human_hits=int(sum(r['human_hits'] for r in rows)),
                human_hits_not_recast=int(sum(r['human_unflagged'] for r in rows)),
                native_ray_fraction_mean=float(np.mean([r['native_rays'] for r in rows])/65536.),
                fast_ms=dict(p50=float(np.median([r['fast_ms'] for r in rows])), p95=float(np.percentile([r['fast_ms'] for r in rows], 95))),
                native_ms=dict(p50=float(np.median([r['native_ms'] for r in rows])), p95=float(np.percentile([r['native_ms'] for r in rows], 95))),
                per_episode={rec['key']: dict(frames=len(rec['frames']), triangles=rec['triangles'], local_frame=rec['local_frame'],
                                              max_abs_m=float(max(r['max_abs'] for r in rec['frames'])),
                                              native_ray_fraction=float(np.mean([r['native_rays'] for r in rec['frames']])/65536.))
                             for rec in recs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fidelity', 'ranges'])
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', default='dev:0')
    ap.add_argument('--decisions', type=int, default=6)
    ap.add_argument('--config', default='k10')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    a = ap.parse_args()
    import os
    from eventwm_graph.querygen import suites
    from .core import WorldModelSandbox
    from .benchmark import parse_items
    os.environ.setdefault('QG_TAPE_CACHE', str(Path(__file__).resolve().parents[2]/'output/sandbox_cache/tapes'))
    Path(os.environ['QG_TAPE_CACHE']).mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'ranges':
        recs = []
        for suite, idx in parse_items(a.episodes):
            rec = ranges_episode(suites.load(suite)[idx], a.steps, a.every)
            recs.append(rec)
            (out/(rec['key'].replace('/', '_')+'.json')).write_text(json.dumps(rec, indent=1, default=float))
            print(rec['key'], 'frames', len(rec['frames']), 'max abs %.2g' % max(r['max_abs'] for r in rec['frames']), flush=True)
        summ = summarize_ranges(recs)
        (out/'summary.json').write_text(json.dumps(summ, indent=1, default=float))
        print(json.dumps(summ, indent=1, default=float))
        return
    sb = WorldModelSandbox.from_release(map_device='cuda:0')
    recs = []
    for suite, idx in parse_items(a.episodes):
        spec = suites.load(suite)[idx]
        rec = fidelity_episode(sb, spec, a.config, a.planner, a.dynamics, a.decisions)
        recs.append(rec)
        (out/(spec.key.replace('/', '_')+'.json')).write_text(json.dumps(rec, indent=1, default=float))
    summ = summarize(recs)
    (out/'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    sb.close()


if __name__ == '__main__':
    main()
