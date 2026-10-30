"""Closed-loop fidelity probe (ANALYSIS ONLY, privileged): how far is the Sandbox's imagination from what really happens?

Decision states of evaluation episodes (the floor by default) are reached in querygen's exact branch simulator.  From
each, every candidate is executed FOR REAL (its interval, then the default policy for ``--steps`` steps) and IMAGINED
by the Sandbox imagination against the TRUE future of every person (oraclebank.OracleBank 'all'), in variants:

  nom3   -- the imagination as the true-future multi-world selector used it: the planner perceives the people tracked
            at t, at every step, true future for 3 s then constant velocity; never anyone else
  nomL   -- as nom3 with the true future over the whole window
  perc   -- perception-faithful people (perceive.VisibleWorld): everyone, through the 90 deg / 10 m frustum with
            legal-map line of sight and the tracker's 2 s memory, from the IMAGINED pose
  percC  -- perc + continuation fidelity (decision boundaries every 16 steps, goal-query inheritance and re-issue,
            as the real default policy; imagine(cont_boundary, cont_inherit))

Per candidate: the real outcome / min body-rectangle clearance to the true people, and per variant the planar
divergence of the imagined robot from the real one at fixed lags and the imagined min clearance -- does the imagination
reproduce the closed loop, and does it predict which candidates collide?

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.fidelity --dir output/eval/floor_dev24/spool \
        --configs orca-native-k10 --out output/fidelity [--limit 12 --at 0.3,0.6 --steps 200 --shard i --shards n]
    python scripts/l3_run.py eventwm_graph.l3.fidelity --summary output/fidelity
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

LAGS = (8, 16, 32, 48, 64, 100, 150, 200, 300)
VARIANTS = ('nom3', 'nomL', 'perc', 'percC')


def clearance_series(poses, tape, i0):
    """Min body-rectangle clearance of robot poses [n, 4] (pose j after j+1 steps from tape index i0) to the true
    people, per pose."""
    from eventwm_graph.sandbox.imagine import rectangle_clearance
    out = np.full(len(poses), np.inf)
    for j, p in enumerate(poses):
        hp = tape[min(i0+1+j, len(tape)-1)]
        if not len(hp):
            continue
        hp = hp[:, [0, 2]]
        hp = hp[np.isfinite(hp).all(1)]
        out[j] = rectangle_clearance(p, hp)
    return out


def probe_state(rt, gen, st, cfg, goal, N, scan=False, variants=VARIANTS):
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.querygen.executor import QueryExecution
    from eventwm_graph.sandbox.imagine import imagine, NominalHumans, frozen_map, new_execution
    from .oraclebank import OracleBank
    from .perceive import VisibleWorld
    from .cfonline import default_continue
    qs, _ = gen.propose(st.legal, st.execution)
    if scan:
        from .env import scan_queries
        qs = list(qs)+scan_queries(st.legal, gen.dyn, [q.label for q in qs])
    H = type('H', (), {'horizon': 16})()
    tape = np.asarray(rt.tape.positions, np.float64)
    i0 = int(st.index)
    end = min(MAX_EPISODE_STEPS, i0+N)
    n0 = len(st.legal.poses)
    T = N+8
    bank3 = OracleBank(tape, i0, st.legal, 'all')
    bankL = OracleBank(tape, i0, st.legal, 'all', horizon_steps=T+1, long_steps=T)
    from eventwm_graph.querygen.dynamics import FAMILIES
    dyn = FAMILIES[cfg.dynamics]
    gmap = frozen_map(st.legal.map)
    labels = [q.label for q in qs]
    out = []
    for q in qs:
        cont = q.kind == 'continue'
        h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-i0)
        t0 = time.perf_counter()
        child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
        fin = child if child.terminal else default_continue(rt, child, goal, end)
        real = np.asarray(fin.legal.poses[n0:], np.float64).reshape(-1, 4)
        rc = clearance_series(real, tape, i0)
        rec = dict(label=q.label, kind=q.kind, h=int(h), real=dict(
            outcome=fin.outcome or 'running', steps=int(fin.index-i0), n=len(real),
            minclr=float(rc.min()) if len(rc) else math.inf, minclr_step=int(np.argmin(rc)) if len(rc) else -1),
            real_s=round(time.perf_counter()-t0, 2), imag={})
        for v in variants:
            t1 = time.perf_counter()
            if cont:
                ex = st.execution.copy()
                ex.decision_boundary()
            else:
                ex = new_execution(q, cfg.planner, dyn)
                if getattr(QueryExecution, 'INHERIT', False) and st.execution is not None:
                    ex.inherit(st.execution)
            if v == 'nom3':
                humans = NominalHumans(st.legal.tracks, st.legal.time, T, bank=bank3, mode='world:0')
            elif v == 'nomL':
                humans = NominalHumans(st.legal.tracks, st.legal.time, T, bank=bankL, mode='world:0')
            else:
                humans = VisibleWorld(st.legal, bankL, 0, T)
            fid = v.endswith('C')
            ro = imagine(st.legal, ex, dyn, humans, h, level=0, gmap=gmap, continuation=max(0, N-h), cont_query=goal,
                         planner=cfg.planner, cont_boundary=16 if fid else 0, cont_inherit=fid)
            c = getattr(ro, 'cont', None)
            im = np.concatenate([ro.poses[1:], c]) if c is not None and len(c) else np.asarray(ro.poses[1:])
            im = np.asarray(im, np.float64).reshape(-1, 4)
            ic = clearance_series(im, tape, i0)
            div = {}
            for lag in LAGS:
                if lag > len(real):
                    break
                j = min(lag, len(im))-1
                if j < 0:
                    break
                div[lag] = float(math.hypot(real[lag-1, 0]-im[j, 0], real[lag-1, 2]-im[j, 2]))
            rec['imag'][v] = dict(n=len(im), ended=ro.ended_by, status=ro.status, stop=bool(ro.stop),
                                  minclr=float(ic.min()) if len(ic) else math.inf,
                                  minclr_step=int(np.argmin(ic)) if len(ic) else -1, div=div,
                                  s=round(time.perf_counter()-t1, 2))
        out.append(rec)
    return dict(step=i0, labels=labels, default='continue' if 'continue' in labels else 'goal', cands=out)


def summary(path):
    recs = []
    for f in sorted(Path(path).glob('*.json')):
        r = json.loads(f.read_text())
        for s in r.get('states', []):
            for c in s['cands']:
                recs.append(c)
    print('states x candidates: %d' % len(recs))
    if not recs:
        return
    coll = np.array([c['real']['outcome'] == 'human_collision' for c in recs])
    rmin = np.array([c['real']['minclr'] for c in recs])
    print('real: collisions %d, success %d, running %d, timeout %d; real min clearance < 0.25 m: %d' % (
        coll.sum(), sum(c['real']['outcome'] == 'success' for c in recs),
        sum(c['real']['outcome'] == 'running' for c in recs), sum(c['real']['outcome'] == 'timeout' for c in recs),
        int((rmin < 0.25).sum())))
    for v in VARIANTS:
        rows = [c for c in recs if v in c['imag']]
        if not rows:
            continue
        line = ['%-6s' % v]
        for lag in LAGS:
            d = [c['imag'][v]['div'][str(lag)] if str(lag) in c['imag'][v]['div'] else c['imag'][v]['div'].get(lag)
                 for c in rows]
            d = [x for x in d if x is not None]
            if d:
                line.append('d%d med %.2f p90 %.2f' % (lag, float(np.median(d)), float(np.percentile(d, 90))))
        print('  '.join(line))
        im = np.array([c['imag'][v]['minclr'] for c in rows])
        rc = np.array([c['real']['outcome'] == 'human_collision' for c in rows])
        rr = np.array([c['real']['minclr'] for c in rows])
        for thr in (0.25, 0.5):
            pred = im < thr
            tp, fp = int((pred & rc).sum()), int((pred & ~rc).sum())
            fn, tn = int((~pred & rc).sum()), int((~pred & ~rc).sum())
            print('        imagined clearance < %.2f vs real collision: TP %d FP %d FN %d TN %d' % (thr, tp, fp, fn, tn))
        if rc.any() and (~rc).any():
            # AUC of (-imagined clearance) as a predictor of a real collision
            a, b = -im[rc], -im[~rc]
            auc = float(((a[:, None] > b[None]).sum()+0.5*(a[:, None] == b[None]).sum())/(len(a)*len(b)))
            print('        AUC(imagined clearance -> real collision) %.3f' % auc)
        ok = np.isfinite(im) & np.isfinite(rr)
        if ok.sum() > 2:
            print('        corr(imagined, real min clearance) %.3f  (n=%d)' % (float(np.corrcoef(im[ok], rr[ok])[0, 1]),
                                                                        int(ok.sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='output/eval/floor_dev24/spool')
    ap.add_argument('--out', default='output/fidelity')
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--at', default='0.3,0.6')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--summary', default=None)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
        return
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.episode import generator, default_query
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = set(a.configs.split(','))
    files = []
    for p in sorted(Path(a.dir).glob('*.ep')):
        d = read_episode(p)
        if d['config'] in keep:
            files.append(p)
    files = files[:a.limit]
    fracs = [float(x) for x in a.at.split(',')]
    for i, p in enumerate(files):
        if i % a.shards != a.shard or (out/(p.name+'.json')).exists():
            continue
        d = read_episode(p)
        cfg = ConfigSpec.parse(d['config'])
        gen = generator(cfg.gen, cfg.dynamics)
        rt = EpisodeRuntime(EpisodeSpec(**d['spec']), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
        rt.sim.check_static = False
        rt.prefix_cache = 48
        sensorcache.attach(rt)
        goal = default_query()
        decs = d['metrics']['decision_list']
        targets = sorted({int(f*(len(decs)-1)) for f in fracs})
        rec = dict(episode=p.name, config=d['config'], outcome=d['metrics']['outcome'], decisions=len(decs), states=[])
        t0 = time.time()
        st = rt.root
        try:
            for j, dec in enumerate(decs):
                if st.terminal or j > max(targets):
                    break
                if j in targets:
                    r = probe_state(rt, gen, st, cfg, goal, a.steps, scan=a.scan)
                    r['decision'] = j
                    rec['states'].append(r)
                    print(json.dumps(dict(episode=p.name, decision=j, step=r['step'], n=len(r['cands']))), flush=True)
                st = rt.run(st, None if dec['cont'] else _query(dec), horizon=int(dec['horizon']),
                            cont=bool(dec['cont']), geodesic=False)
        finally:
            try:
                sensorcache.save(rt)
            except Exception:
                pass
            rt.close()
        rec['seconds'] = round(time.time()-t0)
        (out/(p.name+'.json')).write_text(json.dumps(rec))


if __name__ == '__main__':
    main()
