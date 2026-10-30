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
  percF / percCF -- perc / percC with the legal map the REAL robot holds at the end of the window (privileged: the
            map knowledge the real executor gains while moving; frozen at t in every other variant)
  kin    -- (open loop) the real command sequence integrated by the imagination's kinematics: motion-model error only

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
VARIANTS = ('nom3', 'nomL', 'perc', 'percC', 'percF', 'percCF')


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


def _kinematic_replay(pose, commands, dyn):
    from eventwm_graph.sandbox.imagine import advance
    out, p = [], np.asarray(pose, np.float64)
    for c in commands:
        p = advance(p, dyn, c if isinstance(c, str) else tuple(c))
        out.append(p)
    return np.asarray(out, np.float64).reshape(-1, 4)


def shifted_tape(tape, d):
    """The recorded human tape with every person ``d`` steps earlier (d > 0) or later (d < 0): exogenous timing noise."""
    from eventwm_graph.querygen.native import HumanTape
    n = len(tape.times)
    idx = np.clip(np.arange(n)+int(d), 1, n-1)      # tape rows are complete from index 1 on (index 0: reset placement)
    idx[0] = 0
    return HumanTape(tape.times, [tape.inputs[i] for i in idx], np.asarray(tape.positions)[idx], tape.speed_scales)


def probe_state(rt, gen, st, cfg, goal, N, scan=False, variants=VARIANTS, shifts=()):
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
            real_s=round(time.perf_counter()-t0, 2), imag={}, shift={})
        for dlt in shifts:
            # the same candidate for real with the people's timing shifted by dlt steps (prefix cache off: its entries
            # were simulated against the recorded tape)
            orig, pc = rt.tape, rt.prefix_cache
            sh = shifted_tape(orig, dlt)
            rt.tape = rt.sim.tape = sh
            rt.prefix_cache = 0
            rt._loaded = None
            try:
                ch = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
                fs = ch if ch.terminal else default_continue(rt, ch, goal, end)
                rp = np.asarray(fs.legal.poses[n0:], np.float64).reshape(-1, 4)
                cs = clearance_series(rp, np.asarray(sh.positions, np.float64), i0)
                rec['shift'][str(dlt)] = dict(outcome=fs.outcome or 'running', steps=int(fs.index-i0),
                                              minclr=float(cs.min()) if len(cs) else math.inf)
            finally:
                rt.tape = rt.sim.tape = orig
                rt.prefix_cache = pc
                rt._loaded = None
        kin = _kinematic_replay(st.legal.pose, fin.actions[len(st.actions):], dyn)
        kdiv = {}
        for lag in LAGS:
            if lag > min(len(real), len(kin)):
                break
            kdiv[lag] = float(math.hypot(real[lag-1, 0]-kin[lag-1, 0], real[lag-1, 2]-kin[lag-1, 2]))
        rec['kin'] = dict(n=len(kin), div=kdiv)
        fmap = frozen_map(fin.legal.map)
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
                if v.endswith('F'):
                    from .infofeat import _blocked_lookup
                    humans.blocked = _blocked_lookup(fin.legal.map)
            fid = v in ('percC', 'percCF')
            ro = imagine(st.legal, ex, dyn, humans, h, level=0, gmap=fmap if v.endswith('F') else gmap,
                         continuation=max(0, N-h), cont_query=goal, planner=cfg.planner,
                         cont_boundary=16 if fid else 0, cont_inherit=fid)
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


def decision_check(states, thr=0.25):
    """Offline one-decision check per variant: keep the default unless its imagined min clearance < thr, else take the
    candidate with the largest imagined clearance; real collision rate of the chosen candidates vs the default's and
    vs the best available (any candidate without a real collision)."""
    def bad(c):
        return c['real']['outcome'] in ('human_collision', 'stop_fail')
    rows = [s for s in states if s['cands']]
    n = len(rows)
    if not n:
        return
    dflt = [next((c for c in s['cands'] if c['label'] == s['default']), s['cands'][0]) for s in rows]
    print('decision check (%d states): default real collision %d, states where some candidate avoids it %d' % (
        n, sum(bad(c) for c in dflt), sum(bad(d) and any(not bad(c) for c in s['cands']) for s, d in zip(rows, dflt))))
    for v in VARIANTS:
        if not all(v in c['imag'] for s in rows for c in s['cands']):
            continue
        chosen = []
        for s, d in zip(rows, dflt):
            if d['imag'][v]['minclr'] >= thr:
                chosen.append(d)
            else:
                chosen.append(max(s['cands'], key=lambda c: (min(c['imag'][v]['minclr'], 3.), c is d)))
        fixed = sum(bad(d) and not bad(c) for c, d in zip(chosen, dflt))
        broke = sum(not bad(d) and bad(c) for c, d in zip(chosen, dflt))
        print('  %-6s chosen real collision %d  (fixes %d, breaks %d, deviations %d)' % (
            v, sum(bad(c) for c in chosen), fixed, broke, sum(c is not d for c, d in zip(chosen, dflt))))


def summary(path):
    recs = []
    states = []
    for f in sorted(Path(path).glob('*.json')):
        r = json.loads(f.read_text())
        for s in r.get('states', []):
            states.append(s)
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
    kl = ['%-6s' % 'kin']
    for lag in LAGS:
        d = [c['kin']['div'].get(str(lag)) for c in recs if 'kin' in c]
        d = [x for x in d if x is not None]
        if d:
            kl.append('d%d med %.2f p90 %.2f' % (lag, float(np.median(d)), float(np.percentile(d, 90))))
    print('  '.join(kl))
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
        ic, rc3 = np.minimum(im, 3.), np.minimum(rr, 3.)
        ok = np.isfinite(ic) & np.isfinite(rc3) & ((ic < 3.) | (rc3 < 3.))
        if ok.sum() > 2:
            print('        corr(imagined, real min clearance; clipped at 3 m, either < 3 m) %.3f  (n=%d)' % (
                float(np.corrcoef(ic[ok], rc3[ok])[0, 1]), int(ok.sum())))
    decision_check(states)
    sh = [c for c in recs if c.get('shift')]
    if sh:
        keys = sorted({k for c in sh for k in c['shift']}, key=int)
        for k in keys:
            rows = [c for c in sh if k in c['shift']]
            b = np.array([c['real']['outcome'] == 'human_collision' for c in rows])
            o = np.array([c['shift'][k]['outcome'] == 'human_collision' for c in rows])
            bs = np.array([c['real']['outcome'] == 'success' for c in rows])
            os_ = np.array([c['shift'][k]['outcome'] == 'success' for c in rows])
            print('people shifted %+d steps: collision flips %d / %d (base collisions %d -> %d), success flips %d' % (
                int(k), int((b != o).sum()), len(rows), int(b.sum()), int(o.sum()), int((bs != os_).sum())))
        rows = [c for c in sh if all(k in c['shift'] for k in keys)]
        anyflip = sum((c['real']['outcome'] == 'human_collision') != (c['shift'][k]['outcome'] == 'human_collision')
                      for c in rows for k in keys[:1]) if rows else 0
        robust = [c for c in rows if len({c['real']['outcome'] == 'human_collision'} |
                                         {c['shift'][k]['outcome'] == 'human_collision' for k in keys}) == 1]
        print('candidates whose collision outcome is the same under every shift: %d / %d' % (len(robust), len(rows)))


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
    ap.add_argument('--episodes', default='', help='comma-separated substrings: probe only matching episodes')
    ap.add_argument('--variants', default=','.join(VARIANTS), help="imagination variants ('' = none)")
    ap.add_argument('--shifts', default='', help='comma-separated human timing shifts in steps (real rollouts)')
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
        if d['config'] in keep and (not a.episodes or any(e in p.name for e in a.episodes.split(','))):
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
                    r = probe_state(rt, gen, st, cfg, goal, a.steps, scan=a.scan,
                                    variants=tuple(v for v in a.variants.split(',') if v),
                                    shifts=tuple(int(x) for x in a.shifts.split(',') if x))
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
