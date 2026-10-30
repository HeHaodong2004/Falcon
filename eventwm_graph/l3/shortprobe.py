"""Short-horizon ranking probe (ANALYSIS ONLY; the ground truth is privileged): selector ``shortprobe``.

At every decision of an evaluation episode (the episode itself follows the floor: the default candidate is executed),
every candidate is

* executed FOR REAL in the exact branch simulator: its interval, then the default policy, to ``W`` = 64 steps from
  the decision (prefix-cached; the legal stream is bypassed so Layer 2 never sees these branches), and
* IMAGINED over the same 64 steps (interval + default continuation with decision boundaries every 16 steps and
  goal-query inheritance, as the real default policy) by Sandbox variants:
    rel    the released Sandbox's people: Stage 1 MEAN future of the people tracked at t, always visible  (legal)
    cam6   camera model (perceive.VisibleWorld) re-imagined in each of the 6 heaviest Stage 1 worlds   (legal)
    camT   camera model with the TRUE future of every person                                         (privileged)
    camTD  camT + progressive discovery of the real robot's map at the end of the window             (privileged)
  all on the legal map frozen at t (camTD: discovered along the imagined path).

Per candidate and variant: min body-rectangle clearance to the variant's people within 16 / 32 / 64 steps (cam6: per
world), the legal route cost-to-go (decision-time map) of the pose reached at 16 / 32 / 64 steps, STOP / end flags,
and the imagined-vs-real planar divergence at 8 / 16 / 32 / 48 / 64 steps.  Real: outcome and step of a collision /
success within the window, min clearance to the true people, cost-to-go at 16 / 32 / 64.

Output: one JSON per episode in ``$L3_PROBE_DIR``.  Summary: ``python -m eventwm_graph.l3.shortprobe <dir>``.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

W_MAX = 64
WINDOWS = (16, 32, 64)
LAGS = (8, 16, 32, 48, 64)
N_WORLDS = 6


def _min_upto(series, w):
    s = series[:w]
    return float(np.min(s)) if len(s) else math.inf


def _ctg_at(field, poses, w):
    if not len(poses):
        return None
    p = poses[min(w, len(poses))-1]
    return float(field.at(p[0], p[2]))


def _world_clearance(poses, HPw):
    """Rect clearance [n] of poses to one world's people HPw [T, N, 2] (row j = after j+1 steps)."""
    from .heuristics import _rect_clearance
    n = min(len(poses), len(HPw))
    if not n or not HPw.shape[1]:
        return np.full(len(poses), np.inf)
    d = _rect_clearance(poses[:n], HPw[:n])
    d = np.where(np.isfinite(d), d, np.inf).min(axis=1)
    if n < len(poses):
        d = np.concatenate([d, np.full(len(poses)-n, np.inf)])
    return d


def probe_decision(ep, d, sb):
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.executor import QueryExecution
    from eventwm_graph.sandbox.imagine import imagine, NominalHumans, frozen_map, new_execution
    from eventwm_graph.sandbox.episode import default_query
    from .cfonline import default_continue
    from .compiler import LegalField
    from .fidelity import clearance_series
    from .oraclebank import OracleBank
    from .perceive import VisibleWorld, Discovery
    from .infofeat import _blocked_lookup
    t0 = time.perf_counter()
    rt = ep.rt
    st = ep.ep.state
    cfg = ep.cfg
    dyn = FAMILIES[cfg.dynamics]
    goal = default_query()
    H = type('H', (), {'horizon': 16})()
    i0 = int(st.index)
    n0 = len(st.legal.poses)
    end = min(MAX_EPISODE_STEPS, i0+W_MAX)
    tape = np.asarray(rt.tape.positions, np.float64)
    field = LegalField(st.legal)
    gmap = frozen_map(st.legal.map)
    bank = d.result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    worlds = np.argsort(-W)[:N_WORLDS] if len(np.unique(W)) > 1 else np.arange(min(N_WORLDS, len(W)))
    ww = W[worlds]/W[worlds].sum()
    rel = np.arange(1, W_MAX+1)*0.048
    HP = None
    if len(getattr(bank, 'track_keys', ())):
        inside = rel[rel <= float(bank.times[-1])+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)
        if len(inside) < W_MAX and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/0.048
            extra = np.arange(1, W_MAX-len(inside)+1)*0.048
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    truth = OracleBank(tape, i0, st.legal, 'all', horizon_steps=W_MAX+1, long_steps=W_MAX)
    # exact branches: bypass the legal stream (Layer 2 must not see them)
    raw, hooked = ep.ep._observe, rt.sensor.observe
    rt.sensor.observe = raw
    cands = []
    try:
        reals = []
        for q in d.queries:
            cont = q.kind == 'continue'
            h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-i0)
            child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
            fin = child if child.terminal else default_continue(rt, child, goal, end)
            real = np.asarray(fin.legal.poses[n0:], np.float64).reshape(-1, 4)
            reals.append((q, h, fin, real))
    finally:
        rt.sensor.observe = hooked
    for k, (q, h, fin, real) in enumerate(reals):
        rc = clearance_series(real, tape, i0)
        steps = int(fin.index-i0)
        rec = dict(label=q.label, kind=q.kind, h=int(h),
                   real=dict(outcome=fin.outcome or 'running', steps=steps,
                             clr={w: _min_upto(rc, w) for w in WINDOWS}, ctg={w: _ctg_at(field, real, w) for w in WINDOWS}),
                   imag={})
        cand = d.candidates[k]
        for v in os.environ.get('L3_PROBE_VARIANTS', 'rel,cam6,camT,camTD').split(','):
            runs = []
            if v == 'cam6':
                specs = [(int(w), float(x)) for w, x in zip(worlds, ww)] if HP is not None else []
            else:
                specs = [(None, 1.)]
            for w, wt in specs:
                ex, _ = sb._execution(cand)
                if isinstance(ex, Exception):
                    break
                vmap = gmap
                if v == 'rel':
                    hum = NominalHumans(st.legal.tracks, st.legal.time, W_MAX, bank=bank, mode='mean')
                elif v == 'cam6':
                    hum = VisibleWorld(st.legal, bank, w, W_MAX)
                else:
                    hum = VisibleWorld(st.legal, truth, 0, W_MAX)
                    if v == 'camTD':
                        vmap = st.legal.map.copy()
                        hum.blocked = _blocked_lookup(fin.legal.map)
                        hum.discovery = Discovery(vmap, fin.legal.map)
                fid = v != 'rel'
                ro = imagine(st.legal, ex, dyn, hum, h, level=int(cand.level), gmap=vmap,
                             continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner,
                             cont_boundary=16 if fid else 0, cont_inherit=fid)
                c = getattr(ro, 'cont', None)
                im = np.concatenate([ro.poses[1:], c]) if c is not None and len(c) else np.asarray(ro.poses[1:])
                im = np.asarray(im, np.float64).reshape(-1, 4)
                if v == 'cam6':
                    ic = _world_clearance(im, HP[w])
                elif v == 'rel':
                    hp = np.asarray(hum.positions[1:, :, :], np.float64)[:, :, [0, 2]]
                    ic = _world_clearance(im, hp)
                else:
                    ic = clearance_series(im, tape, i0)
                div = {}
                for lag in LAGS:
                    if lag > len(real) or not len(im):
                        break
                    j = min(lag, len(im))-1
                    div[lag] = float(math.hypot(real[lag-1, 0]-im[j, 0], real[lag-1, 2]-im[j, 2]))
                runs.append(dict(w=wt, n=len(im), stop=bool(ro.stop), ended=ro.ended_by,
                                 near_goal=bool(len(im) and math.hypot(im[-1, 0]-st.legal.goal[0], im[-1, 2]-st.legal.goal[2]) <= 0.3),
                                 clr={w2: _min_upto(ic, w2) for w2 in WINDOWS},
                                 ctg={w2: _ctg_at(field, im, w2) for w2 in WINDOWS}, div=div))
            rec['imag'][v] = runs
        cands.append(rec)
    return dict(step=i0, decision=int(ep.decisions), labels=list(d.labels), default=d.labels[d.default],
                ctg0=float(field.start), cands=cands, seconds=round(time.perf_counter()-t0, 1))


def write(path, data):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, (np.floating,)):
            return conv(float(x))
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(conv(data)))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ summary
def _key_real(c, w):
    r = c['real']
    if r['outcome'] in ('human_collision', 'stop_fail') and r['steps'] <= w:
        return (0., r['steps'])
    if r['outcome'] == 'success' and r['steps'] <= w:
        return (3., -r['steps'])
    v = r['ctg'][str(w)]
    return (2., -(v if v is not None else 99.))


def _key_imag(runs, w, thr):
    """Imagined key over worlds: expected category (contact 0, at goal 3, running 2) + weighted cost-to-go."""
    if not runs:
        return (-1., 0.)
    cat, ctg = 0., 0.
    for r in runs:
        clr = r['clr'][str(w)]
        hit = clr is not None and clr < thr
        if hit:
            c = 0.
        elif r['stop'] and r['n'] <= w:
            c = 3. if r['near_goal'] else 0.
        else:
            c = 2.
        cat += r['w']*c
        v = r['ctg'][str(w)]
        ctg += r['w']*(v if v is not None else 99.)
    return (cat, -ctg)


def summarize(path, thr=0.2):
    states = []
    for f in sorted(Path(path).glob('*.json')):
        states += json.loads(f.read_text()).get('probe', [])
    print('%s: %d decision states' % (path, len(states)))
    variants = ('rel', 'cam6', 'camT', 'camTD')
    for w in WINDOWS:
        print('-- window %d steps (%.2f s); imagined contact = rect clearance < %.2f m' % (w, w*0.048, thr))
        for v in variants + ('default',):
            n = cat_ok = crit = fixes = breaks = dev = 0
            ctg_reg = []
            for s in states:
                cs = s['cands']
                if not cs:
                    continue
                kr = [_key_real(c, w) for c in cs]
                best = max(kr)
                di = s['labels'].index(s['default']) if s['default'] in s['labels'] else 0
                if v == 'default':
                    pick = di
                else:
                    ki = [_key_imag(c['imag'].get(v, []), w, thr) for c in cs]
                    top = max(ki)
                    pick = di if ki[di] >= top else max(range(len(ki)), key=lambda i: ki[i])
                n += 1
                cat_ok += int(kr[pick][0] == best[0])
                if kr[pick][0] == best[0] == 2.:
                    ctg_reg.append(float(best[1]-kr[pick][1]))
                dev += int(pick != di)
                if kr[di][0] == 0. and best[0] > 0.:
                    crit += 1
                    fixes += int(kr[pick][0] > 0.)
                if kr[di][0] > 0. and kr[pick][0] == 0.:
                    breaks += 1
            print('   %-8s exact category = best %.3f  cost-to-go regret (same category) %.3f m  critical %d: fixed %d  '
                  'breaks %d  deviations %.2f' % (v, cat_ok/max(n, 1), float(np.mean(ctg_reg)) if ctg_reg else 0.,
                                                  crit, fixes, breaks, dev/max(n, 1)))
    # divergence
    print('-- imagined vs real divergence (m): median / p90')
    for v in variants:
        line = ['   %-6s' % v]
        for lag in LAGS:
            d = [r['div'][str(lag)] for s in states for c in s['cands'] for r in c['imag'].get(v, []) if str(lag) in r['div']]
            if d:
                line.append('d%d %.2f/%.2f' % (lag, float(np.median(d)), float(np.percentile(d, 90))))
        print('  '.join(line))
    # collision prediction AUC within each window
    print('-- AUC of imagined clearance for a real collision within the window')
    for w in WINDOWS:
        line = ['   w%-3d' % w]
        y = np.array([c['real']['outcome'] == 'human_collision' and c['real']['steps'] <= w for s in states for c in s['cands']])
        for v in variants:
            x = np.array([-min([r['clr'][str(w)] if r['clr'][str(w)] is not None else 99. for r in c['imag'].get(v, [])] or [99.])
                          if v != 'cam6' else
                          -sum(r['w']*min(r['clr'][str(w)] if r['clr'][str(w)] is not None else 99., 3.) for r in c['imag'].get(v, []))
                          for s in states for c in s['cands']])
            if y.any() and (~y).any():
                pos, neg = x[y], x[~y]
                auc = ((pos[:, None] > neg[None]).mean()+0.5*(pos[:, None] == neg[None]).mean())
                line.append('%s %.3f' % (v, auc))
        print('  '.join(line), ' (real collisions %d / %d)' % (int(y.sum()), len(y)))


if __name__ == '__main__':
    summarize(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.2)
