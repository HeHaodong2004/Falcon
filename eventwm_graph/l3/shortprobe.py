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
    camTk  TRUE people through the EMULATED camera + tracker (perceive L3_CAM_TRACKER: the decision's camera rig,
           3-D frustum, legal-map line of sight, the real LegalHumanTracker seeded at t)            (privileged people)
    camTkM camTk + the real robot's map stream (as camM)                                            (privileged)
    camTDo camTD revealed as the legal sensor certifies (perceive.ObserverReveal)                    (privileged)
    camS2  camT + Stage 2's predicted map (LEGAL map stream; people still TRUE) revealed through the imagined camera:
           one Stage 2 call on every candidate's frozen-map imagined path; 'camS2' reads the selector options
           (L3_S2_PB / L3_S2_PF thresholds, L3_S2_REVEAL, L3_S2_ROUNDS, union map); 'camS2<tag>' (tag: lower-case
           letters / digits) reads L3_PROBE_S2_<tag> = 'pb=..,pf=..,map=union|own,reveal=frustum|observer,rounds=..
           [,free_min=..,free_max=..,wall_max=..,rays=..,sway=..]' (unset keys: the release behaviour; rounds=2:
           Stage 2 again on every candidate's paths imagined with round 1's map stream, increments united) or
           'cond=real' (PRIVILEGED diagnostic: Stage 2 conditioned on the real branch paths).  Each run also records
           the predicted map against the candidate's real end-of-window map ('s2': cell counts, privileged evaluation).
  all on the legal map frozen at t (camTD: discovered along the imagined path).

Track agreement (analysis): for the camera-model / replayed-tracker variants, the imagined track set at every step is
compared with the real branch's track log at the same step ('trk': closed loop, imagined pose) and -- for the camera
models -- with the same perception model replayed along the REAL branch's poses ('trk_ol': open loop, isolates the
perception model from the imagined planner); per window 16 / 32 / 64 steps: summed counts (see ``track_agreement``).

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
TRK_GATE = 0.3       # m: an imagined and a real track of the same person (planar distance of their last detections)
TRK_NEAR = 2.5       # m: 'near' tracks -- within this distance of the respective robot (real / imagined)


def _match_tracks(I, R, gate=TRK_GATE):
    """Greedy planar matching of two track lists within ``gate``: [(i, r, distance)]."""
    pairs = []
    for i, a in enumerate(I):
        for r, b in enumerate(R):
            dd = math.hypot(float(a['position'][0])-float(b['position'][0]), float(a['position'][2])-float(b['position'][2]))
            if dd <= gate:
                pairs.append((dd, i, r))
    ui, ur, out = set(), set(), []
    for dd, i, r in sorted(pairs):
        if i in ui or r in ur:
            continue
        ui.add(i); ur.add(r)
        out.append((i, r, dd))
    return out


def _vel(t):
    v = t.get('estimated_velocity')
    return None if (v is None or not t.get('velocity_valid')) else (float(v[0]), float(v[2]))


def track_agreement(rec, track_log, b0, real_pose_at, imag_pose_at):
    """ANALYSIS: step-by-step agreement of an imagined track set with the real tracker's (``track_log[b0 + k]``),
    k >= 1, summed per window (16 / 32 / 64 steps): steps; exact (equal sets: equal sizes, every track matched within
    TRK_GATE with the same velocity_valid); nr / ni / nm (real / imagined / matched tracks); vv (matched, same
    velocity_valid); perr (summed position error of matched); vn / verr (matched with both velocities valid, summed
    |dv|); the same for tracks within TRK_NEAR of the respective robot (nnr / nni / nnm / nexact).  'ex': per-step
    exact flags ('1' / '0', '-' = no real record)."""
    out = {w: dict(steps=0, exact=0, nr=0, ni=0, nm=0, vv=0, perr=0., vn=0, verr=0., nnr=0, nni=0, nnm=0, nexact=0)
           for w in WINDOWS}
    flags = {}
    for k, I in rec:
        k = int(k)
        if k < 1 or k in flags:
            continue
        if b0+k >= len(track_log):
            flags[k] = '-'
            continue
        R = track_log[b0+k][1]
        M = _match_tracks(I, R)
        vv = sum(int(bool(I[i].get('velocity_valid')) == bool(R[r].get('velocity_valid'))) for i, r, _ in M)
        exact = int(len(R) == len(I) == len(M) == vv)
        dv = [math.hypot(a[0]-b[0], a[1]-b[1]) for a, b in ((_vel(I[i]), _vel(R[r])) for i, r, _ in M)
              if a is not None and b is not None]
        pr, pi = real_pose_at(k), imag_pose_at(k)
        nR = [t for t in R if math.hypot(float(t['position'][0])-pr[0], float(t['position'][2])-pr[2]) <= TRK_NEAR]
        nI = [t for t in I if math.hypot(float(t['position'][0])-pi[0], float(t['position'][2])-pi[2]) <= TRK_NEAR]
        Mn = _match_tracks(nI, nR)
        nex = int(len(nR) == len(nI) == len(Mn) and
                  all(bool(nI[i].get('velocity_valid')) == bool(nR[r].get('velocity_valid')) for i, r, _ in Mn))
        flags[k] = str(exact)
        for w in WINDOWS:
            if k > w:
                continue
            o = out[w]
            o['steps'] += 1; o['exact'] += exact
            o['nr'] += len(R); o['ni'] += len(I); o['nm'] += len(M); o['vv'] += vv
            o['perr'] += float(sum(d for _, _, d in M)); o['vn'] += len(dv); o['verr'] += float(sum(dv))
            o['nnr'] += len(nR); o['nni'] += len(nI); o['nnm'] += len(Mn); o['nexact'] += nex
    out['ex'] = ''.join(flags.get(k, '-') for k in range(1, (max(flags) if flags else 0)+1))
    return out


def _min_upto(series, w):
    s = series[:w]
    return float(np.min(s)) if len(s) else math.inf


def _centre_series(poses, tape, i0):
    """Min centre distance of robot poses [n, 4] (pose j after j+1 steps from tape index i0) to the true people."""
    out = np.full(len(poses), np.inf)
    for j, p in enumerate(poses):
        hp = tape[min(i0+1+j, len(tape)-1)]
        if len(hp):
            d = np.hypot(hp[:, 0]-p[0], hp[:, 2]-p[2])
            d = d[np.isfinite(d)]
            if len(d):
                out[j] = float(d.min())
    return out


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


def _s2_spec(tag):
    """Settings of a camS2 variant: map threshold / union-or-own / reveal model / Stage 2 rounds (see the module
    docstring)."""
    if not tag:
        from .perceive import s2_thresholds
        pb, pf = s2_thresholds()
        return dict(pb=pb, pf=pf, map='union', reveal=os.environ.get('L3_S2_REVEAL', 'frustum'),
                    rounds=int(os.environ.get('L3_S2_ROUNDS', '1')), cond='imag')
    raw = os.environ.get('L3_PROBE_S2_%s' % tag)
    if raw is None:
        raise ValueError('variant camS2%s needs L3_PROBE_S2_%s' % (tag, tag))
    spec = dict(pb=None, pf=None, map='union', reveal='frustum', rounds=1, cond='imag')
    for kv in filter(None, raw.split(',')):
        k, v = kv.split('=')
        k, v = k.strip(), v.strip()
        spec[k] = (float(v) if k in ('pb', 'pf', 'free_min', 'free_max', 'wall_max', 'rays', 'sway') else
                   int(v) if k == 'rounds' else v)
    if (spec['map'] not in ('union', 'own') or spec['reveal'] not in ('frustum', 'observer') or spec['rounds'] < 1
            or spec['cond'] not in ('imag', 'real')):
        raise ValueError('bad camS2%s settings %r' % (tag, raw))
    return spec


def _s2_key(spec):
    return tuple(sorted((k, v) for k, v in spec.items()))


def _s2_truth(legal_map, deltas, k, spec, cache, dkey='r1'):
    """The predicted truth map of a camS2 setting for candidate k (cached per decision); ``deltas``: per candidate the
    list of its Stage 2 increments (one per round), ``dkey`` names that set."""
    from .perceive import predicted_truth, predicted_truth_union
    own = spec['map'] == 'own'
    key = (spec['pb'], spec['pf'], k if own else None, dkey)
    if key not in cache:
        if own and len(deltas[k]) == 1:
            cache[key] = predicted_truth(legal_map, deltas[k][0], spec['pb'], spec['pf'])
        else:
            cache[key] = predicted_truth_union(legal_map, deltas[k] if own else [x for ds in deltas for x in ds],
                                               spec['pb'], spec['pf'])
    return cache[key]


def _s2_reveal(work, truth, spec):
    from .perceive import Discovery, ObserverReveal
    if spec['reveal'] == 'observer':
        return ObserverReveal(work, truth, **{k: spec[k] for k in ('free_min', 'free_max', 'wall_max', 'rays', 'sway')
                                              if k in spec})
    return Discovery(work, truth)


def _keys_state(g, keys):
    i, j = keys[:, 0]-g.ox, keys[:, 1]-g.oz
    ok = (i >= 0) & (i < g.state.shape[0]) & (j >= 0) & (j < g.state.shape[1])
    out = np.zeros(len(keys), np.uint8)
    out[ok] = g.state[i[ok], j[ok]]
    return out


def map_stats(truth, legal_map, final):
    """PRIVILEGED evaluation of a predicted truth map: its cells UNKNOWN at t against the real branch's end-of-window
    map ``final``.  rB / rF: cells the real robot certified BLOCKED / FREE in the window; pB / pF: predicted BLOCKED /
    FREE cells; BB / FF: both agree; FB: predicted BLOCKED, really FREE (false wall); BF: predicted FREE, really
    BLOCKED (false opening); B0 / F0: predicted, not certified in the real window."""
    ii, jj = np.nonzero(final.state != 0)
    keys = np.stack([ii+final.ox, jj+final.oz], 1).astype(np.int64)
    new = _keys_state(legal_map, keys) == 0
    real = final.state[ii[new], jj[new]]
    pred = _keys_state(truth, keys[new])
    ti, tj = np.nonzero(truth.state != 0)
    tk = np.stack([ti+truth.ox, tj+truth.oz], 1).astype(np.int64)
    tnew = _keys_state(legal_map, tk) == 0
    ps = truth.state[ti[tnew], tj[tnew]]
    seen = _keys_state(final, tk[tnew]) != 0
    return dict(rB=int((real == 2).sum()), rF=int((real == 1).sum()), pB=int((ps == 2).sum()), pF=int((ps == 1).sum()),
                BB=int(((real == 2) & (pred == 2)).sum()), FF=int(((real == 1) & (pred == 1)).sum()),
                FB=int(((real == 1) & (pred == 2)).sum()), BF=int(((real == 2) & (pred == 1)).sum()),
                B0=int(((ps == 2) & ~seen).sum()), F0=int(((ps == 1) & ~seen).sum()))


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
    from .perceive import VisibleWorld, Discovery, TimedReveal, ReplayTracks, ObserverReveal
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
    # L3_PROBE_VARIANTS=none (default off): the exact labels only -- no imagined variant, no Sandbox / people bank
    variants = [v for v in os.environ.get('L3_PROBE_VARIANTS', 'rel,cam6,camT,camTD').split(',') if v != 'none']
    bank = getattr(d.result, 'humans', None)             # None: a --blind episode (exact labels only)
    if bank is None and variants:
        raise ValueError('shortprobe in a --blind episode: set L3_PROBE_VARIANTS=none')
    worlds = ww = np.zeros(0)
    if bank is not None:
        W = np.asarray(bank.weights, np.float64)
        W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
        worlds = np.argsort(-W)[:N_WORLDS] if len(np.unique(W)) > 1 else np.arange(min(N_WORLDS, len(W)))
        ww = W[worlds]/W[worlds].sum()
    rel = np.arange(1, W_MAX+1)*0.048
    HP = None
    if bank is not None and len(getattr(bank, 'track_keys', ())):
        inside = rel[rel <= float(bank.times[-1])+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)
        if len(inside) < W_MAX and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/0.048
            extra = np.arange(1, W_MAX-len(inside)+1)*0.048
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    truth = OracleBank(tape, i0, st.legal, 'all', horizon_steps=W_MAX+1, long_steps=W_MAX)
    info = expo_state(st.legal, bank) if (os.environ.get('L3_PROBE_EXPO') == '1' and bank is not None) else None
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
    trk_on = os.environ.get('L3_PROBE_TRACKS') == '1'          # analysis: track agreement records (default off)
    b0 = len(st.legal.track_log)-1                               # track log index of the decision (step 0)
    seed = None
    if any(v.startswith('camTk') for v in variants):
        # the emulated tracker's seed vs the real tracker object of the runtime state at t (must be identical)
        from .perceive import seeded_tracker
        real_tr = st.sensor[0] if isinstance(getattr(st, 'sensor', None), (tuple, list)) else None
        s = seeded_tracker(st.legal)
        seed = dict(next_id=[int(s.next_id), None if real_tr is None else int(real_tr.next_id)],
                    timestamp=[float(s.timestamp), None if real_tr is None else float(real_tr.timestamp)],
                    tracks_equal=bool(real_tr is not None and s.tracks == real_tr.tracks), n=len(s.tracks))
    s2 = None                    # Stage 2 increments, one per candidate (camS2 variants)
    s2cache = {}
    if any(v.startswith('camS2') for v in variants):
        # LEGAL predicted map (Layer 2 / Stage 2): every candidate imagined on the frozen map (TRUE people, as camT),
        # one batched Stage 2 call on those paths; per camS2 setting one predicted local map (union of every
        # candidate's increment, perceive.predicted_truth_union) or the candidate's own (predicted_truth), revealed
        # through the imagined camera (Discovery, or ObserverReveal)
        from eventwm_graph.layer2.state import trajectory_from_poses
        trajs = []
        for k, (q, h, fin, real) in enumerate(reals):
            ex, _ = sb._execution(d.candidates[k])
            if isinstance(ex, Exception):
                trajs.append(trajectory_from_poses(np.asarray([st.legal.pose], np.float64), st.legal.time))
                continue
            ro = imagine(st.legal, ex, dyn, VisibleWorld(st.legal, truth, 0, W_MAX), h, level=int(d.candidates[k].level),
                         gmap=gmap, continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner,
                         cont_boundary=16, cont_inherit=True, physics='core')
            c = getattr(ro, 'cont', None)
            P = np.concatenate([np.asarray(ro.poses, np.float64), np.asarray(c, np.float64)]) if c is not None and len(c) else np.asarray(ro.poses, np.float64)
            trajs.append(trajectory_from_poses(P, st.legal.time))
        try:
            pred = sb.l2.predict(ep.last_state, trajs, materialize=False, worlds=sb.config.worlds, seed=sb.config.seed)
            s2 = [c.map for c in pred.candidates]
        except Exception as exc:
            print('shortprobe: Stage 2 failed (%s)' % (exc,), flush=True)
    # camS2 settings with more Stage 2 rounds (LEGAL: every candidate re-imagined with the previous rounds' map stream,
    # Stage 2 again on those paths, the rounds' increments united) or conditioned on the REAL branch paths (cond=real:
    # PRIVILEGED diagnostic of the conditioning path): their per-candidate increments, keyed by the setting
    s2sets = {}
    if s2 is not None:
        from eventwm_graph.layer2.state import trajectory_from_poses
        for v in variants:
            vb = v[:-2] if v.endswith('Pc') else v[:-1] if (v.endswith('P') and v != 'P') else v
            if not vb.startswith('camS2'):
                continue
            spec = _s2_spec(vb[len('camS2'):])
            key = _s2_key(spec)
            if key in s2sets or (spec['rounds'] == 1 and spec['cond'] == 'imag'):
                continue
            phys = 'band' if v.endswith('P') else 'core' if v.endswith('Pc') else ''
            dl, dkey = [[x] for x in s2], 'r1'
            for r in range(1 if spec['cond'] == 'imag' else 0, spec['rounds']):
                trajs = []
                for k, (q, h, fin, real) in enumerate(reals):
                    if spec['cond'] == 'real' and r == 0:
                        trajs.append(trajectory_from_poses(np.concatenate([np.asarray(st.legal.pose, np.float64)[None], real]),
                                                           st.legal.time))
                        continue
                    ex, _ = sb._execution(d.candidates[k])
                    if isinstance(ex, Exception):
                        trajs.append(trajectory_from_poses(np.asarray([st.legal.pose], np.float64), st.legal.time))
                        continue
                    m = _s2_truth(st.legal.map, dl, k, spec, s2cache, dkey)
                    vmap = st.legal.map.copy()
                    hum = VisibleWorld(st.legal, truth, 0, W_MAX)
                    hum.blocked = _blocked_lookup(m)
                    hum.discovery = _s2_reveal(vmap, m, spec)
                    ro = imagine(st.legal, ex, dyn, hum, h, level=int(d.candidates[k].level), gmap=vmap,
                                 continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner, cont_boundary=16,
                                 cont_inherit=True, physics=phys)
                    c = getattr(ro, 'cont', None)
                    P = np.asarray(ro.poses, np.float64)
                    if c is not None and len(c):
                        P = np.concatenate([P, np.asarray(c, np.float64)])
                    trajs.append(trajectory_from_poses(P, st.legal.time))
                try:
                    pr = sb.l2.predict(ep.last_state, trajs, materialize=False, worlds=sb.config.worlds, seed=sb.config.seed)
                except Exception as exc:
                    print('shortprobe: Stage 2 round failed (%s)' % (exc,), flush=True)
                    break
                new = [c.map for c in pr.candidates]
                dl = [[x] for x in new] if (spec['cond'] == 'real' and r == 0) else [a+[x] for a, x in zip(dl, new)]
                dkey = (key, r)
            s2sets[key] = (dl, dkey)
    for k, (q, h, fin, real) in enumerate(reals):
        rc = clearance_series(real, tape, i0)
        steps = int(fin.index-i0)
        rec = dict(label=q.label, kind=q.kind, h=int(h),
                   real=dict(outcome=fin.outcome or 'running', steps=steps,
                             clr={w: _min_upto(rc, w) for w in WINDOWS}, ctg={w: _ctg_at(field, real, w) for w in WINDOWS}),
                   imag={})
        if os.environ.get('L3_PROBE_CENTRE') == '1':
            cs = _centre_series(real, tape, i0)
            rec['real']['cd'] = {w: _min_upto(cs, w) for w in WINDOWS}
        cand = d.candidates[k]
        for v in variants:
            runs = []
            if v == 'cam6':
                specs = [(int(w), float(x)) for w, x in zip(worlds, ww)] if HP is not None else []
            else:
                specs = [(None, 1.)]
            # imagined contact physics (sandbox/imagine.py PHYSICS): a variant name + 'P' ('band') or + 'Pc' ('core')
            base = v[:-2] if v.endswith('Pc') else v[:-1] if (v.endswith('P') and v != 'P') else v
            for w, wt in specs:
                ex, _ = sb._execution(cand)
                if isinstance(ex, Exception):
                    break
                vmap = gmap
                mstats = None
                if v == 'rel':
                    hum = NominalHumans(st.legal.tracks, st.legal.time, W_MAX, bank=bank, mode='mean')
                elif v == 'cam6':
                    hum = VisibleWorld(st.legal, bank, w, W_MAX)
                elif base == 'camCV':
                    # LEGAL: the tracked people at the tracker's constant velocity (oraclebank.CVBank, the CV-null
                    # selector's people) through the camera model; contact still scored against the true people
                    from .oraclebank import CVBank
                    hum = VisibleWorld(st.legal, CVBank(st.legal, horizon_steps=W_MAX+1), 0, W_MAX)
                elif base in ('camR', 'camRM'):
                    # fidelity ladder (privileged): the real tracker's output (camR), + the real map stream (camRM)
                    hum = ReplayTracks(fin.legal.track_log, len(st.legal.track_log)-1, W_MAX, st.legal.time)
                    if base == 'camRM':
                        vmap = st.legal.map.copy()
                        hum.discovery = TimedReveal(vmap, fin.legal.map)
                elif base in ('camTk', 'camTkM'):
                    # TRUE people through the emulated camera + tracker (perceive L3_CAM_TRACKER): the camera rig of the
                    # decision's legal row (runtime state), the real LegalHumanTracker seeded with the legal tracks;
                    # camTkM: + the real map stream for the planner and the final map for occlusion (as camM)
                    hum = VisibleWorld(st.legal, truth, 0, W_MAX, camera=st, tracker=True)
                    if base == 'camTkM':
                        vmap = st.legal.map.copy()
                        hum.blocked = _blocked_lookup(fin.legal.map)
                        hum.discovery = TimedReveal(vmap, fin.legal.map)
                else:
                    hum = VisibleWorld(st.legal, truth, 0, W_MAX)
                    if base in ('camTD', 'camTDo'):
                        vmap = st.legal.map.copy()
                        hum.blocked = _blocked_lookup(fin.legal.map)
                        hum.discovery = (ObserverReveal if base == 'camTDo' else Discovery)(vmap, fin.legal.map)
                    elif base.startswith('camS2'):
                        if s2 is None:
                            break
                        spec = _s2_spec(base[len('camS2'):])
                        dl, dkey = s2sets.get(_s2_key(spec), ([[x] for x in s2], 'r1'))
                        s2map = _s2_truth(st.legal.map, dl, k, spec, s2cache, dkey)
                        vmap = st.legal.map.copy()
                        hum.blocked = _blocked_lookup(s2map)
                        hum.discovery = _s2_reveal(vmap, s2map, spec)
                        mstats = map_stats(s2map, st.legal.map, fin.legal.map)
                    elif base in ('camM', 'camMF', 'camMB'):
                        # fidelity ladder (privileged): camera-model people + the real robot's map stream by time
                        # (camMF / camMB: only its FREE / only its BLOCKED certifications)
                        vmap = st.legal.map.copy()
                        hum.blocked = _blocked_lookup(fin.legal.map)
                        hum.discovery = TimedReveal(vmap, fin.legal.map, only=dict(camMF=1, camMB=2).get(base))
                fid = v != 'rel'
                phys = 'band' if v.endswith('P') else 'core' if v.endswith('Pc') else ''
                track_rec = trk_on and hasattr(hum, 'rec')
                if track_rec:
                    hum.rec = []                   # analysis: the imagined track set of every step
                ro = imagine(st.legal, ex, dyn, hum, h, level=int(cand.level), gmap=vmap,
                             continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner,
                             cont_boundary=16 if fid else 0, cont_inherit=fid, physics=phys)
                c = getattr(ro, 'cont', None)
                im = np.concatenate([ro.poses[1:], c]) if c is not None and len(c) else np.asarray(ro.poses[1:])
                im = np.asarray(im, np.float64).reshape(-1, 4)
                trk = {}
                if track_rec:
                    p0 = np.asarray(st.legal.pose, np.float64)
                    rpose = lambda k: p0 if k < 1 else real[min(k, len(real))-1]
                    ipose = lambda k: p0 if (k < 1 or not len(im)) else im[min(k, len(im))-1]
                    trk['trk'] = track_agreement(hum.rec, fin.legal.track_log, b0, rpose, ipose)
                    hum.rec = None
                    if isinstance(hum, VisibleWorld) and len(real):
                        # open loop: the same perception model (own memory, same occlusion map) along the REAL poses
                        ol = hum.fork(None, hum.blocked)
                        ol.rec = []
                        for kk in range(0, min(len(real), W_MAX)+1):
                            ol.tracks_at(kk, rpose(kk))
                        trk['trk_ol'] = track_agreement(ol.rec, fin.legal.track_log, b0, rpose, rpose)
                legal_cd = None
                if v == 'cam6':
                    ic = _world_clearance(im, HP[w])
                elif v == 'rel':
                    hp = np.asarray(hum.positions[1:, :, :], np.float64)[:, :, [0, 2]]
                    ic = _world_clearance(im, hp)
                elif base == 'camCV':
                    # the legal selector's own score: contact with its (constant-velocity) people
                    ic = _world_clearance(im, hum.P[1:])
                    if os.environ.get('L3_PROBE_CENTRE') == '1':
                        Pq = hum.P[1:]
                        n = min(len(im), len(Pq))
                        legal_cd = np.full(len(im), np.inf)
                        if n and Pq.shape[1]:
                            with np.errstate(invalid='ignore'):
                                dd = np.hypot(Pq[:n, :, 0]-im[:n, None, 0], Pq[:n, :, 1]-im[:n, None, 2])
                            legal_cd[:n] = np.where(np.isfinite(dd), dd, np.inf).min(axis=1)
                else:
                    ic = clearance_series(im, tape, i0)
                icd = legal_cd if legal_cd is not None else (
                    _centre_series(im, tape, i0) if (v not in ('rel', 'cam6') and os.environ.get('L3_PROBE_CENTRE') == '1') else None)
                div = {}
                for lag in LAGS:
                    if lag > len(real) or not len(im):
                        break
                    j = min(lag, len(im))-1
                    div[lag] = float(math.hypot(real[lag-1, 0]-im[j, 0], real[lag-1, 2]-im[j, 2]))
                if v == 'rel' and info is not None:
                    rec['expo'] = expo_cand(info, st.legal, im)
                runs.append(dict(w=wt, n=len(im), stop=bool(ro.stop), ended=ro.ended_by,
                                 near_goal=bool(len(im) and math.hypot(im[-1, 0]-st.legal.goal[0], im[-1, 2]-st.legal.goal[2]) <= 0.3),
                                 clr={w2: _min_upto(ic, w2) for w2 in WINDOWS},
                                 ctg={w2: _ctg_at(field, im, w2) for w2 in WINDOWS}, div=div,
                                 **({} if icd is None else dict(cd={w2: _min_upto(icd, w2) for w2 in WINDOWS})),
                                 **({} if mstats is None else dict(s2=mstats)), **trk))
            rec['imag'][v] = runs
        cands.append(rec)
    out = dict(step=i0, decision=int(ep.decisions), labels=list(d.labels), default=d.labels[d.default],
               ctg0=float(field.start), cands=cands, seconds=round(time.perf_counter()-t0, 1))
    if seed is not None:
        out['seed'] = seed
    if info is not None:
        out['ego'] = [float(x) for x in info.ego()]
    return out


def expo_state(legal, bank):
    """The legal information state at the decision (infofeat.InfoState: staleness of the map per robot-frame sector,
    Stage 1 residual mass per sector, lost tracks, recent camera coverage) -- exposure features."""
    from .compiler import Frame
    from .infofeat import InfoState
    keys = list(getattr(bank, 'track_keys', ()))
    HP = W = None
    if len(keys):
        rel = np.arange(1, int(round(float(bank.times[-1])/0.048))+1)*0.048
        HP = np.asarray(bank.sandbox_positions(rel), np.float64)
        W = np.asarray(bank.weights, np.float64)
        W = W/W.sum() if W.sum() > 0 else W
    return InfoState(legal, Frame(legal.pose), keys, np.asarray(getattr(bank, 'unseen', np.zeros(len(keys))), bool), HP, W)


def expo_cand(info, legal, im):
    """Per candidate (its imagined interval with the released people): yaw sweep left / right, stale free area its
    camera would cover, Stage 1 residual / lost slots it would see (infofeat CAND_INFO)."""
    poses = np.concatenate([np.asarray(legal.pose, np.float64)[None], np.asarray(im[:16], np.float64)])
    return [float(x) for x in info.cand(poses)]


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


def fidelity(path, variants=None, ref='camTPc', lags=(16, 32)):
    """Imagined-vs-real divergence per variant on the candidates every listed variant imagined (paired), non-thru and
    thru ('_th') candidates apart: median / p90 at ``lags``; per variant vs ``ref`` the share of non-thru candidates
    whose 32-step divergence is better / worse by more than 0.25 m; camS2 variants: their predicted maps against the
    real end-of-window maps (summed cell counts, ``map_stats``; precision among the predicted cells the real window
    certified, recall of the cells it certified new).  ``path``: one probe directory or several, comma-separated
    (pooled).

        python scripts/l3_run.py eventwm_graph.l3.shortprobe fidelity <dir>[,<dir>...] [variant,variant,...]"""
    recs = []
    for f in sorted(f for p in str(path).split(',') for f in Path(p).glob('*.json')):
        for s in json.loads(f.read_text()).get('probe', []):
            recs += s['cands']
    names = variants or sorted({v for c in recs for v in c['imag']})
    rows = [c for c in recs if all((c['imag'].get(v) or [{}])[0].get('div') for v in names)]
    print('%s: %d candidates, %d imagined by every variant (%d non-thru)' % (
        path, len(recs), len(rows), sum(not c['label'].endswith('_th') for c in rows)))
    out = {}
    for v in names:
        line = '   %-12s' % v
        res = {}
        for th in (False, True):
            for lag in lags:
                a = np.asarray([c['imag'][v][0]['div'].get(str(lag)) for c in rows if c['label'].endswith('_th') == th
                                and c['imag'][v][0]['div'].get(str(lag)) is not None], np.float64)
                if len(a):
                    res['%s%d' % ('thru' if th else 'non', lag)] = (float(np.median(a)), float(np.percentile(a, 90)), len(a))
                    line += '  %s d%d %.2f / %.2f' % ('thru' if th else 'non-thru', lag, np.median(a), np.percentile(a, 90))
        if ref in names and v != ref:
            d = np.asarray([c['imag'][v][0]['div'].get('32', np.nan) - c['imag'][ref][0]['div'].get('32', np.nan)
                            for c in rows if not c['label'].endswith('_th')], np.float64)
            d = d[np.isfinite(d)]
            if len(d):
                res['better'], res['worse'] = float((d < -0.25).mean()), float((d > 0.25).mean())
                line += '  | vs %s: better %.2f worse %.2f' % (ref, res['better'], res['worse'])
        st = [c['imag'][v][0].get('s2') for c in rows if c['imag'][v][0].get('s2')]
        if st:
            S = {k: sum(x[k] for x in st) for k in st[0]}
            res['s2'] = S
            line += ('\n%16sBLOCKED precision %.2f recall %.2f | FREE precision %.2f recall %.2f | false walls %.3f of real '
                     'FREE, false openings %.3f of real BLOCKED' % (
                         '', S['BB']/max(S['pB']-S['B0'], 1), S['BB']/max(S['rB'], 1), S['FF']/max(S['pF']-S['F0'], 1),
                         S['FF']/max(S['rF'], 1), S['FB']/max(S['rF'], 1), S['BF']/max(S['rB'], 1)))
        out[v] = res
        print(line)
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'fidelity':
        fidelity(sys.argv[2], sys.argv[3].split(',') if len(sys.argv) > 3 else None)
    else:
        summarize(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.2)
