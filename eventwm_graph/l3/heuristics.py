"""Hand-written selectors over the compiled consequence graph (baselines; no learning).

``greedy_index``: the obvious imagination-based rule -- among candidates whose imagined execution is predicted safe
(Stage 1 collision probability over the interval and the 3 s hold extension below thresholds, no imagined pose on
legally BLOCKED cells), take the largest legal route-cost reduction; STOP only when the imagined end is at the goal;
if nothing is predicted safe, take the smallest predicted collision risk.
"""
import math

import numpy as np

from .compiler import CAND

_I = {c: CAND.index(c) for c in CAND}


def greedy_index(graph, p_max=0.15, x_max=0.3):
    c = graph['cand']
    n = len(c)
    pcol = c[:, _I['pcol']]
    xcol = c[:, _I['x_pcol']]
    gain = c[:, _I['ctg_gain']]
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    blocked = c[:, _I['blocked_frac']] > 0
    safe = (pcol <= p_max) & (xcol <= x_max) & ~blocked & ~(stop & ~at_goal)
    score = gain+0.5*at_goal*stop
    if safe.any():
        idx = np.flatnonzero(safe)
        return int(idx[np.argmax(score[idx])])
    risk = pcol+0.5*xcol
    return int(np.argmin(risk))


def progress_index(graph):
    """Pure progress: the largest legal route-cost reduction of the imagined interval (ties: the earlier candidate)."""
    c = graph['cand']
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    score = c[:, _I['ctg_gain']].astype(np.float64)+2.0*(stop & at_goal)-5.0*(stop & ~at_goal)
    return int(np.argmax(score))


_E = None


def escape_index(graph):
    """Progress, with a stall breaker: when the robot has been stalled (< 0.3 m over 48 steps) and no candidate
    shows route progress, take the non-continue candidate whose imagined interval moves the robot the most
    (route alternatives / retreats / exploration first, then any motion); otherwise the largest route progress."""
    global _E
    from .compiler import EGO
    if _E is None:
        _E = {c: EGO.index(c) for c in EGO}
    c = graph['cand']
    e = graph['ego']
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    if e[_E['stall_48']] > 0.5 and gain.max() <= 0.02:
        disp = np.hypot(c[:, _I['disp_fwd']], c[:, _I['disp_left']]).astype(np.float64)
        cont = c[:, _I['intent_continue']] > 0.5
        stop = c[:, _I['intent_stop']] > 0.5
        score = np.where(cont | stop, -1., disp)
        if score.max() > 0.05:
            return int(np.argmax(score))
    return progress_index(graph)


def routesafe_index(graph, default, thr=0.15, ratio=0.5):
    """Default unless its imagined interval or its INTENTION (v4 route columns) carries a predicted contact risk above
    ``thr``; then the candidate with the most route progress among those whose risk is below ``ratio`` x the default's
    (never STOP away from the goal, never a scan), else the least risky one.  Needs L3_LAYOUT=v4."""
    c = graph['cand']
    n = len(c)
    risk = np.maximum(c[:, _I['pcol']], c[:, _I['r_pcol']]).astype(np.float64)
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    ok = (risk < ratio*risk[d]) & ~(stop & ~at_goal) & ~scan
    if ok.any():
        idx = np.flatnonzero(ok)
        safe = idx[risk[idx] < thr]
        pool = safe if len(safe) else idx
        return int(pool[np.argmax(c[pool, _I['ctg_gain']]-0.5*risk[pool])])
    return d


def routesafe_progress_index(graph, default, thr=0.15, slack=0.1):
    """Like ``routesafe_index`` but never trades progress for safety: deviate from the default only to a candidate whose
    interval + intention risk is below ``thr`` and half the default's AND whose legal route progress over the interval
    is at least the default's minus ``slack`` (feature units: 0.1 = 0.3 m); otherwise keep the default."""
    c = graph['cand']
    n = len(c)
    risk = np.maximum(c[:, _I['pcol']], c[:, _I['r_pcol']]).astype(np.float64)
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    ok = (risk < min(thr, 0.5*risk[d])) & ~(stop & ~at_goal) & ~scan & (gain >= gain[d]-slack)
    if ok.any():
        idx = np.flatnonzero(ok)
        return int(idx[np.argmax(gain[idx]-0.5*risk[idx])])
    return d


def routesafe_bank_index(graph, default, thr=0.15, slack=0.1):
    """``routesafe_progress_index`` with every risk taken from the compiler's bank-derived columns (waypoint P(contact)
    over the interval and the intention's P(contact)), so the same rule can run on Stage 1's bank or, for analysis, on
    the true human futures (L3_ORACLE_BANK)."""
    from .compiler import TRAJ
    c = graph['cand']
    n = len(c)
    ip = graph['traj'][:, :, TRAJ.index('p_col')].max(1).astype(np.float64) if n else np.zeros(0)
    risk = np.maximum(ip, c[:, _I['r_pcol']].astype(np.float64))
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    ok = (risk < min(thr, 0.5*risk[d])) & ~(stop & ~at_goal) & ~scan & (gain >= gain[d]-slack)
    if ok.any():
        idx = np.flatnonzero(ok)
        return int(idx[np.argmax(gain[idx]-0.5*risk[idx])])
    return d


def imagined_rollout_index(result, legal, default, slack_p=0.05, slack_m=0.5, radius=0.75, step_s=0.048):
    """LEGAL rollout policy (no learning, no privileged input): every candidate's imagined interval + its imagined
    default continuation (SandboxConfig.continuation: keep the running query, else the goal query; the executor and
    planner in the loop against the nominal Stage 1 people) is scored against the 40 Stage 1 worlds -- the bank's 3 s,
    then each world's people continue at their last velocity -- for P(contact) (centre distance < ``radius``), and
    against the CURRENT legal map for route progress at the end of the imagined horizon.  The least risky candidates
    (within ``slack_p`` of the minimum) compete on progress; the default is kept unless another one is clearly better
    (more than ``slack_m`` metres of progress or lower risk beyond the slack)."""
    from .compiler import LegalField
    bank = result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else W
    field = LegalField(legal)
    seqs = []
    for c in result.candidates:
        ro = c.rollout
        poses = np.asarray(ro.poses, np.float64)
        cont = getattr(ro, 'cont', None)
        seq = np.concatenate([poses[1:], cont]) if cont is not None and len(cont) else poses[1:]
        if not len(seq):
            seq = poses[:1]
        seqs.append(seq)
    T = max(len(s) for s in seqs)
    keys = list(getattr(bank, 'track_keys', ()))
    pcol = np.zeros(len(seqs))
    if len(keys):
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*step_s)   # analysis bank: longer
        rel = np.arange(1, T+1)*step_s
        inside = rel[rel <= horizon+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)              # [W, Tb, N, 2]
        if len(inside) < T:
            if HP.shape[1] >= 2:
                vel = (HP[:, -1]-HP[:, -2])/step_s
            else:
                vel = np.zeros_like(HP[:, -1])
            extra = np.arange(1, T-len(inside)+1)*step_s
            ext = HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]
            HP = np.concatenate([HP, ext], 1)
        with np.errstate(invalid='ignore'):
            for k, seq in enumerate(seqs):
                n = len(seq)
                d = np.hypot(HP[:, :n, :, 0]-seq[None, :, None, 0], HP[:, :n, :, 1]-seq[None, :, None, 2])
                d = np.where(np.isfinite(d), d, np.inf)
                hit = (d.min(axis=(1, 2)) < radius)
                pcol[k] = float((W*hit).sum())
    prog = np.asarray([field.start-field.at(s[-1, 0], s[-1, 2]) for s in seqs])
    stop = np.asarray([bool(c.rollout.stop) for c in result.candidates])
    at_goal = np.asarray([math.hypot(s[-1, 0]-legal.goal[0], s[-1, 2]-legal.goal[2]) < 0.3 for s in seqs])
    bad_stop = stop & ~at_goal
    pc = np.where(bad_stop, 1.0, pcol)
    d = int(default)
    best_p = float(pc.min())
    pool = np.flatnonzero(pc <= best_p+slack_p)
    k = int(pool[np.argmax(prog[pool])])
    if pc[d] <= best_p+slack_p and prog[d] >= prog[k]-slack_m:
        return d
    return k


def multiworld_index(sb, legal, d, n_worlds=6, cont=60, radius=0.75, slack_p=0.1, slack_m=0.5, step_s=0.048):
    """LEGAL multi-world imagined rollout (no learning, no privileged input): each candidate is imagined once per
    sampled Stage 1 world -- the robot's own planner perceiving THAT world's tracked people, interval + ``cont`` steps of
    default continuation -- and checked for contact against that world's people (tracked and hidden); P(contact) is the
    world-weighted share of contact rollouts, progress the mean legal route gain at the end.  Unlike a single nominal
    world, the planner's reactions to each possible future are simulated (the timing that decides hit vs miss)."""
    from eventwm_graph.sandbox.imagine import NominalHumans
    from .compiler import LegalField
    bank = d.result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    worlds = np.argsort(-W)[:n_worlds] if len(np.unique(W)) > 1 else np.arange(min(n_worlds, len(W)))
    ww = W[worlds]/W[worlds].sum()
    field = LegalField(legal)
    K = len(d.candidates)
    steps = max(sb.config.steps_for(c) for c in d.candidates)
    T = steps+cont
    keys = list(getattr(bank, 'track_keys', ()))
    if len(keys):
        rel = np.arange(1, T+1)*step_s
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*step_s)   # analysis bank: longer
        inside = rel[rel <= horizon+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)              # [W, Tb, N, 2]
        if len(inside) < T and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/step_s
            extra = np.arange(1, T-len(inside)+1)*step_s
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    else:
        HP = None
    pc = np.zeros(K)
    prog = np.zeros(K)
    stop_bad = np.zeros(K, bool)
    # contacts counted over the first L3_MW_CONTACT_STEPS imagined steps only (default: the whole rollout); with Stage 1
    # worlds people beyond the 3 s bank horizon are constant-velocity guesses -- the long continuation then judges
    # progress (deadlocks, timeouts) while contacts are judged where the worlds are predictions
    contact_steps = int(__import__('os').environ.get('L3_MW_CONTACT_STEPS', '0')) or 10**9
    old = sb.config.continuation
    sb.config.continuation = int(cont)
    try:
        for j, w in enumerate(worlds):
            if __import__('os').environ.get('L3_MW_PERCEIVE') == '1':
                # perception-faithful imagination: the planner sees what the imagined camera would (perceive.py)
                from .perceive import VisibleWorld
                humans = VisibleWorld(legal, bank, int(w), T)
            else:
                humans = NominalHumans(legal.tracks, legal.time, T, bank=bank, mode='world:%d' % int(w))
            ros = sb.imagine_all(legal, d.candidates, humans)
            for k, ro in enumerate(ros):
                poses = np.asarray(ro.poses, np.float64)
                c = getattr(ro, 'cont', None)
                seq = np.concatenate([poses[1:], c]) if c is not None and len(c) else poses[1:]
                if not len(seq):
                    seq = poses[:1]
                if ro.stop and math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) > 0.3:
                    stop_bad[k] = True
                if HP is not None:
                    n = min(len(seq), HP.shape[1], contact_steps)
                    with np.errstate(invalid='ignore'):
                        dd = np.hypot(HP[w, :n, :, 0]-seq[:n, None, 0], HP[w, :n, :, 1]-seq[:n, None, 2])
                    if np.nanmin(np.where(np.isfinite(dd), dd, np.inf)) < radius:
                        pc[k] += ww[j]
                prog[k] += ww[j]*(field.start-field.at(seq[-1, 0], seq[-1, 2]))
    finally:
        sb.config.continuation = old
    pc = np.where(stop_bad, 1., pc)
    dflt = int(d.default)
    best_p = float(pc.min())
    pool = np.flatnonzero(pc <= best_p+slack_p)
    k = int(pool[np.argmax(prog[pool])])
    if pc[dflt] <= best_p+slack_p and prog[dflt] >= prog[k]-slack_m:
        return dflt
    return k
