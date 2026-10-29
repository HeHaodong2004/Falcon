"""Hand-written selectors over the compiled consequence graph (baselines; no learning).

``greedy_index``: the obvious imagination-based rule -- among candidates whose imagined execution is predicted safe
(Stage 1 collision probability over the interval and the 3 s hold extension below thresholds, no imagined pose on
legally BLOCKED cells), take the largest legal route-cost reduction; STOP only when the imagined end is at the goal;
if nothing is predicted safe, take the smallest predicted collision risk.
"""
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
