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
