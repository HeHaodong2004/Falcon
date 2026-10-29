"""ANALYSIS ONLY (privileged; never training data, never the primary benchmark): a one-world "Stage 1 bank" holding the
TRUE future positions of people from the episode's recorded human tape, swapped into the Sandbox result before the
Graph Compiler (``L3_ORACLE_BANK=tracked`` -- perfect 3 s prediction of the people the robot currently tracks;
``all`` -- of everyone, i.e. perfect prediction and perfect awareness).  It separates the quality of Stage 1's human
predictions from the quality of the decisions made on them: a selector evaluated with Stage 1 and with this bank
on the same episodes.  The imagined rollouts themselves (the planner's perceived humans) are unchanged.
"""
import math

import numpy as np

from eventwm_graph.querygen.dynamics import STEP_SECONDS


class OracleBank:
    def __init__(self, tape_positions, index, legal, mode='tracked', horizon_steps=63):
        P = np.asarray(tape_positions, np.float64)                      # [S, N, 3] true positions per env step
        self.P, self.index = P, int(index)
        N = P.shape[1]
        cur = P[min(self.index, len(P)-1)][:, [0, 2]]
        keys = ['residual:%d' % i for i in range(N)]
        unseen = np.ones(N, bool)
        for t in legal.tracks:
            if not t.get('valid', True) or not len(cur):
                continue
            d = np.hypot(cur[:, 0]-t['position'][0], cur[:, 1]-t['position'][2])
            j = int(np.argmin(d))
            if d[j] < 0.75 and keys[j].startswith('residual'):
                keys[j] = 'track:%s' % t['track_id']
                unseen[j] = False
        keep = np.ones(N, bool) if mode == 'all' else ~unseen
        self.cols = np.flatnonzero(keep)
        self.track_keys = tuple(keys[i] for i in self.cols)
        self.unseen = unseen[self.cols]
        self.weights = np.ones(1)
        self.present = np.ones((1, len(self.cols)), bool)
        self.times = np.arange(1, horizon_steps)*STEP_SECONDS

    def sandbox_positions(self, rel):
        idx = np.minimum(self.index+np.round(np.asarray(rel, np.float64)/STEP_SECONDS).astype(int), len(self.P)-1)
        return self.P[idx][:, self.cols][:, :, [0, 2]][None]                 # [1, T, Nb, 2]
