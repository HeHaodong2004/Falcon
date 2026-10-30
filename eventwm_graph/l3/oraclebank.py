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
    def __init__(self, tape_positions, index, legal, mode='tracked', horizon_steps=63, long_steps=0):
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
        self.noise_worlds = None
        self.long_steps = int(long_steps)        # > 62: the intention (route) features use this longer true horizon

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64)
        idx = np.minimum(self.index+np.round(rel/STEP_SECONDS).astype(int), len(self.P)-1)
        P = self.P[idx][:, self.cols][:, :, [0, 2]][None]                    # [1, T, Nb, 2]
        if self.noise_worlds is None:
            return P
        # noisy multi-world variant (prediction-error study): world w = truth + a per-person constant velocity error
        return P+self.noise_worlds[:, None, :, :]*rel[None, :, None, None]   # [W, T, Nb, 2]

    def add_noise(self, sigma, worlds, seed=0):
        """ANALYSIS: replace the single true world by ``worlds`` noisy copies whose people drift from the truth with a
        per-person constant velocity error ~ N(0, sigma^2) (m/s per axis): position error grows linearly with the
        horizon (sigma * t per axis), as a long-horizon predictor's would.  Weights uniform."""
        rng = np.random.default_rng(int(seed))
        self.noise_worlds = rng.normal(0., float(sigma), size=(int(worlds), len(self.cols), 2))
        self.weights = np.full(int(worlds), 1./int(worlds))
        self.present = np.ones((int(worlds), len(self.cols)), bool)
        return self


class CVBank:
    """LEGAL single-world people bank (Layer 3 option ``L3_MW_BANK=cv``, default off): every person the robot tracks
    (valid, at most 2 s old: ``control.human_estimates``) continues at the tracker's constant velocity; nobody else
    exists.  The OracleBank interface, so the multi-world selector and the camera model can use it in place of
    Stage 1's bank (a no-learning baseline predictor for short horizons)."""

    def __init__(self, legal, horizon_steps=63, max_age=None):
        from eventwm_graph.querygen.control import human_estimates
        # lost-track memory (L3_CV_MAX_AGE, default 2 s = the executor's): tracks last seen up to max_age ago continue
        # at their last estimated velocity from where they were last seen (legal: past observations only)
        max_age = float(__import__('os').environ.get('L3_CV_MAX_AGE', '2.0')) if max_age is None else float(max_age)
        est = human_estimates(legal.tracks, legal.time, max_age=max_age)
        self.track_keys = tuple('track:%s' % h.track_id for h in est)
        self.P0 = np.asarray([[h.position[0], -h.position[1]] for h in est], np.float64).reshape(-1, 2)   # world x, z
        self.V = np.asarray([[h.velocity[0], -h.velocity[1]] for h in est], np.float64).reshape(-1, 2)
        self.unseen = np.zeros(len(est), bool)
        self.weights = np.ones(1)
        self.present = np.ones((1, len(est)), bool)
        self.times = np.arange(1, horizon_steps)*STEP_SECONDS
        self.long_steps = 0

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64)
        return (self.P0[None]+rel[:, None, None]*self.V[None])[None]           # [1, T, N, 2]
