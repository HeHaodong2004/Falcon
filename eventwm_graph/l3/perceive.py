"""Perception-faithful nominal people for imagination (Layer 3 option).

The Sandbox's imagined planner normally perceives every tracked person of the nominal world at every imagined step.
The real planner perceives people only through the 90 deg jaw camera: a person is detected when inside the frustum
(10 m) with a clear line of sight, and a lost track is extrapolated at constant velocity for at most 2 s
(``control.human_estimates``).  ``VisibleWorld`` reproduces that inside the imagination: it holds one world's
people (tracked AND hidden slots of a Stage 1 world, or the analysis-only true future), and at every imagined step
returns the legal-format tracks the robot would have from its IMAGINED pose (frustum, legal-map occlusion, last-seen
memory).  State resets when a rollout starts (k == 0) and carries through the default continuation.
"""
import math

import numpy as np

from eventwm_graph.querygen.dynamics import STEP_SECONDS

from .infofeat import _blocked_lookup, _los

HFOV = math.radians(90.)
RANGE = 10.


class VisibleWorld:
    def __init__(self, legal, bank, world, steps, step_s=STEP_SECONDS):
        self.legal = legal
        self.now = float(legal.time)
        self.steps = int(steps)
        self.step = float(step_s)
        keys = [str(k) for k in getattr(bank, 'track_keys', ())]
        rel = np.arange(0, self.steps+1)*self.step
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*self.step)
        inside = rel[(rel > 0) & (rel <= horizon+1e-9)]
        if len(keys):
            P = np.asarray(bank.sandbox_positions(inside), np.float64)[int(world)]        # [Ti, N, 2]
            cur = np.asarray(bank.sandbox_positions(np.asarray([1e-3])), np.float64)[int(world), 0]
            P = np.concatenate([cur[None], P], 0)
            if len(P) < len(rel):
                vel = (P[-1]-P[-2])/self.step if len(P) >= 2 else np.zeros_like(P[-1])
                extra = np.arange(1, len(rel)-len(P)+1)*self.step
                P = np.concatenate([P, P[-1][None]+extra[:, None, None]*vel[None]], 0)
        else:
            P = np.zeros((len(rel), 0, 2))
        self.P = P                                                                          # [steps+1, N, 2]
        self.ids = [k[len('track:'):] if k.startswith('track:') else 'h%d' % n for n, k in enumerate(keys)]
        y = float(legal.pose[1])+0.9
        self.y = y
        # what the nominal-humans interface exposes for clearance statistics
        N = self.P.shape[1]
        self.positions = np.zeros((self.steps+1, N, 3))
        self.positions[:, :, 0] = np.where(np.isfinite(self.P[..., 0]), self.P[..., 0], 1e6)
        self.positions[:, :, 1] = y
        self.positions[:, :, 2] = np.where(np.isfinite(self.P[..., 1]), self.P[..., 1], 1e6)
        self.blocked = _blocked_lookup(legal.map)
        self._init = {str(t['track_id']): t for t in legal.tracks if t.get('valid', True)}
        self.mode = 'visible'
        self.fallback = 0
        self.reset()

    def reset(self):
        # last-seen memory: id -> (time, position xz, velocity xz); start from the robot's real tracks
        self.mem = {}
        for tid, t in self._init.items():
            p = np.asarray(t['position'], float)
            v = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), float)
            self.mem[tid] = (float(t['last_seen_time']), np.array([p[0], p[2]]), np.array([v[0], v[2]]))

    def tracks(self, k):
        """Fallback (no pose): the remembered tracks at step k (k == 0 starts a new rollout)."""
        if int(k) == 0:
            self.reset()
        return self._format(int(k), set())

    def tracks_at(self, k, pose):
        k = int(min(k, self.steps))
        if k == 0:
            self.reset()
        t_now = self.now+k*self.step
        visible = set()
        if self.P.shape[1]:
            pts = self.P[k]
            ok = np.isfinite(pts).all(1)
            dx, dz = pts[:, 0]-pose[0], pts[:, 1]-pose[2]
            fw = dx*math.cos(pose[3])-dz*math.sin(pose[3])
            lf = -dx*math.sin(pose[3])-dz*math.cos(pose[3])
            inside = ok & (np.abs(np.arctan2(lf, fw)) <= HFOV/2) & (np.hypot(dx, dz) <= RANGE)
            idx = np.flatnonzero(inside)
            if len(idx):
                clear = _los(self.blocked, np.full(len(idx), pose[0]), np.full(len(idx), pose[2]), pts[idx, 0], pts[idx, 1])
                for n in idx[clear]:
                    tid = self.ids[n]
                    prev = self.mem.get(tid)
                    v = (pts[n]-prev[1])/max(t_now-prev[0], self.step) if prev is not None else np.zeros(2)
                    self.mem[tid] = (t_now, pts[n].copy(), v)
                    visible.add(tid)
        return self._format(k, visible)

    def _format(self, k, visible):
        t_now = self.now+k*self.step
        out = []
        for tid, (ts, p, v) in self.mem.items():
            if t_now-ts > 2.0:
                continue
            out.append(dict(track_id=tid, position=[float(p[0]), self.y, float(p[1])],
                            estimated_velocity=[float(v[0]), 0., float(v[1])], velocity_valid=True,
                            visible_now=tid in visible, valid=True, first_seen_time=self.now, last_seen_time=ts,
                            observation_age=t_now-ts, num_observations=5, hypothetical=True))
        return out
