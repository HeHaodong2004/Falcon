"""Kinematic sandbox for planner trials against predicted human futures (L2.5).

The sandbox runs the *same* ``QueryExecution`` (planner + dynamics realiser +
static lattice) on a frozen legal map with nominal robot kinematics and
synthetic tracks generated from one predicted human future.  Two ways of
turning M predicted futures into query consequences are compared:

``open_loop``   one nominal rollout (tracks from the mean/nominal future), then
                vectorised scoring of that single trajectory against all M
                futures (cost: 1 rollout + M x T distance evaluations);
``closed_loop`` one rollout per future, the planner reacting to that future
                (cost: M rollouts).

Futures are legal: constant-velocity extrapolation of legal tracks with
sampled velocity noise (a stand-in for frozen Stage 1 samples; the interface
accepts any [M, T+1, N, 3] array).  ``validate_native`` compares the sandbox
with the true future against native branch execution (privileged, validation
only).
"""
from dataclasses import dataclass
import math
import time
import numpy as np

from .control import to_plane, heading_vec
from .native import rectangle_distance
from .dynamics import STEP_SECONDS
from .executor import QueryExecution, command_for
from .planners import make_planner

BODY_PROXY = 0.25


class SandboxLegal:
    """Minimal legal-state stand-in: frozen map, synthetic tracks, own poses."""

    def __init__(self, legal):
        self.goal = legal.goal
        self.map = legal.map
        self.tracks = legal.tracks
        self.time = float(legal.time)
        self.start_time = legal.start_time
        self.pose = np.asarray(legal.pose, dtype=float).copy()
        self.poses = [self.pose.copy()]
        self.times = [self.time]
        self.steps = 0

    def velocity(self, window=1):
        if len(self.poses) < 2:
            return np.zeros(2), 0.
        a, b = self.poses[-2], self.poses[-1]
        dt = max(self.times[-1]-self.times[-2], 1e-6)
        dyaw = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
        return np.array([(b[0]-a[0])/dt, (b[2]-a[2])/dt]), dyaw/dt


def cv_futures(tracks, now, steps, M=16, sigma_v=0.3, seed=0, include_nominal=True):
    """Legal constant-velocity futures [M, steps+1, N, 3] (+ nominal as index 0)."""
    rng = np.random.default_rng(seed)
    P, V = [], []
    for t in tracks:
        if not t.get('valid', True):
            continue
        p = np.asarray(t['position'], dtype=float)
        v = np.asarray(t['estimated_velocity'], dtype=float) if t.get('velocity_valid') and t.get('estimated_velocity') is not None else np.zeros(3)
        age = max(0., float(now)-float(t['last_seen_time']))
        P.append(p+v*age); V.append(v)
    if not P:
        return np.zeros((M, steps+1, 0, 3))
    P = np.asarray(P); V = np.asarray(V)
    ts = np.arange(steps+1)*STEP_SECONDS
    out = np.empty((M, steps+1, len(P), 3))
    for m in range(M):
        if include_nominal and m == 0:
            Vm = V
        else:
            Vm = V+np.concatenate([rng.normal(0, sigma_v, (len(P), 1)), np.zeros((len(P), 1)),
                                   rng.normal(0, sigma_v, (len(P), 1))], axis=1)
        out[m] = P[None]+ts[:, None, None]*Vm[None]
    return out


def tracks_from_future(F, k, now):
    """Synthetic legal-format tracks at future step k (positions + velocities)."""
    out = []
    for n in range(F.shape[1]):
        p = F[k, n]
        v = (F[min(k+1, F.shape[0]-1), n]-F[max(k-1, 0), n])/max(1, (min(k+1, F.shape[0]-1)-max(k-1, 0)))/STEP_SECONDS
        out.append(dict(track_id=f'sb_{n}', position=p.tolist(), estimated_velocity=v.tolist(), velocity_valid=True,
                        visible_now=True, valid=True, first_seen_time=now, last_seen_time=now,
                        observation_age=0., num_observations=5))
    return out


def rollout(legal, query, planner_name, dyn, F, steps):
    """Run the query executor in the sandbox against human future F [T+1, N, 3]."""
    sl = SandboxLegal(legal)
    ex = QueryExecution(query, make_planner(planner_name), dyn)
    level = 0
    poses = [sl.pose.copy()]
    for k in range(steps):
        sl.tracks = tracks_from_future(F, k, sl.time) if F.shape[1] else []
        prim, level = ex.step(sl, level)
        if prim == 'stop':
            break
        cmd = command_for(dyn, prim)
        # nominal kinematics of the realised command
        p = sl.pose.copy()
        if isinstance(cmd, str):
            if cmd == 'forward':
                d = dyn.step_m
                p[0] += d*math.cos(p[3]); p[2] -= d*math.sin(p[3])
            elif cmd == 'left':
                p[3] += dyn.turn_rad
            elif cmd == 'right':
                p[3] -= dyn.turn_rad
        else:
            lin, ang = cmd[0]/120., cmd[1]/120.
            p[3] += ang
            p[0] += lin*math.cos(p[3]); p[2] -= lin*math.sin(p[3])
        p[3] = math.atan2(math.sin(p[3]), math.cos(p[3]))
        sl.pose = p
        sl.time += STEP_SECONDS
        sl.poses.append(p.copy()); sl.times.append(sl.time)
        poses.append(p.copy())
        if ex.status != 'running':
            break
    return np.asarray(poses)


def score(traj, F):
    """Per-future metrics of one trajectory: min rectangle distance, overlap, PSC steps."""
    M = F.shape[0]
    T = min(len(traj), F.shape[1])
    out = np.zeros((M, 3))
    if F.shape[2] == 0:
        out[:, 0] = np.inf
        return out
    for m in range(M):
        dmin = np.inf; psc = 0
        for k in range(T):
            d = rectangle_distance(traj[k], F[m, k])
            dmin = min(dmin, float(d.min()))
            c = np.hypot(F[m, k, :, 0]-traj[k][0], F[m, k, :, 2]-traj[k][2])
            psc += int((c < 1.0).any())
        out[m] = (dmin, float(dmin < BODY_PROXY), psc)
    return out


@dataclass
class Comparison:
    query: str
    open_overlap: float
    closed_overlap: float
    open_dmin: float
    closed_dmin: float
    open_end: tuple
    closed_end_mean: tuple
    end_dev_mean: float
    open_s: float
    closed_s: float


def compare(legal, queries, planner_name, dyn, F, steps):
    """Open-loop (1 rollout + vectorised scoring) vs closed-loop (M rollouts)."""
    rows = []
    for q in queries:
        if q.kind in ('continue', 'stop'):
            continue
        t = time.perf_counter()
        nom = rollout(legal, q, planner_name, dyn, F[0], steps)
        S = score(nom, F)
        t_open = time.perf_counter()-t
        t = time.perf_counter()
        ends, cl = [], []
        for m in range(F.shape[0]):
            tr = rollout(legal, q, planner_name, dyn, F[m], steps)
            cl.append(score(tr, F[m:m+1])[0]); ends.append(tr[-1])
        t_closed = time.perf_counter()-t
        cl = np.asarray(cl); ends = np.asarray(ends)
        dev = float(np.mean(np.hypot(ends[:, 0]-nom[-1][0], ends[:, 2]-nom[-1][2])))
        rows.append(Comparison(q.label, float(S[:, 1].mean()), float(cl[:, 1].mean()),
                               float(np.median(S[:, 0])), float(np.median(cl[:, 0])),
                               (float(nom[-1][0]), float(nom[-1][2])),
                               (float(ends[:, 0].mean()), float(ends[:, 2].mean())), dev, t_open, t_closed))
    return rows
