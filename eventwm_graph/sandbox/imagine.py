"""Imagined Query execution inside the World Model Sandbox (legal, hypothetical).

One candidate = one Query executed by its own low-level planner (ORCA / SFM / DWA ...) and dynamics family through the
UNCHANGED querygen ``QueryExecution`` (the executor of the Oracle and of the real episode), but in imagination:

* the legal map is FROZEN at the decision time t (no future observation exists; predicted Stage 2 cells are never
  given to the planner -- they stay hypothetical outputs);
* the people the planner perceives are the currently tracked humans (never Stage 1's unseen residual slots) at the
  positions of ONE nominal future of the shared Stage 1 bank (or constant velocity without a bank), refreshed every
  imagined step as fully observed tracks;
* the robot moves with the family's nominal kinematics: per step it either translates straight ahead by the
  commanded level or turns in place (Falcon's base-velocity integration without the navmesh, which is privileged).

Every quantity is a function of the legal state at t, the candidate and the Stage 1 bank.  Nothing reads the
simulator.  The executor objects are deep copies: the real episode's state is never touched.
"""
from dataclasses import dataclass, field
import math
import time
from typing import Optional, Tuple

import numpy as np

from eventwm_graph.querygen.dynamics import STEP_SECONDS, CTRL_FREQ, FAMILIES
from eventwm_graph.querygen.executor import QueryExecution, command_for
from eventwm_graph.querygen.planners import make_planner
from eventwm_graph.querygen.control import human_estimates
from eventwm_graph.querygen.legal import BLOCKED

HISTORY_TAIL = 8          # real poses kept for the executor's measured-velocity estimate (it reads a 1-step window)
TRACK_PREFIX = 'track:'   # Stage 1 bank key of a legal track: 'track:' + legal track id (residual slots: 'residual:i')


# ---------------------------------------------------------------------------------------------- human futures
class NominalHumans:
    """The hypothetical people the imagined planner perceives at steps k = 0..steps (shared by all candidates).

    ``positions`` [steps+1, M, 3] world (y = the track's legal height), ``ids`` the legal track ids.  Built from the
    perceivable legal tracks at t (valid, at most 2 s old: exactly ``control.human_estimates``) and either the Stage 1
    bank (``mode`` 'mean': present-weighted mean over worlds; 'medoid': the world closest to the mean; 'world:i') or
    constant-velocity extrapolation (``mode`` 'cv', or no bank).  Beyond the bank horizon the nominal path continues
    at its last velocity.  A tracked slot present in less than half the worlds (weighted) is not perceived.
    """

    def __init__(self, legal_tracks, now, steps, bank=None, mode='mean', step=STEP_SECONDS):
        self.now = float(now)
        self.steps = int(steps)
        self.step = float(step)
        self.mode = mode if bank is not None else 'cv'
        self.fallback = 0
        est = human_estimates(legal_tracks, now)
        by_id = {t['track_id']: t for t in legal_tracks}
        rel = np.arange(self.steps+1)*self.step
        ids, heights, paths = [], [], []
        if bank is not None and self.mode != 'cv':
            keys = [str(k) for k in bank.track_keys]
            slot = {k[len(TRACK_PREFIX):]: n for n, k in enumerate(keys) if k.startswith(TRACK_PREFIX)}
            unseen = np.asarray(bank.unseen, bool)
            w = np.asarray(bank.weights, np.float64)
            horizon = float(bank.times[-1])
            covered = rel[rel <= horizon+1e-9]
            P = np.asarray(bank.sandbox_positions(covered), np.float64)          # [W, Tc, N, 2], +inf absent
            present = np.asarray(bank.present, bool)                              # [W, N]
            for h in est:
                n = slot.get(str(h.track_id))
                if n is None or unseen[n]:
                    # a perceivable track Stage 1 does not model: the planner still sees it (constant velocity)
                    p0 = np.array([h.position[0], -h.position[1]]); v = np.array([h.velocity[0], -h.velocity[1]])
                    ids.append(h.track_id); heights.append(float(by_id[h.track_id]['position'][1]))
                    paths.append(p0[None]+rel[:, None]*v[None])
                    self.fallback += 1
                    continue
                pw = present[:, n]
                if float(w[pw].sum()) < 0.5 or not pw.any():
                    continue
                Q = P[pw, :, n, :]                                                  # [Wp, Tc, 2]
                wp = w[pw]/w[pw].sum()
                if self.mode == 'mean':
                    path = np.tensordot(wp, Q, axes=(0, 0))
                elif self.mode == 'medoid':
                    mean = np.tensordot(wp, Q, axes=(0, 0))
                    path = Q[int(np.argmin(np.linalg.norm(Q-mean[None], axis=-1).sum(-1)))]
                elif str(self.mode).startswith('world:'):
                    i = int(str(self.mode).split(':')[1])
                    if not present[i, n]:
                        continue
                    path = P[i, :, n, :]
                else:
                    raise ValueError('unknown nominal mode %r' % self.mode)
                path = _extend(path, len(rel), self.step)
                ids.append(h.track_id); heights.append(float(by_id[h.track_id]['position'][1])); paths.append(path)
        else:
            for h in est:
                p0 = np.array([h.position[0], -h.position[1]])                      # planar -> world (x, z)
                v = np.array([h.velocity[0], -h.velocity[1]])
                ids.append(h.track_id); heights.append(float(by_id[h.track_id]['position'][1]))
                paths.append(p0[None]+rel[:, None]*v[None])
        self._build(ids, heights, paths)

    @classmethod
    def from_paths(cls, ids, heights, paths, now, steps, step=STEP_SECONDS, mode='given'):
        """Nominal humans from explicit planar world paths [steps+1, 2] per id (e.g. a privileged true future for
        validation; never used by the sandbox itself)."""
        self = cls.__new__(cls)
        self.now, self.steps, self.step, self.mode = float(now), int(steps), float(step), mode
        self.fallback = 0
        self._build(list(ids), list(heights), [np.asarray(p, np.float64) for p in paths])
        return self

    def _build(self, ids, heights, paths):
        self.ids = tuple(ids)
        M = len(ids)
        self.positions = np.zeros((self.steps+1, M, 3))
        for m, (y, path) in enumerate(zip(heights, paths)):
            self.positions[:, m, 0] = path[:, 0]; self.positions[:, m, 1] = y; self.positions[:, m, 2] = path[:, 1]
        vel = np.zeros_like(self.positions)
        if self.steps > 0 and M:
            vel[1:-1] = (self.positions[2:]-self.positions[:-2])/(2*self.step)
            vel[0] = (self.positions[1]-self.positions[0])/self.step
            vel[-1] = (self.positions[-1]-self.positions[-2])/self.step
        self.velocities = vel
        self._tracks = [None]*(self.steps+1)

    def tracks(self, k):
        """Legal-format synthetic tracks at imagined step k (read-only dicts shared by every candidate)."""
        k = min(int(k), self.steps)
        out = self._tracks[k]
        if out is None:
            t = self.now+k*self.step
            out = [dict(track_id=i, position=self.positions[k, m].tolist(), estimated_velocity=self.velocities[k, m].tolist(),
                        velocity_valid=True, visible_now=True, valid=True, first_seen_time=self.now, last_seen_time=t,
                        observation_age=0., num_observations=5, hypothetical=True)
                   for m, i in enumerate(self.ids)]
            self._tracks[k] = out
        return out


def _extend(path, n, step):
    """Path [Tc, 2] -> [n, 2]: continue at the last velocity past the bank horizon."""
    if len(path) >= n:
        return path[:n]
    v = path[-1]-path[-2] if len(path) > 1 else np.zeros(2)          # per-step displacement
    extra = path[-1][None]+np.arange(1, n-len(path)+1)[:, None]*v[None]
    return np.concatenate([path, extra])


# ---------------------------------------------------------------------------------------------- legal view
class ImaginedLegal:
    """The executor-facing legal state inside imagination (the attributes ``QueryExecution`` reads).

    ``map`` is the legal map at t, shared read-only by every candidate (the executor never writes it: its tight /
    thru / standing-human variants are private copies); ``bumps`` the legal proprioceptive bump memory at t.
    ``pose`` / ``time`` / ``tracks`` advance in imagination.
    """

    __slots__ = ('goal', 'map', 'bumps', 'start_time', 'time', 'pose', 'poses', 'times', 'tracks', 'steps')

    def __init__(self, legal, tracks, gmap=None):
        self.goal = legal.goal
        self.map = legal.map if gmap is None else gmap
        self.bumps = list(getattr(legal, 'bumps', ()) or ())
        self.start_time = legal.start_time
        self.time = float(legal.time)
        self.pose = np.asarray(legal.pose, np.float64).copy()
        self.poses = [np.asarray(p, np.float64) for p in legal.poses[-HISTORY_TAIL:]]
        self.times = [float(t) for t in legal.times[-HISTORY_TAIL:]]
        self.tracks = tracks
        self.steps = 0

    def velocity(self, window=1):
        """``LegalState.velocity``: measured planar velocity and yaw rate over the last ``window`` steps."""
        if len(self.poses) < 2:
            return np.zeros(2), 0.
        k = min(window, len(self.poses)-1)
        a, b = self.poses[-1-k], self.poses[-1]
        dt = max(self.times[-1]-self.times[-1-k], 1e-6)
        dyaw = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
        return np.array([(b[0]-a[0])/dt, (b[2]-a[2])/dt]), dyaw/dt


# ---------------------------------------------------------------------------------------------- kinematics
def advance(pose, dyn, cmd):
    """Nominal kinematics of one realised command (Falcon base-velocity integration, translation first; no navmesh).

    Native actions: 'forward' = ``step_m`` straight ahead, 'left'/'right' = +/- ``turn_rad``.  Velocity families:
    (lin, ang) integrator arguments, i.e. lin / CTRL_FREQ metres and ang / CTRL_FREQ radians per step.
    Yaw convention of the frames: forward = (cos yaw, -sin yaw) in world (x, z).
    """
    x, y, z, yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
    if isinstance(cmd, str):
        d = w = 0.
        if cmd == 'forward':
            d = dyn.step_m
        elif cmd == 'left':
            w = dyn.turn_rad
        elif cmd == 'right':
            w = -dyn.turn_rad
        elif cmd not in ('stop', 'idle'):
            raise ValueError('unknown native command %r' % (cmd,))
    else:
        d, w = float(cmd[0])/CTRL_FREQ, float(cmd[1])/CTRL_FREQ
    x += d*math.cos(yaw); z -= d*math.sin(yaw)
    yaw += w
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return np.array([x, y, z, yaw])


# ---------------------------------------------------------------------------------------------- one rollout
@dataclass
class ImaginedRollout:
    """The imagined execution of one candidate (all HYPOTHETICAL)."""
    poses: np.ndarray                 # [T+1, 4] x, y, z, yaw; row 0 = the real pose at t
    times: np.ndarray                 # [T+1] absolute times t + k * 0.048 s
    commands: list                    # realised robot commands (as sent to the environment)
    levels: np.ndarray                # [T+1] speed level after each step (stateful families; 0 otherwise)
    status: str                       # executor status at the end: running | done | failed
    reason: str                       # executor reason (arrived, timeout, ...)
    ended_by: str                     # horizon | done | stop | failed | error
    stop: bool                        # the candidate issued STOP (ends the episode in reality)
    goal_distance: float              # legal Euclidean distance to the public goal at the end
    progress: float                   # legal Euclidean goal-distance decrease over the rollout
    path_length: float
    blocked_steps: int                # imagined poses whose base cell is legally BLOCKED (the planner avoids them)
    nominal_clearance: float          # min robot-rectangle distance to the nominal humans' centres (m)
    planner_ms: float
    error: Optional[str] = None
    hypothetical: bool = True
    tail: Optional[np.ndarray] = None   # [B, 4] imagined emergency stop from the end pose (straight ahead, braking at the
                                        # family's limit); empty for robots without speed state.  Not part of the
                                        # trajectory given to Layer 2 (a consequence summary for the Graph Compiler).
    cont: Optional[np.ndarray] = None   # [C, 4] continuation (SandboxConfig.continuation > 0): what the robot would do
                                        # after the interval under the default policy -- keep the running query, else the
                                        # goal query -- same executor, planner, dynamics and nominal humans.  Not given
                                        # to Layer 2; the Graph Compiler scores it against the Stage 1 worlds.

    @property
    def end_level(self):
        return int(self.levels[-1]) if len(self.levels) else 0

    @property
    def steps(self):
        return len(self.poses)-1

    def trajectory(self):
        """Canonical Layer 2 candidate: N x 5 [t, x, y, z, yaw]."""
        return np.c_[self.times, self.poses]


def rectangle_clearance(pose, points):
    """Min distance from the robot body rectangle at ``pose`` to planar points [M, 2] (world x, z)."""
    from eventwm_graph.querygen.native import ROBOT_BOUNDS
    if not len(points):
        return math.inf
    c, s = math.cos(pose[3]), math.sin(pose[3])
    dx = points[:, 0]-pose[0]; dz = points[:, 1]-pose[2]
    lx = dx*c-dz*s; lz = dx*s+dz*c
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    ex = np.maximum(np.maximum(xmin-lx, 0.), lx-xmax)
    ez = np.maximum(np.maximum(zmin-lz, 0.), lz-zmax)
    return float(np.hypot(ex, ez).min())


def frozen_map(gmap):
    """A read-only alias of the legal map at t for imagination: a distinct GridMap object sharing the arrays (never
    written: the executor only reads maps and builds private variants), flagged for the exact per-decision caches
    (``accel``).  The real episode's map object is not touched."""
    from eventwm_graph.querygen.legal import GridMap
    g = GridMap.__new__(GridMap)
    g.__dict__.update(gmap.__dict__)
    g._sandbox_frozen = True
    return g


def imagine(legal, execution, dyn, humans, steps, level=0, gmap=None, continuation=0, cont_query=None, planner=None):
    """Roll ``execution`` (a fresh or copied ``QueryExecution``; consumed) forward for at most ``steps`` imagined
    steps from the legal state ``legal`` at t, against the nominal humans.  Returns an ``ImaginedRollout``.
    ``gmap``: the decision's frozen map alias (default: ``legal.map`` itself)."""
    tick = time.perf_counter()
    sl = ImaginedLegal(legal, humans.tracks(0), gmap)
    poses = [sl.pose.copy()]
    levels = [int(level)]
    commands = []
    ended, stop, error = 'horizon', False, None
    gmap = sl.map
    blocked = 0
    clearance = math.inf
    for k in range(int(steps)):
        sl.tracks = humans.tracks(k)
        try:
            prim, level = execution.step(sl, level)
        except Exception as exc:          # a planner failure ends this candidate only (recorded, never raised)
            ended, error = 'error', '%s: %s' % (type(exc).__name__, exc)
            break
        if prim == 'stop':
            ended, stop = 'stop', True
            break
        try:
            cmd = command_for(dyn, prim)
            p = advance(sl.pose, dyn, cmd)
        except Exception as exc:
            ended, error = 'error', '%s: %s' % (type(exc).__name__, exc)
            break
        if not np.isfinite(p).all():
            ended, error = 'error', 'non-finite imagined pose'
            break
        commands.append(cmd if isinstance(cmd, str) else tuple(map(float, cmd)))
        sl.time += STEP_SECONDS
        sl.pose = p
        sl.poses.append(p); sl.times.append(sl.time)
        if len(sl.poses) > HISTORY_TAIL:
            del sl.poses[0]; del sl.times[0]
        sl.steps += 1
        poses.append(p); levels.append(int(level))
        if gmap.state_at(p[0], p[2]) == BLOCKED:
            blocked += 1
        if humans.positions.shape[1]:
            hp = humans.positions[min(k+1, humans.steps)]
            clearance = min(clearance, rectangle_clearance(p, hp[:, [0, 2]]))
        if execution.status != 'running':
            ended = 'done' if execution.status == 'done' else 'failed'
            break
    cont = None
    if continuation > 0 and not stop and error is None and cont_query is not None:
        cont = _continue(sl, execution, dyn, humans, len(poses)-1, int(continuation), int(levels[-1]), cont_query,
                         planner)
    P = np.asarray(poses)
    g = np.asarray(legal.goal, np.float64)
    d0 = float(math.hypot(g[0]-P[0, 0], g[2]-P[0, 2]))
    d1 = float(math.hypot(g[0]-P[-1, 0], g[2]-P[-1, 2]))
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 2])) if len(P) > 1 else np.zeros(0)
    return ImaginedRollout(poses=P, times=float(legal.time)+np.arange(len(P))*STEP_SECONDS, commands=commands,
                           levels=np.asarray(levels, np.int16), status=execution.status, reason=execution.reason,
                           ended_by=ended, stop=stop, goal_distance=d1, progress=d0-d1, path_length=float(seg.sum()),
                           blocked_steps=int(blocked), nominal_clearance=float(clearance),
                           planner_ms=(time.perf_counter()-tick)*1000, error=error,
                           tail=brake_tail(P[-1], levels[-1], dyn) if not stop else np.zeros((0, 4)), cont=cont)


def _continue(sl, execution, dyn, humans, k0, n, level, goal_query, planner):
    """Continue the imagination for ``n`` steps under the default policy (keep the running query at a decision
    boundary, else the goal query; the goal query again when the running one ends).  Returns poses [<=n, 4]."""
    out = []
    ex = execution
    if ex.status == 'running':
        ex.decision_boundary()
    else:
        ex = new_execution(goal_query, planner, dyn)
    for j in range(n):
        sl.tracks = humans.tracks(k0+j)
        try:
            prim, level = ex.step(sl, level)
            if prim == 'stop':
                break
            p = advance(sl.pose, dyn, command_for(dyn, prim))
        except Exception:
            break
        if not np.isfinite(p).all():
            break
        sl.time += STEP_SECONDS
        sl.pose = p
        sl.poses.append(p); sl.times.append(sl.time)
        if len(sl.poses) > HISTORY_TAIL:
            del sl.poses[0]; del sl.times[0]
        sl.steps += 1
        out.append(p)
        if ex.status != 'running':
            if ex.query is goal_query:
                break
            ex = new_execution(goal_query, planner, dyn)
    return np.asarray(out, np.float64).reshape(-1, 4)


def brake_tail(pose, level, dyn, max_steps=24):
    """Poses of an imagined emergency stop from ``pose`` at speed ``level``: straight ahead, braking by ``dyn.brake``
    levels per step (the motion realiser's braking rule) until rest.  Empty for families without speed state."""
    out = []
    if not getattr(dyn, 'stateful', False):
        return np.zeros((0, 4))
    p = np.asarray(pose, np.float64)
    lv = int(level)
    while lv > 0 and len(out) < max_steps:
        lv = max(0, lv-int(dyn.brake))
        p = advance(p, dyn, dyn.to_command(dyn.levels[lv], 0.))
        out.append(p)
    return np.asarray(out, np.float64).reshape(-1, 4)


def new_execution(query, planner, dyn, planner_kw=None):
    return QueryExecution(query, make_planner(planner, **(planner_kw or {})), dyn)


def failed_rollout(legal, error):
    """A candidate whose execution could not start: stationary one-row trajectory, error recorded."""
    p = np.asarray(legal.pose, np.float64).reshape(1, 4).copy()
    return ImaginedRollout(poses=p, times=np.asarray([float(legal.time)]), commands=[], levels=np.zeros(1, np.int16),
                           status='failed', reason='error', ended_by='error', stop=False, goal_distance=math.nan,
                           progress=0., path_length=0., blocked_steps=0, nominal_clearance=math.inf, planner_ms=0.,
                           error=error)


def dynamics(name_or_model):
    return FAMILIES[name_or_model] if isinstance(name_or_model, str) else name_or_model
