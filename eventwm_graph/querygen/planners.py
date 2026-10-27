"""Low-level planners behind one interface (legal inputs only).

* ``ORCAPlanner``  (P1) full-responsibility ORCA velocity + kinematic adapter.
* ``SFMPlanner``   (P2) Helbing social force model with velocity-dependent
  elliptical repulsion (Johansson, Helbing & Shukla 2007) and anisotropy.
* ``DWAPlanner``   (P3) dynamic-window sampling with forward simulation over
  the robot's admissible commands (Fox, Burgard & Thrun 1997).
* ``FollowPlanner`` diagnostic legal-route follower without avoidance.

Every planner receives a ``PlanContext`` built by the query executor from the
legal state and returns a motion primitive ('forward'/'left'/'right') for
discrete robots or a (v, w) command in m/step, rad/step for unicycles.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from .control import (NativeAdapter, heading_vec, wrap, body_ok, body_cost, turn_ok, to_world,
                      BODY_X, BODY_HALF_WIDTH, FORWARD_X)
from .dynamics import STEP_SECONDS
from .orca import ORCASolver, ORCAConfig
from .legal import BLOCKED


class Realized(tuple):
    """(command primitive, new speed level) already produced under the dynamics."""
    __slots__ = ()

    def __new__(cls, cmd, level):
        return tuple.__new__(cls, (cmd, int(level)))


@dataclass
class PlanContext:
    P: np.ndarray            # planar position
    psi: float               # yaw
    V: np.ndarray            # measured planar velocity (m/s)
    dyn: object              # DynamicsModel
    dyn_state: tuple         # (v m/step, w rad/step) for unicycles
    v_pref: np.ndarray       # preferred planar velocity (m/s)
    humans: list             # HumanEstimate list
    statics: np.ndarray      # nearby observed BLOCKED points (planar)
    gmap: object             # legal GridMap
    hold: bool = False       # query asks to hold position
    face: float = None       # heading to face while holding / at the end
    progress_fn: object = None   # planar point -> route cost-to-go (m), for DWA
    route_dir: float = None  # route tangent heading at the robot
    plan_prims: list = None  # body-aware static plan (shared static layer)
    plan_poses: list = None

    @property
    def vmax(self):
        return self.dyn.v_max/STEP_SECONDS


class DiscreteAdapter(NativeAdapter):
    """NativeAdapter parametrised by a discrete DynamicsModel."""

    def __init__(self, dyn, **kw):
        super().__init__(**kw)
        self.step_m = dyn.step_m
        self.turn = dyn.turn_rad
        self.vmax = dyn.step_m/STEP_SECONDS


class UnicycleAdapter:
    """Tracks a desired planar velocity with a limited unicycle."""

    def __init__(self, dyn, k_heading=2.5):
        self.dyn = dyn
        self.k = k_heading

    def reset(self):
        pass

    def command(self, ctx, v_des, hard_ok=None, face=None):
        dyn = ctx.dyn
        speed = float(np.hypot(*v_des))*STEP_SECONDS   # m/step
        if speed < 0.01:
            w = 0. if face is None else self.k*wrap(face-ctx.psi)*dyn.turn_rad
            v, w = dyn.limit((0., w), ctx.dyn_state)
            return (v, w)
        theta = math.atan2(v_des[1], v_des[0])
        err = wrap(theta-ctx.psi)
        v = speed*max(0., math.cos(err))
        w = max(-dyn.w_max, min(dyn.w_max, err))       # reach heading in one step if allowed
        v, w = dyn.limit((v, w), ctx.dyn_state)
        # body check: do not translate into observed walls
        if v > 0 and ctx.gmap is not None:
            P1 = ctx.P+v*heading_vec(ctx.psi+w*.5)
            if not body_ok(ctx.gmap, P1, ctx.psi+w):
                v, w = dyn.limit((0., w), ctx.dyn_state)
        return (v, w)


def make_adapter(dyn):
    return DiscreteAdapter(dyn)


class Planner:
    name = 'base'
    mode = None

    def reset(self):
        pass

    def act(self, ctx, adapter):
        raise NotImplementedError

    def _emit(self, ctx, adapter, v_des, hard_ok=None):
        if ctx.dyn.discrete:
            a, mode = adapter.command(ctx.P, ctx.psi, v_des, hard_ok, ctx.gmap, ctx.face)
            self.mode = mode
            return a
        return adapter.command(ctx, v_des, hard_ok, ctx.face)


class FollowPlanner(Planner):
    """Static plan / legal-route follower; no pedestrian avoidance (diagnostic)."""
    name = 'follow'
    mode = None

    def act(self, ctx, adapter):
        if not ctx.hold and ctx.plan_prims and ctx.dyn.discrete:
            self.mode = 'plan'
            return ctx.plan_prims[0]
        return self._emit(ctx, adapter, np.zeros(2) if ctx.hold else ctx.v_pref)


def one_step_hard_ok(humans, clearance):
    """Predicate: robot rectangle at (P, psi) keeps ``clearance`` from every
    human's constant-velocity position one control step ahead."""
    if not humans:
        return None
    H = np.stack([h.at(STEP_SECONDS) for h in humans])

    def ok(P, psi):
        c, s = math.cos(psi), math.sin(psi)
        d = H-P
        lx = d[:, 0]*c+d[:, 1]*s
        ly = -d[:, 0]*s+d[:, 1]*c
        dx = np.maximum(np.maximum(BODY_X[0]-lx, 0.), lx-BODY_X[1])
        dy = np.maximum(np.abs(ly)-BODY_HALF_WIDTH, 0.)
        return bool((np.hypot(dx, dy) >= clearance).all())
    return ok


def governed_speed(lines, u, s_pref):
    """Largest s in [0, s_pref] with s*u inside every ORCA half-plane; None if no s >= 0 is."""
    from .orca import det
    lo, hi = 0., s_pref
    for ln in lines:
        a = det(ln.direction, ln.point)
        b = det(ln.direction, u)
        if abs(b) < 1e-9:
            if a > 1e-9:
                return None
            continue
        if b > 0:
            lo = max(lo, a/b)
        else:
            hi = min(hi, a/b)
    if lo > hi+1e-9:
        return None
    return max(0., hi) if lo <= 1e-9 else None


class ORCAPlanner(Planner):
    """Full-responsibility ORCA; the solver velocity is the agent's model
    velocity (RVO2 semantics) and is tracked by the dynamics adapter."""
    name = 'orca'

    def __init__(self, config=None, hard_clearance=0.30, turn_commit=True, nh_project=True, commit=False):
        self.solver = ORCASolver(config or ORCAConfig())
        self.hard_clearance = hard_clearance
        self.turn_commit = turn_commit
        self.nh_project = nh_project
        self.commit = commit          # 'commit' queries: follow the static plan under the hard check only
        self.v = None
        self.last = None

    def reset(self):
        self.v = None

    def act(self, ctx, adapter):
        if self.v is None:
            self.v = ctx.V.copy()
        v_pref = np.zeros(2) if ctx.hold else ctx.v_pref
        # Static layer already keeps the body off walls; ORCA handles humans.
        statics = ctx.statics if not ctx.plan_prims else np.zeros((0, 2))
        v, lines, nobst = self.solver.solve(ctx.P, self.v, v_pref, ctx.vmax, ctx.humans, statics, STEP_SECONDS)
        self.v = v
        self.last = dict(v=v.tolist(), n_lines=len(lines), n_obst=nobst)
        hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
        if self.commit and not ctx.hold and ctx.plan_prims and ctx.dyn.discrete:
            # decisive execution (timing is the decision layer's job): the body-aware static plan
            # is followed whenever its next pose keeps the one-step rectangle clearance; otherwise
            # hold (or step forward out of an imminent violation)
            prim = ctx.plan_prims[0]
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self.mode = 'commit'
                return prim
            return self._emit(ctx, adapter, np.zeros(2), hard)
        if (not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and
                float(np.hypot(*(v-v_pref))) <= 0.05*ctx.vmax):
            # Nominal motion is ORCA-admissible: execute the static plan.
            prim = ctx.plan_prims[0]
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self.mode = 'plan'
                return prim
        if (self.turn_commit and not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and
                ctx.plan_prims[0] in ('left', 'right') and (hard is None or hard(ctx.P, ctx.psi))):
            # v9: an in-place turn of the static plan does not translate the robot, so ORCA's
            # velocity constraint does not apply to it; executing it (if the one-step rectangle
            # check holds) avoids the plan/ORCA left-right dithering that wasted up to ~25
            # steps per corridor entry (e.g. Qpor2mEya8F steps 17-28 and 37-46).
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, ctx.plan_prims[0])
            if hard is None or hard(P1, psi1):
                self.mode = 'plan_turn'
                return ctx.plan_prims[0]
        if (self.nh_project and not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and
                ctx.plan_prims[0] == 'forward'):
            # v9: non-holonomic projection.  A turn-in-place robot cannot realise ORCA's lateral
            # deviation without stopping to turn (it dithered between the plan heading and ORCA's
            # direction).  Keep the plan heading and take the largest speed along it that satisfies
            # every ORCA half-plane (possibly zero = hold); only if even holding violates ORCA
            # (a human on a collision course) fall back to ORCA's evasive velocity.
            u = heading_vec(ctx.psi)
            s_gov = governed_speed(lines, u, float(np.hypot(*v_pref)))
            if s_gov is not None:
                self.mode = 'governed'
                return self._emit(ctx, adapter, s_gov*u, hard)
        return self._emit(ctx, adapter, v, hard)


@dataclass
class SFMConfig:
    relaxation: float = 0.25        # s, goal-force time constant (benchmark time)
    human_strength: float = 1.6     # x v0/relaxation
    human_range: float = 0.45       # m (B)
    contact_radius: float = 0.95    # m robot+human proxy radius (r_ij)
    ellipse_dt: float = 0.5         # s, step for velocity-dependent ellipses
    anisotropy: float = 0.3         # lambda
    wall_strength: float = 0.8      # x v0/relaxation
    wall_range: float = 0.15        # m
    wall_radius: float = 0.30       # m robot centre to BLOCKED-cell centre


class SFMPlanner(Planner):
    name = 'sfm'

    def __init__(self, config=None, hard_clearance=0.30, turn_commit=True, nh_project=True):
        self.cfg = config or SFMConfig()
        self.v = None
        self.hard_clearance = hard_clearance
        self.turn_commit = turn_commit      # v9 (as ORCA): execute static-plan in-place turns under the hard check
        self.nh_project = nh_project        # v9: keep the plan heading; social force only scales the speed

    def reset(self):
        self.v = None

    def force(self, ctx, v):
        cfg = self.cfg
        v0 = float(np.hypot(*ctx.v_pref))
        scale = max(v0, 1e-3)/cfg.relaxation
        e = ctx.v_pref/v0 if v0 > 1e-9 else heading_vec(ctx.psi)
        F = (ctx.v_pref-v)/cfg.relaxation
        for h in ctx.humans:
            d = ctx.P-h.position
            y = h.velocity*cfg.ellipse_dt
            dn = float(np.hypot(*d)); dy = d-y; dyn_ = float(np.hypot(*dy)); yn = float(np.hypot(*y))
            if dn < 1e-6:
                continue
            b = 0.5*math.sqrt(max((dn+dyn_)**2-yn*yn, 1e-12))
            mag = cfg.human_strength*scale*math.exp((cfg.contact_radius-b)/cfg.human_range)
            grad = (dn+dyn_)/(4.*b)*(d/dn+(dy/dyn_ if dyn_ > 1e-9 else d/dn))
            cosphi = float(np.dot(e, -d/dn))
            w = cfg.anisotropy+(1-cfg.anisotropy)*(1+cosphi)/2
            F = F+mag*w*grad
        for Q in ctx.statics:
            d = ctx.P-Q
            dn = float(np.hypot(*d))
            if dn < 1e-6:
                continue
            F = F+cfg.wall_strength*scale*math.exp((cfg.wall_radius-dn)/cfg.wall_range)*d/dn
        return F

    def act(self, ctx, adapter):
        if self.v is None:
            self.v = ctx.V.copy()
        if ctx.hold:
            ctx_pref = np.zeros(2)
        else:
            ctx_pref = ctx.v_pref
        saved = ctx.v_pref
        ctx.v_pref = ctx_pref
        if ctx.plan_prims:
            ctx.statics = np.zeros((0, 2))       # static layer handles walls
        F = self.force(ctx, self.v)
        F_goal = (ctx.v_pref-self.v)/self.cfg.relaxation
        ctx.v_pref = saved
        v = self.v+F*STEP_SECONDS
        n = float(np.hypot(*v))
        if n > ctx.vmax:
            v = v/n*ctx.vmax
        self.v = v
        hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
        social = float(np.hypot(*(F-F_goal)))*STEP_SECONDS
        if not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and social <= 0.05*ctx.vmax:
            prim = ctx.plan_prims[0]
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self.mode = 'plan'
                self.v = ctx.v_pref.copy()
                return prim
        if not ctx.hold and ctx.plan_prims and ctx.dyn.discrete:
            prim = ctx.plan_prims[0]
            if self.turn_commit and prim in ('left', 'right'):
                P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
                if hard is None or (hard(ctx.P, ctx.psi) and hard(P1, psi1)):
                    self.mode = 'plan_turn'
                    return prim
            if self.nh_project and prim == 'forward':
                # non-holonomic projection (v9): the social force may only slow the robot along its
                # heading (a turn-in-place robot cannot follow a lateral deflection without dithering)
                u = heading_vec(ctx.psi)
                s_ = max(0., min(float(np.dot(v, u)), float(np.hypot(*ctx.v_pref))))
                self.mode = 'governed'
                return self._emit(ctx, adapter, s_*u, hard)
        return self._emit(ctx, adapter, v, hard)


@dataclass
class DWAConfig:
    horizon: int = 16               # control steps simulated per candidate
    w_progress: float = 1.0         # per metre of legal-route cost-to-go reduction
    w_clearance: float = 0.6        # per metre of min human clearance (capped)
    clearance_cap: float = 1.2
    w_heading: float = 0.25
    w_plan: float = 0.15            # bonus for following the static body-aware plan
    safety: float = 0.30            # admissible min rectangle-to-human clearance (m)
    turn_options: tuple = (0, 1, 2, 4, 7, 11, 16)
    speed_caps: tuple = (1.0, 0.5)  # fraction of top speed level for 'go' segments
    w_commit: float = 0.3           # bonus for continuing the previously chosen trajectory


class DWAPlanner(Planner):
    """Dynamic Window Approach over the robot's admissible command sequences.

    Candidates: follow the static body-aware plan; rotate k steps either way
    (braking first when moving) then advance with the speed capped; hold.
    Each candidate is rolled out with the exact dynamics family (speed
    levels, accel/brake limits, rotation only at rest) against legal
    constant-velocity human predictions and the legal map.  A candidate is
    admissible if the body stays off observed walls and the rectangle keeps
    ``safety`` clearance from every predicted human over the horizon
    (including the braking tail).  The best admissible candidate by progress
    + clearance + heading (+ plan agreement) is chosen and its first command
    executed; with none admissible the maximum-clearance candidate is used.
    """
    name = 'dwa'
    realizes = True

    def __init__(self, config=None):
        self.cfg = config or DWAConfig()
        self._hold = 1.
        self._committed = None
        self._memo = {}

    def reset(self):
        self._committed = None

    @staticmethod
    def _rect_dist(P, psi, Q):
        c, s = math.cos(psi), math.sin(psi)
        d = Q-P
        lx = d[..., 0]*c+d[..., 1]*s
        ly = -d[..., 0]*s+d[..., 1]*c
        dx = np.maximum(np.maximum(BODY_X[0]-lx, 0.), lx-BODY_X[1])
        dy = np.maximum(np.abs(ly)-BODY_HALF_WIDTH, 0.)
        return np.hypot(dx, dy)

    def _sequences(self, ctx):
        """Rotate k steps (either way), advance m steps (speed-capped), then hold."""
        dyn, T = ctx.dyn, self.cfg.horizon
        top = len(dyn.levels)-1
        seqs = []
        if ctx.plan_prims and not ctx.hold:
            plan = list(ctx.plan_prims[:T])
            seqs.append(('plan', plan+['hold']*(T-len(plan)), top))
        seqs.append(('hold', ['hold']*T, 0))
        for k in self.cfg.turn_options:
            for side in (('left', 'right') if k else ('none',)):
                rot = [side]*k if k else []
                if k:
                    seqs.append((f'{side}{k}_hold', rot+['hold']*(T-k), 0))
                if ctx.hold:
                    continue
                for m in sorted({4, max(1, T-k)}):
                    if k+m > T:
                        continue
                    caps = self.cfg.speed_caps if (k == 0 and dyn.stateful) else (1.0,)
                    for cap in caps:
                        seqs.append((f'{side}{k}_go{m}_c{cap}', rot+['go']*m+['hold']*(T-k-m),
                                     max(1, int(round(top*cap)))))
        return seqs

    def _body_ok(self, gmap, P, psi, budget0, x_range=None):
        """body_ok memoised within one act() call (exact: keyed on the exact float pose)."""
        key = (float(P[0]), float(P[1]), float(psi), x_range is not None)
        memo = self._memo
        r = memo.get(key)
        if r is None:
            r = body_ok(gmap, P, psi, True, budget0, x_range) if x_range is not None else body_ok(gmap, P, psi, True, budget0)
            memo[key] = r
        return r

    def _rollout(self, ctx, intents, cap, budget0=0):
        """Exact dynamics rollout -> list of (P, psi, level, command) per step.

        An advance segment ends (and the rest of the sequence holds) as soon
        as continuing could not stop before the body reaches an observed wall.
        """
        dyn = ctx.dyn
        P, psi, level = ctx.P.copy(), ctx.psi, int(ctx.dyn_state or 0)
        out = []
        flip = self._hold
        blocked = False
        intents = list(intents)
        for idx, it in enumerate(intents):
            if it == 'go' and blocked:
                it = 'hold'
            if it == 'go' and ctx.gmap is not None:
                lvl = min(cap, level+dyn.accel) if level < cap else max(cap, level-dyn.brake)
                Pn = P+dyn.levels[lvl]*heading_vec(psi)
                stop = Pn+dyn.stop_distance(lvl)*heading_vec(psi)
                if not (self._body_ok(ctx.gmap, Pn, psi, budget0, FORWARD_X) and
                        self._body_ok(ctx.gmap, stop, psi, budget0, FORWARD_X)):
                    blocked = True
                    it = 'hold'
            if it in ('left', 'right'):
                if level > 0 and dyn.turn_needs_stop:
                    level = max(0, level-dyn.brake)
                    P = P+dyn.levels[level]*heading_vec(psi)
                    cmd = ('forward', level) if level > 0 else 'idle'
                else:
                    psi += dyn.turn_rad if it == 'left' else -dyn.turn_rad
                    cmd = it
            elif it == 'hold':
                if level > 0:
                    level = max(0, level-dyn.brake)
                    P = P+dyn.levels[level]*heading_vec(psi)
                    cmd = ('forward', level) if level > 0 else 'idle'
                else:
                    psi += flip*dyn.turn_rad
                    cmd = 'left' if flip > 0 else 'right'
                    flip = -flip
            else:   # 'go' / 'forward': accelerate toward the cap
                if it == 'forward':
                    cap_now = len(dyn.levels)-1
                else:
                    cap_now = cap
                level = min(cap_now, level+dyn.accel) if level < cap_now else max(cap_now, level-dyn.brake)
                P = P+dyn.levels[level]*heading_vec(psi)
                cmd = ('forward', level) if level > 0 else 'idle'
            out.append((P.copy(), psi, level, cmd))
        # braking tail (must also be safe)
        tail = []
        while level > 0:
            level = max(0, level-dyn.brake)
            P = P+dyn.levels[level]*heading_vec(psi)
            tail.append((P.copy(), psi, level, None))
        return out, tail

    def act(self, ctx, adapter):
        try:
            return self._act(ctx, adapter)
        finally:
            self._memo = {}          # per-call memo only (never carried into branch copies)

    def _act(self, ctx, adapter):
        cfg = self.cfg
        dyn = ctx.dyn
        H = ctx.humans
        best = fallback = None
        start_cost = ctx.progress_fn(ctx.P) if (ctx.progress_fn and not ctx.hold) else 0.
        budget0 = body_cost(ctx.gmap, ctx.P, ctx.psi) if ctx.gmap is not None else 0
        self._memo = {}
        dmemo = {}
        seqs = self._sequences(ctx)
        if self._committed and not ctx.hold:
            seqs.append(('commit', self._committed[0]+['hold']*(cfg.horizon-len(self._committed[0])), self._committed[1]))
        for name, intents, cap in seqs:
            steps, tail = self._rollout(ctx, intents, cap, budget0)
            traj = steps+tail
            ok = True
            for k, (P, psi, _, _) in enumerate(traj):
                if ctx.gmap is not None and k % 2 == 0 and not self._body_ok(ctx.gmap, P, psi, budget0):
                    ok = False
                    break
            if not ok:
                continue
            if H:
                ds = []
                for k, (P, psi, _, _) in enumerate(traj):
                    key = (float(P[0]), float(P[1]), float(psi), k)
                    d = dmemo.get(key)
                    if d is None:
                        d = dmemo[key] = float(min(self._rect_dist(P, psi, h.at((k+1)*STEP_SECONDS)) for h in H))
                    ds.append(d)
                dmin = min(ds)
            else:
                dmin = cfg.clearance_cap+cfg.safety
            Pend, psi_end = steps[-1][0], steps[-1][1]
            progress = 0. if ctx.hold or not ctx.progress_fn else start_cost-ctx.progress_fn(Pend)
            target = ctx.face if ctx.hold and ctx.face is not None else ctx.route_dir
            heading = math.cos(wrap(target-psi_end)) if target is not None else 0.
            score = (cfg.w_progress*progress+cfg.w_clearance*min(dmin, cfg.clearance_cap)+cfg.w_heading*heading
                     +(cfg.w_plan if name == 'plan' else 0.)+(cfg.w_commit if name == 'commit' else 0.))
            if fallback is None or dmin > fallback[0]:
                fallback = (dmin, steps)
            if dmin < cfg.safety:
                continue
            if best is None or score > best[0]:
                best = (score, steps, name, intents, cap)
        chosen = best if best is not None else (None, fallback[1], 'fallback') if fallback is not None else None
        if chosen is None:
            self.mode = 'blocked'
            if dyn.stateful:
                # v10: a moving robot brakes at its brake limit (v9 commanded zero velocity here while the speed
                # level only dropped by `brake`: an instantaneous stop the dynamics model forbids)
                l2 = max(0, int(ctx.dyn_state or 0)-dyn.brake)
                return Realized(('forward', l2) if l2 > 0 else 'idle', l2)
            return Realized(adapter.wait_action(ctx.P, ctx.psi, ctx.gmap), 0)
        steps, self.mode = chosen[1], chosen[2]
        cmd, level = steps[0][3], steps[0][2]
        if best is not None:
            rest = [i for i in best[3][1:] if i != 'hold']
            self._committed = (list(best[3][1:]), best[4]) if rest else None
        else:
            self._committed = None
        if self.mode == 'hold' and cmd in ('left', 'right'):
            self._hold = -self._hold
        if not dyn.stateful:
            if isinstance(cmd, tuple):
                cmd = 'forward'
            elif cmd == 'idle':
                cmd = adapter.wait_action(ctx.P, ctx.psi, ctx.gmap)
        return Realized(cmd, level)


PLANNERS = {'orca': ORCAPlanner, 'sfm': SFMPlanner, 'dwa': DWAPlanner, 'follow': FollowPlanner}


def make_planner(name, **kw):
    return PLANNERS[name](**kw)
