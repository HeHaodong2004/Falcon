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

from .control import (NativeAdapter, heading_vec, wrap, body_ok, turn_ok, to_world,
                      BODY_X, BODY_HALF_WIDTH)
from .dynamics import STEP_SECONDS
from .orca import ORCASolver, ORCAConfig
from .legal import BLOCKED


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
    return DiscreteAdapter(dyn) if dyn.discrete else UnicycleAdapter(dyn)


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


class ORCAPlanner(Planner):
    """Full-responsibility ORCA; the solver velocity is the agent's model
    velocity (RVO2 semantics) and is tracked by the dynamics adapter."""
    name = 'orca'

    def __init__(self, config=None, hard_clearance=0.30):
        self.solver = ORCASolver(config or ORCAConfig())
        self.hard_clearance = hard_clearance
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
        if (not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and
                float(np.hypot(*(v-v_pref))) <= 0.05*ctx.vmax):
            # Nominal motion is ORCA-admissible: execute the static plan.
            prim = ctx.plan_prims[0]
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self.mode = 'plan'
                return prim
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

    def __init__(self, config=None):
        self.cfg = config or SFMConfig()
        self.v = None

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
        hard = one_step_hard_ok(ctx.humans, 0.30)
        social = float(np.hypot(*(F-F_goal)))*STEP_SECONDS
        if not ctx.hold and ctx.plan_prims and ctx.dyn.discrete and social <= 0.05*ctx.vmax:
            prim = ctx.plan_prims[0]
            P1, psi1 = adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self.mode = 'plan'
                self.v = ctx.v_pref.copy()
                return prim
        return self._emit(ctx, adapter, v, hard)


@dataclass
class DWAConfig:
    horizon: int = 14               # control steps simulated
    w_progress: float = 1.0         # per metre of route cost-to-go reduction
    w_clearance: float = 0.8        # per metre of min human clearance (capped)
    clearance_cap: float = 1.2
    w_heading: float = 0.3
    human_radius: float = 0.40      # m from robot rectangle to human centre = contact
    safety: float = 0.25            # m extra margin required for admissibility
    turn_options: tuple = (0, 1, 2, 3, 5, 8, 12, 18, 26, 36)


class DWAPlanner(Planner):
    """Sampling over admissible command sequences with forward simulation."""
    name = 'dwa'

    def __init__(self, config=None):
        self.cfg = config or DWAConfig()

    # robot rectangle distance to points, vectorised over time
    @staticmethod
    def _rect_dist(P, psi, Q):
        c, s = math.cos(psi), math.sin(psi)
        d = Q-P
        lx = d[..., 0]*c+d[..., 1]*s
        ly = -d[..., 0]*s+d[..., 1]*c
        dx = np.maximum(np.maximum(BODY_X[0]-lx, 0.), lx-BODY_X[1])
        dy = np.maximum(np.abs(ly)-BODY_HALF_WIDTH, 0.)
        return np.hypot(dx, dy)

    def _candidates(self, ctx):
        dyn, T = ctx.dyn, self.cfg.horizon
        out = []
        if dyn.discrete:
            out.append(('hold', ['hold']*T))
            for k in self.cfg.turn_options:
                for side in (('left', 'right') if k else ('none',)):
                    seq = ([side]*min(k, T))+['forward']*max(T-k, 0)
                    out.append((f'{side}{k}', seq))
                    if k:
                        out.append((f'{side}{k}_hold', [side]*min(k, T)+['hold']*max(T-k, 0)))
        else:
            v0, w0 = ctx.dyn_state
            vs = np.linspace(max(0., v0-min(dyn.brake, dyn.v_max)), min(dyn.v_max, v0+min(dyn.accel, dyn.v_max)), 6)
            ws = np.linspace(max(-dyn.w_max, w0-min(dyn.w_accel, 2*dyn.w_max)), min(dyn.w_max, w0+min(dyn.w_accel, 2*dyn.w_max)), 9)
            for v in np.unique(np.round(vs, 6)):
                for w in np.unique(np.round(ws, 6)):
                    out.append((f'v{v:.3f}w{w:.3f}', [(float(v), float(w))]*T))
        return out

    def _rollout(self, ctx, seq):
        dyn = ctx.dyn
        P, psi = ctx.P.copy(), ctx.psi
        state = ctx.dyn_state
        poses = []
        hold_flip = 1.
        for cmd in seq:
            if dyn.discrete:
                if cmd == 'forward':
                    P = P+dyn.step_m*heading_vec(psi)
                elif cmd == 'left':
                    psi += dyn.turn_rad
                elif cmd == 'right':
                    psi -= dyn.turn_rad
                elif cmd == 'hold':
                    psi += hold_flip*dyn.turn_rad; hold_flip = -hold_flip
            else:
                v, w = dyn.limit(cmd, state)
                state = (v, w)
                psi += w
                P = P+v*heading_vec(psi-w/2)
            poses.append((P.copy(), psi))
        return poses

    def act(self, ctx, adapter):
        cfg = self.cfg
        cands = self._candidates(ctx)
        H = ctx.humans
        best = None
        fallback = None
        T = cfg.horizon
        times = np.arange(1, T+1)*STEP_SECONDS
        Hpos = np.stack([np.stack([h.at(t) for h in H]) for t in times]) if H else None
        start_cost = ctx.progress_fn(ctx.P) if ctx.progress_fn else 0.
        for name, seq in cands:
            if ctx.hold and not (name == 'hold' or name.endswith('_hold') or name.startswith('v0.000')):
                continue
            poses = self._rollout(ctx, seq)
            # static admissibility: body never over observed walls
            ok = True
            for k, (P, psi) in enumerate(poses):
                if k % 2 == 1 and k != len(poses)-1:
                    continue
                if not body_ok(ctx.gmap, P, psi):
                    ok = False
                    break
            if not ok:
                continue
            if Hpos is not None:
                dmin = min(float(self._rect_dist(P, psi, Hpos[k]).min()) for k, (P, psi) in enumerate(poses))
            else:
                dmin = cfg.clearance_cap+cfg.human_radius
            clear = dmin-cfg.human_radius
            Pend, psi_end = poses[-1]
            progress = (start_cost-ctx.progress_fn(Pend)) if ctx.progress_fn else 0.
            if ctx.hold:
                progress = 0.
            heading = 0.
            target_dir = ctx.face if ctx.hold and ctx.face is not None else ctx.route_dir
            if target_dir is not None:
                heading = math.cos(wrap(target_dir-psi_end))
            score = cfg.w_progress*progress+cfg.w_clearance*min(clear, cfg.clearance_cap)+cfg.w_heading*heading
            if fallback is None or clear > fallback[0]:
                fallback = (clear, seq)
            if clear < cfg.safety:
                continue
            if best is None or score > best[0]:
                best = (score, seq, name)
        seq = best[1] if best is not None else (fallback[1] if fallback is not None else None)
        if seq is None:
            return adapter.wait_action(ctx.P, ctx.psi, ctx.gmap, ctx.face) if ctx.dyn.discrete else ctx.dyn.limit((0., 0.), ctx.dyn_state)
        cmd = seq[0]
        if ctx.dyn.discrete:
            if cmd == 'hold':
                return adapter.wait_action(ctx.P, ctx.psi, ctx.gmap, ctx.face)
            return cmd
        return ctx.dyn.limit(cmd, ctx.dyn_state)


PLANNERS = {'orca': ORCAPlanner, 'sfm': SFMPlanner, 'dwa': DWAPlanner, 'follow': FollowPlanner}


def make_planner(name, **kw):
    return PLANNERS[name](**kw)
