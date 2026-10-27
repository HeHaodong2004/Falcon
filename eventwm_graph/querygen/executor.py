"""High-level query contract and its low-level execution (legal inputs only).

A query is a high-level navigation decision executed by a fixed low-level
planner under a robot dynamics model until it completes or the decision
layer interrupts it at its horizon.  Semantics:

``goto``    follow the legal-map route through ordered via-points (world x,z);
            completes within ``ARRIVE_M`` of the last via-point.
``goal``    ``goto`` the public PointGoal; issues STOP (ends the episode) when
            the legal Euclidean goal distance is below ``STOP_M`` (or the robot
            is within ``STUCK_STOP_M`` and the last moves made no progress).
``wait``    hold position for ``duration`` steps (native robot: alternating
            in-place turns or turning toward ``heading``); the planner still
            evades if holding becomes inadmissible.
``rotate``  turn in place toward ``heading``.
``creep``   manoeuvre primitive: ``duration`` forward steps, each only if it
            keeps the one-step human clearance (otherwise hold); like explicit
            rotations it is not filtered by the (dilated) legal wall band.
            Proposed only when the legal stuck test fires.
``wait_clear`` hold (like ``wait``) until no tracked human is legally predicted
            to conflict with the next metres of the goal route for two
            consecutive steps, at most ``duration`` steps.
``stop``    issue STOP now (only proposed near the goal).

Routes are recomputed on the *current legal map* (UNKNOWN traversed
optimistically, BLOCKED never).  Via-point progress persists in the
``QueryExecution`` so an interrupted query can be continued.
"""
from dataclasses import dataclass, field, replace
import copy
import math
import numpy as np

from .control import (to_plane, to_world, wrap, heading_vec, human_estimates, blocked_points_near)
from .gridplan import PlanningView, INF, path_length, point_along
from . import lattice
from .control import body_cost
from .motion import MotionRealizer
from .planners import PlanContext, make_adapter
from .dynamics import STEP_SECONDS

ARRIVE_M = 0.30
HARD_CLEARANCE = 0.30        # one-step rectangle-to-predicted-human clearance (all query kinds)
CLOSE_HARD_CLEARANCE = 0.27  # 'close' queries (declared body proxy: 0.25 m)
# 'close' queries: ORCA disk proxies sized to the body *width* (the rectangle hard check above
# guards the nose and tail) instead of the conservative 0.55 + 0.30 + 0.25 m combined radius, and a
# 0.5 s human horizon (~10 native steps; the native robot stops instantly and is ~4x faster than
# the humans, so 1.5 s forbade approaching any human within ~6 m along the heading)
CLOSE_ORCA = dict(robot_radius=0.30, human_radius=0.25, margin=0.05, time_horizon=0.5)
# 'stgoal' queries (v9): legal space-time lattice plan (stplan.py) re-planned every ST_REPLAN steps
ST_HORIZON = 40
ST_MARGIN = 0.35            # rectangle to predicted human centre at t = 0 ...
ST_MARGIN_CLOSE = 0.28      # ... for 'close' queries (declared body proxy 0.25 m)
ST_GROWTH = 0.004           # ... plus this per predicted step (CV uncertainty)
ST_REPLAN = 4
ST_HUMAN_RANGE = 8.0
VIA_ADVANCE_M = 0.45
STOP_M = 0.15
STUCK_STOP_M = 0.22
LOOKAHEAD_M = 0.8


class _TightLegal:
    """Legal state whose map relabels the outer rings of the inflated BLOCKED band as UNKNOWN.

    The legal observer certifies BLOCKED as wall hits dilated by 0.25 m (~ the 0.243 m body
    half width) on 0.1 m cells, so its band is conservative; a *tight* query may plan into
    its outer ``cells`` rings (traversed optimistically like any UNKNOWN cell, never treated
    as FREE).  Everything else is the unchanged legal state.
    """

    def __init__(self, legal, cells):
        from scipy import ndimage
        from .legal import GridMap, BLOCKED, UNKNOWN
        g0 = legal.map
        g = GridMap.__new__(GridMap)
        g.__dict__.update(g0.__dict__)
        B = g0.state == BLOCKED
        core = ndimage.binary_erosion(B, iterations=int(cells), border_value=1)
        st = g0.state.copy()
        st[B & ~core] = UNKNOWN
        g.state = st
        self._legal = legal
        self.map = g

    def __getattr__(self, k):
        return getattr(self._legal, k)


def tight_legal(legal, cells):
    return _TightLegal(legal, cells) if cells else legal


@dataclass
class Query:
    kind: str
    via: tuple = ()
    heading: float = None
    duration: int = 0
    speed: float = 1.0
    label: str = ''
    meta: dict = field(default_factory=dict)

    def signature(self):
        v = tuple((round(float(x), 2), round(float(z), 2)) for x, z in self.via)
        h = None if self.heading is None else round(float(self.heading), 2)
        return (self.kind, v, h, int(self.duration), round(float(self.speed), 2))

    def describe(self):
        return dict(kind=self.kind, via=[[float(x), float(z)] for x, z in self.via], heading=self.heading,
                    duration=self.duration, speed=self.speed, label=self.label, meta=self.meta)


class QueryExecution:
    def __init__(self, query, planner, dyn):
        self.query = query
        self.planner = planner
        self.dyn = dyn
        self.adapter = make_adapter(dyn)
        self.via_index = 0
        self.steps = 0
        self.status = 'running'
        self.reason = ''
        self.view = None
        self.field = None
        self.field_key = None
        self.route = None
        self.route_len = None
        self.recent = []
        self.plan = None           # body-aware lattice plan: dict(prims, poses, ...)
        self.plan_i = 0
        self.plan_age = 0
        self.lattice_calls = 0
        self.realizer = MotionRealizer(dyn)
        self.speed_debt = 0.
        self.clear_streak = 0
        self.kt_i = 0               # kturn: primitives executed so far
        self.st = None              # stgoal: current space-time plan
        self.st_i = 0
        self.st_age = 0
        self.hard_clearance = HARD_CLEARANCE
        if (query.meta or {}).get('close'):
            self._make_close()
        if (query.meta or {}).get('commit') and hasattr(self.planner, 'commit'):
            self.planner.commit = True
        self.planner.reset()

    def _make_close(self):
        """Close-pass variant: smaller human-avoidance proxies, rectangle hard check at 0.27 m."""
        from .planners import ORCAPlanner
        from .orca import ORCASolver, ORCAConfig
        from dataclasses import replace as _replace
        self.hard_clearance = CLOSE_HARD_CLEARANCE
        if isinstance(self.planner, ORCAPlanner):
            self.planner.solver = ORCASolver(_replace(self.planner.solver.cfg, **CLOSE_ORCA))
            self.planner.hard_clearance = CLOSE_HARD_CLEARANCE

    def copy(self):
        return copy.deepcopy(self)

    def decision_boundary(self):
        """Drop derived caches (route view/field, static plan) at a decision.

        Applied identically in Oracle branches and authoritative replay, so a
        continued query re-derives them from the then-current legal map.
        """
        self.view = None
        self.field = None
        self.field_key = None
        self.plan = None
        self.plan_i = 0
        self.plan_age = 0
        self.st = None
        if hasattr(self.adapter, 'reset'):
            self.adapter.reset()

    @property
    def target(self):
        q = self.query
        if q.kind == 'goal':
            return None
        return q.via[self.via_index] if self.via_index < len(q.via) else None

    def _goal_xz(self, legal):
        return (float(legal.goal[0]), float(legal.goal[2]))

    def _ensure_field(self, legal, target_xz):
        robot_xz = (float(legal.pose[0]), float(legal.pose[2]))
        key = (legal.map.revision, round(target_xz[0], 3), round(target_xz[1], 3))
        need = self.field is None or self.field_key[1:] != key[1:]
        if not need and self.field_key[0] != key[0] and self.steps % 3 == 0:
            need = True          # new legal map evidence (refresh every 3 steps)
        if not need:
            i, j = self.view.to_local(*robot_xz)
            need = not self.view.inside(i, j)
        if need:
            self.view = PlanningView(legal.map, robot_xz, target_xz)
            self.field = self.view.field([target_xz])
            self.field_key = key
        self.route = self.view.route(self.field, robot_xz)
        self.route_len = path_length(self.route) if self.route is not None else None
        return self.route

    def _context(self, legal, dyn_state, v_pref, hold=False, face=None, route_dir=None, progress_fn=None):
        P = to_plane([legal.pose[0], legal.pose[2]])
        V_world, _ = legal.velocity(1)
        V = to_plane(V_world)
        humans = human_estimates(legal.tracks, legal.time)
        statics = blocked_points_near(legal.map, P, radius=1.4)
        return PlanContext(P=P, psi=float(legal.pose[3]), V=V, dyn=self.dyn, dyn_state=dyn_state,
                           v_pref=np.asarray(v_pref, dtype=float), humans=humans, statics=statics,
                           gmap=legal.map, hold=hold, face=face, route_dir=route_dir, progress_fn=progress_fn)

    def step(self, legal, level=0):
        """Next (command primitive, new speed level) under the dynamics family."""
        self._plan_now = None
        legal = tight_legal(legal, int((self.query.meta or {}).get('tight', 0)))
        prim = self._step_kin(legal, level)
        if prim == 'stop':
            return 'stop', 0
        from .planners import Realized
        if isinstance(prim, Realized):
            return prim[0], prim[1]          # planner already realised the command under the dynamics
        P = to_plane([legal.pose[0], legal.pose[2]])
        humans = human_estimates(legal.tracks, legal.time)
        speed = float(self.query.speed)
        if speed < 1. and prim == 'forward' and not self.dyn.stateful:
            # reduced query speed for instant-speed robots: duty-cycled advances
            self.speed_debt += speed
            if self.speed_debt >= 1.-1e-9:
                self.speed_debt -= 1.
            else:
                prim = self.adapter.wait_action(P, float(legal.pose[3]), legal.map)
                self.plan = None
        cap = None if speed >= 1. else max(1, int(round((len(self.dyn.levels)-1)*speed)))
        return self.realizer.realize(prim, int(level), P, float(legal.pose[3]), legal.map, humans,
                                     self._plan_now, cap)

    def _step_kin(self, legal, dyn_state=0):
        """Next kinematic primitive, or 'stop'; updates completion status first."""
        q = self.query
        self.steps += 1
        pose = legal.pose
        robot_xz = np.array([pose[0], pose[2]])
        if q.kind == 'stop':
            self.status, self.reason = 'done', 'stop'
            return 'stop'
        if q.kind == 'creep':
            if self.steps >= max(1, int(q.duration)):
                self.status, self.reason = 'done', 'crept'
            ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
            from .planners import one_step_hard_ok
            hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
            P1, psi1 = ctx.P+self.dyn.step_m*heading_vec(ctx.psi), ctx.psi
            if hard is None or hard(P1, psi1):
                remaining = max(1, int(q.duration)-self.steps+1)
                self._plan_now = ['forward']*remaining     # stateful families brake to rest at the end
                return 'forward'
            return self.planner.act(ctx, self.adapter)
        if q.kind == 'wait_clear':
            clear = self._route_clear(legal)
            self.clear_streak = self.clear_streak+1 if clear else 0
            if self.steps >= max(1, q.duration) or self.clear_streak >= 2:
                self.status, self.reason = 'done', ('clear' if clear else 'waited')
            ctx = self._context(legal, dyn_state, np.zeros(2), hold=True, face=q.heading)
            return self.planner.act(ctx, self.adapter)
        if q.kind == 'kturn':
            return self._step_kturn(legal, dyn_state)
        if q.kind in ('wait', 'rotate'):
            face = q.heading
            if q.kind == 'rotate' and face is not None and abs(wrap(face-pose[3])) <= self.dyn.turn_rad*.55:
                self.status, self.reason = 'done', 'heading_reached'
            if q.kind == 'wait' and self.steps >= max(1, q.duration):
                self.status, self.reason = 'done', 'waited'
            if q.kind == 'rotate' and self.status == 'running' and self.dyn.discrete:
                # Explicit-direction in-place rotation (the long way round is a
                # legitimate manoeuvre choice when one sweep side is blocked).
                direction = q.meta.get('direction')
                if direction is None:
                    direction = 'left' if wrap(face-pose[3]) > 0 else 'right'
                ctx = self._context(legal, dyn_state, np.zeros(2), hold=True, face=face)
                from .planners import one_step_hard_ok
                hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
                P1, psi1 = self.adapter.predicted_pose(ctx.P, ctx.psi, direction)
                if hard is None or hard(P1, psi1):
                    return direction
                return self.planner.act(ctx, self.adapter)
            ctx = self._context(legal, dyn_state, np.zeros(2), hold=True, face=face)
            return self.planner.act(ctx, self.adapter)
        if q.kind == 'stgoal':
            return self._step_st(legal, dyn_state)
        # goto / goal
        if q.kind == 'goal' and q.via:
            # goal approach through ordered pre-goal via-points
            while self.via_index < len(q.via) and np.hypot(*(robot_xz-np.asarray(q.via[self.via_index]))) < VIA_ADVANCE_M*0.6:
                self.via_index += 1
        if q.kind == 'goal' and q.via and self.via_index < len(q.via):
            target = tuple(map(float, q.via[self.via_index]))
        elif q.kind == 'goal':
            target = self._goal_xz(legal)
            d = float(np.hypot(*(robot_xz-target)))
            self.recent.append(d)
            stopped = (not self.dyn.stateful) or int(dyn_state) == 0
            if stopped and (d < STOP_M or (d < STUCK_STOP_M and len(self.recent) >= 4 and
                                           self.recent[-4]-d < 0.02)):
                self.status, self.reason = 'done', 'stop_at_goal'
                return 'stop'
        else:
            while self.via_index < len(q.via)-1 and np.hypot(*(robot_xz-np.asarray(q.via[self.via_index]))) < VIA_ADVANCE_M:
                self.via_index += 1
            target = tuple(map(float, q.via[self.via_index]))
            if self.via_index == len(q.via)-1 and np.hypot(*(robot_xz-np.asarray(target))) < ARRIVE_M:
                self.status, self.reason = 'done', 'arrived'
                face = q.heading
                ctx = self._context(legal, dyn_state, np.zeros(2), hold=True, face=face)
                return self.planner.act(ctx, self.adapter)
        route = self._ensure_field(legal, target)
        if route is None:
            self.status, self.reason = 'done', 'unreachable'
            ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
            return self.planner.act(ctx, self.adapter)
        Pl = to_plane(robot_xz)
        final = (q.kind == 'goal' and self.via_index >= len(q.via)) or (q.kind != 'goal' and self.via_index == len(q.via)-1)
        prims, plan_poses = self._static_plan(legal, target, final)
        look = point_along(route, LOOKAHEAD_M)
        direction = to_plane(look)-Pl
        n = float(np.hypot(*direction))
        route_dir = math.atan2(direction[1], direction[0]) if n > 1e-6 else None
        vmax = self.dyn.v_max/STEP_SECONDS*q.speed
        # Preferred velocity = the static plan's mean velocity over its next steps
        # (zero while the plan turns in place); falls back to route pursuit.
        if prims:
            k = min(len(plan_poses), 4)
            tgt = to_plane([plan_poses[k-1][0], plan_poses[k-1][1]])
            v_pref = (tgt-Pl)/(k*STEP_SECONDS)*q.speed
            sp = float(np.hypot(*v_pref))
            if sp > vmax:
                v_pref = v_pref/sp*vmax
        elif n > 1e-6:
            v_pref = direction/n*vmax
        else:
            v_pref = np.zeros(2)
        view, field_ = self.view, self.field

        def progress_fn(P):
            return view.cost_to_go(field_, tuple(to_world(P)))
        ctx = self._context(legal, dyn_state, v_pref, route_dir=route_dir, progress_fn=progress_fn)
        ctx.plan_prims = prims
        ctx.plan_poses = plan_poses
        self._plan_now = prims
        prim = self.planner.act(ctx, self.adapter)
        if prims and prim == prims[0]:
            self.plan_i += 1
        else:
            self.plan = None          # deviated (human avoidance / adapter) -> replan
        return prim

    def _step_st(self, legal, dyn_state):
        """stgoal: follow a legal space-time lattice plan to the goal (or the last via-point).

        The plan (stplan.py: exact native primitives incl. two-step holds, legal map, legal CV human
        predictions) is recomputed every ST_REPLAN steps, when the robot is off the planned pose, or
        when the one-step rectangle check rejects its next primitive (then the robot holds once).
        """
        from . import stplan
        from .planners import one_step_hard_ok
        q = self.query
        pose = legal.pose
        target = tuple(map(float, q.via[-1])) if q.via else self._goal_xz(legal)
        d = math.hypot(target[0]-pose[0], target[1]-pose[2])
        if not q.via:
            self.recent.append(d)
            if d < STOP_M or (d < STUCK_STOP_M and len(self.recent) >= 4 and self.recent[-4]-d < 0.02):
                self.status, self.reason = 'done', 'stop_at_goal'
                return 'stop'
        ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
        if q.via and d < ARRIVE_M:
            self.status, self.reason = 'done', 'arrived'
            return self.planner.act(ctx, self.adapter)
        need = self.st is None or self.st_i >= len(self.st['prims']) or self.st_age >= ST_REPLAN
        if not need and self.st_i > 0:
            ex = self.st['poses'][self.st_i-1]
            need = math.hypot(ex[0]-pose[0], ex[1]-pose[2]) > 0.06 or abs(wrap(ex[2]-pose[3])) > 0.03
        if need:
            route = self._ensure_field(legal, target)
            if route is None:
                self.status, self.reason = 'done', 'unreachable'
                return self.planner.act(ctx, self.adapter)
            P = to_plane([pose[0], pose[2]])
            humans = [h for h in ctx.humans if float(np.hypot(*(h.position-P))) <= ST_HUMAN_RANGE]
            close = bool((q.meta or {}).get('close'))
            kw = dict(horizon=ST_HORIZON, goal_tol=STOP_M*0.8 if not q.via else ARRIVE_M*0.8,
                      margin=ST_MARGIN_CLOSE if close else ST_MARGIN,
                      growth=ST_GROWTH, start_budget=body_cost(legal.map, P, pose[3]))
            self.st = stplan.plan(self.view, self.field, pose, target, self.dyn.step_m, self.dyn.turn_rad, humans,
                                  STEP_SECONDS, **kw)
            if not self.st['reached'] and not self._st_progress(self.st, pose):
                # No progress within the horizon: if the static problem alone also makes none (a long
                # in-place turn / multi-point manoeuvre longer than the horizon), follow the full static
                # lattice plan (as 'goal'); otherwise the humans force the hold and the plan stands.
                free = stplan.plan(self.view, self.field, pose, target, self.dyn.step_m, self.dyn.turn_rad, [],
                                   STEP_SECONDS, **kw) if humans else self.st
                if not free['reached'] and not self._st_progress(free, pose):
                    self.plan = None
                    final = not q.via
                    prims_s, poses_s = self._static_plan(legal, target, final)
                    self.st = dict(prims=list(prims_s), poses=list(poses_s), reached=False, static=True)
            self.st_i = 0
            self.st_age = 0
            self.lattice_calls += 1
        self.st_age += 1
        prims = self.st['prims']
        if self.st_i < len(prims):
            prim = prims[self.st_i]
            hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
            P1, psi1 = self.adapter.predicted_pose(ctx.P, ctx.psi, prim)
            if hard is None or hard(P1, psi1):
                self._plan_now = prims[self.st_i:]
                self.st_i += 1
                return prim
            self.st = None
        return self.planner.act(ctx, self.adapter)

    def _step_kturn(self, legal, dyn_state):
        """Multi-point turn (manoeuvre): ``first`` = (direction, k) in-place turn steps, ``creep`` forward
        steps, then turn in direction ``final`` until facing ``heading`` (the route heading when proposed).
        Like explicit rotations/creeps it is not filtered by the legal wall band; every primitive keeps the
        one-step rectangle clearance to tracked humans (otherwise the robot holds).
        """
        from .planners import one_step_hard_ok
        q = self.query
        m = q.meta
        d1, k1 = m['first']
        n = int(m['creep'])
        i = self.kt_i
        if i < k1:
            prim = d1
        elif i < k1+n:
            prim = 'forward'
        else:
            err = wrap(float(q.heading)-float(legal.pose[3]))
            if abs(err) <= self.dyn.turn_rad*.55 or i >= k1+n+int(2*math.pi/self.dyn.turn_rad)+2:
                self.status, self.reason = 'done', 'heading_reached'
                ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
                return self.planner.act(ctx, self.adapter)
            prim = m['final']
        ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
        hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
        P1, psi1 = self.adapter.predicted_pose(ctx.P, ctx.psi, prim)
        if hard is None or hard(P1, psi1):
            self.kt_i += 1
            return prim
        return self.planner.act(ctx, self.adapter)

    def _st_progress(self, st, pose, min_m=0.2):
        """Does a space-time plan end at least ``min_m`` closer (static legal cost-to-go) than it starts?"""
        if not st['poses']:
            return False
        x, z = st['poses'][-1][0], st['poses'][-1][1]
        return self.view.cost_to_go(self.field, (x, z)) < self.view.cost_to_go(self.field, (pose[0], pose[2]))-min_m

    def _route_clear(self, legal, horizon_m=3.0, robot_speed=2.0, conflict_dist=1.3, human_range=5.0):
        """Legal: no tracked human (constant velocity) meets the next metres of the goal route."""
        from .control import route_conflicts
        humans = [h for h in human_estimates(legal.tracks, legal.time)
                  if float(np.hypot(*(h.position-to_plane([legal.pose[0], legal.pose[2]])))) <= human_range]
        if not humans:
            return True
        route = self._ensure_field(legal, self._goal_xz(legal))
        return not route_conflicts(route, humans, horizon_m, robot_speed, conflict_dist)

    def _static_plan(self, legal, target, final):
        """Remaining body-feasible primitives (list) and their poses."""
        dyn = self.dyn
        if not dyn.discrete:
            return [], []
        pose = legal.pose
        P = to_plane([pose[0], pose[2]])
        ok = (self.plan is not None and self.plan_i < len(self.plan['prims']) and self.plan_age < 24
              and not dyn.stateful)
        if ok and self.plan_i > 0:
            ex = self.plan['poses'][self.plan_i-1]
            if (math.hypot(ex[0]-pose[0], ex[1]-pose[2]) > 0.06 or
                    abs(wrap(ex[2]-pose[3])) > 0.03):
                ok = False
        if ok and legal.map.revision != self.plan.get('revision'):
            # re-validate the next few planned poses on the new evidence
            budget = body_cost(legal.map, P, pose[3])
            for (x, z, psi) in self.plan['poses'][self.plan_i:self.plan_i+12]:
                if body_cost(legal.map, to_plane([x, z]), psi) > budget:
                    ok = False
                    break
        if not ok:
            d = math.hypot(target[0]-pose[0], target[1]-pose[2])
            at_goal_stage = self.query.kind == 'goal' and self.via_index >= len(self.query.via)
            tol = (STOP_M*0.8 if at_goal_stage else ARRIVE_M*0.6 if self.query.kind == 'goal' else ARRIVE_M*0.8)
            near_goal = at_goal_stage and d <= 1.2
            i, j = self.view.to_local(pose[0], pose[2])
            if d <= 4.0:
                goal_tol, goal_field = (tol, -1.)
            else:
                here = self.field[min(max(i, 0), self.field.shape[0]-1), min(max(j, 0), self.field.shape[1]-1)]
                goal_tol, goal_field = (tol, max(here-3.0/self.view.res, 0.))
            budget = body_cost(legal.map, P, pose[3])
            # On final goal approach the robot stops on arrival, so the nose
            # margin (kept only to preserve rotation freedom) is not required.
            self.plan = lattice.plan(self.view, self.field, pose, target, dyn.step_m, dyn.turn_rad,
                                     goal_tol=goal_tol, goal_field=goal_field, start_budget=budget,
                                     nose_margin=0. if near_goal else None)
            self.plan['revision'] = legal.map.revision
            self.plan_i = 0
            self.plan_age = 0
            self.lattice_calls += 1
        self.plan_age += 1
        return self.plan['prims'][self.plan_i:], self.plan['poses'][self.plan_i:]


def command_for(dyn, primitive):
    """Map a realised primitive to the robot command for this dynamics family."""
    if primitive == 'stop':
        return 'stop'
    if dyn.native_actions:
        return primitive
    if primitive == 'forward':
        return dyn.to_command(dyn.step_m, 0.)
    if primitive == 'idle':
        return dyn.to_command(0., 0.)
    if primitive == 'left':
        return dyn.to_command(0., dyn.turn_rad)
    if primitive == 'right':
        return dyn.to_command(0., -dyn.turn_rad)
    if isinstance(primitive, tuple) and primitive[0] == 'forward':
        return dyn.to_command(dyn.levels[int(primitive[1])], 0.)
    raise ValueError(primitive)
