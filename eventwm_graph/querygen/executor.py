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
VIA_ADVANCE_M = 0.45
STOP_M = 0.15
STUCK_STOP_M = 0.22
LOOKAHEAD_M = 0.8


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
        self.planner.reset()

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
        prim = self._step_kin(legal, level)
        if prim == 'stop':
            return 'stop', 0
        from .planners import Realized
        if isinstance(prim, Realized):
            return prim[0], prim[1]          # planner already realised the command under the dynamics
        P = to_plane([legal.pose[0], legal.pose[2]])
        humans = human_estimates(legal.tracks, legal.time)
        return self.realizer.realize(prim, int(level), P, float(legal.pose[3]), legal.map, humans,
                                     self._plan_now)

    def _step_kin(self, legal, dyn_state=0):
        """Next kinematic primitive, or 'stop'; updates completion status first."""
        q = self.query
        self.steps += 1
        pose = legal.pose
        robot_xz = np.array([pose[0], pose[2]])
        if q.kind == 'stop':
            self.status, self.reason = 'done', 'stop'
            return 'stop'
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
                hard = one_step_hard_ok(ctx.humans, 0.30)
                P1, psi1 = self.adapter.predicted_pose(ctx.P, ctx.psi, direction)
                if hard is None or hard(P1, psi1):
                    return direction
                return self.planner.act(ctx, self.adapter)
            ctx = self._context(legal, dyn_state, np.zeros(2), hold=True, face=face)
            return self.planner.act(ctx, self.adapter)
        # goto / goal
        if q.kind == 'goal':
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
        final = q.kind == 'goal' or self.via_index == len(q.via)-1
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
            tol = (STOP_M*0.8 if self.query.kind == 'goal' else ARRIVE_M*0.8)
            near_goal = self.query.kind == 'goal' and d <= 1.2
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
