"""Deterministic, non-learning Query Generator (legal inputs only).

Inputs: the legal state (partial certificate map, legal human tracks, pose and
action history, public PointGoal), the currently executing query (for
commitment), and the robot dynamics.  Output: <= ``budget`` executable,
de-duplicated ``Query`` objects plus diagnostics.  Categories:

  continue    keep executing the unfinished query (via-point progress kept)
  goal        legal-map route to the PointGoal, STOP on arrival
  stop        STOP now (only when the legal goal distance is small)
  alt_route   via-point on a homotopically different goal route (the main
              route's corridor is penalised and the goal field recomputed)
  frontier    best frontier clusters by through-frontier cost to the goal
  pass_*      pass a conflicting tracked human on the left/right, or behind
              it, then rejoin the main route
  yield       retreat to the nearby reachable point that maximises predicted
              clearance from conflicting humans
  wait        hold position for a short/long time facing the route, or (on a
              predicted route conflict) until the conflict is legally clear
  observe     rotate toward the goal direction / a recently lost human
  advance     short straight move to gain manoeuvring space
  maneuver    exact 1/4/10-step rotations both ways and 1/3-step forward creeps,
              composable into multi-point turns; offered in manoeuvre mode (entered
              after a legal stall near the start or the goal, kept while the
              previous decision was a manoeuvre)

UNKNOWN is never treated as verified free: routes may cross it only
optimistically (higher cost) and via-points must be on FREE cells.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy import ndimage

from .legal import FREE, BLOCKED, UNKNOWN
from .gridplan import PlanningView, INF, path_length, point_along, _dijkstra
from .control import (to_plane, to_world, wrap, human_estimates, body_cost, heading_vec, resample_route,
                      route_conflicts)
from .executor import Query, STOP_M
from .dynamics import STEP_SECONDS


@dataclass(frozen=True)
class GeneratorConfig:
    budget: int = 10
    horizon_m: float = 3.0           # human relevance / conflict look-ahead along the route
    human_range: float = 5.0
    conflict_dist: float = 1.3       # human predicted path within this of the route -> conflict
    pass_offset: float = 1.15        # lateral offset from the human at passing time
    rejoin_m: float = 1.6
    robot_speed: float = 2.0         # m/s effective (turning included) for ETA estimates
    wait_short: int = 6
    wait_long: int = 16
    wait_clear_max: int = 48
    maneuver_turns: tuple = (1, 4, 10)
    maneuver_creeps: tuple = (1, 3)
    maneuver_spawn_only: bool = True # v8: manoeuvre mode only before the robot first leaves the spawn region
    maneuver_start_m: float = 1.0    # exact manoeuvres only within this distance of the start pose
    maneuver_goal_m: float = 1.0     # ... or of the goal
    root_probes: bool = True         # v7: 1-step creep/rotations at the reset state (False = frozen v6)
    select: str = 'quota'            # 'quota' (v6/v7: priority quotas) | 'compact' (slots by situation, for K ~ 6)
    categories: tuple = ()           # ablation: keep only these candidate categories (() = all); the fixed
                                     # goal / continue / stop queries are always offered
    stall_window: int = 16           # ... and only after this many steps without translation or turning
    stall_min: int = 6
    stall_disp_m: float = 0.15
    stall_turn_rad: float = 0.35
    alt_routes: int = 2
    alt_penalty: float = 8.0
    alt_corridor_m: float = 0.8
    alt_via_m: float = 2.5
    frontiers: int = 2
    stop_radius: float = 0.40
    advance_m: float = 0.7
    retreat_m: float = 1.6           # backtrack distance along the robot's own legal trajectory
    approach_range: float = 3.0      # goal-approach alignment candidates within this goal distance
    approach_m: float = 0.9          # straight final segment into the goal
    approaches: int = 2
    sidestep_m: float = 0.9
    sidestep_human_m: float = 3.0
    dedup_m: float = 0.45
    tight: int = 0                   # v9 ablation: every query plans with the outer N rings of the inflated
                                     # BLOCKED band relabelled UNKNOWN (executor._TightLegal); 0 = off
    close: int = 0                   # v9 ablation: every query executes with close-pass human avoidance
                                     # (executor.CLOSE_ORCA, rectangle hard check 0.27 m); 0 = off
    commit: int = 0                  # v9 ablation: every query follows its static plan under the hard
                                     # rectangle check only (ORCA soft avoidance suspended); 0 = off
    glance: bool = False             # v10: bounded (<= 45 deg) in-place glance toward the goal bearing instead of the
                                     # full look_goal rotation (observe openings on the goal side)
    goal_opt: bool = False           # v9: optimistic goal route (UNKNOWN priced like FREE) as an 'alt' candidate when
                                     # its first alt_via_m metres diverge from the main route (unexplored shortcuts)
    tc_variants: tuple = ()          # v9 mixed mode: also offer tight(1)+close copies of these fixed queries
                                     # (labels + '_tc'), e.g. ('goal', 'st_goal'); the rest stays conservative
    st_goal: bool = False            # v9: also offer 'st_goal' (legal space-time lattice plan to the goal,
                                     # stplan.py) as a fixed candidate (native-like discrete robots only)
    v9_select: bool = True           # v9: approach gated by legal *route* length; K=10 quotas from solution usage
                                     # (alt/frontier get slots; wait_long, retreat, look_lost_human only as leftovers)
    kturns: tuple = ()               # v9: multi-point turns offered in manoeuvre mode: first in-place turn of
                                     # k steps away from the final direction, one creep step, then turn to the
                                     # route heading (e.g. (6, 12)); () = off
    st_goal_momentum: bool = False   # v10: also offer st_goal to robots with momentum (stplan.plan_inertia:
                                     # speed level in the lattice state; braking-tail safety at every step)
    rot_exact: bool = True           # v10: in-place rotations end on the primitive that reaches the target heading
                                     # (v9 noticed it one step late and spent that step on a native hold turn,
                                     # which always starts right: rot_left1 was a no-op, rot_left<k> turned k-1,
                                     # rot_right<k> k+1).  False = v9 behaviour
    arrive_exact: bool = True        # v10: a goto without a final heading ends on the step that reaches its target
                                     # (v9 spent one more step on a hold = native in-place turn)
    root_cover: bool = True          # v10 compact selection at the reset state (nothing behind the robot observed yet,
                                     # so any first primitive may meet unseen geometry): both route turns instead of
                                     # the goal_tc copy, so the slots start with forward, left and right primitives
                                     # (TQSiMZJawkS: every K=6 root candidate started 'r' or 'f'; only 'l' was free)


_route_points = resample_route


def is_maneuver(q):
    """Exact manoeuvre queries (k-step rotations ``rot_left<k>``/``rot_right<k>``, creeps)."""
    lab = q.label
    return q.kind in ('creep', 'kturn') or (q.kind == 'rotate' and lab[-1:].isdigit() and
                                            lab.rstrip('0123456789') in ('rot_left', 'rot_right'))


class QueryGenerator:
    name = 'qg_v0'

    def __init__(self, dyn, config=None):
        self.dyn = dyn
        self.cfg = config or GeneratorConfig()

    # ------------------------------------------------------------------ utils
    def _free_reachable(self, view, rfield, xz):
        i, j = view.to_local(*xz)
        return view.inside(i, j) and view.free[i, j] and rfield[i, j] < INF/2

    def _stand_ok(self, legal, xz, heading):
        P = to_plane(xz)
        return body_cost(legal.map, P, heading) == 0

    def _nearest_free(self, view, rfield, xz, radius=0.6):
        i, j = view.to_local(*xz)
        r = int(math.ceil(radius/view.res))
        best = None
        for a in range(i-r, i+r+1):
            for b in range(j-r, j+r+1):
                if not view.inside(a, b) or not view.free[a, b] or rfield[a, b] >= INF/2:
                    continue
                d = (a-i)**2+(b-j)**2
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            return None
        return view.to_world(best[1], best[2])

    # --------------------------------------------------------------- propose
    def propose(self, legal, execution=None):
        queries, diag = self._propose_all(legal, execution)
        if self.cfg.rot_exact and self.dyn.discrete:
            for q in queries:
                if q.kind == 'rotate':
                    q.meta = dict(q.meta, exact_end=1)
        if self.cfg.arrive_exact:
            for q in queries:
                if q.kind == 'goto' and q.heading is None:
                    q.meta = dict(q.meta, arrive_exact=1)
        return queries, diag

    def _propose_all(self, legal, execution=None):
        cfg = self.cfg
        if cfg.tight or cfg.close or cfg.commit:
            from .executor import tight_legal
            queries, diag = self._propose(tight_legal(legal, cfg.tight), execution)
            # close passes and plan commitment rely on stopping within one step (one-step rectangle check);
            # robots with momentum (stateful dynamics: braking takes several steps) never get them
            stop_now = not self.dyn.stateful
            extra = {k: int(v) for k, v in (('tight', cfg.tight), ('close', cfg.close if stop_now else 0),
                                             ('commit', cfg.commit if stop_now else 0)) if v}
            for q in queries:
                if q.kind != 'continue':
                    q.meta = dict(q.meta, **extra)
            return queries, diag
        return self._propose(legal, execution)

    def _propose(self, legal, execution=None):
        cfg = self.cfg
        pose = legal.pose
        robot_xz = (float(pose[0]), float(pose[2]))
        goal_xz = (float(legal.goal[0]), float(legal.goal[2]))
        view = PlanningView(legal.map, robot_xz, goal_xz)
        gfield = view.field([goal_xz])
        rfield = view.field([robot_xz])
        route = view.route(gfield, robot_xz)
        humans = [h for h in human_estimates(legal.tracks, legal.time)
                  if float(np.hypot(*(h.position-to_plane(robot_xz)))) <= cfg.human_range]
        cats = {k: [] for k in ('human', 'wait', 'alt', 'frontier', 'observe', 'advance', 'yield', 'turn', 'retreat',
                                'approach', 'maneuver', 'wait_long', 'observe_lost')}
        fixed = []
        diag = dict(route_len=path_length(route) if route is not None else None, humans=len(humans))

        if execution is not None and execution.status == 'running' and execution.query.kind != 'wait':
            fixed.append(Query('continue', label='continue', meta=dict(of=execution.query.label)))
        gd = math.hypot(goal_xz[0]-robot_xz[0], goal_xz[1]-robot_xz[1])
        fixed.append(Query('goal', label='goal'))
        if cfg.st_goal and (not self.dyn.stateful or cfg.st_goal_momentum):
            fixed.append(Query('stgoal', label='st_goal'))
        if cfg.tc_variants:
            # mixed mode: explicit squeeze variants (tight wall band + close passes for robots that stop
            # within one step); tight-only for robots with momentum
            tc_meta = dict(tight=1) if self.dyn.stateful else dict(tight=1, close=1)
            for q in list(fixed):
                if q.label in cfg.tc_variants:
                    fixed.append(Query(q.kind, via=q.via, heading=q.heading, duration=q.duration, speed=q.speed,
                                       label=q.label+'_tc', meta=dict(q.meta, **tc_meta)))
        if gd < cfg.stop_radius:
            fixed.append(Query('stop', label='stop', meta=dict(legal_goal_distance=gd)))

        route_dir = None
        if route is not None and len(route) >= 2:
            look = point_along(route, 0.8)
            d = to_plane(look)-to_plane(robot_xz)
            if float(np.hypot(*d)) > 1e-6:
                route_dir = math.atan2(d[1], d[0])

        # ---- humans: passing and yielding
        conflicts = self._conflicts(route, humans, robot_xz)
        for h, t_star, c_xz, c_dist in conflicts[:2]:
            hp = h.at(t_star)
            hv = h.velocity
            sp = float(np.hypot(*hv))
            # lateral axis: perpendicular to the robot's travel at the conflict
            rd = to_plane(c_xz)-to_plane(robot_xz)
            if float(np.hypot(*rd)) < 1e-6:
                rd = heading_vec(pose[3])
            rd = rd/np.linalg.norm(rd)
            left = np.array([-rd[1], rd[0]])
            rejoin = point_along(route, c_dist+cfg.rejoin_m)
            for side, sgn in (('left', 1.), ('right', -1.)):
                cand = to_world(hp+sgn*cfg.pass_offset*left)
                via = self._nearest_free(view, rfield, cand, 0.5)
                if via is not None:
                    cats['human'].append(Query('goto', via=(via, tuple(rejoin)), speed=1.0,
                                               label=f'pass_{side}', meta=dict(track=h.track_id, t_star=t_star)))
            if sp > 0.15:
                behind = to_world(h.position-hv/sp*0.9)
                via = self._nearest_free(view, rfield, behind, 0.5)
                if via is not None:
                    cats['human'].append(Query('goto', via=(via, tuple(rejoin)), label='pass_behind',
                                               meta=dict(track=h.track_id)))
            y = self._yield_point(view, rfield, legal, robot_xz, [h])
            if y is not None:
                cats['yield'].append(Query('goto', via=(y,), heading=route_dir, label='yield',
                                           meta=dict(track=h.track_id)))

        # ---- goal approach alignment: final straight drive along a heading whose
        # goal pose keeps the long body off observed walls
        # v9: the approach trigger uses the legal route's cost-to-go (UNKNOWN cells cost 2.5x), so a goal
        # behind walls or behind unobserved space is not 'near' (LEFTm3JecaC spawn: 1.6 m straight line
        # through an unobserved wall, 3.3 m true route)
        gd_route = gd
        if cfg.v9_select:
            ri, rj = view.to_local(*robot_xz)
            if view.inside(ri, rj) and gfield[ri, rj] < INF/2:
                gd_route = float(gfield[ri, rj])*view.res
            elif route is None:
                gd_route = float('inf')
        if gd_route < cfg.approach_range:
            for th, via in self._approaches(view, rfield, legal, goal_xz, robot_xz):
                cats['approach'].append(Query('goal', via=(via,), label='approach',
                                              meta=dict(heading=round(th, 3))))

        # ---- sidestep out of a tracked human's way (left/right of the heading)
        if any(float(np.hypot(*(h.position-to_plane(robot_xz)))) <= cfg.sidestep_human_m for h in humans):
            hv = heading_vec(pose[3]); lv = np.array([-hv[1], hv[0]])
            for side, sgn in (('left', 1.), ('right', -1.)):
                cand = to_world(to_plane(robot_xz)+sgn*cfg.sidestep_m*lv)
                via = self._nearest_open(view, rfield, cand, 0.4)
                if via is not None:
                    cats['yield'].append(Query('goto', via=(via,), label=f'sidestep_{side}'))

        # ---- retreat along the robot's own legally observed trajectory
        back = self._backtrack_point(legal, cfg.retreat_m)
        if back is not None:
            cats['retreat'].append(Query('goto', via=(back,), label='retreat'))

        # ---- following: a tracked human ahead moves along the route direction
        if route_dir is not None:
            for h in humans:
                rel = h.position-to_plane(robot_xz)
                dist = float(np.hypot(*rel))
                sp = float(np.hypot(*h.velocity))
                if dist < 4.0 and sp > 0.2:
                    ahead = math.cos(math.atan2(rel[1], rel[0])-route_dir) > 0.5
                    same = math.cos(math.atan2(h.velocity[1], h.velocity[0])-route_dir) > 0.5
                    if ahead and same:
                        cats['wait'].insert(0, Query('goal', speed=0.4, label='goal_slow',
                                                     meta=dict(track=h.track_id)))
                        break

        # ---- temporal alternatives
        # Holding alternates small in-place turns (no native pause action);
        # it deliberately does not rotate toward the route (a sweep choice).
        if conflicts:
            # hold until the legally predicted conflict with the route ahead is gone
            cats['wait'].append(Query('wait_clear', duration=cfg.wait_clear_max, heading=route_dir, label='wait_clear'))
        cats['wait'].append(Query('wait', duration=cfg.wait_short, label='wait_short'))
        (cats['wait_long'] if cfg.v9_select else cats['wait']).append(Query('wait', duration=cfg.wait_long,
                                                                            label='wait_long'))

        # ---- optimistic route through unexplored space (v9): offered when it diverges from the main route
        if cfg.goal_opt and route is not None:
            oview = PlanningView(legal.map, robot_xz, goal_xz, unknown_cost=1.0)
            oroute = oview.route(oview.field([goal_xz]), robot_xz)
            if oroute is not None and len(oroute) >= 2:
                a = point_along(route, cfg.alt_via_m); b = point_along(oroute, cfg.alt_via_m)
                if float(np.hypot(*(np.asarray(a)-np.asarray(b)))) > 0.8:
                    cats['alt'].insert(0, Query('goal', label='goal_opt', meta=dict(unknown_cost=1.0)))

        # ---- alternative routes
        if route is not None:
            for k, alt in enumerate(self._alt_routes(view, route, goal_xz, robot_xz)):
                via = point_along(alt, cfg.alt_via_m)
                via = self._nearest_free(view, rfield, tuple(via), 0.5) or tuple(via)
                cats['alt'].append(Query('goto', via=(via,), label=f'alt_route{k}'))

        # ---- frontiers
        for k, (xz, score) in enumerate(self._frontiers(view, rfield, gfield, route)):
            cats['frontier'].append(Query('goto', via=(xz,), label=f'frontier{k}', meta=dict(score=score)))

        # ---- manoeuvre rotations toward the route heading, both directions
        # (with a 1.23 m body and a blind near field, the sweep side matters)
        if route_dir is not None and abs(wrap(route_dir-pose[3])) > math.radians(30):
            err = wrap(route_dir-pose[3])
            short, long_ = ('left', 'right') if err > 0 else ('right', 'left')
            n_short = int(math.ceil(abs(err)/self.dyn.turn_rad))+2
            n_long = int(math.ceil((2*math.pi-abs(err))/self.dyn.turn_rad))+2
            cats['turn'].append(Query('rotate', heading=route_dir, duration=n_short, label=f'turn_{short}',
                                      meta=dict(direction=short)))
            cats['turn'].append(Query('rotate', heading=route_dir, duration=n_long, label=f'turn_{long_}',
                                      meta=dict(direction=long_)))
        # ---- observation rotations
        gdir = math.atan2(-(goal_xz[1]-robot_xz[1]), goal_xz[0]-robot_xz[0])
        if abs(wrap(gdir-pose[3])) > math.radians(50) and (route_dir is None or abs(wrap(gdir-route_dir)) > math.radians(30)):
            if cfg.glance:
                # v10: bounded glance toward the goal bearing (observe side openings without leaving the route)
                err = wrap(gdir-float(pose[3]))
                k = int(min(abs(err), math.radians(45))/self.dyn.turn_rad)
                if k >= 3:
                    side = 'left' if err > 0 else 'right'
                    cats['observe'].append(Query('rotate', heading=wrap(float(pose[3])+math.copysign(k*self.dyn.turn_rad, err)),
                                                 duration=k+1, label='glance_goal', meta=dict(direction=side)))
            else:
                cats['observe'].append(Query('rotate', heading=gdir, duration=40, label='look_goal'))
        lost = [t for t in legal.tracks if not t.get('visible_now') and
                legal.time-float(t['last_seen_time']) < 2.0]
        if lost:
            t = min(lost, key=lambda t: math.hypot(t['position'][0]-robot_xz[0], t['position'][2]-robot_xz[1]))
            ldir = math.atan2(-(t['position'][2]-robot_xz[1]), t['position'][0]-robot_xz[0])
            if abs(wrap(ldir-pose[3])) > math.radians(35):
                (cats['observe_lost'] if cfg.v9_select else cats['observe']).append(
                    Query('rotate', heading=ldir, duration=40, label='look_lost_human'))

        # ---- stalled near the start or the goal (legal history test) -> exact
        # primitive-count manoeuvres to compose an escape / final approach
        # Exact manoeuvres are offered only where the benchmark placed the robot
        # or the goal arbitrarily: near the legal start pose (confined spawn) or
        # near the public goal (tight final approach).  Elsewhere the robot got
        # there with ordinary queries, which can also take it out again.
        # They also require that the robot's own recent legal history shows a
        # stall (no translation beyond 0.3 m over the last <= 24 steps).
        P0 = np.asarray(legal.poses[0], dtype=float)
        near_start = math.hypot(float(pose[0])-P0[0], float(pose[2])-P0[2]) < cfg.maneuver_start_m
        near_goal = gd < cfg.maneuver_goal_m
        # v8: manoeuvres only while the robot has never left the spawn region (own legal pose history);
        # every manoeuvre used by a solution in all runs so far occurred there
        PH = np.asarray(legal.poses, dtype=float)
        in_spawn = float(np.max(np.hypot(PH[:, 0]-P0[0], PH[:, 2]-P0[2]))) < cfg.maneuver_start_m
        # The stall itself is the signal: obstacles that block the body may be
        # absent from the legal map (e.g. at arm height), so the legal lattice
        # cannot be required to agree.
        # Manoeuvre mode: entered after a strict stall near the start or the goal;
        # it persists while the previous decision was itself a manoeuvre (the
        # generator knows its own last query) and ends with an ordinary query.
        prev = execution.query if execution is not None else None
        in_mode = prev is not None and is_maneuver(prev)
        if cfg.maneuver_spawn_only:
            stuck = in_spawn and (in_mode or self._stalled(legal))
        else:
            # v9: manoeuvre mode only near the start or the goal -- it used to persist anywhere while the
            # previous decision was a manoeuvre, so exact rotate/creep chains wandered 2-4 m from the spawn
            # (YRUkbU5xsYj: 88% of expansions, 2,719 static-collision edges)
            stuck = (near_start or near_goal) and (in_mode or self._stalled(legal))
        diag['stuck'] = stuck
        diag['reset'] = len(legal.poses) <= 1
        diag['maneuver_mode'] = in_mode
        # v7: with no motion history yet (reset state) also offer the finest
        # probes - a single forward creep and single-step rotations - because in
        # a tight spawn every ordinary query (and even waiting, which alternates
        # in-place turns) can collide, leaving no child from which manoeuvre mode
        # could be entered.
        if cfg.root_probes and len(legal.poses) <= 1 and not stuck:
            if cfg.kturns and route_dir is not None and not self.dyn.stateful:
                # v9: multi-point turns toward the route heading are also root probes
                for k in cfg.kturns:
                    for final in ('left', 'right'):
                        away = 'right' if final == 'left' else 'left'
                        cats['maneuver'].append(Query('kturn', heading=route_dir,
                                                      duration=k+1+int(2*math.pi/self.dyn.turn_rad)+2,
                                                      label=f'kturn_{final}{k}',
                                                      meta=dict(first=(away, int(k)), creep=1, final=final)))
            cats['maneuver'] += [Query('creep', duration=1, label='creep1'),
                                 Query('rotate', heading=wrap(float(pose[3])+self.dyn.turn_rad), duration=2,
                                       label='rot_left1', meta=dict(direction='left')),
                                 Query('rotate', heading=wrap(float(pose[3])-self.dyn.turn_rad), duration=2,
                                       label='rot_right1', meta=dict(direction='right'))]
        if stuck:
            # Exact primitive-count manoeuvres: in-place rotations of k turn
            # steps each way and k-step forward creeps.  Composed by the
            # Oracle they express any multi-point turn (there is no reverse
            # action); confined escapes are sensitive to single steps.
            # Order (v9): the 1-step creep first, then rotations by size, then the longer creeps, so a
            # short slot budget still keeps a forward probe.
            rots = []
            for k in cfg.maneuver_turns:
                for side, sgn in (('left', 1.), ('right', -1.)):
                    rots.append(Query('rotate', heading=wrap(float(pose[3])+sgn*k*self.dyn.turn_rad),
                                      duration=k+1, label=f'rot_{side}{k}', meta=dict(direction=side)))
            creeps = [Query('creep', duration=n, label=f'creep{n}') for n in cfg.maneuver_creeps]
            cats['maneuver'] += creeps[:1]+rots[:4]+creeps[1:]+rots[4:]
            if cfg.kturns and route_dir is not None and not self.dyn.stateful:
                # multi-point turns toward the route heading: turn away, step, turn back (both senses);
                # listed first so the quota selector keeps them next to the finest probes
                kt = []
                for k in cfg.kturns:
                    for final in ('left', 'right'):
                        away = 'right' if final == 'left' else 'left'
                        kt.append(Query('kturn', heading=route_dir, duration=k+1+int(2*math.pi/self.dyn.turn_rad)+2,
                                        label=f'kturn_{final}{k}', meta=dict(first=(away, int(k)), creep=1, final=final)))
                fine = cats['maneuver']
                cats['maneuver'] = [q for q in fine if q.label in ('rot_left1', 'rot_right1', 'creep1')]+kt+\
                    [q for q in fine if q.label not in ('rot_left1', 'rot_right1', 'creep1')]

        # ---- advance (manoeuvring space)
        h = heading_vec(pose[3])
        ahead = to_world(to_plane(robot_xz)+cfg.advance_m*h)
        i, j = view.to_local(*ahead)
        if view.inside(i, j) and not view.blocked[i, j]:
            cats['advance'].append(Query('goto', via=(tuple(ahead),), label='advance'))

        if cfg.categories:
            for k in cats:
                if k not in cfg.categories:
                    cats[k] = []
        diag['conflicts'] = len(conflicts)
        if stuck:
            # manoeuvre mode: the space-time variants would crowd out the exact manoeuvres (in mixed mode K=10
            # kept only ~5 slots; vd3HHTEpmyA/yZME6UR9dUN lost every creep) -- keep continue / goal / goal_tc /
            # stop (3t8DB4Uzvkt's solution leaves its spawn manoeuvres with goal_tc)
            fixed = [q for q in fixed if q.label in ('continue', 'goal', 'goal_tc', 'stop')]
        queries = self._select_compact(fixed, cats, diag) if cfg.select == 'compact' else self._select(fixed, cats)
        diag.update(raw={k: len(v) for k, v in cats.items()}, fixed=len(fixed), kept=len(queries),
                    conflicts=len(conflicts))
        return queries, diag

    # --------------------------------------------------------------- helpers
    def _conflicts(self, route, humans, robot_xz):
        """Humans whose constant-velocity path meets the next metres of route."""
        cfg = self.cfg
        return route_conflicts(route, humans, cfg.horizon_m, cfg.robot_speed, cfg.conflict_dist)

    def _stuck(self, view, gfield, legal, goal_xz):
        from . import lattice
        pose = legal.pose
        i, j = view.to_local(float(pose[0]), float(pose[2]))
        if not view.inside(i, j) or gfield[i, j] >= INF/2:
            return True
        here = gfield[i, j]
        goal_field = max(here-1.0/view.res, 0.)
        P = to_plane([pose[0], pose[2]])
        plan = lattice.plan(view, gfield, pose, goal_xz, self.dyn.step_m, self.dyn.turn_rad,
                            goal_tol=0.12, goal_field=goal_field, start_budget=body_cost(legal.map, P, float(pose[3])),
                            max_expand=15000)
        return not plan['prims']

    def _stalled(self, legal):
        """Legal history test: over the last ``stall_window`` steps (at least
        ``stall_min`` available) the robot neither translated (``stall_disp_m``)
        nor turned (``stall_turn_rad`` net), i.e. its queries are not moving it."""
        cfg = self.cfg
        n = min(cfg.stall_window, len(legal.poses)-1)
        if n < cfg.stall_min:
            return False
        P = np.asarray(legal.poses[-1-n:], dtype=float)
        moved = float(np.max(np.hypot(P[:, 0]-P[-1, 0], P[:, 2]-P[-1, 2])))
        turned = abs(wrap(float(P[-1, 3])-float(P[0, 3])))
        return moved < cfg.stall_disp_m and turned < cfg.stall_turn_rad

    def _boxed_in(self, legal):
        from .control import turn_ok, FORWARD_X
        P = to_plane([legal.pose[0], legal.pose[2]]); psi = float(legal.pose[3])
        budget = body_cost(legal.map, P, psi)
        fwd_budget = body_cost(legal.map, P, psi, FORWARD_X)
        fwd = body_cost(legal.map, P+self.dyn.step_m*heading_vec(psi), psi, FORWARD_X) <= fwd_budget
        left = turn_ok(legal.map, P, psi, math.pi/2, True, budget)
        right = turn_ok(legal.map, P, psi, -math.pi/2, True, budget)
        return (not fwd) or (not left and not right)

    def _approaches(self, view, rfield, legal, goal_xz, robot_xz):
        cfg = self.cfg
        G = to_plane(goal_xz)
        out = []
        for k in range(8):
            th = k*math.pi/4
            if body_cost(legal.map, G, th) > 0:
                continue
            B = G-cfg.approach_m*heading_vec(th)
            Bxz = tuple(to_world(B))
            i, j = view.to_local(*Bxz)
            if not view.inside(i, j) or view.blocked[i, j] or rfield[i, j] >= INF/2:
                continue
            if body_cost(legal.map, B, th) > 0:
                continue
            # straight final segment must stay off observed walls
            if any(body_cost(legal.map, B+t*cfg.approach_m*heading_vec(th), th) > 0 for t in (0.33, 0.66)):
                continue
            score = rfield[i, j]*view.res+0.3*abs(wrap(th-float(legal.pose[3])))
            out.append((score, th, Bxz))
        out.sort()
        return [(th, b) for _, th, b in out[:cfg.approaches]]

    def _nearest_open(self, view, rfield, xz, radius=0.4):
        """Nearest reachable non-BLOCKED cell (FREE preferred, UNKNOWN allowed)."""
        f = self._nearest_free(view, rfield, xz, radius)
        if f is not None:
            return f
        i, j = view.to_local(*xz)
        r = int(math.ceil(radius/view.res)); best = None
        for a in range(i-r, i+r+1):
            for b in range(j-r, j+r+1):
                if not view.inside(a, b) or view.blocked[a, b] or rfield[a, b] >= INF/2:
                    continue
                d = (a-i)**2+(b-j)**2
                if best is None or d < best[0]:
                    best = (d, a, b)
        return None if best is None else view.to_world(best[1], best[2])

    def _backtrack_point(self, legal, dist):
        """Point ~dist metres back along the robot's own past path (legal poses)."""
        P = np.asarray(legal.poses)[:, [0, 2]]
        if len(P) < 2:
            return None
        acc = 0.
        for k in range(len(P)-1, 0, -1):
            acc += float(np.hypot(*(P[k]-P[k-1])))
            if acc >= dist:
                return (float(P[k-1][0]), float(P[k-1][1]))
        return None

    def _yield_point(self, view, rfield, legal, robot_xz, humans):
        P0 = to_plane(robot_xz)
        best = None
        for r in (0.8, 1.2, 1.6):
            for a in np.linspace(0, 2*math.pi, 16, endpoint=False):
                Q = P0+r*np.array([math.cos(a), math.sin(a)])
                xz = to_world(Q)
                i, j = view.to_local(*xz)
                if not view.inside(i, j) or view.blocked[i, j] or rfield[i, j] >= INF/2:
                    continue
                if body_cost(legal.map, Q, a) > 0:
                    continue
                clear = min(float(np.hypot(*(h.at(t)-Q))) for h in humans for t in np.linspace(0, 3., 7))
                score = clear-0.1*r-(0.3 if not view.free[i, j] else 0.)
                if best is None or score > best[0]:
                    best = (score, tuple(xz))
        return None if best is None else best[1]

    def _alt_routes(self, view, route, goal_xz, robot_xz):
        cfg = self.cfg
        out = []
        prev = [route]
        cost = view.cost.copy()
        for _ in range(cfg.alt_routes):
            mask = np.zeros(view.state.shape, dtype=bool)
            for r in prev:
                for x, z in _route_points(r, 0.2):
                    i, j = view.to_local(x, z)
                    if view.inside(i, j):
                        mask[i, j] = True
            # corridor of the previous routes (beyond the robot's immediate cell)
            dist = ndimage.distance_transform_edt(~mask)*view.res
            i0, j0 = view.to_local(*robot_xz)
            near_robot = np.zeros_like(mask)
            r0 = int(1.0/view.res)
            near_robot[max(0, i0-r0):i0+r0+1, max(0, j0-r0):j0+r0+1] = True
            pen = (dist < cfg.alt_corridor_m) & ~near_robot
            c2 = np.where(pen, cost*cfg.alt_penalty, cost)
            gi, gj = view.to_local(*goal_xz)
            gi = min(max(gi, 0), view.state.shape[0]-1); gj = min(max(gj, 0), view.state.shape[1]-1)
            f2 = _dijkstra(np.ascontiguousarray(c2), np.asarray([gi], dtype=np.int64), np.asarray([gj], dtype=np.int64),
                           np.asarray([0.]), view.blocked)
            alt = view.route(f2, robot_xz)
            if alt is None:
                break
            # must genuinely diverge from the previous routes within the first metres
            pts = _route_points(alt, 0.2)
            pts = pts[:int(cfg.alt_via_m/0.2)+1]
            far = 0
            for x, z in pts:
                i, j = view.to_local(x, z)
                if view.inside(i, j) and dist[i, j] > cfg.alt_corridor_m*0.75:
                    far += 1
            if far < max(3, len(pts)//3) or path_length(alt) > 2.2*path_length(route)+2.:
                break
            out.append(alt)
            prev.append(alt)
        return out

    def _frontiers(self, view, rfield, gfield, route):
        cfg = self.cfg
        unk = view.unknown
        free = view.free & (rfield < INF/2)
        nb = np.zeros_like(unk)
        nb[1:, :] |= unk[:-1, :]; nb[:-1, :] |= unk[1:, :]; nb[:, 1:] |= unk[:, :-1]; nb[:, :-1] |= unk[:, 1:]
        front = free & nb
        if not front.any():
            return []
        lab, n = ndimage.label(front, structure=np.ones((3, 3)))
        rpts = _route_points(route, 0.3) if route is not None else None
        out = []
        for c in range(1, n+1):
            ii, jj = np.nonzero(lab == c)
            if len(ii) < 4:
                continue
            score = rfield[ii, jj]+gfield[ii, jj]
            k = int(np.argmin(score))
            xz = view.to_world(int(ii[k]), int(jj[k]))
            if rpts is not None and len(rpts):
                if float(np.min(np.hypot(rpts[:, 0]-xz[0], rpts[:, 1]-xz[1]))) < 1.0:
                    continue
            out.append((tuple(xz), float(score[k])*view.res))
        out.sort(key=lambda r: r[1])
        return out[:cfg.frontiers]

    def _select_compact(self, fixed, cats, diag):
        """Compact candidate set for small budgets (K ~ 6): slots by situation.

        progress: STOP (near goal), CONTINUE (only for a running non-goal query: a running goal
        query is equivalent to a fresh GOAL), GOAL;  timing: one hold (follow-slowly, hold until
        the route conflict clears, or a short wait);  situational: manoeuvres when in manoeuvre
        mode (1- and 4-step turns both ways, 1-step creep), else human pass/yield options on a
        conflict, approach headings near the goal, the shorter turn toward the route, then
        advance / alternative route / frontier / retreat.
        """
        cfg = self.cfg
        K = cfg.budget
        chosen = []

        def dup(q):
            for c in chosen:
                if c.kind != q.kind or not q.via or not c.via:
                    if c.signature() == q.signature():
                        return True
                    continue
                if math.hypot(c.via[0][0]-q.via[0][0], c.via[0][1]-q.via[0][1]) < cfg.dedup_m:
                    return True
            return False

        def take(q):
            if q is not None and len(chosen) < K and not dup(q):
                chosen.append(q)
        by_kind = {q.kind: q for q in fixed}
        take(by_kind.get('stop'))
        cont = by_kind.get('continue')
        if cont is not None and cont.meta.get('of') not in ('goal', 'st_goal', 'goal_tc', 'st_goal_tc', None):
            take(cont)
        by_label = {q.label: q for q in fixed}
        take(by_label.get('goal'))
        take(by_label.get('st_goal'))
        cover = cfg.root_cover and diag.get('reset') and not diag.get('stuck') and len(cats.get('turn', [])) >= 2
        for lab in ('goal_tc', 'st_goal_tc'):
            if not cover:
                take(by_label.get(lab))
        if diag.get('stuck'):
            want = ('creep1', 'rot_left1', 'rot_right1', 'rot_left4', 'rot_right4', 'rot_left10', 'rot_right10',
                    'creep3')
            pool = {q.label: q for q in cats.get('maneuver', [])}
            kt = [q for q in cats.get('maneuver', []) if q.kind == 'kturn']
            for q in kt[:2]:
                take(q)
            for lab in want:
                take(pool.get(lab))
            return chosen[:K]
        waits = cats.get('wait', [])
        take(waits[0] if waits else None)
        # reset state: keep the single forward probe (a creep starts manoeuvre mode)
        take(next((q for q in cats.get('maneuver', []) if q.label == 'creep1'), None))
        if cover:
            for q in cats['turn'][:2]:
                take(q)
        order = []
        if diag.get('conflicts'):
            order += [cats.get('yield', []), cats.get('human', [])]
        if cats.get('approach'):
            order += [cats.get('approach', [])]
        if cats.get('turn'):
            order += [cats['turn'][:1]]
        order += [cats.get('advance', []), cats.get('alt', [])[:1], cats.get('frontier', [])[:1],
                  cats.get('retreat', []), cats.get('turn', [])[1:], cats.get('observe', []),
                  cats.get('human', []), cats.get('yield', []), waits[1:]]
        for pool in order:
            for q in pool:
                take(q)
                if len(chosen) >= K:
                    return chosen
        return chosen[:K]

    def _select(self, fixed, cats):
        """Situation-aware quotas in priority order, then round-robin leftovers.

        Quotas react only to the categories' legal triggers (boxed in, human
        conflict, near goal); a category with no candidates takes no slots.
        """
        cfg = self.cfg
        chosen = list(fixed)

        def dup(q):
            for c in chosen:
                if c.kind != q.kind or not q.via or not c.via:
                    if c.signature() == q.signature():
                        return True
                    continue
                if math.hypot(c.via[0][0]-q.via[0][0], c.via[0][1]-q.via[0][1]) < cfg.dedup_m:
                    return True
            return False
        if cfg.v9_select:
            # usage in ~60 DEV/HOLDOUT solutions: goal, advance, turns, approach, alt_route, wait_short, pass_*
            # dominate; wait_long, retreat, look_lost_human never occur -> leftovers only
            quotas = (('maneuver', 8), ('human', 3), ('approach', 2), ('yield', 2), ('turn', 2), ('wait', 2),
                      ('alt', 1), ('frontier', 1), ('advance', 1), ('observe', 1), ('retreat', 0), ('wait_long', 0),
                      ('observe_lost', 0))
        else:
            quotas = (('maneuver', 8), ('human', 3), ('approach', 2), ('yield', 2), ('turn', 2), ('wait', 2),
                      ('retreat', 1), ('advance', 1), ('alt', 1), ('frontier', 1), ('observe', 1))
        pools = {k: list(cats.get(k, [])) for k, _ in quotas}
        for k, quota in quotas:
            taken = 0
            while pools[k] and taken < quota and len(chosen) < cfg.budget:
                q = pools[k].pop(0)
                if not dup(q):
                    chosen.append(q); taken += 1
        while len(chosen) < cfg.budget and any(pools.values()):
            for k, _ in quotas:
                while pools[k]:
                    q = pools[k].pop(0)
                    if not dup(q):
                        chosen.append(q)
                        break
                if len(chosen) >= cfg.budget:
                    break
        return chosen[:cfg.budget]
