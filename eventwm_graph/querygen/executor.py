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
from .control import body_cost, HOLD_SWEEP, HOLD_GAZE
from .motion import MotionRealizer
from .planners import PlanContext, make_adapter
from .dynamics import STEP_SECONDS, NATIVE_TURN_RAD

ARRIVE_M = 0.30
HARD_CLEARANCE = 0.30        # one-step rectangle-to-predicted-human clearance (all query kinds)
CLOSE_HARD_CLEARANCE = 0.27  # 'close' queries (declared body proxy: 0.25 m)
# 'close' queries: ORCA disk proxies sized to the body *width* (the rectangle hard check above
# guards the nose and tail) instead of the conservative 0.55 + 0.30 + 0.25 m combined radius, and a
# 0.5 s human horizon (~10 native steps; the native robot stops instantly and is ~4x faster than
# the humans, so 1.5 s forbade approaching any human within ~6 m along the heading)
CLOSE_ORCA = dict(robot_radius=0.30, human_radius=0.25, margin=0.05, time_horizon=0.5)
CLOSE_SFM = dict(contact_radius=0.55)
CLOSE_DWA = dict(safety=CLOSE_HARD_CLEARANCE, clearance_cap=0.8)
# v12 'vclose' queries: contact-level passes.  The official rules count only physical robot-human contact (Falcon's
# humans collide as thin articulated skeletons); native witnesses pass with the robot rectangle < 0.25 m from human
# centres without contact (VqCaAuuoeWk: 20 such steps in a 94-step solution that no 0.30 m-clearance query could
# express).  Legal inputs unchanged; the clearance is to *predicted* (constant-velocity) human centres.
VCLOSE_HARD_CLEARANCE = 0.20
VCLOSE_ORCA = dict(robot_radius=0.25, human_radius=0.18, margin=0.0, time_horizon=0.4)
VCLOSE_SFM = dict(contact_radius=0.45)
VCLOSE_DWA = dict(safety=VCLOSE_HARD_CLEARANCE, clearance_cap=0.6)
# 'stgoal' queries (v9): legal space-time lattice plan (stplan.py) re-planned every ST_REPLAN steps
ST_HORIZON = 40
ST_MARGIN = 0.35            # rectangle to predicted human centre at t = 0 ...
ST_MARGIN_CLOSE = 0.28      # ... for 'close' queries (declared body proxy 0.25 m)
ST_MARGIN_VCLOSE = 0.21     # ... for v12 'vclose' (contact-level) queries
ST_GROWTH = 0.004           # ... plus this per predicted step (CV uncertainty)
ST_REPLAN = 4
ST_HUMAN_RANGE = 8.0
VIA_ADVANCE_M = 0.45
STOP_M = 0.15
STUCK_STOP_M = 0.22
STUCK_STOP_STRICT_M = 0.19   # v12 meta 'stop19': official success needs geodesic < 0.2 m, so a stuck STOP at
                             # 0.2-0.22 m can only fail (YM4nG4pSAEJ: default policy stopped at 0.21 m)
LOOKAHEAD_M = 0.8


# Layer 3 option (process-wide, default OFF -> released behaviour unchanged; L3_ANTIDITHER=<steps>): the executor's
# turn decisions of robots with speed state can alternate left / right without translating until the episode times
# out (dev floors: the run ends in such a dither in 5-14 of the timeouts of every momentum family vs 1-2 for native /
# slowturn).  With the option, after 6 strictly alternating in-place turns the execution commits to its last turn
# direction for <steps> steps (or until it translates), so the heading leaves the tie and the planner finds a way on.
ANTIDITHER = int(__import__('os').environ.get('L3_ANTIDITHER', '0') or 0)
# Layer 3 option (default OFF; with L3_ANTIDITHER): L3_ANTIDITHER_HOLD=0 -- alternations that ARE holds (adapter modes
# hold / duty / jammed, wait / wait_clear queries, reflex holds) neither count nor commit: a native hold is a strict
# left/right alternation, so every hold of 6+ steps became a forced 12-step (57-degree) rotation and a turn back
# (tuning set, quick floor: 221 forced rotations in 43 episodes; ranker: 25, inside 4 of 35 non-spawn collisions).
ANTIDITHER_HOLD = __import__('os').environ.get('L3_ANTIDITHER_HOLD', '1') != '0'
# Layer 3 option (default OFF; L3_BUMP_CREEP=1): creep queries respect bump memory -- a creep step whose body front lies
# in a bump disc holds instead of pushing (slow / quick timeouts re-issued creep3 47-98 times into unseen obstacles).
BUMP_CREEP = __import__('os').environ.get('L3_BUMP_CREEP') == '1'
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_DYN=1 -- the reflex horizon and its escape-turn lengths scale
# with the robot's turn rate (native turn / turn, horizon capped at 62 steps), so a slow-turning robot considers the
# same escape angles (10-90 degrees) and sees a whole turn ahead (slowturn: a 90-degree turn takes 38 steps > 31).
EVADE2_DYN = __import__('os').environ.get('L3_EVADE_DYN') == '1'
# Layer 3 option (default OFF; L3_STOP_GUARD=1): goal-reach STOP for EVERY query kind -- STOP when the legal Euclidean
# goal distance is below STOP_M at rest (today only goal / stgoal queries stop: a creep or rotate query running at the
# goal never does), or when the goal disc is observed BLOCKED on the legal map (an off-navmesh goal), the robot is within
# STOP_GUARD_OFF_M and its goal distance has not decreased by 0.02 m over the last 16 steps.  Tuning-set timeouts sat
# within 0.2 m geodesic of the goal for up to 208 steps without stopping (VVfe2KiqLaN/4632, MjzBosUX3WW/202).
STOP_GUARD = __import__('os').environ.get('L3_STOP_GUARD') == '1'
# Layer 3 option (default OFF; L3_ST_COMMIT=<delta steps>, e.g. 3): space-time plan commitment on real legal states.
# (a) A scheduled re-plan (every ST_REPLAN steps, robot on the planned pose) keeps the running plan unless the fresh plan
# is shorter by more than <delta> steps of (remaining primitives + legal cost-to-go of the end pose / step) or the old
# plan's remaining poses no longer keep the space-time margin from the NEW constant-velocity predictions.  (b) The plan
# survives decision boundaries of the same execution and passes to a new stgoal execution with the same target and
# margin class.  Exposure analysis (tuning set, 20 replays): the space-time plan produced 57% of the wasted turning (60% of
# its turns undone; 406 of 680 wasted steps with nobody near).  Imagined states keep the released behaviour.
ST_COMMIT = float(__import__('os').environ.get('L3_ST_COMMIT', '0') or 0)
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_TURNCOST=<w> -- among clearance-equivalent reflex candidates,
# rank by legal progress minus w x (time to turn back onto the route at the candidate's end, in metres of driving):
# holds, advances and 'hold then resume' win ties over turn-away escapes (78% of the reflex's turns were undone later).
EVADE2_TURNCOST = float(__import__('os').environ.get('L3_EVADE_TURNCOST', '0') or 0)
STOP_GUARD_OFF_M = float(__import__('os').environ.get('L3_STOP_GUARD_OFF_M', '0.40') or 0.40)

# Layer 3 option (process-wide, default OFF -> released behaviour unchanged; L3_EVADE2=1): plan-aware threat reflex for
# queries with meta 'evade' (replaces the v11 reflex, which tests only the next primitive followed by holding over 14
# steps at 0.30 m and escapes only by turning first).  Every step the robot's COMMITTED motion -- this step's primitive,
# then the rest of the static / space-time plan it follows, then holding -- is rolled out for L3_EVADE_H steps with the
# exact turn rate and speed levels against the legal constant-velocity predictions of the tracked humans.  If the robot
# rectangle comes within L3_EVADE_R of a predicted human centre, the reflex picks the best of a fixed set of short native
# sequences (hold; advance n; turn k then advance n; turn k; hold k then resume the plan) by capped predicted clearance,
# then legal route progress, then fewest off-plan steps, and executes it while it stays best (hysteresis); it releases
# once the planner's own motion is clear again.  Held-out floor collisions (dev analysis, 31 Oct): 77% are a walking
# human meeting a turning or holding robot that tracked it; an escape exists 1 s before contact in ~75% of them.
# Applied by every Layer 3 process that runs the executor (actors, Sandbox imagination, labellers, official replay).
_ENV = __import__('os').environ
EVADE2 = _ENV.get('L3_EVADE2') == '1'
EVADE2_H = int(_ENV.get('L3_EVADE_H', '31') or 31)
EVADE2_R = float(_ENV.get('L3_EVADE_R', '0.45') or 0.45)
EVADE2_CAP = float(_ENV.get('L3_EVADE_CAP', '0.30') or 0.30)    # clearance above R + this is fully safe (then progress)
EVADE2_GAIN = float(_ENV.get('L3_EVADE_GAIN', '0.10') or 0.10)  # an escape must beat the committed motion by this much
EVADE2_KEEP = 0.05          # hysteresis bonus of the running escape
EVADE2_RANGE = 8.0          # m: tracked humans considered (the robot covers ~6.5 m in 31 native steps)
EVADE2_STATIC = float(_ENV.get('L3_EVADE_STATIC', '0.20') or 0.20)   # m: threat distance of a standing human
EVADE2_STATS = _ENV.get('L3_EVADE_STATS')     # diagnostics: append per-process trigger counts (JSON line) to this file
# ANALYSIS ONLY (privileged; never in the benchmark): L3_EVADE_TRUTH=all | tracked -- the reflex sees the TRUE next
# EVADE2_H positions of every human ('all') or of the humans tracked now ('tracked') instead of the legal constant-
# velocity predictions (the real-episode runtime attaches the tape as ``legal._ev2_truth``; imagined states never
# carry it).  The ceiling of a reflex with perfect prediction (all) and perfect prediction of seen people (tracked).
EVADE2_TRUTH = _ENV.get('L3_EVADE_TRUTH') or None
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_FC=fc1 | modes -- the reflex predicts the tracked people of the
# REAL legal state with the map-aware multimodal forecaster (l3/peoplemap.py, model L3_FC_MODEL; legal inputs: track
# history + the robot's legal map) instead of constant velocity: 'fc1' the weighted mean of each person's modes, 'modes'
# every mode with weight >= L3_EVADE_FCW (default 0.2) as its own hypothesis.  Imagined states (no track log) keep
# constant velocity, as their people move.  Tracks without a detection in the last 2 s stay constant velocity.
EVADE2_FC = _ENV.get('L3_EVADE_FC') or None
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_RISK=<model dir> -- the learned threat reflex (l3/reflex.py:
# legal features, trained on privileged labels) decides on real legal states (imagined states keep the rule);
# L3_EVADE_PTRIG / L3_EVADE_PGAIN its trigger probability and required gain.  L3_EVADE_LOG=<dir>: TRAINING DATA ONLY --
# log every candidate's legal features with its TRUE clearance from the human tape at sampled steps (privileged labels).
EVADE2_RISK = _ENV.get('L3_EVADE_RISK') or None
EVADE2_PTRIG = float(_ENV.get('L3_EVADE_PTRIG', '0.3') or 0.3)
EVADE2_PGAIN = float(_ENV.get('L3_EVADE_PGAIN', '0.1') or 0.1)
EVADE2_LOG = _ENV.get('L3_EVADE_LOG') or None
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_REAL=1 -- the reflex runs on real legal states only; the
# Sandbox imagination (states without a track log) executes the released executor, so a ranker trained on imagined
# features of the old executor sees unchanged inputs and the reflex is a safety layer underneath it.
EVADE2_REAL = _ENV.get('L3_EVADE_REAL') == '1'
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_MEM=<s> -- the reflex also avoids people whose track the
# tracker dropped (> 2 s without a detection) but who were detected within the last <s> seconds (legal track log): last
# detected position extrapolated at the last valid velocity for at most L3_EVADE_MEM_EXT seconds (default 2.5).  Held-out
# attribution (tuning set, ranker + reflex): 5 of 22 non-spawn collisions were with people seen 1.2-3.0 s earlier.
EVADE2_MEM = float(_ENV.get('L3_EVADE_MEM', '0') or 0)
# Layer 3 option (default OFF; with L3_EVADE2=1): L3_EVADE_RESTART=1 -- pause-robust prediction: a tracked person whose
# legal velocity is below 0.2 m/s now but who walked faster than 0.8 m/s within the last 1.5 s (legal track log) also
# counts as a hypothetical walker restarting now along the last walking velocity (Falcon people stop and go; earlier
# attribution: ~40% of contacts came ~0.2 s after a restart of a person the robot had tracked).
EVADE2_RESTART = _ENV.get('L3_EVADE_RESTART') == '1'
# Layer 3 option (default OFF; L3_CORNER_SLOW=<duty>, e.g. 0.5): on real legal states a native robot advancing past free
# space that lies INSIDE the camera's view near the path ahead (0.6-1.4 m ahead, up to 1.2 m sideways) but has not been
# observed for 0.3 s (so something occludes it) advances only on a <duty> share of steps (holding otherwise) until that
# space is seen.  Held-out attribution (tuning set, ranker + reflex): 4 of 22 non-spawn collisions were people stepping
# out of an occluded spot almost straight ahead of a robot moving at 4.3 m/s.
CORNER_SLOW = float(_ENV.get('L3_CORNER_SLOW', '0') or 0)
EVADE2_MEM_EXT = float(_ENV.get('L3_EVADE_MEM_EXT', '2.5') or 2.5)
EVADE2_FCW = float(_ENV.get('L3_EVADE_FCW', '0.2') or 0.2)

# Layer 3 option (process-wide, default OFF -> released behaviour unchanged; L3_UNREACH_FALLBACK=1): a goal / goto /
# stgoal query whose route is 'unreachable' on the conservative legal map retries in the same step on the tight map (two
# outer band rings released), then on the 'thru' map (observed walls soft, bump discs hard), and keeps that relaxation
# for the rest of its life.  Without it the query ends after one step and is re-issued: held-out timeouts contained
# 13 floor / 10 ranker loops of 200-400 one-step 'unreachable' goal queries (dev analysis, 31 Oct).
UNREACH_FALLBACK = _ENV.get('L3_UNREACH_FALLBACK') == '1'
_EV2 = dict(steps=0, threat=0, escape=0, keep=0, release=0)
if EVADE2_STATS:
    import atexit as _atexit
    import json as _json

    def _dump_ev2():
        if _EV2['steps']:
            with open(EVADE2_STATS, 'a') as f:
                f.write(_json.dumps(dict(_EV2, pid=__import__('os').getpid()))+'\n')
    _atexit.register(_dump_ev2)


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


_CORE = {}


def _core_map(legal):
    """``tight_legal(legal, 2).map`` (observed wall cores), cached per map object and revision (the reflex reads it
    every step; the erosion of the whole grid costs ~2 ms)."""
    from .legal import BLOCKED
    g = legal.map
    key = (id(g), g.revision, g.ox, g.oz, g.state.shape, int(np.count_nonzero(g.state == BLOCKED)))
    m = _CORE.get(key)
    if m is None:
        if len(_CORE) > 8:
            _CORE.clear()
        m = _CORE[key] = tight_legal(legal, 2).map
    return m


SOFT_COST = 6.0     # v15 candidate 'thru' queries: route cost factor of soft (seen-but-unproven) obstacle cells


class _ThruLegal:
    """v15 candidate (meta 'thru'): every observed BLOCKED cell becomes a *soft* obstacle -- UNKNOWN for the body
    checks, routed at SOFT_COST -- except the proprioceptive bump-memory discs (a forward step there did not move the
    base), which stay BLOCKED.  The legal map's band is visual (wall hits 0.2-1.4 m above the base, dilated 0.25 m);
    the native robot moves kinematically on the navmesh and may scrape geometry (static contact is not a failure under
    the official rules): every hard development witness crosses band cells at body height (p32JzpQyhPk, VqCaAuuoeWk,
    ANmWrL7Kz7h, FXuXGH9YQTW, kdFEfVoT1WE: 115-505 hits within 0.25 m of the path, contact in 18-24% of steps).  A
    'thru' query tries such passages and learns real obstacles from its own motion (bump memory)."""

    def __init__(self, legal):
        from .legal import GridMap, BLOCKED, UNKNOWN, LegalState
        g0 = legal.map
        g = GridMap.__new__(GridMap)
        g.__dict__.update(g0.__dict__)
        st = g0.state.copy()
        soft = st == BLOCKED
        st[soft] = UNKNOWN
        r = float(LegalState.BUMP_MEMORY)
        if r > 0:
            n = int(math.ceil(r/g0.res))
            for x, z in getattr(legal, 'bumps', []) or []:
                ci, cj = g0.index(x, z)
                for a in range(-n, n+1):
                    for b in range(-n, n+1):
                        i, j = ci+a, cj+b
                        if math.hypot(a*g0.res, b*g0.res) <= r and 0 <= i < st.shape[0] and 0 <= j < st.shape[1]:
                            st[i, j] = BLOCKED; soft[i, j] = False
        g.state = st
        g.soft = soft
        self._legal = legal
        self.map = g

    def __getattr__(self, k):
        return getattr(self._legal, k)


STATIC_HUMAN_SPEED = 0.05    # m/s: a tracked human with a valid velocity estimate below this is standing still
STATIC_HUMAN_R = 0.35        # m: BLOCKED disc around a standing human for escalated plans (body ~0.25 m + 0.1 m)


def static_humans(legal):
    """Legally tracked humans that stand still (valid constant-velocity estimate below STATIC_HUMAN_SPEED).
    Falcon's humans stop for good once their waypoint routes end (typically after 100-400 steps)."""
    return [h for h in human_estimates(legal.tracks, legal.time)
            if h.valid_velocity and float(np.hypot(*h.velocity)) < STATIC_HUMAN_SPEED]


class _StaticHumanLegal:
    """Legal state whose map also marks a disc around every standing tracked human BLOCKED (v12 meta 'shum'):
    an escalated plan routes around standing humans like around furniture instead of holding behind them (the
    one-step hard human check still applies).  The revision key includes the disc cells, so plans and fields
    refresh when a human starts or stops."""

    def __init__(self, legal, humans):
        from .legal import GridMap, BLOCKED
        g0 = legal.map
        g = GridMap.__new__(GridMap)
        g.__dict__.update(g0.__dict__)
        st = g0.state.copy()
        n = int(math.ceil(STATIC_HUMAN_R/g0.res))
        cells = set()
        for h in humans:
            x, z = to_world(h.position)
            ci, cj = g0.index(x, z)
            for a in range(-n, n+1):
                for b in range(-n, n+1):
                    i, j = ci+a, cj+b
                    if not g0.inside(i, j):
                        continue
                    cx, cz = g0.center_xz(i, j)
                    if math.hypot(cx-x, cz-z) <= STATIC_HUMAN_R:
                        st[i, j] = BLOCKED
                        cells.add((i+g0.ox, j+g0.oz))
        g.state = st
        g.revision = (g0.revision, tuple(sorted(cells)))
        self._legal = legal
        self.map = g

    def __getattr__(self, k):
        return getattr(self._legal, k)


@dataclass
class Query:
    kind: str
    via: tuple = ()
    heading: float = None
    duration: int = 0
    speed: float = 1.0
    label: str = ''
    meta: dict = field(default_factory=dict)

    EXEC_META = ('tight', 'close', 'commit', 'unknown_cost', 'exact_end', 'arrive_exact', 'blind', 'unstick',
                 'evade', 'ukeep', 'bumpf', 'shum', 'stop19', 'uprog', 'vclose', 'action', 'thru',
                 'short')  # meta that change execution (short: the Oracle runs the edge for that many steps)

    def signature(self):
        v = tuple((round(float(x), 2), round(float(z), 2)) for x, z in self.via)
        h = None if self.heading is None else round(float(self.heading), 2)
        m = tuple((k, (self.meta or {})[k]) for k in self.EXEC_META if (self.meta or {}).get(k))
        return (self.kind, v, h, int(self.duration), round(float(self.speed), 2))+((m,) if m else ())

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
        self.escalated = False      # v11 'unstick': static deadlock -> tight map + plan following (hard check only)
        self._still = 0
        self._last_xz = None
        self._still_psi = []        # v12 'uprog': headings during the current no-translation streak
        if (query.meta or {}).get('vclose'):
            self._make_close(very=True)
        elif (query.meta or {}).get('close'):
            self._make_close()
        if (query.meta or {}).get('commit') and hasattr(self.planner, 'commit'):
            self.planner.commit = True
        self.planner.reset()

    def _stuck_stop_m(self):
        return STUCK_STOP_STRICT_M if (self.query.meta or {}).get('stop19') else STUCK_STOP_M

    def _bump_blocked(self, legal, P, psi):
        """v12 meta 'bumpf': the body's front contact point at plane pose (P, psi) lies in a bump-memory disc --
        proprioceptive evidence that a forward step from here does not move the base (an obstacle the camera does
        not see).  The base-centre checks on the (tight) legal map miss it: the disc sits 0.684 m ahead of the base."""
        bumps = getattr(legal, 'bumps', None)
        if not bumps or not (self.query.meta or {}).get('bumpf'):
            return False
        from .legal import LegalState
        r = float(LegalState.BUMP_MEMORY)
        fx, fz = to_world(P+LegalState.BUMP_FRONT_M*heading_vec(psi))
        return any(math.hypot(fx-bx, fz-bz) <= r for bx, bz in bumps)

    EVADE_H = 14            # steps of constant-velocity human prediction
    EVADE_R = 0.30          # rectangle-to-human-centre distance treated as a contact threat (official contact needs touch)
    EVADE_SEQS = ('f'*8, 'l'+'f'*7, 'r'+'f'*7, 'll'+'f'*6, 'rr'+'f'*6, 'llll'+'f'*4, 'rrrr'+'f'*4, 'l'*8, 'r'*8)

    def _evade(self, legal, prim):
        """v11 meta 'evade': human-threat reflex (legal).  Falcon's humans do not yield, and 13 of 16 default-policy
        collisions hit a robot that held or turned in place while a human it had tracked for 10-20 steps walked into
        it.  If the query's own next primitive followed by holding brings the robot rectangle within EVADE_R of a
        tracked human's constant-velocity prediction within EVADE_H steps, execute the first primitive of the short
        native sequence (turn 0-8 steps, then advance) that maximises the predicted clearance instead.  Static
        obstacles are soft here (wall contact is not a benchmark failure): only observed wall cores (the legal map
        with its two outer band rings released) stop a predicted advance."""
        from .planners import DWAPlanner
        P0 = to_plane([legal.pose[0], legal.pose[2]])
        humans = [h for h in human_estimates(legal.tracks, legal.time) if float(np.hypot(*(h.position-P0))) < 4.0]
        if not humans:
            return prim
        H = np.stack([np.stack([h.at(k*STEP_SECONDS) for h in humans]) for k in range(1, self.EVADE_H+1)])  # [T,N,2]
        core = tight_legal(legal, 2).map
        from .legal import BLOCKED

        def clearance(seq):
            P, psi, dmin = P0.copy(), float(legal.pose[3]), math.inf
            for k in range(self.EVADE_H):
                a = seq[k] if k < len(seq) else 'h'
                if a == 'f':
                    Pn = P+self.dyn.step_m*heading_vec(psi)
                    x, z = to_world(Pn)
                    if core.state_at(x, z) != BLOCKED and not self._bump_blocked(legal, P, psi):
                        P = Pn
                elif a == 'l':
                    psi += self.dyn.turn_rad
                elif a == 'r':
                    psi -= self.dyn.turn_rad
                dmin = min(dmin, float(DWAPlanner._rect_dist(P, psi, H[k]).min()))
            return dmin
        own = {'forward': 'f', 'left': 'l', 'right': 'r'}[prim]
        d_own = clearance(own)
        if d_own >= self.EVADE_R:
            return prim
        best = max(self.EVADE_SEQS, key=clearance)
        if clearance(best) > d_own+0.05:
            self.plan = None
            self.reason = 'evade'
            return {'f': 'forward', 'l': 'left', 'r': 'right'}[best[0]]
        return prim

    def _stop_guard(self, legal, level):
        """L3_STOP_GUARD (see the option): ('stop', 0) when the goal is reached by the legal tests, else None."""
        if self.dyn.stateful and int(level) > 0:
            return None
        gx, gz = float(legal.goal[0]), float(legal.goal[2])
        d = math.hypot(gx-float(legal.pose[0]), gz-float(legal.pose[2]))
        hist = getattr(self, '_sg_hist', None)
        if hist is None:
            hist = self._sg_hist = []
        hist.append(d)
        if len(hist) > 40:
            del hist[0]
        stop = d < STOP_M
        if not stop and d < STOP_GUARD_OFF_M and len(hist) >= 16 and hist[-16]-d < 0.02:
            from .legal import BLOCKED
            g = legal.map
            n = blk = 0
            for a in np.linspace(-0.2, 0.2, 5):
                for b in np.linspace(-0.2, 0.2, 5):
                    if a*a+b*b <= 0.04+1e-9:
                        n += 1
                        blk += g.state_at(gx+a, gz+b) == BLOCKED
            stop = g.state_at(gx, gz) == BLOCKED and blk >= 0.5*n
        if stop:
            self.status, self.reason = 'done', 'stop_guard'
            return 'stop', 0
        return None

    def _bump_front(self, legal, P, psi):
        """The body's front contact point at (P, psi) lies in a bump-memory disc (as ``_bump_blocked``, without the
        'bumpf' meta requirement)."""
        bumps = getattr(legal, 'bumps', None)
        if not bumps:
            return False
        from .legal import LegalState
        r = float(LegalState.BUMP_MEMORY)
        fx, fz = to_world(P+LegalState.BUMP_FRONT_M*heading_vec(psi))
        return any(math.hypot(fx-bx, fz-bz) <= r for bx, bz in bumps)

    _EV2_CODE = {'forward': 'f', 'left': 'l', 'right': 'r'}

    def _evade2(self, legal, prim, level):
        """L3_EVADE2 (module docstring at ``EVADE2``).  ``prim``: this step's kinematic primitive, or a planner-realised
        command (DWA) for robots without speed state; returns the (possibly replaced) primitive of the same kind."""
        from .planners import Realized
        realized = isinstance(prim, Realized)
        if realized and self.dyn.stateful:
            return prim                    # momentum DWA: its own rollouts already brake for humans
        kin = prim[0] if realized else prim
        if kin not in self._EV2_CODE:
            self._esc = None
            return prim
        _EV2['steps'] += 1
        P0 = to_plane([legal.pose[0], legal.pose[2]])
        tscale = (NATIVE_TURN_RAD/self.dyn.turn_rad) if EVADE2_DYN else 1.
        H = min(62, int(round(EVADE2_H*tscale))) if EVADE2_DYN else EVADE2_H
        esc = getattr(self, '_esc', None)
        tape = getattr(legal, '_ev2_truth', None)                     # PRIVILEGED tape (diagnostics / labels only)
        real = getattr(legal, 'track_log', None) is not None          # a real legal state (imagined ones: no log)
        learned = bool(EVADE2_RISK) and real
        log = bool(EVADE2_LOG) and tape is not None
        cur = keep = None
        if EVADE2_TRUTH and tape is not None:
            T_, idx = np.asarray(tape[0], np.float64), int(tape[1])   # ANALYSIS ONLY (see EVADE2_TRUTH)
            rows = np.minimum(np.arange(idx+1, idx+H+1), len(T_)-1)
            cur = T_[min(idx, len(T_)-1)]
            keep = np.hypot(cur[:, 0]-legal.pose[0], cur[:, 2]-legal.pose[2]) < EVADE2_RANGE if len(cur) else \
                np.zeros(0, bool)
            if EVADE2_TRUTH == 'tracked' and len(cur):
                seen = np.zeros(len(cur), bool)
                for t in human_estimates(legal.tracks, legal.time):
                    x, z = to_world(t.position)
                    d = np.hypot(cur[:, 0]-x, cur[:, 2]-z)
                    j = int(np.argmin(d))
                    if d[j] < 0.75:
                        seen[j] = True
                keep &= seen
            Hp = to_plane(T_[rows][:, keep][:, :, [0, 2]]) if keep.any() else np.zeros((H, 0, 2))
        elif EVADE2_FC and real:
            Hp = self._fc_predict(legal, P0, H)                       # L3_EVADE_FC (legal forecaster)
        else:
            humans = [h for h in human_estimates(legal.tracks, legal.time)
                      if float(np.hypot(*(h.position-P0))) < EVADE2_RANGE]
            T = np.arange(1, H+1)[:, None, None]*STEP_SECONDS
            Hp = (np.stack([h.position for h in humans])[None]+np.stack([h.velocity for h in humans])[None]*T
                  if humans else np.zeros((H, 0, 2)))                # [H,N,2]
            if EVADE2_MEM > 0 and real:
                mem = self._remembered(legal, P0, [h.track_id for h in humans])
                if mem:
                    Mp = np.stack([p for p, v in mem])[None]+np.stack([v for p, v in mem])[None]*T
                    Hp = np.concatenate([Hp, Mp], axis=1)
            if EVADE2_RESTART and real and humans:
                ghosts = self._restart_ghosts(legal, humans)
                if ghosts:
                    Gp = np.stack([p for p, v in ghosts])[None]+np.stack([v for p, v in ghosts])[None]*T
                    Hp = np.concatenate([Hp, Gp], axis=1)
        if not Hp.shape[1] and not learned and not log:
            self._esc = None
            return prim
        # standing humans (predicted to move < 0.3 m over the horizon) are a threat only within EVADE2_STATIC of the
        # rectangle: they are obstacles for the planner (the one-step check keeps 0.30 m), and a reflex that yields to
        # them holds forever beside a standing person
        if Hp.shape[1]:
            p_now = Hp[0] if cur is None else to_plane(cur[keep][:, [0, 2]])
            moving = np.hypot(*(Hp-p_now[None]).transpose(2, 0, 1)).max(axis=0) >= 0.3          # [N]
        else:
            moving = np.zeros(0, bool)
        offs = np.where(moving, 0., EVADE2_R-EVADE2_STATIC)
        dyn = self.dyn
        psi0 = float(legal.pose[3])
        core = _core_map(legal)
        from .legal import BLOCKED
        free_memo = {}

        def free(P, psi, Pn):
            key = (round(float(Pn[0]), 3), round(float(Pn[1]), 3), round(psi, 3))
            r = free_memo.get(key)
            if r is None:
                x, z = to_world(Pn)
                r = free_memo[key] = core.state_at(x, z) != BLOCKED and not self._bump_blocked(legal, P, psi)
            return r
        top = len(dyn.levels)-1

        def roll(seq):
            P, psi, lv = P0.copy(), psi0, int(level)
            out = np.empty((H, 3))
            for k in range(H):
                a = seq[k] if k < len(seq) else 'h'
                d = 0.
                if dyn.stateful:
                    if a == 'f':
                        lv = min(top, lv+dyn.accel) if lv < top else top
                        d = dyn.levels[lv]
                    elif lv > 0 and (a == 'h' or dyn.turn_needs_stop):
                        lv = max(0, lv-dyn.brake)
                        d = dyn.levels[lv]
                    elif a == 'l':
                        psi += dyn.turn_rad
                    elif a == 'r':
                        psi -= dyn.turn_rad
                elif a == 'f':
                    d = dyn.step_m
                elif a == 'l':
                    psi += dyn.turn_rad
                elif a == 'r':
                    psi -= dyn.turn_rad
                if d > 0.:
                    Pn = P+d*heading_vec(psi)
                    if free(P, psi, Pn):
                        P = Pn
                    elif dyn.stateful:
                        lv = 0
                out[k] = (P[0], P[1], psi)
            return out

        def clearance(trajs, pts=None, off=None):
            from eventwm_graph.l3.reflex import rect_dist
            pts = Hp if pts is None else pts
            if not pts.shape[1]:
                return np.full(len(trajs), np.inf)
            D = rect_dist(np.asarray(trajs), pts)                   # [S,H,N]
            return (D+(offs if off is None else off)).min(axis=(1, 2))
        plan = [self._EV2_CODE.get(p, 'h') for p in (self._plan_now or [])]
        own = [self._EV2_CODE[kin]]+(plan[1:H] if plan and plan[0] == self._EV2_CODE[kin] else [])
        own_traj = roll(own)
        d_own = float(clearance([own_traj])[0])
        sample = log and (_EV2['steps'] % 3 == 0 or d_own < EVADE2_R+0.3)
        if not learned and not sample:
            if esc is None and d_own >= EVADE2_R:
                return prim
            if esc is not None and d_own >= EVADE2_R+0.1:
                self._esc = None
                _EV2['release'] += 1
                return prim
        cands, kinds = [own, ['h']*H], [0, 1]
        for n in (2, 4, 8, 12, 16, 24):
            cands.append(['f']*n); kinds.append(2)
        for a in 'lr':
            for k in ((2, 4, 6, 9, 12, 16, 19) if tscale == 1. else
                      sorted({max(1, int(round(k0*tscale))) for k0 in (2, 4, 6, 9, 12, 16, 19)})):
                for n in (0, 4, 8, 16):
                    if k+n <= H:
                        cands.append([a]*k+['f']*n); kinds.append(3 if n else 4)
        waitk = {}
        if plan:
            for k in (4, 8, 16):
                waitk[len(cands)] = k
                cands.append(['h']*k+plan[:H-k]); kinds.append(5)
        i_esc = None
        if esc:
            i_esc = len(cands)
            cands.append(list(esc)); kinds.append(6)
        if dyn.stateful:
            trajs = np.stack([own_traj]+[roll(c) for c in cands[1:]])
        else:
            trajs = self._ev2_fast_trajs(legal, cands, kinds, waitk, own_traj, plan, roll, P0, psi0, H)
        prog = np.zeros(len(cands))
        if self.view is not None and self.field is not None:
            c0 = self.view.cost_to_go(self.field, tuple(to_world(P0)))
            if np.isfinite(c0):
                for i, tr in enumerate(trajs):
                    c1 = self.view.cost_to_go(self.field, tuple(to_world(tr[-1, :2])))
                    prog[i] = c0-c1 if np.isfinite(c1) else -10.
        off = np.array([sum(1 for k in range(H) if (c[k] if k < len(c) else 'h') != (own[k] if k < len(own) else 'h'))
                        for c in cands])
        rank = prog
        if EVADE2_TURNCOST > 0 and self.view is not None and self.field is not None:
            rank = prog-EVADE2_TURNCOST*dyn.step_m*np.array([self._turn_back_steps(tr[-1]) for tr in trajs])
        Xf = None
        if learned or sample:
            from eventwm_graph.l3 import reflex
            ests = [h for h in human_estimates(legal.tracks, legal.time) if float(np.hypot(*(h.position-P0))) < EVADE2_RANGE]
            Xf = reflex.features(legal, dyn, level, P0, psi0, cands, kinds, trajs, ests, prog, off)
            if sample:
                # PRIVILEGED labels: the TRUE clearance of every candidate to every person on the tape (training data)
                T_, idx = np.asarray(tape[0], np.float64), int(tape[1])
                rows_ = np.minimum(np.arange(idx+1, idx+H+1), len(T_)-1)
                truep = to_plane(T_[rows_][:, :, [0, 2]]) if T_.shape[1] else np.zeros((H, 0, 2))
                tc = clearance(trajs, truep, 0.)
                key = tape[2] if len(tape) > 2 else ''
                reflex.log_rows(EVADE2_LOG, dict(key=key, scene=key.split('/')[2] if key.count('/') >= 3 else key,
                                                 step=idx, dyn=dyn.name, planner=type(self.planner).__name__,
                                                 kind=self.query.kind, label=self.query.label, d_own=d_own),
                                Xf, (tc < reflex.LABEL_R).astype(np.int8), np.minimum(tc, 9.).astype(np.float32))
        if learned:
            return self._evade2_learned(legal, prim, kin, realized, top, P0, psi0, cands, Xf, prog, off, i_esc, esc)
        if esc is None and d_own >= EVADE2_R:
            return prim
        if esc is not None and d_own >= EVADE2_R+0.1:
            self._esc = None
            _EV2['release'] += 1
            return prim
        _EV2['threat'] += 1
        dmin = clearance(trajs)
        capped = np.minimum(dmin, EVADE2_R+EVADE2_CAP)
        eff = capped.copy()
        if i_esc is not None:
            eff[i_esc] += EVADE2_KEEP
        top_eff = float(eff.max())
        pool = [i for i in range(len(cands)) if eff[i] >= top_eff-0.02]
        best = max(pool, key=lambda i: (rank[i], -off[i]))
        if best == 0 or (i_esc is None and capped[best] < capped[0]+EVADE2_GAIN):
            if esc is not None:
                _EV2['release'] += 1
            self._esc = None
            return prim
        return self._evade2_exec(legal, kin, realized, top, P0, psi0, cands[best], best == i_esc)

    def _ev2_fast_trajs(self, legal, cands, kinds, waitk, own_traj, plan, roll, P0, psi0, H):
        """Rollouts of the reflex candidates for robots without speed state, built from shared prefixes (the semantics
        of ``roll``: holds keep the pose, turns keep the position, an advance whose target is an observed wall core or
        whose front contact point is in a bump disc keeps the pose -- and so does every later advance of the run)."""
        dyn = self.dyn
        out = np.empty((len(cands), H, 3))
        static = np.array([P0[0], P0[1], psi0])
        need = {}
        for c, kd in zip(cands, kinds):
            if kd in (2, 3, 4):
                k = 0 if kd == 2 else next((j for j in range(len(c)) if c[j] != c[0]), len(c))
                psi_k = psi0+(0. if kd == 2 else (1. if c[0] == 'l' else -1.))*k*dyn.turn_rad
                key = round(psi_k, 9)
                need[key] = (psi_k, max(need.get(key, (0, 0))[1], len(c)-k))
        runs = {key: self._ev2_run(legal, P0, psi, n) for key, (psi, n) in need.items() if n > 0}
        plan_traj = None
        for i, (c, kd) in enumerate(zip(cands, kinds)):
            if i == 0:
                out[0] = own_traj
            elif kd == 1:
                out[i] = static
            elif kd in (2, 3, 4):                    # [turn k] then forward n, then hold
                k = 0 if kd == 2 else next((j for j in range(len(c)) if c[j] != c[0]), len(c))
                n = len(c)-k
                sgn = 0. if kd == 2 else (1. if c[0] == 'l' else -1.)
                if k:
                    out[i, :k, 0], out[i, :k, 1] = P0[0], P0[1]
                    out[i, :k, 2] = psi0+sgn*np.arange(1, k+1)*dyn.turn_rad
                psi_k = psi0+sgn*k*dyn.turn_rad
                if n:
                    r = runs[round(psi_k, 9)]
                    out[i, k:k+n, :2] = r[:n]
                    out[i, k:k+n, 2] = psi_k
                out[i, k+n:] = out[i, k+n-1] if k+n > 0 else static
            elif kd == 5:                            # hold k, then the plan from the start
                if plan_traj is None:
                    plan_traj = roll(plan)
                k = waitk[i]
                out[i, :k] = static
                out[i, k:] = plan_traj[:H-k]
            else:
                out[i] = roll(c)
        return out

    def _ev2_run(self, legal, P0, psi, n):
        """Planar positions [n, 2] after each of ``n`` advances along ``psi`` from P0 (see ``_ev2_fast_trajs``)."""
        from .legal import BLOCKED, LegalState
        g = _core_map(legal)
        h = heading_vec(psi)
        Pn = P0[None]+np.arange(1, n+1)[:, None]*self.dyn.step_m*h[None]
        W = to_world(Pn)
        ii = np.floor(W[:, 0]/g.res).astype(np.int64)-g.ox
        jj = np.floor(W[:, 1]/g.res).astype(np.int64)-g.oz
        inside = (ii >= 0) & (ii < g.state.shape[0]) & (jj >= 0) & (jj < g.state.shape[1])
        st = np.zeros(n, np.uint8)
        st[inside] = g.state[ii[inside], jj[inside]]
        blocked = st == BLOCKED
        bumps = getattr(legal, 'bumps', None)
        if bumps and (self.query.meta or {}).get('bumpf'):
            Pb = np.vstack([P0[None], Pn[:-1]])+LegalState.BUMP_FRONT_M*h[None]
            Fb = to_world(Pb)
            B = np.asarray(bumps, np.float64)
            blocked |= (np.hypot(Fb[:, None, 0]-B[None, :, 0], Fb[:, None, 1]-B[None, :, 1]) <=
                        float(LegalState.BUMP_MEMORY)).any(axis=1)
        if blocked.any():
            b = int(np.argmax(blocked))
            Pn[b:] = Pn[b-1] if b > 0 else P0
        return Pn

    def _evade2_exec(self, legal, kin, realized, top, P0, psi0, seq, keep):
        """Execute the first action of the chosen escape ``seq`` and keep the rest (hysteresis)."""
        from .planners import Realized
        _EV2['keep' if keep else 'escape'] += 1
        self._esc = list(seq[1:]) if len(seq) > 1 else None
        a = seq[0]
        if a == 'h':
            out = self.adapter.wait_action(P0, psi0, legal.map)
            self._ev2_hold = True
        else:
            out = {'f': 'forward', 'l': 'left', 'r': 'right'}[a]
        if out != kin:
            self.plan = None
            self.st = None
            self._plan_now = [{'f': 'forward', 'l': 'left', 'r': 'right'}.get(x, 'left') for x in seq]
        if realized:
            return Realized(out, (top if out == 'forward' else 0))
        return out

    def _evade2_learned(self, legal, prim, kin, realized, top, P0, psi0, cands, Xf, prog, off, i_esc, esc):
        """L3_EVADE_RISK: the learned threat probability (l3/reflex.py) replaces the constant-velocity clearance --
        trigger when the committed motion's probability exceeds L3_EVADE_PTRIG (release below half of it), then the
        lowest-probability candidates (within 0.03), most progress, fewest off-plan steps; act if the winner is at
        least L3_EVADE_PGAIN below the committed motion (the running escape gets a 0.03 bonus)."""
        from eventwm_graph.l3 import reflex
        p = reflex.load(EVADE2_RISK).prob(Xf)
        p_own = float(p[0])
        if esc is None and p_own < EVADE2_PTRIG:
            return prim
        if esc is not None and p_own < 0.5*EVADE2_PTRIG:
            self._esc = None
            _EV2['release'] += 1
            return prim
        _EV2['threat'] += 1
        eff = p.copy()
        if i_esc is not None:
            eff[i_esc] -= 0.03
        lo = float(eff.min())
        pool = [i for i in range(len(cands)) if eff[i] <= lo+0.03]
        best = max(pool, key=lambda i: (prog[i], -off[i]))
        if best == 0 or (i_esc is None and p[best] > p_own-EVADE2_PGAIN):
            if esc is not None:
                _EV2['release'] += 1
            self._esc = None
            return prim
        return self._evade2_exec(legal, kin, realized, top, P0, psi0, cands[best], best == i_esc)

    def _fc_predict(self, legal, P0, H):
        """L3_EVADE_FC: planar predictions [H, M, 2] of the tracked people within EVADE2_RANGE at steps 1..H (legal)."""
        from eventwm_graph.l3 import peoplemap as pm
        rows = [r for r in pm.instances(legal) if float(np.hypot(*(r['est'].position-P0))) < EVADE2_RANGE]
        if not rows:
            return np.zeros((H, 0, 2))
        steps = np.arange(1, H+1)
        cols = []
        fidx = [r for r in rows if r.get('hist') is not None]
        for r in rows:
            if r.get('hist') is None:
                e = r['est']
                cols.append(e.position[None]+e.velocity[None]*(steps*STEP_SECONDS)[:, None])
        if fidx:
            fc = pm.load_forecaster()
            if not getattr(pm, '_EV2_THREADS', False):
                pm._torch().set_num_threads(1)
                pm._EV2_THREADS = True
            w, knots, _ = fc.modes(fidx)                               # knots [M, K, NK, 2] world (x, z)
            # the forecast starts at the last detection (age); shift the step grid by the age in steps
            for m, r in enumerate(fidx):
                off = int(round(float(r['age'])/STEP_SECONDS))
                tr = pm.at_steps(knots[m], steps+off)                  # [K, H, 2] world
                if EVADE2_FC == 'modes':
                    for k in np.flatnonzero(w[m] >= min(EVADE2_FCW, float(w[m].max()))):
                        cols.append(to_plane(tr[k]))
                else:
                    cols.append(to_plane((w[m][:, None, None]*tr).sum(0)))
        return np.stack(cols, axis=1)

    def _remembered(self, legal, P0, current_ids):
        """L3_EVADE_MEM: (planar position now, planar velocity) of people detected within EVADE2_MEM seconds whose
        track is no longer in the tracker's list (legal track log; extrapolation capped at EVADE2_MEM_EXT seconds; the
        velocity is zeroed once the cap is reached)."""
        now = float(legal.time)
        seen = set(current_ids)
        out = []
        for tm, trs in reversed(legal.track_log or []):
            if now-float(tm) > EVADE2_MEM:
                break
            for t in trs:
                tid = t.get('track_id')
                if tid in seen or not t.get('visible_now'):
                    continue
                seen.add(tid)
                p = to_plane(np.asarray(t['position'], float)[[0, 2]])
                v = to_plane(np.asarray(t['estimated_velocity'], float)[[0, 2]]) if t.get('velocity_valid') and \
                    t.get('estimated_velocity') is not None else np.zeros(2)
                age = now-float(t.get('last_seen_time', tm))
                if age <= 2.0:
                    continue                                   # still a live track (or about to be one)
                if age >= EVADE2_MEM_EXT:
                    p, v = p+v*EVADE2_MEM_EXT, np.zeros(2)
                else:
                    p = p+v*age
                if float(np.hypot(*(p-P0))) < EVADE2_RANGE:
                    out.append((p, v))
        return out

    def _occluded_ahead(self, legal, stale_s=0.3, half_fov=math.radians(40.)):
        """L3_CORNER_SLOW: is there non-wall space inside the camera wedge near the path ahead that has not been observed
        for ``stale_s`` seconds (occluded), within 0.6-1.4 m ahead and 1.2 m sideways?"""
        from .legal import BLOCKED
        g = legal.map
        P = to_plane([legal.pose[0], legal.pose[2]])
        psi = float(legal.pose[3])
        h, n = heading_vec(psi), heading_vec(psi+math.pi/2)
        now = float(legal.time)
        hits = 0
        for d in (0.6, 1.0, 1.4):
            for l in (-1.2, -0.8, -0.4, 0.4, 0.8, 1.2):
                if abs(math.atan2(l, d)) > half_fov:
                    continue
                x, z = to_world(P+d*h+l*n)
                i, j = g.index(x, z)
                if not g.inside(i, j) or g.state[i, j] == BLOCKED:
                    continue
                ls = g.last_seen[i, j]
                if np.isnan(ls) or ls < now-stale_s:
                    hits += 1
        return hits >= 2

    def _restart_ghosts(self, legal, humans):
        """L3_EVADE_RESTART: (position, last walking velocity) of tracked people paused now (< 0.2 m/s) who walked
        faster than 0.8 m/s within the last 1.5 s (legal track log)."""
        paused = {h.track_id: h for h in humans if float(np.hypot(*h.velocity)) < 0.2}
        if not paused:
            return []
        now = float(legal.time)
        walk = {}
        for tm, trs in reversed(legal.track_log or []):
            if now-float(tm) > 1.5:
                break
            for t in trs:
                tid = t.get('track_id')
                if tid in paused and tid not in walk and t.get('velocity_valid') and t.get('estimated_velocity') is not None:
                    v = to_plane(np.asarray(t['estimated_velocity'], float)[[0, 2]])
                    if float(np.hypot(*v)) > 0.8:
                        walk[tid] = v
        return [(paused[k].position, v) for k, v in walk.items()]

    def _gaze_target(self, legal):
        """L3_HOLD_GAZE: heading toward the nearest tracked person within 4 m who is not visible now (None if none)."""
        P = to_plane([legal.pose[0], legal.pose[2]])
        best = None
        for h in human_estimates(legal.tracks, legal.time):
            if h.visible:
                continue
            d = float(np.hypot(*(h.position-P)))
            if d < 4.0 and (best is None or d < best[0]):
                best = (d, h)
        if best is None:
            return None
        rel = best[1].position-P
        return math.atan2(rel[1], rel[0])

    def _update_unstick(self, legal):
        """v11 meta 'unstick' (= N steps): a goal query whose robot has not translated for N steps while no
        legally tracked human is within 2 m (a static deadlock, not a yield) escalates for the rest of its life to
        the tight legal map and plan following under the one-step hard human check only.  Static contact is not a
        failure in the official benchmark, so the default behaviour must never freeze for static reasons alone
        (DWA on auFeVz9Go4m held in place for 470 steps: every advance was cut by the inflated wall band)."""
        n = int((self.query.meta or {}).get('unstick', 0))
        if not n or self.escalated or self.query.kind != 'goal' or self.query.via or not self.dyn.discrete:
            return
        xz = (float(legal.pose[0]), float(legal.pose[2]))
        if self._last_xz is not None and math.hypot(xz[0]-self._last_xz[0], xz[1]-self._last_xz[1]) < 0.02:
            self._still += 1
            self._still_psi.append(float(legal.pose[3]))
        else:
            self._still = 0
            self._still_psi = [float(legal.pose[3])]
        self._last_xz = xz
        stuck = self._still >= n
        if stuck and (self.query.meta or {}).get('uprog'):
            # v12 'uprog': a genuine deadlock only -- no translation for n steps AND at most 1.5 turn steps of net
            # rotation over them (holding, dithering, pushing into an obstacle), or no translation for 40 steps
            # (a full turn-around takes ~37).  v11 also escalated on ordinary in-place turns (88 of 100 default
            # runs within the first 16 steps -- the spawn turn toward the route), which replaced ORCA / DWA by the
            # same plan follower (identical ORCA and DWA action sequences in 24 of 48 episodes)
            psi = self._still_psi
            net = abs(wrap(psi[-1]-psi[max(0, len(psi)-1-n)])) if len(psi) > 1 else 0.
            stuck = net <= 1.5*self.dyn.turn_rad or self._still >= 40
        if stuck:
            P = to_plane(xz)
            shum = bool((self.query.meta or {}).get('shum'))
            near = any(float(np.hypot(*(h.position-P))) < 2.0 and not
                       (shum and h.valid_velocity and float(np.hypot(*h.velocity)) < STATIC_HUMAN_SPEED)
                       for h in human_estimates(legal.tracks, legal.time))
            if not near:
                self.escalated = True
                self.plan = None
                self.reason = 'unstick'

    def _make_close(self, very=False):
        """Close-pass variant: smaller human-avoidance proxies, rectangle hard check at 0.27 m (v12 ``very``:
        contact-level proxies, 0.20 m)."""
        from .planners import ORCAPlanner, SFMPlanner, DWAPlanner
        from .orca import ORCASolver
        from dataclasses import replace as _replace
        hard = VCLOSE_HARD_CLEARANCE if very else CLOSE_HARD_CLEARANCE
        self.hard_clearance = hard
        if isinstance(self.planner, ORCAPlanner):
            self.planner.solver = ORCASolver(_replace(self.planner.solver.cfg, **(VCLOSE_ORCA if very else CLOSE_ORCA)))
            self.planner.hard_clearance = hard
        elif isinstance(self.planner, SFMPlanner):
            # contact radius sized to the body width (0.95 m disc -> 0.55 m), same hard check
            self.planner.cfg = _replace(self.planner.cfg, **(VCLOSE_SFM if very else CLOSE_SFM))
            self.planner.hard_clearance = hard
        elif isinstance(self.planner, DWAPlanner):
            self.planner.cfg = _replace(self.planner.cfg, **(VCLOSE_DWA if very else CLOSE_DWA))

    def copy(self):
        return copy.deepcopy(self)

    # Layer 3 option (process-wide, default OFF -> released behaviour unchanged; L3_INHERIT=1): a NEW goal query issued
    # while another goal query is running (or after non-goal queries interrupted one: ``suspended``) inherits that goal
    # execution's history toward the same target -- the
    # stillness / 'unstick' streak and its escalation (with the kept static plan, 'ukeep') and the recent goal
    # distances of the stuck-STOP rule -- so that interrupting a query is not a hidden reset (measured on dev:
    # re-issuing a fresh goal query every interval instead of continuing costs 5-8 SR points, timeouts 13% -> 27%).
    # Route caches are re-derived as at any decision boundary.  Applied identically by the branch runtime, the
    # Sandbox imagination, the counterfactual labellers and the official replay.
    INHERIT = __import__('os').environ.get('L3_INHERIT') == '1'

    def _goal_like(self):
        return self.query.kind == 'goal' and not self.query.via

    def inherit(self, prev):
        """Layer 3 option (``INHERIT``): take over the history of the goal query that ``prev`` interrupted or was.
        A goal query interrupted by non-goal queries (scans, waits, rotations) is remembered across them
        (``suspended``) and its history resumes when a goal query is issued again."""
        if prev is None:
            return
        if STOP_GUARD and getattr(prev, '_sg_hist', None):
            self._sg_hist = list(prev._sg_hist)                    # (L3_STOP_GUARD: goal-distance history)
        if ST_COMMIT > 0 and self.query.kind == 'stgoal' and prev.query.kind == 'stgoal' and \
                getattr(prev, '_st_real', False) and prev.st is not None and not prev.st.get('static') and \
                prev.st_i < len(prev.st['prims']) and tuple(map(tuple, prev.query.via or ())) == \
                tuple(map(tuple, self.query.via or ())) and \
                bool((prev.query.meta or {}).get('close')) == bool((self.query.meta or {}).get('close')) and \
                bool((prev.query.meta or {}).get('vclose')) == bool((self.query.meta or {}).get('vclose')):
            self.st, self.st_i, self.st_age = copy.deepcopy(prev.st), prev.st_i, prev.st_age   # (L3_ST_COMMIT)
            self._st_real = True
        if prev._goal_like() and prev.status == 'running':
            src = prev
        else:
            src = getattr(prev, 'suspended', None)
        if not self._goal_like():
            if src is not None:
                self.suspended = src                     # carried across the interlude (never stepped again)
            return
        if src is None:
            return
        self.recent = list(src.recent)
        self._still, self._last_xz, self._still_psi = src._still, src._last_xz, list(src._still_psi)
        mq = self.query.meta or {}
        if src.escalated and mq.get('unstick'):
            self.escalated = True
            self.reason = 'unstick'
            if mq.get('ukeep') and (src.query.meta or {}).get('ukeep') and src.plan is not None:
                self.plan, self.plan_i, self.plan_age = copy.deepcopy(src.plan), src.plan_i, src.plan_age

    def decision_boundary(self):
        """Drop derived caches (route view/field, static plan) at a decision.

        Applied identically in Oracle branches and authoritative replay, so a
        continued query re-derives them from the then-current legal map.
        """
        self.view = None
        self.field = None
        self.field_key = None
        if not (self.escalated and (self.query.meta or {}).get('ukeep')):
            self.plan = None
            self.plan_i = 0
            self.plan_age = 0
        # v12 meta 'ukeep': an escalated (unstick) goal query keeps its static plan across decision boundaries
        # (still re-validated on new evidence every step).  Re-planning at the boundary from an intermediate pose of
        # an in-place turn could fail (start pose inside the inflated band) while the route fallback turned the
        # other way: HMkoS756sz6 / bXu6SSWkJY8 / LEFTm3JecaC rotated back and forth for 400 steps (32-step cycle)
        if not (ST_COMMIT > 0 and getattr(self, '_st_real', False) and self.st is not None and not self.st.get('static')):
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
            uc = (self.query.meta or {}).get('unknown_cost')
            self.view = PlanningView(legal.map, robot_xz, target_xz, **({'unknown_cost': float(uc)} if uc else {}))
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
        if ANTIDITHER and not ANTIDITHER_HOLD:
            self.planner.mode = None
            self._ev2_hold = False
        out = self._step_inner(legal, level)
        if ANTIDITHER and not ANTIDITHER_HOLD and (
                getattr(self.planner, 'mode', None) in ('hold', 'duty', 'jammed') or getattr(self, '_ev2_hold', False)
                or self.query.kind in ('wait', 'wait_clear')):
            self._ad_hist, self._ad_commit = [], None          # (L3_ANTIDITHER_HOLD=0) a hold is not a dither
            return out
        return self._antidither(out) if ANTIDITHER else out

    def _antidither(self, out):
        """L3_ANTIDITHER: break a left / right turn alternation without translation (module docstring above)."""
        cmd = out[0]
        code = ('L' if cmd == 'left' else 'R' if cmd == 'right' else
                'F' if (cmd == 'forward' or (isinstance(cmd, tuple) and cmd and cmd[0] == 'forward')) else 'O')
        hist = getattr(self, '_ad_hist', None)
        if hist is None:
            hist = self._ad_hist = []
        commit = getattr(self, '_ad_commit', None)
        if code in ('L', 'R'):
            if commit is not None and commit[1] > 0:
                d = commit[0]
                self._ad_commit = (d, commit[1]-1)
                out = (('left' if d == 'L' else 'right'),)+tuple(out[1:])
                code = d
            elif len(hist) >= 6 and all(c in 'LR' for c in hist[-6:]) and \
                    all(hist[-6+i] != hist[-5+i] for i in range(5)) and code != hist[-1]:
                d = hist[-1]
                self._ad_commit = (d, ANTIDITHER-1)
                out = (('left' if d == 'L' else 'right'),)+tuple(out[1:])
                code = d
        else:
            self._ad_commit = None
        hist.append(code)
        if len(hist) > 12:
            del hist[0]
        return out

    def _step_inner(self, legal, level=0):
        if self.query.kind == 'prim':
            # v13 primitive query (generator 'primitives'): exactly one native action, no planner, no safety filter
            # -- the Oracle's privileged fallback phase (native witness) expresses its solution with these
            self.steps += 1
            self.status, self.reason = 'done', 'prim'
            return self.query.meta['action'], 0
        self._plan_now = None
        self._update_unstick(legal)
        raw_legal = legal
        if HOLD_SWEEP or HOLD_GAZE:
            self.adapter.real = getattr(raw_legal, 'track_log', None) is not None     # (hold options: real states only)
            if HOLD_GAZE and self.adapter.real:
                self.adapter.gaze = self._gaze_target(raw_legal)
        legal = self._exec_legal(raw_legal)
        if STOP_GUARD and self.query.kind not in ('stop', 'prim') and getattr(raw_legal, 'track_log', None) is not None:
            g = self._stop_guard(raw_legal, level)
            if g is not None:
                return g
        n_recent = len(self.recent)
        prim = self._step_kin(legal, level)
        while UNREACH_FALLBACK and self.status == 'done' and self.reason == 'unreachable' and \
                self.query.kind in ('goal', 'goto', 'stgoal') and getattr(self, '_relax', 0) < 2:
            # L3_UNREACH_FALLBACK: retry this step on a more permissive legal map (see UNREACH_FALLBACK)
            self._relax = getattr(self, '_relax', 0)+1
            self.status, self.reason = 'running', ''
            self.steps -= 1
            del self.recent[n_recent:]
            self.view = self.field = self.field_key = None
            self.plan, self.st = None, None
            legal = self._exec_legal(raw_legal)
            prim = self._step_kin(legal, level)
        if EVADE2 and (self.query.meta or {}).get('evade') and \
                (not EVADE2_REAL or getattr(raw_legal, 'track_log', None) is not None):
            prim = self._evade2(raw_legal, prim, level)
        elif (self.query.meta or {}).get('evade') and self.dyn.discrete and prim in ('forward', 'left', 'right') \
                and self.status == 'running':
            prim = self._evade(raw_legal, prim)
        if CORNER_SLOW > 0 and prim == 'forward' and not self.dyn.stateful and \
                getattr(raw_legal, 'track_log', None) is not None and self._occluded_ahead(raw_legal):
            self.speed_debt += CORNER_SLOW
            if self.speed_debt >= 1.-1e-9:
                self.speed_debt -= 1.
            else:
                P_ = to_plane([raw_legal.pose[0], raw_legal.pose[2]])
                prim = self.adapter.wait_action(P_, float(raw_legal.pose[3]), raw_legal.map)
                self.plan = None
        if prim == 'stop':
            if self.dyn.stateful and int(level) > 0:
                # v10: a robot with momentum declares STOP only at rest -- it brakes at its brake limit first
                # (v9 let the Oracle's goal test stop at any speed level: an instantaneous halt)
                self.status, self.reason = 'running', ''
                l2 = max(0, int(level)-self.dyn.brake)
                return (('forward', l2) if l2 > 0 else 'idle'), l2
            return 'stop', 0
        from .planners import Realized
        if isinstance(prim, Realized):
            return self._arrive_check(legal, (prim[0], prim[1]))   # planner already realised the command
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
        # v11 meta 'blind': an exact creep is not gated by the (inflated) legal wall band -- as for the native robot
        gmap = None if (self.query.kind == 'creep' and (self.query.meta or {}).get('blind')) else legal.map
        return self._arrive_check(legal, self.realizer.realize(prim, int(level), P, float(legal.pose[3]), gmap,
                                                               humans, self._plan_now, cap))

    def _exec_legal(self, raw_legal):
        """The legal view this query plans on: its 'thru' / 'tight' variant (tight 1 once escalated), the standing-human
        discs of an escalated 'shum' query, and the L3_UNREACH_FALLBACK relaxation level."""
        relax = getattr(self, '_relax', 0)
        if (self.query.meta or {}).get('thru') or relax >= 2:
            legal = _ThruLegal(raw_legal)
        else:
            cells = int((self.query.meta or {}).get('tight', 0)) or (1 if self.escalated else 0)
            legal = tight_legal(raw_legal, max(cells, 2) if relax == 1 else cells)
        if self.escalated and (self.query.meta or {}).get('shum'):
            sh = static_humans(raw_legal)
            if sh:
                legal = _StaticHumanLegal(legal, sh)
        return legal

    def _arrive_check(self, legal, out):
        """v10 (meta 'arrive_exact'): a goto without a final heading ends on the step whose own motion reaches
        the final via-point (v9 noticed arrival at the start of the next step and spent it on a hold, which for
        the native robot is an in-place turn)."""
        q = self.query
        if not (q.kind == 'goto' and q.meta.get('arrive_exact') and q.heading is None and q.via and
                self.status == 'running' and self.via_index == len(q.via)-1):
            return out
        cmd = out[0]
        d = (self.dyn.step_m if cmd == 'forward' else
             self.dyn.levels[int(cmd[1])] if isinstance(cmd, tuple) and cmd[0] == 'forward' else 0.)
        if d > 0.:
            P1 = to_plane([legal.pose[0], legal.pose[2]])+d*heading_vec(float(legal.pose[3]))
            if float(np.hypot(*(P1-to_plane(tuple(map(float, q.via[-1])))))) < ARRIVE_M:
                self.status, self.reason = 'done', 'arrived'
        return out

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
            bumped = BUMP_CREEP and self._bump_front(legal, ctx.P, ctx.psi)
            if (hard is None or hard(P1, psi1)) and not bumped:
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
                    if q.meta.get('exact_end') and face is not None and \
                            not (self.dyn.stateful and self.dyn.turn_needs_stop and int(dyn_state) > 0):
                        # v10: this primitive reaches the target heading -> the decision ends with it
                        # (no trailing hold step; a native hold is itself an in-place turn).  A moving
                        # robot that must stop before turning brakes on this step instead (motion.py).
                        after = pose[3]+(self.dyn.turn_rad if direction == 'left' else -self.dyn.turn_rad)
                        if abs(wrap(face-after)) <= self.dyn.turn_rad*.55:
                            self.status, self.reason = 'done', 'heading_reached'
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
            if stopped and (d < STOP_M or (d < self._stuck_stop_m() and len(self.recent) >= 4 and
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
        if self.escalated and self.dyn.discrete:
            # unstick: follow the (tight-map) static plan whenever its next pose keeps the one-step human clearance;
            # without a body-feasible plan (narrow dead ends: the in-place sweep meets the inflated walls) use the
            # official motion model -- an in-place rotation is always executable (the base is navmesh-constrained,
            # body-wall overlap is not a benchmark failure): turn toward the legal route, then advance while the next
            # base position is not an observed wall core
            from .planners import one_step_hard_ok
            hard = one_step_hard_ok(ctx.humans, self.hard_clearance)
            cand = prims[0] if prims else None
            if cand is None and route_dir is not None:
                err = wrap(route_dir-float(legal.pose[3]))
                if abs(err) > 0.6*self.dyn.turn_rad:
                    cand = 'left' if err > 0 else 'right'
                else:
                    from .legal import BLOCKED
                    P1 = Pl+self.dyn.step_m*heading_vec(float(legal.pose[3]))
                    x1, z1 = to_world(P1)
                    if tight_legal(legal, 2).map.state_at(x1, z1) != BLOCKED and \
                            not self._bump_blocked(legal, Pl, float(legal.pose[3])):
                        cand = 'forward'
            if cand is not None:
                P1, psi1 = self.adapter.predicted_pose(ctx.P, ctx.psi, cand)
                if hard is None or hard(P1, psi1):
                    if prims and cand == prims[0]:
                        self.plan_i += 1
                    return cand
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
        if self.dyn.stateful:
            return self._step_st_inertia(legal, dyn_state)
        from . import stplan
        from .planners import one_step_hard_ok
        q = self.query
        pose = legal.pose
        target = tuple(map(float, q.via[-1])) if q.via else self._goal_xz(legal)
        d = math.hypot(target[0]-pose[0], target[1]-pose[2])
        if not q.via:
            self.recent.append(d)
            if d < STOP_M or (d < self._stuck_stop_m() and len(self.recent) >= 4 and self.recent[-4]-d < 0.02):
                self.status, self.reason = 'done', 'stop_at_goal'
                return 'stop'
        ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
        if q.via and d < ARRIVE_M:
            self.status, self.reason = 'done', 'arrived'
            return self.planner.act(ctx, self.adapter)
        real = getattr(legal, 'track_log', None) is not None
        self._st_real = real
        need = self.st is None or self.st_i >= len(self.st['prims']) or self.st_age >= ST_REPLAN
        on_plan = False
        if self.st is not None and self.st_i > 0 and self.st_i <= len(self.st['poses']):
            ex = self.st['poses'][self.st_i-1]
            on_plan = math.hypot(ex[0]-pose[0], ex[1]-pose[2]) <= 0.06 and abs(wrap(ex[2]-pose[3])) <= 0.03
            if not need:
                need = not on_plan
        old_st = self.st if (ST_COMMIT > 0 and real and need and on_plan and self.st is not None and
                             not self.st.get('static') and self.st_i < len(self.st['prims'])) else None
        if need:
            route = self._ensure_field(legal, target)
            if route is None:
                self.status, self.reason = 'done', 'unreachable'
                return self.planner.act(ctx, self.adapter)
            P = to_plane([pose[0], pose[2]])
            humans = [h for h in ctx.humans if float(np.hypot(*(h.position-P))) <= ST_HUMAN_RANGE]
            close = bool((q.meta or {}).get('close'))
            vclose = bool((q.meta or {}).get('vclose'))
            kw = dict(horizon=ST_HORIZON, goal_tol=STOP_M*0.8 if not q.via else ARRIVE_M*0.8,
                      margin=ST_MARGIN_VCLOSE if vclose else ST_MARGIN_CLOSE if close else ST_MARGIN,
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
            if old_st is not None and not self.st.get('static') and \
                    self._st_keep_old(old_st, self.st, humans, kw['margin'], kw['growth']):
                self.st = old_st                                  # (L3_ST_COMMIT) the running plan stands
                self.st_age = 0
            else:
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

    def _turn_back_steps(self, end, delta=0.15):
        """L3_EVADE_TURNCOST: turn steps from the end heading onto the legal route's descent direction at the end
        position (0 when the cost-to-go field is not defined around it)."""
        x, z = to_world(end[:2])
        c = [self.view.cost_to_go(self.field, (x+a, z+b)) for a, b in ((delta, 0.), (-delta, 0.), (0., delta), (0., -delta))]
        if not all(np.isfinite(c)):
            return 0.
        gx, gz = -(c[0]-c[1]), -(c[2]-c[3])                  # world descent direction
        if abs(gx)+abs(gz) < 1e-9:
            return 0.
        route = math.atan2(-gz, gx)                          # planar heading of (gx, -gz)
        return abs(wrap(float(end[2])-route))/self.dyn.turn_rad

    def _st_keep_old(self, old, new, humans, margin, growth):
        """L3_ST_COMMIT: keep the running space-time plan ``old`` (from index ``self.st_i``) instead of ``new``?"""
        from .planners import DWAPlanner
        rest = old['poses'][self.st_i:]
        if not rest:
            return False
        for k, (x, z, psi) in enumerate(rest):
            t = (k+1)*STEP_SECONDS
            P = to_plane([x, z])
            for h in humans:
                if float(DWAPlanner._rect_dist(P, psi, h.at(t))) < margin+growth*(k+1):
                    return False

        def J(prims, poses):
            if not poses:
                return float('inf')
            c = self.view.cost_to_go(self.field, (poses[-1][0], poses[-1][1]))
            return len(prims)+(c/self.dyn.step_m if np.isfinite(c) else 1e3)
        return J(new['prims'], new['poses']) >= J(old['prims'][self.st_i:], rest)-ST_COMMIT

    def _step_st_inertia(self, legal, dyn_state):
        """stgoal for robots with momentum (v10): follow a legal momentum-aware space-time plan
        (stplan.plan_inertia: speed level in the state, turns at rest, arrival at rest).  Each command is
        executed only if it and its braking tail keep the body off observed walls and the rectangle clearance
        from the legal constant-velocity predictions; otherwise the DWA planner holds (it brakes)."""
        from . import stplan
        from .planners import Realized
        q = self.query
        pose = legal.pose
        L = int(dyn_state)
        target = tuple(map(float, q.via[-1])) if q.via else self._goal_xz(legal)
        d = math.hypot(target[0]-pose[0], target[1]-pose[2])
        if not q.via:
            self.recent.append(d)
            if L == 0 and (d < STOP_M or (d < self._stuck_stop_m() and len(self.recent) >= 4 and
                                          self.recent[-4]-d < 0.02)):
                self.status, self.reason = 'done', 'stop_at_goal'
                return 'stop'
        ctx = self._context(legal, dyn_state, np.zeros(2), hold=True)
        if q.via and d < ARRIVE_M:
            self.status, self.reason = 'done', 'arrived'
            return self.planner.act(ctx, self.adapter)
        need = self.st is None or self.st_i >= len(self.st['cmds']) or self.st_age >= ST_REPLAN
        if not need and self.st_i > 0:
            ex = self.st['poses'][self.st_i-1]
            need = math.hypot(ex[0]-pose[0], ex[1]-pose[2]) > 0.06 or abs(wrap(ex[2]-pose[3])) > 0.03 or \
                self.st['levels'][self.st_i-1] != L
        if need:
            route = self._ensure_field(legal, target)
            if route is None:
                self.status, self.reason = 'done', 'unreachable'
                return self.planner.act(ctx, self.adapter)
            P = to_plane([pose[0], pose[2]])
            humans = [h for h in ctx.humans if float(np.hypot(*(h.position-P))) <= ST_HUMAN_RANGE]
            self.st = stplan.plan_inertia(self.view, self.field, pose, L, target, self.dyn, humans, STEP_SECONDS,
                                          horizon=ST_HORIZON, goal_tol=STOP_M*0.8 if not q.via else ARRIVE_M*0.8,
                                          margin=ST_MARGIN, growth=ST_GROWTH, start_budget=body_cost(legal.map, P, pose[3]),
                                          stop_at_goal=not q.via)
            if not self.st['found'] or not self.st['cmds']:
                self.st = None
            self.st_i = 0
            self.st_age = 0
            self.lattice_calls += 1
        self.st_age += 1
        if self.st is not None and self.st_i < len(self.st['cmds']):
            cmd, lvl = self.st['cmds'][self.st_i], self.st['levels'][self.st_i]
            if self._momentum_cmd_safe(legal, ctx, cmd, lvl):
                self.st_i += 1
                return Realized(cmd, lvl)
            self.st = None
        return self.planner.act(ctx, self.adapter)

    def _momentum_cmd_safe(self, legal, ctx, cmd, lvl):
        """The command and the braking tail that follows it: body admissible on the legal map (no worse than
        now) and rectangle >= hard clearance from every legal constant-velocity human prediction."""
        from .control import body_ok
        from .planners import DWAPlanner
        dyn = self.dyn
        P, psi = np.array(ctx.P, dtype=float), float(ctx.psi)
        if isinstance(cmd, tuple):
            P = P+dyn.levels[int(lvl)]*heading_vec(psi)
        elif cmd in ('left', 'right'):
            psi += dyn.turn_rad if cmd == 'left' else -dyn.turn_rad
        traj = [(P.copy(), psi)]
        lv = int(lvl)
        while lv > 0:
            lv = max(0, lv-dyn.brake)
            P = P+dyn.levels[lv]*heading_vec(psi)
            traj.append((P.copy(), psi))
        budget0 = body_cost(legal.map, ctx.P, ctx.psi)
        for k, (Pk, psik) in enumerate(traj):
            if not body_ok(legal.map, Pk, psik, True, budget0):
                return False
            for h in ctx.humans or []:
                if float(DWAPlanner._rect_dist(Pk, psik, h.at((k+1)*STEP_SECONDS))) < self.hard_clearance:
                    return False
        return True

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
        keep = self.escalated and (self.query.meta or {}).get('ukeep')
        ok = (self.plan is not None and self.plan_i < len(self.plan['prims']) and (self.plan_age < 24 or keep)
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
