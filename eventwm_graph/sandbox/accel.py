"""Exact per-decision memoisation for imagined executions on the frozen legal map.

Inside one sandbox decision every candidate's executor reads the SAME frozen legal map, so several of its derived
quantities are pure functions of inputs that repeat across candidates and steps:

* the tight / thru map variants of ``executor.tight_legal`` and ``executor._ThruLegal`` (an erosion / relabelling
  of the whole map, rebuilt by the executor on every step);
* ``gridplan.PlanningView`` windows (cropped arrays, distance transform, costs) for the same (map, robot position,
  target, options) and their cost-to-go fields (Dijkstra) for the same targets;
* body-aware ``lattice.plan`` results for the same (view, field, start pose, target, options);
* ``PlanningView.route`` polylines for the same (view, field, start position) (in-place turns and holds repeat them).

Two pure per-step helpers are replaced by exact numba equivalents (same floating-point operations in the same order:
``control.body_cells_state``, the body-axis map lookup behind every body check, and ``control.blocked_points_near``);
numpy's ``linspace``, ``arctan2`` and ``hypot`` agree bit for bit with the compiled forms on this platform (checked in
``tests/sandbox``).

``FrozenMapCache`` memoises exactly these, keyed on the identical inputs (object identity of the frozen map / view
plus exact float poses), and is active only inside ``active(cache)`` for legal states that are ``ImaginedLegal``
(or wrappers of one): real executions in the same process call the original functions.  Results are shared
read-only (the querygen code never mutates views or fields) and lattice plans are returned as fresh copies (the
executor annotates its plan dict).  ``tests/sandbox`` checks bit-identical rollouts with and without the cache.
"""
from collections import Counter
import contextlib
import math
import threading

import numpy as np
from numba import njit

from eventwm_graph.querygen import executor as _ex
from eventwm_graph.querygen import lattice as _lat
from eventwm_graph.querygen import gridplan as _gp
from eventwm_graph.querygen import control as _ctl

_STATE = threading.local()
_ORIG = dict(tight_legal=_ex.tight_legal, TightLegal=_ex._TightLegal, ThruLegal=_ex._ThruLegal,
             PlanningView=_ex.PlanningView, lattice_plan=_lat.plan, body_cells_state=_ctl.body_cells_state,
             blocked_points_near=_ctl.blocked_points_near, ex_blocked_points_near=_ex.blocked_points_near)


class FrozenMapCache:
    def __init__(self):
        self.maps = {}
        self.views = {}
        self.view_ids = {}
        self.fields = {}
        self.plans = {}
        self.routes = {}
        self.stats = Counter()

    def summary(self):
        return dict(self.stats)


def _cache():
    return getattr(_STATE, 'cache', None)


@contextlib.contextmanager
def active(cache):
    """Memoise inside this block (one decision; the frozen map must not change)."""
    prev = _cache()
    _STATE.cache = cache
    try:
        yield cache
    finally:
        _STATE.cache = prev


def _base(legal):
    from .imagine import ImaginedLegal
    seen = 0
    while not isinstance(legal, ImaginedLegal):
        inner = legal.__dict__.get('_legal') if hasattr(legal, '__dict__') else None
        if inner is None or seen > 8:
            return None
        legal, seen = inner, seen+1
    return legal


# ------------------------------------------------------------------------------------------------- derived maps
def tight_legal(legal, cells):
    c = _cache()
    if c is None or not cells or _base(legal) is None:
        return _ORIG['tight_legal'](legal, cells)
    key = ('tight', id(legal.map), int(cells))
    hit = c.maps.get(key)
    if hit is not None and hit[0] is legal.map:
        w = object.__new__(_ORIG['TightLegal'])
        w._legal = legal; w.map = hit[1]
        c.stats['tight_hit'] += 1
        return w
    w = _ORIG['TightLegal'](legal, cells)
    c.maps[key] = (legal.map, w.map)
    c.stats['tight_miss'] += 1
    return w


def thru_legal(legal):
    c = _cache()
    b = _base(legal) if c is not None else None
    if b is None:
        return _ORIG['ThruLegal'](legal)
    from eventwm_graph.querygen.legal import LegalState
    key = ('thru', id(legal.map), tuple(getattr(legal, 'bumps', ()) or ()), float(LegalState.BUMP_MEMORY))
    hit = c.maps.get(key)
    if hit is not None and hit[0] is legal.map:
        w = object.__new__(_ORIG['ThruLegal'])
        w._legal = legal; w.map = hit[1]
        c.stats['thru_hit'] += 1
        return w
    w = _ORIG['ThruLegal'](legal)
    c.maps[key] = (legal.map, w.map)
    c.stats['thru_miss'] += 1
    return w


# ------------------------------------------------------------------------------------------------- views / fields
class _View(_gp.PlanningView):
    """A memoised PlanningView (immutable after construction) whose fields and routes are memoised."""

    def route(self, field, start_xz, max_len=4000, smooth=True):
        c = _cache()
        if c is None or id(self) not in c.view_ids:
            return _gp.PlanningView.route(self, field, start_xz, max_len, smooth)
        key = (id(self), id(field), float(start_xz[0]), float(start_xz[1]), int(max_len), bool(smooth))
        hit = c.routes.get(key)
        if hit is not None and hit[0] is field:
            c.stats['route_hit'] += 1
            r = hit[1]
        else:
            r = _gp.PlanningView.route(self, field, start_xz, max_len, smooth)
            c.routes[key] = (field, r)
            c.stats['route_miss'] += 1
        return None if r is None else r.copy()

    def field(self, targets_xz, radius_m=0.):
        c = _cache()
        key = (id(self), tuple((float(x), float(z)) for x, z in targets_xz), float(radius_m), float(_gp.BLOCKED_TARGET_RELAX_M))
        if c is not None and id(self) in c.view_ids:
            hit = c.fields.get(key)
            if hit is not None:
                c.stats['field_hit'] += 1
                return hit
            f = _gp.PlanningView.field(self, targets_xz, radius_m)
            c.fields[key] = f
            c.stats['field_miss'] += 1
            return f
        return _gp.PlanningView.field(self, targets_xz, radius_m)


def planning_view(gmap, robot_xz, goal_xz=None, **kw):
    c = _cache()
    if c is None or not getattr(gmap, '_sandbox_frozen', False):
        return _ORIG['PlanningView'](gmap, robot_xz, goal_xz, **kw)
    key = (id(gmap), (float(robot_xz[0]), float(robot_xz[1])),
           None if goal_xz is None else (float(goal_xz[0]), float(goal_xz[1])), tuple(sorted(kw.items())))
    hit = c.views.get(key)
    if hit is not None and hit[0] is gmap:
        c.stats['view_hit'] += 1
        return hit[1]
    v = _View(gmap, robot_xz, goal_xz, **kw)
    c.views[key] = (gmap, v)
    c.view_ids[id(v)] = v
    c.stats['view_miss'] += 1
    return v


# ------------------------------------------------------------------------------------------------- lattice plans
def lattice_plan(view, field, start_pose, goal_xz, step_m, turn_rad, **kw):
    c = _cache()
    if c is None or id(view) not in c.view_ids:
        return _ORIG['lattice_plan'](view, field, start_pose, goal_xz, step_m, turn_rad, **kw)
    key = (id(view), id(field), float(start_pose[0]), float(start_pose[2]), float(start_pose[3]),
           float(goal_xz[0]), float(goal_xz[1]), float(step_m), float(turn_rad), tuple(sorted(kw.items())))
    hit = c.plans.get(key)
    if hit is None or hit[0] is not field:
        res = _ORIG['lattice_plan'](view, field, start_pose, goal_xz, step_m, turn_rad, **kw)
        c.plans[key] = (field, res)
        c.stats['plan_miss'] += 1
    else:
        res = hit[1]
        c.stats['plan_hit'] += 1
    return dict(res, prims=list(res['prims']), poses=list(res['poses']))


# ------------------------------------------------------------------------------------------------- exact helpers
@njit(cache=True)
def _body_states(state, ox, oz, res, px, py, h0, h1, x0, x1, n):
    """control.body_cells_state: np.linspace(x0, x1, n) axis samples (y_k = k*step + x0, y_last = x1), planar point
    P + x_k * h, world (X, -Y), GridMap.state_at."""
    out = np.empty(n, np.uint8)
    n0, n1 = state.shape
    step = (x1-x0)/(n-1) if n > 1 else 0.
    for k in range(n):
        xk = x0 if n == 1 else (x1 if k == n-1 else k*step+x0)
        X = px+xk*h0
        Y = py+xk*h1
        i = math.floor(X/res)-ox
        j = math.floor((-Y)/res)-oz
        out[k] = state[i, j] if (0 <= i < n0 and 0 <= j < n1) else 0
    return out


def body_cells_state(gmap, P, psi, spacing=0.1, x_range=_ctl.BODY_X):
    if _cache() is None:
        return _ORIG['body_cells_state'](gmap, P, psi, spacing, x_range)
    n = int(math.ceil((x_range[1]-x_range[0])/spacing))+1
    return _body_states(gmap.state, int(gmap.ox), int(gmap.oz), float(gmap.res), float(P[0]), float(P[1]),
                        math.cos(psi), math.sin(psi), float(x_range[0]), float(x_range[1]), n)


@njit(cache=True)
def _blocked_near(state, ox, oz, res, px, py, radius, sectors):
    """control.blocked_points_near: nearest BLOCKED cell centre per angular sector (first minimum in np.nonzero
    order), sectors ascending."""
    x = px; z = -py
    i0 = math.floor(x/res)-ox; j0 = math.floor(z/res)-oz
    r = int(math.ceil(radius/res))
    a0 = max(0, i0-r); a1 = min(state.shape[0], i0+r+1)
    b0 = max(0, j0-r); b1 = min(state.shape[1], j0+r+1)
    best = np.full(sectors, np.inf)
    q = np.zeros((sectors, 2))
    two_pi = 2*math.pi
    if a1 <= a0 or b1 <= b0:
        return np.zeros((0, 2))
    for i in range(a0, a1):
        for j in range(b0, b1):
            if state[i, j] != 2:
                continue
            cx = (float(i+ox)+.5)*res
            cz = (float(j+oz)+.5)*res
            qx = cx; qy = -cz
            dx = qx-px; dy = qy-py
            d = math.hypot(dx, dy)
            if not d <= radius:
                continue
            sec = int((math.atan2(dy, dx)+math.pi)/two_pi*sectors) % sectors
            if d < best[sec]:
                best[sec] = d; q[sec, 0] = qx; q[sec, 1] = qy
    m = 0
    for k in range(sectors):
        if best[k] < np.inf:
            m += 1
    out = np.zeros((m, 2))
    m = 0
    for k in range(sectors):
        if best[k] < np.inf:
            out[m, 0] = q[k, 0]; out[m, 1] = q[k, 1]; m += 1
    return out


def blocked_points_near(gmap, P, radius=1.6, sectors=24):
    if _cache() is None:
        return _ORIG['blocked_points_near'](gmap, P, radius, sectors)
    return _blocked_near(gmap.state, int(gmap.ox), int(gmap.oz), float(gmap.res), float(P[0]), float(P[1]),
                         float(radius), int(sectors))


def mark_frozen(gmap):
    """Flag the decision's legal map (and only it) as frozen for the view cache."""
    gmap._sandbox_frozen = True
    return gmap


def install():
    """Route the executor's calls through the memoising versions (idempotent; originals used when inactive)."""
    _ex.tight_legal = tight_legal
    _ex._ThruLegal = thru_legal
    _ex.PlanningView = planning_view
    _lat.plan = lattice_plan
    _ctl.body_cells_state = body_cells_state
    _ctl.blocked_points_near = blocked_points_near
    _ex.blocked_points_near = blocked_points_near


def uninstall():
    _ex.tight_legal = _ORIG['tight_legal']
    _ex._ThruLegal = _ORIG['ThruLegal']
    _ex.PlanningView = _ORIG['PlanningView']
    _lat.plan = _ORIG['lattice_plan']
    _ctl.body_cells_state = _ORIG['body_cells_state']
    _ctl.blocked_points_near = _ORIG['blocked_points_near']
    _ex.blocked_points_near = _ORIG['ex_blocked_points_near']
