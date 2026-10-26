"""Body-aware hybrid-A* over the robot's motion primitives on the legal map.

Spot's all-gait body (1.23 x 0.49 m) is far larger than the 0.25 m navmesh
radius that constrains only its centre, and the native robot can only turn in
place or step straight ahead.  Static feasibility therefore depends on
heading.  This planner searches continuous poses (x, z, psi) with the robot's
exact nominal primitives (forward ``step_m``; in-place turn ``+-turn_rad``;
for unicycle models: arcs), keyed by (cell, heading bin) for duplicate
detection.  Every pose must keep its body-axis samples off observed BLOCKED
cells (walls dilated by 0.25 m ~ the 0.243 m half width); forward poses also
keep a nose margin.  UNKNOWN cells are traversed optimistically at a small
penalty.  The admissible heuristic is the 2-D cost-to-go field.
"""
import math
import numpy as np
from numba import njit

from .control import BODY_X, FRONT_MARGIN

N_BINS = 72
SAMPLE_SPACING = 0.1


@njit(cache=True)
def _body_count(blocked, unknown, kx0, kz0, res, x, z, psi, x_lo, x_hi, spacing):
    c = math.cos(psi); s = math.sin(psi)
    n = int(math.ceil((x_hi-x_lo)/spacing))+1
    nb = 0; nu = 0
    n0, n1 = blocked.shape
    for k in range(n):
        o = x_lo+(x_hi-x_lo)*k/(n-1)
        px = x+o*c; pz = z-o*s
        i = int(math.floor(px/res))-kx0; j = int(math.floor(pz/res))-kz0
        if i < 0 or j < 0 or i >= n0 or j >= n1:
            nu += 1
            continue
        if blocked[i, j]:
            nb += 1
        elif unknown[i, j]:
            nu += 1
    return nb, nu


@njit(cache=True)
def _search(blocked, unknown, cost, field, kx0, kz0, res,
            sx, sz, spsi, gx, gz, goal_tol, goal_field, step_m, turn_rad,
            max_expand, unknown_pen, start_budget, field_scale, body_lo, body_hi, nose_hi, n_bins):
    n0, n1 = blocked.shape
    cap = max_expand*3+8
    X = np.empty(cap); Z = np.empty(cap); PSI = np.empty(cap); G = np.empty(cap)
    PAR = np.empty(cap, dtype=np.int64); PRIM = np.empty(cap, dtype=np.int64)
    closed = np.zeros((n0, n1, n_bins), dtype=np.bool_)
    heap_f = np.empty(cap); heap_k = np.empty(cap, dtype=np.int64)
    size = 0; nn = 0
    X[0] = sx; Z[0] = sz; PSI[0] = spsi; G[0] = 0.; PAR[0] = -1; PRIM[0] = -1; nn = 1
    i0 = int(math.floor(sx/res))-kx0; j0 = int(math.floor(sz/res))-kz0
    h0 = field[min(max(i0, 0), n0-1), min(max(j0, 0), n1-1)]*field_scale
    heap_f[0] = h0; heap_k[0] = 0; size = 1
    best_node = 0; best_h = h0
    expanded = 0
    while size > 0 and expanded < max_expand:
        # pop
        k = heap_k[0]
        size -= 1
        heap_f[0] = heap_f[size]; heap_k[0] = heap_k[size]
        pos = 0
        while True:
            l = 2*pos+1; r = l+1; m = pos
            if l < size and heap_f[l] < heap_f[m]:
                m = l
            if r < size and heap_f[r] < heap_f[m]:
                m = r
            if m == pos:
                break
            heap_f[m], heap_f[pos] = heap_f[pos], heap_f[m]
            heap_k[m], heap_k[pos] = heap_k[pos], heap_k[m]
            pos = m
        x = X[k]; z = Z[k]; psi = PSI[k]
        ci = int(math.floor(x/res))-kx0; cj = int(math.floor(z/res))-kz0
        if ci < 0 or cj < 0 or ci >= n0 or cj >= n1:
            continue
        hb = int(math.floor((psi % (2*math.pi))/(2*math.pi)*n_bins)) % n_bins
        if closed[ci, cj, hb]:
            continue
        closed[ci, cj, hb] = True
        expanded += 1
        hval = field[ci, cj]*field_scale
        if hval < best_h:
            best_h = hval; best_node = k
        dx = x-gx; dz = z-gz
        if (goal_tol > 0. and dx*dx+dz*dz <= goal_tol*goal_tol) or (goal_field >= 0. and field[ci, cj] <= goal_field):
            return k, nn, X, Z, PSI, PAR, PRIM, expanded, True
        for prim in range(3):
            if prim == 0:
                nx = x+step_m*math.cos(psi); nz = z-step_m*math.sin(psi); npsi = psi
                nb, nu = _body_count(blocked, unknown, kx0, kz0, res, nx, nz, npsi, body_lo, nose_hi, 0.1)
            else:
                nx = x; nz = z; npsi = psi+(turn_rad if prim == 1 else -turn_rad)
                nb, nu = _body_count(blocked, unknown, kx0, kz0, res, nx, nz, npsi, body_lo, body_hi, 0.1)
            if nb > start_budget:
                continue
            ni = int(math.floor(nx/res))-kx0; nj = int(math.floor(nz/res))-kz0
            if ni < 0 or nj < 0 or ni >= n0 or nj >= n1:
                continue
            if blocked[ni, nj]:
                continue
            nhb = int(math.floor((npsi % (2*math.pi))/(2*math.pi)*n_bins)) % n_bins
            if closed[ni, nj, nhb]:
                continue
            step_cost = cost[ni, nj] if prim == 0 else 1.
            g = G[k]+step_cost+unknown_pen*nu+5.*nb
            if field[ni, nj] > 1e17:
                continue
            f = g+field[ni, nj]*field_scale
            if nn >= cap:
                break
            X[nn] = nx; Z[nn] = nz; PSI[nn] = npsi; G[nn] = g; PAR[nn] = k; PRIM[nn] = prim
            # push
            pos = size; size += 1
            heap_f[pos] = f; heap_k[pos] = nn
            while pos > 0:
                parent = (pos-1)//2
                if heap_f[parent] <= heap_f[pos]:
                    break
                heap_f[parent], heap_f[pos] = heap_f[pos], heap_f[parent]
                heap_k[parent], heap_k[pos] = heap_k[pos], heap_k[parent]
                pos = parent
            nn += 1
    return best_node, nn, X, Z, PSI, PAR, PRIM, expanded, False


PRIM_NAMES = ('forward', 'left', 'right')


def _crop(view, field, x, z, half):
    i, j = view.to_local(x, z)
    a0 = max(0, i-half); b0 = max(0, j-half)
    a1 = min(view.state.shape[0], i+half+1); b1 = min(view.state.shape[1], j+half+1)
    return (np.ascontiguousarray(view.blocked[a0:a1, b0:b1]), np.ascontiguousarray(view.unknown[a0:a1, b0:b1]),
            np.ascontiguousarray(view.cost[a0:a1, b0:b1]), np.ascontiguousarray(field[a0:a1, b0:b1]),
            view.kx0+a0, view.kz0+b0)


def plan(view, field, start_pose, goal_xz, step_m, turn_rad, goal_tol=0.3, goal_field=-1.,
         max_expand=40000, unknown_pen=0.15, start_budget=0, half_window=48, nose_margin=None):
    """Return dict(prims, poses, reached, expanded) in world (x, z, psi)."""
    sx, sz, spsi = float(start_pose[0]), float(start_pose[2]), float(start_pose[3])
    field_scale = view.res/step_m      # cell-cost units -> forward steps
    blocked, unknown, cost, fld, kx0, kz0 = _crop(view, field, sx, sz, half_window)
    node, nn, X, Z, PSI, PAR, PRIM, expanded, reached = _search(
        blocked, unknown, cost, fld, kx0, kz0, view.res,
        sx, sz, spsi, float(goal_xz[0]), float(goal_xz[1]), float(goal_tol), float(goal_field),
        float(step_m), float(turn_rad), int(max_expand), float(unknown_pen), int(start_budget),
        float(field_scale), float(BODY_X[0]), float(BODY_X[1]),
        float(BODY_X[1]+(FRONT_MARGIN if nose_margin is None else nose_margin)), N_BINS)
    prims, poses = [], []
    k = node
    while k > 0:
        prims.append(PRIM_NAMES[PRIM[k]]); poses.append((X[k], Z[k], PSI[k]))
        k = PAR[k]
    prims.reverse(); poses.reverse()
    return dict(prims=prims, poses=poses, reached=bool(reached), expanded=int(expanded))
