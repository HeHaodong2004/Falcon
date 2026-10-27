"""Legal space-time lattice planner for the native (turn-in-place) robot.

Search over (x, z, psi, t) with the robot's exact nominal primitives -- forward one step, turn
one step, and a two-step hold (left then right, which is how the native robot waits) -- on the
robot's *legal* map (body-axis samples off observed BLOCKED cells, UNKNOWN traversed at a small
penalty, as in ``lattice``) against *legal* constant-velocity predictions of tracked humans: the
robot rectangle must keep ``margin + growth * t`` from every predicted human centre.  The cost is
time; the heuristic is the static legal-map cost-to-go to the target.  The first popped node that
reaches the target, or the horizon, ends the search (weighted A*), so the plan either arrives or
ends where the remaining static distance is smallest after ``horizon`` steps.  No privileged input
is used: the planner sees exactly what the query generator sees.
"""
import math
import numpy as np
from numba import njit, types
from numba.typed import Dict

from .lattice import _body_count, _crop, heading_bins
from .control import BODY_X, BODY_HALF_WIDTH, FRONT_MARGIN

PRIMS = ('forward', 'left', 'right', 'hold')


@njit(cache=True)
def _rect_dist(x, z, psi, hx, hz, x_lo, x_hi, half_w):
    dx = hx-x
    dz = hz-z
    c = math.cos(psi)
    s = math.sin(psi)
    lx = dx*c-dz*s
    ly = -dx*s-dz*c
    ex = max(max(x_lo-lx, 0.), lx-x_hi)
    ey = max(abs(ly)-half_w, 0.)
    return math.sqrt(ex*ex+ey*ey)


@njit(cache=True)
def _humans_ok(x, z, psi, t, H, margin0, growth, x_lo, x_hi, half_w):
    if H.shape[1] == 0:
        return True
    tt = min(t, H.shape[0]-1)
    m = margin0+growth*t
    for n in range(H.shape[1]):
        if _rect_dist(x, z, psi, H[tt, n, 0], H[tt, n, 1], x_lo, x_hi, half_w) < m:
            return False
    return True


@njit(cache=True)
def _st_search(blocked, unknown, field, kx0, kz0, res, sx, sz, spsi, gx, gz, goal_tol, step_m, turn_rad,
               field_scale, body_lo, body_hi, nose_hi, half_w, H, margin0, growth, horizon, max_expand,
               weight, start_budget, unknown_pen, n_bins):
    n0, n1 = blocked.shape
    cap = max_expand*4+8
    X = np.empty(cap); Z = np.empty(cap); PSI = np.empty(cap); G = np.empty(cap)
    T = np.empty(cap, dtype=np.int64)
    PAR = np.empty(cap, dtype=np.int64); PRIM = np.empty(cap, dtype=np.int64)
    heap_f = np.empty(cap); heap_k = np.empty(cap, dtype=np.int64)
    closed = Dict.empty(key_type=types.int64, value_type=types.boolean)
    X[0] = sx; Z[0] = sz; PSI[0] = spsi; G[0] = 0.; T[0] = 0; PAR[0] = -1; PRIM[0] = -1
    nn = 1
    i0 = int(math.floor(sx/res))-kx0; j0 = int(math.floor(sz/res))-kz0
    h0 = field[min(max(i0, 0), n0-1), min(max(j0, 0), n1-1)]*field_scale
    heap_f[0] = weight*h0; heap_k[0] = 0; size = 1
    best = 0; best_h = h0
    expanded = 0
    while size > 0 and expanded < max_expand:
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
        x = X[k]; z = Z[k]; psi = PSI[k]; t = T[k]
        ci = int(math.floor(x/res))-kx0; cj = int(math.floor(z/res))-kz0
        if ci < 0 or cj < 0 or ci >= n0 or cj >= n1:
            continue
        hb = int(math.floor((psi % (2*math.pi))/(2*math.pi)*n_bins)) % n_bins
        key = ((t*n0+ci)*n1+cj)*n_bins+hb
        if key in closed:
            continue
        closed[key] = True
        expanded += 1
        hval = field[ci, cj]*field_scale
        if hval < best_h:
            best_h = hval; best = k
        dx = x-gx; dz = z-gz
        if dx*dx+dz*dz <= goal_tol*goal_tol:
            return k, nn, X, Z, PSI, PAR, PRIM, T, expanded, 1
        if t >= horizon:
            return k, nn, X, Z, PSI, PAR, PRIM, T, expanded, 0
        for prim in range(4):
            nt = t+1
            npsi = psi
            nx = x; nz = z
            if prim == 0:
                nx = x+step_m*math.cos(psi); nz = z-step_m*math.sin(psi)
                nb, nu = _body_count(blocked, unknown, kx0, kz0, res, nx, nz, psi, body_lo, nose_hi, 0.1)
            elif prim == 1 or prim == 2:
                npsi = psi+(turn_rad if prim == 1 else -turn_rad)
                nb, nu = _body_count(blocked, unknown, kx0, kz0, res, x, z, npsi, body_lo, body_hi, 0.1)
            else:
                nt = t+2
                nb, nu = _body_count(blocked, unknown, kx0, kz0, res, x, z, psi+turn_rad, body_lo, body_hi, 0.1)
                if not _humans_ok(x, z, psi+turn_rad, t+1, H, margin0, growth, body_lo, body_hi, half_w):
                    continue
            if nb > start_budget:
                continue
            ni = int(math.floor(nx/res))-kx0; nj = int(math.floor(nz/res))-kz0
            if ni < 0 or nj < 0 or ni >= n0 or nj >= n1 or blocked[ni, nj]:
                continue
            if field[ni, nj] > 1e17:
                continue
            if not _humans_ok(nx, nz, npsi, nt, H, margin0, growth, body_lo, body_hi, half_w):
                continue
            nhb = int(math.floor((npsi % (2*math.pi))/(2*math.pi)*n_bins)) % n_bins
            nkey = ((nt*n0+ni)*n1+nj)*n_bins+nhb
            if nkey in closed:
                continue
            g = G[k]+(nt-t)+unknown_pen*nu+5.*nb
            if nn >= cap:
                break
            X[nn] = nx; Z[nn] = nz; PSI[nn] = npsi; G[nn] = g; T[nn] = nt; PAR[nn] = k; PRIM[nn] = prim
            pos = size; size += 1
            heap_f[pos] = g+weight*field[ni, nj]*field_scale; heap_k[pos] = nn
            while pos > 0:
                parent = (pos-1)//2
                if heap_f[parent] <= heap_f[pos]:
                    break
                heap_f[parent], heap_f[pos] = heap_f[pos], heap_f[parent]
                heap_k[parent], heap_k[pos] = heap_k[pos], heap_k[parent]
                pos = parent
            nn += 1
    return best, nn, X, Z, PSI, PAR, PRIM, T, expanded, -1


def human_predictions(humans, horizon, step_s):
    """[horizon+1, N, 2] world (x, z) constant-velocity positions of legal human estimates."""
    if not humans:
        return np.zeros((horizon+1, 0, 2))
    ts = np.arange(horizon+1)*step_s
    out = np.empty((horizon+1, len(humans), 2))
    for n, h in enumerate(humans):
        P = h.position[None, :]+h.velocity[None, :]*ts[:, None]      # planar (x, -z)
        out[:, n, 0] = P[:, 0]
        out[:, n, 1] = -P[:, 1]
    return out


def plan(view, field, start_pose, target_xz, step_m, turn_rad, humans, step_s, horizon=40, goal_tol=0.12,
         margin=0.35, growth=0.004, max_expand=60000, weight=1.5, start_budget=0, unknown_pen=0.15,
         half_window=48):
    """Return dict(prims, poses, times, reached, horizon_hit, expanded); poses are (x, z, psi)."""
    sx, sz, spsi = float(start_pose[0]), float(start_pose[2]), float(start_pose[3])
    field_scale = view.res/step_m
    blocked, unknown, _, fld, kx0, kz0 = _crop(view, field, sx, sz, half_window)
    H = np.ascontiguousarray(human_predictions(humans, horizon+2, step_s))
    node, nn, X, Z, PSI, PAR, PRIM, T, expanded, status = _st_search(
        blocked, unknown, fld, kx0, kz0, view.res, sx, sz, spsi, float(target_xz[0]), float(target_xz[1]),
        float(goal_tol), float(step_m), float(turn_rad), float(field_scale), float(BODY_X[0]), float(BODY_X[1]),
        float(BODY_X[1]+FRONT_MARGIN), float(BODY_HALF_WIDTH), H, float(margin), float(growth), int(horizon),
        int(max_expand), float(weight), int(start_budget), float(unknown_pen), heading_bins(turn_rad))
    prims, poses, times = [], [], []
    k = node
    while k > 0:
        p = int(PRIM[k])
        if p == 3:
            prims += ['right', 'left']           # reversed below: hold = left then right
            poses += [(X[k], Z[k], PSI[k]), (X[k], Z[k], PSI[k]+turn_rad)]
            times += [int(T[k]), int(T[k])-1]
        else:
            prims.append(PRIMS[p]); poses.append((X[k], Z[k], PSI[k])); times.append(int(T[k]))
        k = PAR[k]
    prims.reverse(); poses.reverse(); times.reverse()
    return dict(prims=prims, poses=poses, times=times, reached=status == 1, horizon_hit=status == 0,
                expanded=int(expanded))
