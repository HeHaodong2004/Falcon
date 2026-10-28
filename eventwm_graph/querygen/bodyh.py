"""Privileged body-aware static cost-to-go for the Oracle (selection only; never used online).

The geodesic (and the space-time heuristic) treat Spot as a point; Spot's 1.23 x 0.49 m body can only
step straight ahead or turn in place, so near walls the true cost depends on heading: a spawn facing
away from the route with a wall behind needs a long arc, a passage may admit only one approach
heading.  This table is the native-primitive cost-to-go over (cell, heading) on the simulator's
navmesh, which is built for a 0.25 m agent radius ~ the body half width (0.243 m): body-axis samples
on navigable cells means the body fits.  Real solutions often put the body slightly onto that
dilated band (the collision geometry is thinner), so off-navmesh samples are priced
(``band_cost`` per sample, like the lattice planner) instead of forbidden.  Backward Dijkstra from
the goal region; humans are ignored (the space-time heuristic covers them).
"""
import math
import numpy as np
from numba import njit

from .control import BODY_X


@njit(cache=True)
def _band_counts(nav, n_bins, cell, offsets):
    nz, nx = nav.shape
    nb = np.zeros((nz, nx, n_bins), dtype=np.int8)
    for iz in range(nz):
        for ix in range(nx):
            x = (ix+0.5)*cell
            z = (iz+0.5)*cell
            for h in range(n_bins):
                psi = 2*math.pi*h/n_bins
                c, s = math.cos(psi), math.sin(psi)
                k = 0
                for o in offsets:
                    jx = int(math.floor((x+o*c)/cell))
                    jz = int(math.floor((z-o*s)/cell))
                    if jx < 0 or jz < 0 or jx >= nx or jz >= nz or not nav[jz, jx]:
                        k += 1
                nb[iz, ix, h] = k
    return nb


@njit(cache=True)
def _dijkstra(nb, goal, cell, step_m, n_bins, band_cost, max_samples):
    nz, nx, _ = nb.shape
    INF = np.float32(1e9)
    C = np.full((nz, nx, n_bins), INF, dtype=np.float32)
    cap = nz*nx*n_bins*2+16
    hk = np.empty(cap, dtype=np.float32)
    hi = np.empty(cap, dtype=np.int64)
    size = 0
    for iz in range(nz):
        for ix in range(nx):
            if goal[iz, ix]:
                for h in range(n_bins):
                    if nb[iz, ix, h] <= max_samples:
                        C[iz, ix, h] = 0.
                        # push
                        pos = size; size += 1
                        hk[pos] = 0.; hi[pos] = (iz*nx+ix)*n_bins+h
                        while pos > 0:
                            par = (pos-1)//2
                            if hk[par] <= hk[pos]:
                                break
                            hk[par], hk[pos] = hk[pos], hk[par]
                            hi[par], hi[pos] = hi[pos], hi[par]
                            pos = par
    while size > 0:
        d = hk[0]; k = hi[0]
        size -= 1
        hk[0] = hk[size]; hi[0] = hi[size]
        pos = 0
        while True:
            l = 2*pos+1; r = l+1; m = pos
            if l < size and hk[l] < hk[m]:
                m = l
            if r < size and hk[r] < hk[m]:
                m = r
            if m == pos:
                break
            hk[m], hk[pos] = hk[pos], hk[m]
            hi[m], hi[pos] = hi[pos], hi[m]
            pos = m
        h = k % n_bins; c = k//n_bins; ix = c % nx; iz = c//nx
        if d > C[iz, ix, h]:
            continue
        enter = 1.+band_cost*nb[iz, ix, h]          # cost of the primitive that ends in this state
        # predecessors: forward (same heading, one step behind), in-place turns (adjacent heading bins)
        psi = 2*math.pi*h/n_bins
        px = (ix+0.5)*cell-step_m*math.cos(psi)
        pz = (iz+0.5)*cell+step_m*math.sin(psi)
        cand_x = np.empty(3, dtype=np.int64); cand_z = np.empty(3, dtype=np.int64); cand_h = np.empty(3, dtype=np.int64)
        cand_x[0] = int(math.floor(px/cell)); cand_z[0] = int(math.floor(pz/cell)); cand_h[0] = h
        cand_x[1] = ix; cand_z[1] = iz; cand_h[1] = (h-1) % n_bins
        cand_x[2] = ix; cand_z[2] = iz; cand_h[2] = (h+1) % n_bins
        for q in range(3):
            jx = cand_x[q]; jz = cand_z[q]; jh = cand_h[q]
            if jx < 0 or jz < 0 or jx >= nx or jz >= nz:
                continue
            if nb[jz, jx, jh] > max_samples:
                continue
            nd = d+enter
            if nd < C[jz, jx, jh]:
                C[jz, jx, jh] = nd
                if size >= cap:
                    continue
                pos = size; size += 1
                hk[pos] = nd; hi[pos] = (jz*nx+jx)*n_bins+jh
                while pos > 0:
                    par = (pos-1)//2
                    if hk[par] <= hk[pos]:
                        break
                    hk[par], hk[pos] = hk[pos], hk[par]
                    hi[par], hi[pos] = hi[pos], hi[par]
                    pos = par
    return C


class BodyCostToGo:
    """C[z, x, heading bin] = native-primitive steps (+ band price) to the goal region, humans ignored."""

    def __init__(self, rt, cell=0.1, goal_m=0.2, band_cost=1.0, max_samples=6, spacing=0.1):
        import time
        t0 = time.perf_counter()
        task = rt.task
        pf = task.sim.pathfinder
        y = float(task.robot.base_pos[1])
        nav = np.asarray(pf.get_topdown_view(cell, y), dtype=bool)          # [z, x]
        lb = pf.get_bounds()[0]
        self.x0, self.z0, self.cell = float(lb[0]), float(lb[2]), float(cell)
        dyn = rt.dyn
        self.n_bins = int(round(2*math.pi/dyn.turn_rad))
        offsets = np.arange(BODY_X[0], BODY_X[1]+1e-9, spacing)
        self.nb = _band_counts(nav, self.n_bins, self.cell, offsets)
        nz, nx = nav.shape
        zc = self.z0+(np.arange(nz)+.5)*cell
        xc = self.x0+(np.arange(nx)+.5)*cell
        g = task.goal
        goal = np.hypot(xc[None, :]-g[0], zc[:, None]-g[2]) <= goal_m
        self.C = _dijkstra(self.nb, goal, self.cell, float(dyn.step_m), self.n_bins, float(band_cost), int(max_samples))
        self.seconds = time.perf_counter()-t0
        self.shape = self.C.shape

    def value(self, pose):
        """Steps-to-goal at a world pose (x, y, z, psi); None outside the table / unreachable."""
        ix = int(math.floor((float(pose[0])-self.x0)/self.cell))
        iz = int(math.floor((float(pose[2])-self.z0)/self.cell))
        if ix < 0 or iz < 0 or iz >= self.C.shape[0] or ix >= self.C.shape[1]:
            return None
        h = int(round((float(pose[3]) % (2*math.pi))/(2*math.pi)*self.n_bins)) % self.n_bins
        v = float(self.C[iz, ix, h])
        return v if v < 1e8 else None

    def heading_cost(self, pose):
        """Extra steps of the current heading over the best heading at this cell (body-aware turn cost:
        which way to turn, and how far, given the body and the walls); None if unknown."""
        ix = int(math.floor((float(pose[0])-self.x0)/self.cell))
        iz = int(math.floor((float(pose[2])-self.z0)/self.cell))
        if ix < 0 or iz < 0 or iz >= self.C.shape[0] or ix >= self.C.shape[1]:
            return None
        row = self.C[iz, ix]
        m = float(row.min())
        h = int(round((float(pose[3]) % (2*math.pi))/(2*math.pi)*self.n_bins)) % self.n_bins
        v = float(row[h])
        if m >= 1e8 or v >= 1e8:
            return None
        return v-m
