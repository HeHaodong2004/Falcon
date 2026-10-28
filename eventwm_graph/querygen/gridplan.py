"""Legal-map planning substrate: costs, Dijkstra fields, routes (numba).

All functions consume only the legal ``GridMap`` arrays.  UNKNOWN cells are
traversable *optimistically* at a higher cost (frontier exploration); they are
never treated as verified free.  BLOCKED cells are impassable; diagonal moves
may not cut a BLOCKED corner.
"""
import math
import heapq
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt

from .legal import UNKNOWN, FREE, BLOCKED

INF = 1e18
# v10 (GeneratorConfig.goal_relax_m sets it per process; 0 = v9): a field target inside observed BLOCKED
# cells -- typically a goal on the navmesh edge next to furniture, covered by the legal map's 0.25 m inflated
# band once seen -- seeds the field from the passable cells within this radius (initial cost = their distance
# to the target) instead of leaving every cell unreachable (LEFTm3JecaC: every goal query 'unreachable' after
# the first turn; 13 of 68 DEV/unseen goals lie within 0.1 m of the navmesh edge).  Default = the v10
# generator default, so replays that build no generator agree with the search.
BLOCKED_TARGET_RELAX_M = 0.5


@njit(cache=True)
def _dijkstra(cost, sources_i, sources_j, sources_d, blocked):
    n0, n1 = cost.shape
    dist = np.full((n0, n1), INF)
    # binary heap over flat indices
    heap_d = np.empty(n0*n1*4+16)
    heap_k = np.empty(n0*n1*4+16, dtype=np.int64)
    size = 0
    for s in range(len(sources_i)):
        i, j = sources_i[s], sources_j[s]
        if 0 <= i < n0 and 0 <= j < n1 and not blocked[i, j] and sources_d[s] < dist[i, j]:
            dist[i, j] = sources_d[s]
            # push
            pos = size; size += 1
            heap_d[pos] = sources_d[s]; heap_k[pos] = i*n1+j
            while pos > 0:
                parent = (pos-1)//2
                if heap_d[parent] <= heap_d[pos]:
                    break
                heap_d[parent], heap_d[pos] = heap_d[pos], heap_d[parent]
                heap_k[parent], heap_k[pos] = heap_k[pos], heap_k[parent]
                pos = parent
    di = np.array([-1, 1, 0, 0, -1, -1, 1, 1])
    dj = np.array([0, 0, -1, 1, -1, 1, -1, 1])
    dl = np.array([1., 1., 1., 1., math.sqrt(2.), math.sqrt(2.), math.sqrt(2.), math.sqrt(2.)])
    while size > 0:
        d = heap_d[0]; k = heap_k[0]
        size -= 1
        heap_d[0] = heap_d[size]; heap_k[0] = heap_k[size]
        pos = 0
        while True:
            l = 2*pos+1; r = l+1; m = pos
            if l < size and heap_d[l] < heap_d[m]:
                m = l
            if r < size and heap_d[r] < heap_d[m]:
                m = r
            if m == pos:
                break
            heap_d[m], heap_d[pos] = heap_d[pos], heap_d[m]
            heap_k[m], heap_k[pos] = heap_k[pos], heap_k[m]
            pos = m
        i = k//n1; j = k % n1
        if d > dist[i, j]:
            continue
        for e in range(8):
            a = i+di[e]; b = j+dj[e]
            if a < 0 or b < 0 or a >= n0 or b >= n1 or blocked[a, b]:
                continue
            if e >= 4 and (blocked[i, b] or blocked[a, j]):
                continue
            nd = d+dl[e]*0.5*(cost[i, j]+cost[a, b])
            if nd < dist[a, b]:
                dist[a, b] = nd
                pos = size; size += 1
                heap_d[pos] = nd; heap_k[pos] = a*n1+b
                while pos > 0:
                    parent = (pos-1)//2
                    if heap_d[parent] <= heap_d[pos]:
                        break
                    heap_d[parent], heap_d[pos] = heap_d[pos], heap_d[parent]
                    heap_k[parent], heap_k[pos] = heap_k[pos], heap_k[parent]
                    pos = parent
    return dist


@njit(cache=True)
def _descend(dist, blocked, i, j, max_len):
    n0, n1 = dist.shape
    out_i = np.empty(max_len, dtype=np.int64); out_j = np.empty(max_len, dtype=np.int64)
    out_i[0] = i; out_j[0] = j; n = 1
    di = np.array([-1, 1, 0, 0, -1, -1, 1, 1]); dj = np.array([0, 0, -1, 1, -1, 1, -1, 1])
    while n < max_len:
        best = dist[i, j]; bi = -1; bj = -1
        for e in range(8):
            a = i+di[e]; b = j+dj[e]
            if a < 0 or b < 0 or a >= n0 or b >= n1 or blocked[a, b]:
                continue
            if e >= 4 and (blocked[i, b] or blocked[a, j]):
                continue
            if dist[a, b] < best:
                best = dist[a, b]; bi = a; bj = b
        if bi < 0:
            break
        i = bi; j = bj
        out_i[n] = i; out_j[n] = j; n += 1
        if dist[i, j] <= 0.:
            break
    return out_i[:n], out_j[:n]


@njit(cache=True)
def _line_clear(passable, i0, j0, i1, j1):
    """Dense samples (1/4 cell) of the center-to-center segment stay passable."""
    di = i1-i0; dj = j1-j0
    steps = int(max(abs(di), abs(dj))*4)+1
    for s in range(steps+1):
        t = s/steps
        a = int(math.floor(i0+di*t+.5)); b = int(math.floor(j0+dj*t+.5))
        if not passable[a, b]:
            return False
    return True


class PlanningView:
    """Cropped planning arrays for one legal map revision."""

    def __init__(self, gmap, robot_xz, goal_xz=None, margin_m=4., unknown_cost=2.5,
                 clearance_pref_m=0.65, clearance_weight=6.0, max_cells=420):
        res = gmap.res
        known = np.argwhere(gmap.state != UNKNOWN)
        pts = [gmap.index(*robot_xz)]
        if goal_xz is not None:
            pts.append(gmap.index(*goal_xz))
        pts = np.asarray(pts)
        lo = pts.min(axis=0); hi = pts.max(axis=0)
        if len(known):
            lo = np.minimum(lo, known.min(axis=0)); hi = np.maximum(hi, known.max(axis=0))
        m = int(math.ceil(margin_m/res))
        lo = lo-m; hi = hi+m
        # Bound the window around the robot (very distant goals are clipped:
        # their direction still drives the boundary cost).
        r = np.asarray(gmap.index(*robot_xz))
        lo = np.maximum(lo, r-max_cells); hi = np.minimum(hi, r+max_cells)
        self.i0, self.j0 = int(lo[0]), int(lo[1])
        # Absolute cell keys of local index (0, 0): independent of later map growth.
        self.kx0, self.kz0 = self.i0+gmap.ox, self.j0+gmap.oz
        self.res = res
        shape = (int(hi[0]-lo[0]+1), int(hi[1]-lo[1]+1))
        st = np.zeros(shape, dtype=np.uint8)
        # copy overlap with the map array
        a0 = max(self.i0, 0); b0 = max(self.j0, 0)
        a1 = min(self.i0+shape[0], gmap.state.shape[0]); b1 = min(self.j0+shape[1], gmap.state.shape[1])
        if a1 > a0 and b1 > b0:
            st[a0-self.i0:a1-self.i0, b0-self.j0:b1-self.j0] = gmap.state[a0:a1, b0:b1]
        self.state = st
        self.blocked = st == BLOCKED
        self.free = st == FREE
        self.unknown = st == UNKNOWN
        # Clearance (m) from cell centers to the nearest observed BLOCKED cell.
        if self.blocked.any():
            self.clearance = distance_transform_edt(~self.blocked)*res
        else:
            self.clearance = np.full(shape, 99.)
        pen = np.clip(clearance_pref_m-self.clearance, 0., None)/clearance_pref_m
        cost = 1.+clearance_weight*pen*pen
        cost = np.where(self.unknown, cost*unknown_cost, cost)
        soft = getattr(gmap, 'soft', None)
        if soft is not None:
            # v15 candidate 'thru' maps: seen-but-unproven obstacle cells (UNKNOWN in the state) cost SOFT_COST
            from .executor import SOFT_COST
            sc = np.zeros(shape, dtype=bool)
            if a1 > a0 and b1 > b0:
                sc[a0-self.i0:a1-self.i0, b0-self.j0:b1-self.j0] = soft[a0:a1, b0:b1]
            cost = np.where(sc, (1.+clearance_weight*pen*pen)*SOFT_COST, cost)
        self.cost = np.ascontiguousarray(cost)
        self.unknown_cost = unknown_cost
        self.smooth_clearance = clearance_pref_m

    def to_local(self, x, z):
        return math.floor(x/self.res)-self.kx0, math.floor(z/self.res)-self.kz0

    def to_world(self, i, j):
        return (i+self.kx0+.5)*self.res, (j+self.kz0+.5)*self.res

    def inside(self, i, j):
        return 0 <= i < self.state.shape[0] and 0 <= j < self.state.shape[1]

    def field(self, targets_xz, radius_m=0.):
        """Cost-to-go field (cell units) from world targets (multi-source)."""
        si, sj, sd = [], [], []
        for x, z in targets_xz:
            i, j = self.to_local(x, z)
            i = min(max(i, 0), self.state.shape[0]-1); j = min(max(j, 0), self.state.shape[1]-1)
            if self.blocked[i, j] and BLOCKED_TARGET_RELAX_M > 0:
                r = BLOCKED_TARGET_RELAX_M/self.res
                n = int(math.ceil(r))
                found = False
                for a in range(max(i-n, 0), min(i+n+1, self.state.shape[0])):
                    for b in range(max(j-n, 0), min(j+n+1, self.state.shape[1])):
                        d = math.hypot(a-i, b-j)
                        if d <= r and not self.blocked[a, b]:
                            si.append(a); sj.append(b); sd.append(d); found = True
                if found:
                    continue
            si.append(i); sj.append(j); sd.append(0.)
        return _dijkstra(self.cost, np.asarray(si, dtype=np.int64), np.asarray(sj, dtype=np.int64),
                         np.asarray(sd, dtype=np.float64), self.blocked)

    def route(self, field, start_xz, max_len=4000, smooth=True):
        """World-frame polyline from start down the field (None if unreachable)."""
        i, j = self.to_local(*start_xz)
        if not self.inside(i, j):
            return None
        if self.blocked[i, j]:
            # robot center may sit in an inflated BLOCKED cell; step to the
            # cheapest passable neighbour within 2 cells
            best = None
            for a in range(i-2, i+3):
                for b in range(j-2, j+3):
                    if self.inside(a, b) and not self.blocked[a, b] and field[a, b] < INF/2:
                        if best is None or field[a, b] < field[best]:
                            best = (a, b)
            if best is None:
                return None
            i, j = best
        if field[i, j] >= INF/2:
            return None
        ci, cj = _descend(field, self.blocked, i, j, max_len)
        cells = np.stack([ci, cj], axis=1)
        if smooth and len(cells) > 2:
            # Shortcuts may not approach observed walls closer than the
            # preferred body clearance; narrow passages keep their vertices.
            passable = (~self.blocked) & (self.clearance >= self.smooth_clearance)
            passable[cells[:, 0], cells[:, 1]] = True
            keep = [0]; k = 0
            while k < len(cells)-1:
                m = min(len(cells)-1, k+60)
                while m > k+1 and not _line_clear(passable, cells[k, 0], cells[k, 1], cells[m, 0], cells[m, 1]):
                    m -= 1
                keep.append(m); k = m
            cells = cells[keep]
        pts = [self.to_world(int(a), int(b)) for a, b in cells]
        pts[0] = (float(start_xz[0]), float(start_xz[1]))
        return np.asarray(pts, dtype=float)

    def cost_to_go(self, field, xz):
        i, j = self.to_local(*xz)
        if not self.inside(i, j):
            return INF
        return float(field[i, j])*self.res


def path_length(route):
    if route is None or len(route) < 2:
        return 0.
    return float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())


def point_along(route, distance):
    """Point at arclength ``distance`` along a polyline (clamped)."""
    if route is None or len(route) == 0:
        return None
    if len(route) == 1:
        return route[0].copy()
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    acc = 0.
    for k, L in enumerate(seg):
        if acc+L >= distance:
            t = (distance-acc)/max(L, 1e-9)
            return route[k]+(route[k+1]-route[k])*t
        acc += L
    return route[-1].copy()
