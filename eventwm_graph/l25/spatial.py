"""Connected tiles of arrived certified cells; no hidden map or human truth."""
from dataclasses import dataclass
from collections import deque
import heapq
from types import MappingProxyType
import numpy as np
from .schema import readonly

SPATIAL_FEATURES = ('x', 'z', 'height', 'width', 'depth', 'known_free', 'known_occupied', 'unknown',
                    'frontier_fraction', 'goal_dx', 'goal_dz', 'goal_l2', 'ego_dx', 'ego_dz',
                    'track_expected_count', 'track_age_mean', 'track_variance_mean', 'observed_track_mass',
                    'optical_visibility_supported', 'ego_yaw_sin', 'ego_yaw_cos')


@dataclass(frozen=True)
class SpatialGraph:
    ids: tuple
    positions: np.ndarray
    base_features: np.ndarray
    edge_index: np.ndarray
    edge_length: np.ndarray
    cell_to_node: object
    resolution: float
    unknown_index: int

    def locate(self, xz):
        cell = tuple(np.floor(np.asarray(xz)/self.resolution).astype(int))
        return self.cell_to_node.get(cell, self.unknown_index)

    def goal_distances(self, goal):
        """Observed-cell connectivity only, never full-map geodesic distance."""
        target = self.locate(np.asarray(goal)[[0, 2]])
        distances = np.full(len(self.ids), np.inf)
        if target == self.unknown_index or not self.base_features[target, 5]:
            return distances
        adjacency = [[] for _ in self.ids]
        for (a, b), d in zip(self.edge_index.T, self.edge_length):
            adjacency[a].append((int(b), float(d)))
        distances[target] = 0.
        queue = [(0., target)]
        while queue:
            d, a = heapq.heappop(queue)
            if d != distances[a]:
                continue
            for b, length in adjacency[a]:
                nd = d+length
                if nd < distances[b]:
                    distances[b] = nd
                    heapq.heappush(queue, (nd, b))
        return distances


def build_spatial(graph, tile_size=4):
    cells = graph.materialize()
    remaining = set(cells)
    groups, lookup = [], {}
    # Cells merge only within a tile, same support state and <=2cm height
    # deviation from the seed. Native support heights have sub-mm noise.
    # Disconnected free patches in the same tile remain distinct nodes.
    for cell in sorted(cells):
        if cell not in remaining:
            continue
        state, height = cells[cell]
        tile = (cell[0]//tile_size, cell[1]//tile_size)
        region, queue = [], deque([cell])
        remaining.remove(cell)
        while queue:
            a = queue.popleft()
            region.append(a)
            lookup[a] = len(groups)
            for b in ((a[0]+1, a[1]), (a[0]-1, a[1]), (a[0], a[1]+1), (a[0], a[1]-1)):
                if (b in remaining and (b[0]//tile_size, b[1]//tile_size) == tile
                        and cells[b][0] == state and abs(cells[b][1]-height) <= .02):
                    remaining.remove(b)
                    queue.append(b)
        groups.append((region, state, height))
    n = len(groups)
    features = np.zeros((n+1, 9), dtype=np.float32)
    ids = []
    r = graph.resolution
    edges = set()
    for i, (region, state, height) in enumerate(groups):
        a = np.asarray(region)
        # Bounding-box geometry describes the region extent, not interior clearance.
        features[i, :5] = [*(a.mean(0)+.5)*r, np.mean([cells[k][1] for k in region]), *((a.max(0)-a.min(0)+1)*r)]
        features[i, 5:8] = [state == 1, state == 2, state == 0]
        frontier = 0
        for cell in region:
            for other in ((cell[0]+1, cell[1]), (cell[0]-1, cell[1]), (cell[0], cell[1]+1), (cell[0], cell[1]-1)):
                j = lookup.get(other)
                if j is None:
                    frontier += 1
                elif j != i and state == 1 and cells[other][0] == 1 and abs(cells[cell][1]-cells[other][1]) <= .22:
                    edges.add((i, j))
        features[i, 8] = frontier/(4*len(region))
        ids.append('spatial:%d:%d:%d' % (*min(region), state))
    features[n, 7] = 1.
    ids.append('spatial:external-unknown')
    edge_index = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2).T
    pos = features[:, :2]
    length = np.linalg.norm(pos[edge_index[0]]-pos[edge_index[1]], axis=-1)
    return SpatialGraph(tuple(ids), readonly(pos), readonly(features), readonly(edge_index), readonly(length), MappingProxyType(lookup), r, n)


def context_features(spatial, ego, goal, tracks):
    x = np.zeros((len(spatial.ids), len(SPATIAL_FEATURES)), dtype=np.float32)
    x[:, :9] = spatial.base_features
    x[:, 9:11] = np.asarray(goal)[[0, 2]]-spatial.positions
    x[:, 11] = np.linalg.norm(x[:, 9:11], axis=1)
    x[:, 12:14] = np.asarray(ego)[[0, 2]]-spatial.positions
    x[:, 19] = np.sin(ego[3])
    x[:, 20] = np.cos(ego[3])
    # A mean-to-region assignment is a compact belief summary, not occupancy.
    for track in tracks:
        i = spatial.locate(track['mean'][:2])
        mass = track['presence_probability']
        x[i, 14] += mass
        x[i, 15] += mass*track['age']
        x[i, 16] += mass*np.trace(track['covariance'][:2, :2])
        x[i, 17] += mass*(track['age'] < 1e-8)
    np.divide(x[:, 15], x[:, 14], out=x[:, 15], where=x[:, 14] > 0)
    np.divide(x[:, 16], x[:, 14], out=x[:, 16], where=x[:, 14] > 0)
    # External UNKNOWN has no invented centroid/goal geometry.
    x[spatial.unknown_index, :5] = 0
    x[spatial.unknown_index, 9:14] = 0
    return readonly(x)
