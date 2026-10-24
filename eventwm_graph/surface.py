"""A 3D navigation-surface complex, independent of local graph resolution."""
from collections import defaultdict
import heapq
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .native import path_length


def vertex_key(p):
    return tuple(np.round(p, 5))


def edge_key(a, b, island):
    return (int(island), *sorted((vertex_key(a), vertex_key(b))))


def refine_triangles(triangles, islands, max_edge, limit):
    """Conforming shared-edge bisection; no raster and no projection welding."""
    faces, incidence, heap = {}, defaultdict(set), []
    serial = itertools.count()

    def add(tri, island, source):
        i = next(serial)
        faces[i] = (tri, island, source)
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            key = edge_key(a, b, island)
            incidence[key].add(i)
            length = float(np.linalg.norm(a - b))
            if length > max_edge:
                heapq.heappush(heap, (-length, key))

    for i, (tri, island) in enumerate(zip(triangles, islands)):
        add(tri, island, i)
    while heap:
        _, key = heapq.heappop(heap)
        ids = sorted(incidence.get(key, ()))
        if not ids:
            continue
        for i in ids:
            tri, island, source = faces.pop(i)
            for a, b in zip(tri, np.roll(tri, -1, axis=0)):
                incidence[edge_key(a, b, island)].discard(i)
            j = next(j for j in range(3) if edge_key(tri[j], tri[(j+1) % 3], island) == key)
            a, b, c = tri[j], tri[(j+1) % 3], tri[(j+2) % 3]
            mid = (a + b) / 2
            add(np.array([a, mid, c]), island, source)
            add(np.array([mid, b, c]), island, source)
        if len(faces) > limit:
            raise ValueError(f"Surface exceeds {limit} triangles; increase surface_max_edge or budget")
    rows = sorted(faces.values(), key=lambda r: (int(r[1]), *vertex_key(r[0].mean(0)),
                                               tuple(sorted(vertex_key(v) for v in r[0]))))
    return (np.asarray([r[0] for r in rows]).reshape(-1, 3, 3),
            np.asarray([r[1] for r in rows], dtype=int),
            np.asarray([r[2] for r in rows], dtype=int))


def sparse_graph(n, links):
    rr, cc, vv = [], [], []
    for a, b, weight in links:
        rr.extend([a, b]); cc.extend([b, a]); vv.extend([max(weight, 1e-7)] * 2)
    return csr_matrix((vv, (rr, cc)), shape=(n, n))


class Surface:
    def __init__(self, navigation, config, arrays=None):
        self.nav, self.config = navigation, config
        if arrays is not None:
            self.triangles = arrays["triangles"]
            self.islands = arrays["islands"]
            self.source_ids = arrays["source_ids"]
            self.centers = arrays["centers"]
            self.floor_ids = arrays["floor_ids"]
            self.clearances = arrays["clearances"]
            self.region_ids = arrays["region_ids"]
            self.unsupported = []
        else:
            triangles, islands, self.unsupported = navigation.triangles()
            self.triangles, self.islands, self.source_ids = refine_triangles(
                triangles, islands, config.surface_max_edge, config.max_surface_triangles)
            self.centers = np.array([navigation.snap(t.mean(0), int(i))
                                     for t, i in zip(self.triangles, self.islands)]).reshape(-1, 3)
            self.clearances = np.array([navigation.clearance(p, config.clearance_search)["navmesh_boundary_distance"]
                                        for p in self.centers])
            self.region_ids = np.full(len(self.triangles), -1, dtype=int)
        self.areas = np.linalg.norm(np.cross(self.triangles[:, 1]-self.triangles[:, 0],
                                             self.triangles[:, 2]-self.triangles[:, 0]), axis=1) / 2
        self.tree = cKDTree(self.centers) if len(self.centers) else None
        self.low = self.triangles.min(axis=1) if len(self.triangles) else np.empty((0, 3))
        self.high = self.triangles.max(axis=1) if len(self.triangles) else np.empty((0, 3))
        if arrays is None:
            self.floor_levels, self.floor_ids = self._floors()
        self.adjacency = None
        self.link_paths = {}
        self.links = []
        self.repaired_seams = []

    def _floors(self):
        if not len(self.centers):
            return [], np.array([], dtype=int)
        heights = self.centers[:, 1]
        bins = np.floor(heights / 0.25).astype(int)
        weights = defaultdict(float)
        for b, area in zip(bins, self.areas):
            weights[int(b)] += area
        levels = []
        for b in sorted(weights, key=lambda b: (-weights[b], b)):
            h = (b + 0.5) * 0.25
            if weights[b] < max(0.1, self.areas.sum() * 0.02) and levels:
                continue
            if all(abs(h - x) >= self.config.floor_band for x in levels):
                mask = abs(heights - h) < 0.4
                levels.append(float(np.average(heights[mask], weights=self.areas[mask])))
        levels.sort()
        ids = np.argmin(abs(heights[:, None] - np.asarray(levels)[None, :]), axis=1)
        return levels, ids

    def build_adjacency(self):
        incidence = defaultdict(list)
        for i, (tri, island) in enumerate(zip(self.triangles, self.islands)):
            for a, b in zip(tri, np.roll(tri, -1, axis=0)):
                incidence[edge_key(a, b, island)].append(i)
        candidates = sorted({tuple(sorted((a, b))) for ids in incidence.values()
                             for a, b in itertools.combinations(ids, 2)})
        for a, b in candidates:
            path = self.nav.shortest_path(self.centers[a], self.centers[b])
            if path is None:
                continue
            length = path_length(path)
            if length > np.linalg.norm(self.centers[a]-self.centers[b]) * 1.5 + 0.15:
                continue  # Coincident exported geometry is not native adjacency.
            self._link(a, b, path)
        self.adjacency = sparse_graph(len(self.centers), self.links)
        # Exported polygons can have unmatched tessellation edges or stair seams.
        # Repair each native island with explicit native routes, never Euclidean edges.
        count, labels = connected_components(self.adjacency)
        for island in sorted(set(self.islands.tolist())):
            ids = np.flatnonzero(self.islands == island)
            components = sorted(set(labels[ids]))
            if len(components) < 2:
                continue
            reached = set(ids[labels[ids] == components[0]].tolist())
            while len(reached) < len(ids):
                rest = np.array(sorted(set(ids.tolist()) - reached))
                current = np.array(sorted(reached))
                distances, nearest = cKDTree(self.centers[current]).query(self.centers[rest])
                choices = sorted(zip(distances.tolist(), rest.tolist(), current[nearest].tolist()))
                found = False
                for _, b, a in choices:
                    path = self.nav.shortest_path(self.centers[a], self.centers[b])
                    if path is None or not self.nav.validate_path(path)["valid"]:
                        continue
                    self._link(a, b, path)
                    self.repaired_seams.append({"source": a, "target": b, "length": path_length(path)})
                    reached.update(ids[labels[ids] == labels[b]].tolist())
                    found = True
                    break
                if not found:
                    raise ValueError(f"Native island {island} cannot be connected through exported surfaces")
        self.adjacency = sparse_graph(len(self.centers), self.links)
        return self.adjacency

    def _link(self, a, b, path):
        if a > b:
            a, b, path = b, a, path[::-1]
        self.links.append((a, b, path_length(path)))
        self.link_paths[(a, b)] = path

    def path_for_link(self, a, b):
        return self.link_paths[(a, b)] if a < b else self.link_paths[(b, a)][::-1]

    def locate(self, position):
        p = self.nav.snap(position)
        island = self.nav.island(p)
        ids = np.flatnonzero((self.islands == island) &
                            np.all(self.low[:, [0, 2]] <= p[[0, 2]] + 1e-5, axis=1) &
                            np.all(self.high[:, [0, 2]] >= p[[0, 2]] - 1e-5, axis=1))
        matches = []
        for i in ids:
            tri = self.triangles[i]
            matrix = (tri[1:, [0, 2]] - tri[0, [0, 2]]).T
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            uv = np.linalg.solve(matrix, p[[0, 2]] - tri[0, [0, 2]])
            if min(uv) < -1e-4 or uv.sum() > 1.0001:
                continue
            height = tri[0, 1] + np.dot(uv, tri[1:, 1] - tri[0, 1])
            if abs(height-p[1]) <= 0.08:
                matches.append((abs(height-p[1]), int(i)))
        if not matches:
            raise ValueError("Native point has no exported surface support (possibly a degenerate island)")
        return min(matches)[1]

    def route_between(self, start, end):
        """Interior-surface fallback for native shortest-path boundary corner traps."""
        from .topology import concatenate, simplify_path
        a, b = self.locate(start), self.locate(end)
        if self.adjacency is None:
            self.build_adjacency()
        distance, previous = dijkstra(self.adjacency, directed=False, indices=a, return_predecessors=True)
        if not np.isfinite(distance[b]):
            raise ValueError("No connected surface route")
        ids = [b]
        while ids[-1] != a:
            ids.append(int(previous[ids[-1]]))
        ids.reverse()
        parts = [np.array([start, self.centers[a]])]
        parts.extend(self.path_for_link(i, j) for i, j in zip(ids[:-1], ids[1:]))
        parts.append(np.array([self.centers[b], end]))
        path = simplify_path(concatenate(parts), self.nav)
        if not self.nav.validate_path(path)["valid"]:
            raise ValueError("Interior surface route failed native traversal")
        return path

    def arrays(self):
        return {k: getattr(self, k) for k in ("triangles", "islands", "source_ids", "centers",
                                             "floor_ids", "clearances", "region_ids")}
