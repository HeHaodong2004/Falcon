"""Lazy collision-scale cells clipped to real 3D NavMesh surface triangles."""
from collections import OrderedDict, defaultdict
from dataclasses import asdict
import math
import numpy as np
from scipy.spatial import cKDTree

from .config import FloorContext, LocalConfig, digest
from .native import path_length, samples


def clip_polygon(polygon, axis, value, keep_greater):
    result = []
    if not len(polygon):
        return np.empty((0, 3))
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        inside_a = a[axis] >= value-1e-10 if keep_greater else a[axis] <= value+1e-10
        inside_b = b[axis] >= value-1e-10 if keep_greater else b[axis] <= value+1e-10
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            t = (value-a[axis])/(b[axis]-a[axis])
            result.append(a+t*(b-a))
    return np.asarray(result).reshape(-1, 3)


def polygon_area(p):
    if len(p) < 3:
        return 0.0
    return float(sum(np.linalg.norm(np.cross(p[i]-p[0], p[i+1]-p[0])) / 2
                     for i in range(1, len(p)-1)))


def shared_boundary(a, b, tolerance=2e-5):
    """Positive-length overlap of 3D polygon edges (not projected overlap)."""
    for x, y in zip(a, np.roll(a, -1, axis=0)):
        v = y-x
        length = np.linalg.norm(v)
        if length < tolerance:
            continue
        v /= length
        for z, w in zip(b, np.roll(b, -1, axis=0)):
            if max(np.linalg.norm(np.cross(z-x, v)), np.linalg.norm(np.cross(w-x, v))) > tolerance:
                continue
            lo, hi = sorted((np.dot(z-x, v), np.dot(w-x, v)))
            if min(length, hi)-max(0, lo) > tolerance:
                return True
    return False


class LocalGraph:
    def __init__(self, surface, robot, config=LocalConfig(), topology_id=""):
        self.surface, self.nav, self.robot, self.config = surface, surface.nav, robot, config
        self.resolution = config.k * robot.collision_reference
        self.identity = digest({"topology": topology_id, "local": asdict(config),
                                "collision_reference": robot.collision_reference,
                                "geometry": robot.geometry_id})[:16]
        self.cache = OrderedDict()
        self.edge_cache = OrderedDict()
        self.tree = cKDTree(surface.centers[:, [0, 2]]) if len(surface.centers) else None
        self.support_radius = float(np.linalg.norm(surface.triangles-surface.centers[:, None, :], axis=2).max()) if len(surface.centers) else 0

    def _cell(self, ix, iz):
        key = (int(ix), int(iz))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        h = self.resolution
        center = (np.array(key)+0.5)*h
        candidates = self.tree.query_ball_point(center, self.support_radius+h) if self.tree else []
        pieces = []
        for i in sorted(candidates):
            p = self.surface.triangles[i]
            for axis, index in ((0, ix), (2, iz)):
                p = clip_polygon(p, axis, index*h, True)
                p = clip_polygon(p, axis, (index+1)*h, False)
            area = polygon_area(p)
            if area > 1e-10:
                pieces.append((i, p, area))
        # Components of actual clipped support. A cell split by a wall or an
        # overlapping floor yields distinct nodes even if its XZ key is shared.
        remaining, groups = set(range(len(pieces))), []
        while remaining:
            seed = min(remaining); remaining.remove(seed)
            group, stack = [seed], [seed]
            while stack:
                a = stack.pop()
                ia, pa, _ = pieces[a]
                for b in sorted(remaining):
                    ib, pb, _ = pieces[b]
                    if self.surface.islands[ia] != self.surface.islands[ib]:
                        continue
                    if not shared_boundary(pa, pb):
                        continue
                    if not self.nav.segment_free(pa.mean(0), pb.mean(0)):
                        continue
                    remaining.remove(b); group.append(b); stack.append(b)
            groups.append(group)
        nodes = []
        for group in groups:
            supports = [pieces[i] for i in group]
            representative = max(supports, key=lambda p: (p[2], -p[0]))
            sid = representative[0]
            try:
                position = self.nav.snap(representative[1].mean(0), int(self.surface.islands[sid]))
            except ValueError:
                continue
            region_area = defaultdict(float)
            for i, _, area in supports:
                region_area[int(self.surface.region_ids[i])] += area
            total = sum(region_area.values())
            nodes.append({"id": f"l{self.identity}:{ix}:{iz}:{min(p[0] for p in supports)}",
                          "cell": [int(ix), int(iz)], "position": position.tolist(),
                          "island": int(self.surface.islands[sid]),
                          "floor_id": int(self.surface.floor_ids[sid]), "area": total,
                          "support": [{"surface_id": int(i), "polygon": p.tolist(), "area": area}
                                      for i, p, area in supports],
                          "region_weights": {str(k): v/total for k, v in sorted(region_area.items())},
                          "clearance": self.nav.clearance(position)})
        self.cache[key] = nodes
        while len(self.cache) > self.config.max_cached_cells:
            self.cache.popitem(last=False)
        return nodes

    def _edge(self, a, b):
        """Reuse immutable native edge evidence across overlapping patches."""
        key=tuple(sorted((a['id'],b['id'])))
        if key in self.edge_cache:
            self.edge_cache.move_to_end(key);value=self.edge_cache[key]
        else:
            value=None;pa,pb=np.asarray(a['position']),np.asarray(b['position'])
            if (a['island']==b['island'] and
                    abs(pa[1]-pb[1])<=max(self.robot.max_climb,self.resolution)*1.5 and
                    self.nav.segment_free(pa,pb)):
                path=self.nav.shortest_path(pa,pb)
                if path is not None and path_length(path)<=2.5*self.resolution and self.nav.validate_path(path)['valid']:
                    value=(a['id'],b['id'],tuple(tuple(float(v) for v in p) for p in path),path_length(path))
            self.edge_cache[key]=value
            while len(self.edge_cache)>24*self.config.max_cached_cells:
                self.edge_cache.popitem(last=False)
        if value is None:return None
        return {'source':value[0],'target':value[1],'path':[list(p) for p in value[2]],'length':value[3]}

    def get_local_graph_patch(self, position, radius, floor_context=None):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Patch radius must be positive and finite")
        p = self.nav.snap(position)
        context = floor_context or FloorContext(island=self.nav.island(p))
        if context.island >= 0 and self.nav.island(p) != context.island:
            raise ValueError("Position and requested island disagree")
        h = self.resolution
        low = np.floor((p[[0, 2]]-radius)/h).astype(int)
        high = np.floor((p[[0, 2]]+radius)/h).astype(int)
        if np.prod(high-low+1) > self.config.max_patch_cells:
            raise ValueError("Patch exceeds cell budget; use smaller ROIs or route patches")
        nodes, cell_nodes = [], defaultdict(list)
        for ix in range(low[0], high[0]+1):
            for iz in range(low[1], high[1]+1):
                for node in self._cell(ix, iz):
                    if context.island >= 0 and node["island"] != context.island:
                        continue
                    if context.floor_id >= 0 and node["floor_id"] != context.floor_id:
                        continue
                    q = np.asarray(node["position"])
                    if abs(q[1]-p[1]) > context.max_height_delta:
                        continue
                    if np.linalg.norm(q[[0, 2]]-p[[0, 2]]) > radius+math.sqrt(2)*h:
                        continue
                    cell_nodes[(ix, iz)].append(node)
                    nodes.append(node)
        edges = []
        for cell, items in sorted(cell_nodes.items()):
            neighbours = [(cell[0]+dx, cell[1]+dz) for dx, dz in
                          ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))]
            for other in neighbours:
                for a in items:
                    for b in cell_nodes.get(other, []):
                        if a["id"] == b["id"] or (other == cell and a["id"] > b["id"]):
                            continue
                        edge=self._edge(a,b)
                        if edge is not None:edges.append(edge)
        return {"id": self.identity, "resolution": h, "center": p.tolist(), "radius": radius,
                "floor_context": asdict(context), "nodes": sorted(nodes, key=lambda n: n["id"]),
                "edges": edges, "support_semantics": "union of clipped 3D NavMesh polygons; ROI includes intersecting cells"}

    def get_local_nodes_for_path(self, path, radius=None):
        radius = radius if radius is not None else 2*self.resolution
        nodes, edges = {}, {}
        for p in samples(path, max(radius, self.resolution)):
            patch = self.get_local_graph_patch(self.nav.project_path_sample(p), radius)
            nodes.update((n["id"], n) for n in patch["nodes"])
            edges.update((tuple(sorted((e["source"], e["target"]))), e) for e in patch["edges"])
        return {"id": self.identity, "resolution": self.resolution,
                "nodes": [nodes[k] for k in sorted(nodes)],
                "edges": [edges[k] for k in sorted(edges)]}


def build_local_graph(surface, robot_config, config=LocalConfig(), topology_id=""):
    return LocalGraph(surface, robot_config, config, topology_id)


def get_local_graph_patch(local_graph, *args, **kwargs):
    return local_graph.get_local_graph_patch(*args, **kwargs)


def get_local_nodes_for_path(local_graph, *args, **kwargs):
    return local_graph.get_local_nodes_for_path(*args, **kwargs)
