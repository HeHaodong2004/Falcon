"""Clearance-aware geodesic regions and a native-verified portal multigraph."""
from collections import Counter, defaultdict
import math
import numpy as np
from scipy.sparse.csgraph import dijkstra, connected_components

from .config import ALGORITHM_VERSION
from .native import path_length, samples


def concatenate(paths):
    result = []
    for path in paths:
        for p in path:
            if not result or np.linalg.norm(p-result[-1]) > 1e-7:
                result.append(np.asarray(p))
    return np.asarray(result).reshape(-1, 3)


def simplify_path(path, nav):
    """Compress redundant bends only when the native straight traversal succeeds."""
    if len(path) < 3:
        return path
    out, i = [path[0]], 0
    while i < len(path)-1:
        j = min(len(path)-1, i+20)
        while j > i+1:
            # A straight horizontal projection cannot erase a stair profile.
            if (np.ptp(path[i:j+1, 1]) < 0.12 and nav.segment_free(path[i], path[j])
                    and nav.validate_path([path[i], path[j]])["valid"]):
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.asarray(out)


def restore_native_corners(path, nav):
    """Retain native polygon corners even when a finite chord sampler misses one.

    A corpus failure exposed a 3.9 mm corner cut between samples 5 cm apart.
    Querying each segment's native geodesic restores its polygon corner. A
    meaningful detour is not silently substituted for the selected passage.
    Native action certification remains mandatory after this construction step.
    """
    if len(path) < 2:
        return np.asarray(path)
    parts = []
    for a, b in zip(path[:-1], path[1:]):
        native = nav.shortest_path(a, b)
        if native is None:
            raise ValueError('Native corner restoration has no surface connection')
        if path_length(native) > 1.05*np.linalg.norm(np.asarray(b)-a)+.01:
            raise ValueError('Native corner restoration requires a different passage')
        if (np.linalg.norm(native[0]-a) > .002 or np.linalg.norm(native[-1]-b) > .002):
            raise ValueError('Native corner restoration changed a support endpoint')
        parts.append(native)
    return concatenate(parts)


def _boundary_groups(crossings, adjacency, surface):
    """Keep separate passages between the same regions as parallel graph edges."""
    by_face = defaultdict(list)
    for j, (a, b, _) in enumerate(crossings):
        by_face[a].append(j); by_face[b].append(j)
    pending, groups = set(range(len(crossings))), []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        stack, group = [seed], []
        while stack:
            j = stack.pop()
            group.append(crossings[j])
            for face in crossings[j][:2]:
                for other in [face, *adjacency.indices[adjacency.indptr[face]:adjacency.indptr[face+1]]]:
                    for k in by_face.get(other, []):
                        if k in pending:
                            pending.remove(k); stack.append(k)
        groups.append(group)
    # Triangulation can split one open interface into several dual-edge strips.
    # Merge only through short, straight native traversals. This removes false
    # wall-adjacent "portals" in a simple corridor while keeping separate doors.
    parent = list(range(len(groups)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            merge = False
            for a, b, _ in groups[i]:
                p = surface.centers[a]
                for c, d, _ in groups[j]:
                    q = surface.centers[c]
                    if (np.linalg.norm(p-q) < 2.5*surface.config.surface_max_edge and
                            abs(p[1]-q[1]) < .25 and surface.nav.segment_free(p, q) and
                            surface.nav.segment_free(surface.centers[b], surface.centers[d])):
                        merge = True; break
                if merge:
                    break
            if merge:
                parent[root(j)] = root(i)
    result = defaultdict(list)
    for i, group in enumerate(groups):
        result[root(i)].extend(group)
    return [result[k] for k in sorted(result)]


def build_global_graph(surface, config, geometry=None):
    nav, positions = surface.nav, surface.centers
    n = len(positions)
    if not n:
        return {"algorithm": ALGORITHM_VERSION, "nodes": [], "edges": [], "regions": [],
                "floor_levels": [], "unsupported_islands": surface.unsupported,
                "statistics": {"empty": True, "surface_components": 0, "native_islands": nav.pf.num_islands}}
    graph = surface.build_adjacency()
    nearest = np.full(n, np.inf)
    seeds = []
    # Independent radius from the local collision-scale lattice. Prefer central
    # clearance maxima among uncovered surface sites, with deterministic ties.
    while np.max(nearest) > config.region_radius:
        uncovered = np.flatnonzero(nearest > config.region_radius)
        finite = np.minimum(nearest[uncovered], 2 * config.region_radius)
        score = surface.clearances[uncovered] + 0.1 * finite
        seed = int(uncovered[np.argmax(score)])
        seeds.append(seed)
        nearest = np.minimum(nearest, dijkstra(graph, directed=False, indices=seed))
    seeds.sort()
    distances, predecessors, sources = dijkstra(graph, directed=False, indices=seeds,
                                               min_only=True, return_predecessors=True)
    seed_to_region = {s: i for i, s in enumerate(seeds)}
    regions = np.array([seed_to_region[int(s)] for s in sources], dtype=int)
    surface.region_ids = regions
    nodes, region_rows, edges = [], [], []

    def node(position, kind, region_ids, surface_id):
        position = nav.snap(position)
        nid = f"g{len(nodes):06d}"
        row = {"id": nid, "type": kind, "tags": [kind], "position": position.tolist(),
               "island": int(surface.islands[surface_id]),
               "floor_id": int(surface.floor_ids[surface_id]), "surface_id": int(surface_id),
               "regions": sorted(set(int(i) for i in region_ids)),
               "clearance": nav.clearance(position, config.clearance_search),
               "visiting_yaws": [], "pose_validation": "navmesh_only",
               "local_support": {"kind": "surface_region_intersection", "region_ids": sorted(set(int(i) for i in region_ids))}}
        if geometry is not None and geometry.robot is not None:
            row["visiting_yaws"] = [float(y) for y in np.linspace(-math.pi, math.pi, 8, endpoint=False)
                                     if geometry.pose_free(position, float(y))]
            row["pose_validation"] = "native_static_reference_posture_8_yaws"
        nodes.append(row)
        return nid

    for region, seed in enumerate(seeds):
        ids = np.flatnonzero(regions == region)
        nid = node(positions[seed], "region_anchor", [region], seed)
        region_rows.append({"id": region, "anchor": nid, "surface_ids": ids.tolist(),
                            "area": float(surface.areas[ids].sum()),
                            "island": int(surface.islands[seed]),
                            "floor_ids": sorted(set(surface.floor_ids[ids].tolist())),
                            "max_dual_distance": float(distances[ids].max()),
                            "meaning": "connected geodesic region, not a semantic room"})

    def chain(face):
        out = [face]
        while predecessors[out[-1]] >= 0:
            out.append(int(predecessors[out[-1]]))
        return out[::-1]

    def chain_path(indices):
        return concatenate([surface.path_for_link(a, b) for a, b in zip(indices[:-1], indices[1:])]) if len(indices)>1 else positions[indices]

    def add_edge(source, target, path, boundary=None):
        path = simplify_path(path, nav)
        path = restore_native_corners(path, nav)
        validation = nav.validate_path(path, min(0.05, config.edge_sample_step))
        if not validation["valid"]:
            raise ValueError(f"Global edge fails native verification: {validation}")
        clearance = [nav.clearance(nav.project_path_sample(p), config.clearance_search)["navmesh_boundary_distance"]
                     for p in samples(path, config.edge_sample_step)]
        edges.append({"id": f"e{len(edges):06d}", "source": source, "target": target,
                      "path": path.tolist(), "length": path_length(path),
                      "min_navmesh_clearance_sampled": float(min(clearance)),
                      "clearance_sample_step": config.edge_sample_step,
                      "floor_transition": bool(np.ptp(path[:, 1]) >= 0.75),
                      "height_range": [float(path[:, 1].min()), float(path[:, 1].max())],
                      "verified": validation, "boundary_surface_ids": boundary,
                      "compiler_ref": {"edge_id": f"e{len(edges):06d}", "geometry": "stored_native_path"}})

    boundaries = defaultdict(list)
    for a, b, length in surface.links:
        ra, rb = int(regions[a]), int(regions[b])
        if ra != rb:
            if ra > rb:
                a, b, ra, rb = b, a, rb, ra
            boundaries[(ra, rb)].append((a, b, length))
    for (ra, rb), crossings in sorted(boundaries.items()):
        for group in _boundary_groups(crossings, graph, surface):
            # Select a wide, short crossing, avoiding arbitrary triangle corners.
            a, b, _ = max(group, key=lambda row: (min(surface.clearances[row[0]], surface.clearances[row[1]]),
                                                -row[2], -row[0], -row[1]))
            left, right = chain(a), chain(b)
            route = concatenate([chain_path(left), surface.path_for_link(a, b), chain_path(right)[::-1]])
            start, end = region_rows[ra]["anchor"], region_rows[rb]["anchor"]
            narrow = min(surface.clearances[a], surface.clearances[b])
            anchor_clearance = min(surface.clearances[seeds[ra]], surface.clearances[seeds[rb]])
            if narrow < config.portal_clearance and narrow < config.portal_ratio * anchor_clearance:
                # Two feasible centre locations straddling the native transition;
                # physical stopping orientations are reported separately.
                pa = node(positions[a], "portal", [ra, rb], a) if a != seeds[ra] else start
                pb = node(positions[b], "portal", [ra, rb], b) if b != seeds[rb] else end
                if pa != start:
                    add_edge(start, pa, chain_path(left), [a, b])
                add_edge(pa, pb, surface.path_for_link(a, b), [a, b])
                if pb != end:
                    add_edge(pb, end, chain_path(right)[::-1], [a, b])
            else:
                add_edge(start, end, route, [a, b])
    degree = Counter()
    for edge in edges:
        degree[edge["source"]] += 1; degree[edge["target"]] += 1
    for row in nodes:
        if row["type"] != "region_anchor":
            continue
        deg = degree[row["id"]]
        if deg == 1:
            row["type"] = "terminal"
        elif deg >= 3:
            neighbours = []
            by_id = {n["id"]: n for n in nodes}
            for edge in edges:
                if edge["source"] == row["id"]:
                    other = edge["target"]
                elif edge["target"] == row["id"]:
                    other = edge["source"]
                else:
                    continue
                distance, previous, visited = edge["length"], row["id"], {row["id"]}
                # Portal pairs can lie inside a broad intersection. Inspect the
                # actual outgoing branch beyond short degree-two chains before
                # deciding whether departures are geometrically separated.
                while degree[other] == 2 and distance < config.region_radius:
                    next_edges = [e for e in edges if other in (e["source"], e["target"])
                                  and previous not in (e["source"], e["target"])]
                    if not next_edges or other in visited:
                        break
                    visited.add(other)
                    nxt = next_edges[0]
                    previous, other = other, nxt["target"] if nxt["source"] == other else nxt["source"]
                    distance += nxt["length"]
                neighbours.append(by_id[other]["position"])
            blocked_chords = sum(not nav.segment_free(a, b) for i, a in enumerate(neighbours)
                                 for b in neighbours[i+1:])
            # Region tessellation alone creates degree-three nodes in an empty
            # room. A junction requires geometrically separated departures.
            if blocked_chords >= 2:
                row["type"] = "junction"
        elif deg == 2:
            incident = [e for e in edges if row["id"] in (e["source"], e["target"])]
            vectors = []
            for edge in incident:
                p = np.asarray(edge["path"])
                if edge["target"] == row["id"]:
                    p = p[::-1]
                if len(p) > 1:
                    v = (p[1]-p[0])[[0, 2]]
                    vectors.append(v/max(np.linalg.norm(v), 1e-9))
            if len(vectors) == 2 and np.dot(*vectors) > -math.cos(math.radians(config.turn_angle)):
                row["type"] = "corridor_turn"
        row["tags"] = sorted(set(row["tags"] + [row["type"]]))
    if geometry is not None:
        # Ray discontinuities are measured using the actual scene collision mesh.
        # Enrich existing candidates; no hidden actor state or nav occupancy rays.
        for row in nodes:
            rays = geometry.ray_distances(row["position"])
            contrast = float(np.max(abs(np.roll(rays, 1)-rays)))
            row["visibility"] = {"height": 1.25, "range": 8.0, "ray_count": 32,
                                 "distances": rays, "max_angular_range_jump": contrast,
                                 "geometry": "native_static_collision_rays"}
            if contrast > 3 and row["type"] in ("corridor_turn", "portal", "junction"):
                row["tags"].append("visibility_candidate")
    count, _ = connected_components(graph)
    return {"algorithm": ALGORITHM_VERSION, "nodes": nodes, "edges": edges,
            "regions": region_rows, "floor_levels": surface.floor_levels,
            "unsupported_islands": surface.unsupported,
            "statistics": {"surface_triangles": n, "surface_components": count,
                           "native_islands": nav.pf.num_islands,
                           "node_types": dict(Counter(x["type"] for x in nodes)),
                           "max_region_dual_radius": float(distances.max()),
                           "repaired_export_seams": surface.repaired_seams}}
