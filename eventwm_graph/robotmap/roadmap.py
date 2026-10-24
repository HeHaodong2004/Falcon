"""Deterministic configuration-space extraction before global graph building."""
from collections import Counter, defaultdict
from dataclasses import replace
import heapq
import math
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..surface import Surface
from ..native import path_length
from .config import TRAVERSABLE, NON_TRAVERSABLE, UNKNOWN, angle_delta


def join_poses(parts):
    out = []
    for part in parts:
        for pose in part:
            q = np.asarray(pose, dtype=float)
            if not out or np.linalg.norm(q[:3]-out[-1][:3]) > 1e-8 or abs(angle_delta(out[-1][3], q[3])) > 1e-8:
                out.append(q)
    return np.asarray(out).reshape(-1, 4)


def motion_duration(poses, robot):
    q = np.asarray(poses)
    return float(sum(np.linalg.norm(b[:3]-a[:3])/robot.max_speed+
                     abs(angle_delta(a[3], b[3]))/robot.max_yaw_rate for a, b in zip(q[:-1], q[1:])))


def state_adjacency(states, motions):
    rr, cc, vv = [], [], []
    for e in motions:
        rr.append(e["source"]); cc.append(e["target"]); vv.append(max(1e-7, e["duration"]))
    return csr_matrix((vv, (rr, cc)), shape=(len(states), len(states)))


def angle_route(site, start_index, goal_index):
    """Return already validated angular samples; never infer a free turn."""
    queue = [(0., start_index)]
    distances, previous = {start_index: 0.}, {}
    adjacency = defaultdict(list)
    for index, edge in enumerate(site["turns"]):
        if edge["status"] == TRAVERSABLE:
            a, b = edge["source"], edge["target"]
            adjacency[a].append((b, index, False))
            adjacency[b].append((a, index, True))
    while queue:
        cost, i = heapq.heappop(queue)
        if i == goal_index:
            break
        if cost != distances[i]:
            continue
        for j, edge, reverse in adjacency[i]:
            d = cost+abs(angle_delta(site["yaws"][i], site["yaws"][j]))
            if d < distances.get(j, math.inf)-1e-9:
                distances[j] = d; previous[j] = (i, edge, reverse)
                heapq.heappush(queue, (d, j))
    if goal_index not in distances:
        return None
    segments, i = [], goal_index
    while i != start_index:
        j, e, reverse = previous[i]
        samples = site["turns"][e]["poses"]
        segments.append(samples[::-1] if reverse else samples)
        i = j
    return join_poses(segments[::-1]) if segments else np.array([[*site["position"], site["yaws"][start_index]]])


def heading_connectors(site, states, yaw, oracle):
    """Connect a transit heading to each compatible stopping-state component."""
    result = []
    for state in states:
        options = sorted(state["heading_indices"], key=lambda i: (abs(angle_delta(site["yaws"][i], yaw)), i))
        for i in options[:2]:
            if abs(angle_delta(site["yaws"][i], yaw)) > 2*math.pi/len(site["yaws"])+1e-6:
                continue
            proof = oracle.motion([[*site["position"], site["yaws"][i]], [*site["position"], yaw]])
            if proof["status"] == TRAVERSABLE:
                route = angle_route(site, state["heading_index"], i)
                result.append((state["id"], join_poses([route, proof["swept_poses"]])))
                break
    return result


def extract_roadmap(navigation, geometry, robot, build_config, config, oracle, progress=None):
    """All poses, support and directed motions precede topology extraction.

    Surface triangles are sampling/support domains. Untested configurations in
    those domains remain UNKNOWN even when a feasible witness is present.
    """
    began = time.perf_counter()
    surface = Surface(navigation, replace(build_config, surface_max_edge=config.surface_max_edge))
    sites, states = [], []
    yaws = np.linspace(-math.pi, math.pi, config.heading_count, endpoint=False).tolist()
    surface_sites = defaultdict(list)
    for sid, center in enumerate(surface.centers):
        center = navigation.project_path_sample(center)
        primary = [oracle.pose(center, y) for y in yaws]
        candidates = [(center, primary)]
        if config.adaptive_sites and sum(v["status"] == TRAVERSABLE for v in primary) < len(yaws):
            triangle = surface.triangles[sid]
            for j in range(3):
                p = .6*triangle[j]+.2*triangle[(j+1)%3]+.2*triangle[(j+2)%3]
                try:
                    p = navigation.snap(p, int(surface.islands[sid]))
                except ValueError:
                    continue
                if np.linalg.norm(p-center) < .06:
                    continue
                candidates.append((p, [oracle.pose(p, y) for y in yaws]))
        ranked = sorted(enumerate(candidates), key=lambda item: (
            -sum(x["status"] == TRAVERSABLE for x in item[1][1]),
            -navigation.clearance(item[1][0])["navmesh_boundary_distance"], item[0]))
        chosen = [ranked[0][1]]
        available = {i for i, r in enumerate(chosen[0][1]) if r["status"] == TRAVERSABLE}
        for _, candidate in ranked[1:]:
            extra = {i for i, r in enumerate(candidate[1]) if r["status"] == TRAVERSABLE}-available
            if extra and np.linalg.norm(candidate[0]-chosen[0][0]) > .1:
                chosen.append(candidate)
                break
        for position, evidence in chosen:
            site = {"id": len(sites), "surface_id": sid, "position": position.tolist(),
                    "island": int(surface.islands[sid]), "floor_id": int(surface.floor_ids[sid]),
                    "domain_status": UNKNOWN, "domain_scope": "untested continuum retained",
                    "yaws": yaws, "pose_evidence": evidence, "turns": [], "state_ids": []}
            valid = {i for i, r in enumerate(evidence) if r["status"] == TRAVERSABLE}
            neighbours = defaultdict(list)
            for i in range(len(yaws)):
                j = (i+1)%len(yaws)
                if i not in valid or j not in valid:
                    site["turns"].append({"source": i, "target": j, "status": NON_TRAVERSABLE
                                          if any(evidence[k]["status"] == NON_TRAVERSABLE for k in (i, j)) else UNKNOWN,
                                          "reason": "endpoint_pose_not_verified"})
                    continue
                proof = oracle.motion([[*position, yaws[i]], [*position, yaws[j]]])
                row = {"source": i, "target": j, **{k: v for k, v in proof.items() if k != "swept_poses"}}
                if proof["status"] == TRAVERSABLE:
                    row["poses"] = proof["swept_poses"]
                    neighbours[i].append(j); neighbours[j].append(i)
                site["turns"].append(row)
            pending = set(valid)
            while pending:
                seed = min(pending); pending.remove(seed); group = [seed]; stack = [seed]
                while stack:
                    for nxt in neighbours[stack.pop()]:
                        if nxt in pending:
                            pending.remove(nxt); group.append(nxt); stack.append(nxt)
                representative = min(group, key=lambda i: (abs(yaws[i]), i))
                state = {"id": len(states), "site": site["id"], "surface_id": sid,
                         "position": position.tolist(), "yaw": yaws[representative],
                         "heading_index": representative, "heading_indices": sorted(group),
                         "status": TRAVERSABLE, "island": site["island"], "floor_id": site["floor_id"]}
                site["state_ids"].append(state["id"]); states.append(state)
            surface_sites[sid].append(site["id"]); sites.append(site)
        if len(sites) > config.max_sites:
            raise ValueError("Traversability site budget exceeded; increase max_sites or surface_max_edge")
        if progress and sid % 200 == 0:
            progress({"phase": "poses", "surface": sid, "surfaces": len(surface.centers), "sites": len(sites), "oracle_calls": oracle.calls})
    motions, rejected = [], []
    positions = np.array([s["position"] for s in sites]).reshape(-1, 3)
    tree = cKDTree(positions) if len(positions) else None
    pairs = set()
    for i, site in enumerate(sites):
        near = tree.query_ball_point(positions[i], config.connection_radius)
        near = sorted((j for j in near if j != i and sites[j]["island"] == site["island"]),
                      key=lambda j: (np.linalg.norm(positions[j]-positions[i]), j))[:config.max_neighbors]
        pairs.update(tuple(sorted((i, j))) for j in near)
    for pair_index, (a, b) in enumerate(sorted(pairs)):
        pa, pb = positions[a], positions[b]
        if not navigation.segment_free(pa, pb):
            rejected.append({"source_site": a, "target_site": b, "status": NON_TRAVERSABLE,
                             "scope": "specified center segment", "reason": "native_segment_blocked"})
            continue
        for source, target in ((a, b), (b, a)):
            sa, sb = sites[source], sites[target]
            if not sa["state_ids"] or not sb["state_ids"]:
                rejected.append({"source_site": source, "target_site": target, "status": UNKNOWN,
                                 "reason": "no_verified_heading_witness", "scope": "site_pair_connectivity"})
                continue
            p, q = positions[source], positions[target]
            yaw = math.atan2(p[2]-q[2], q[0]-p[0])
            proof = oracle.motion([[*p, yaw], [*q, yaw]])
            if proof["status"] != TRAVERSABLE:
                rejected.append({"source_site": source, "target_site": target, **proof,
                                 "scope": "specified_forward_primitive; other motions remain unknown"})
                continue
            left = heading_connectors(sa, [states[i] for i in sa["state_ids"]], yaw, oracle)
            right = heading_connectors(sb, [states[i] for i in sb["state_ids"]], yaw, oracle)
            if not left or not right:
                rejected.append({"source_site": source, "target_site": target, "status": UNKNOWN,
                                 "reason": "transit_heading_not_connected_to_stopping_states"})
            for u, entry in left:
                for v, leave in right:
                    poses = join_poses([entry, proof["swept_poses"], leave[::-1]])
                    motions.append({"id": len(motions), "source": u, "target": v,
                                    "poses": poses.tolist(), "length": path_length(poses[:, :3]),
                                    "duration": motion_duration(poses, robot), "status": TRAVERSABLE,
                                    "validation_level": config.validation_level,
                                    "spatial_step": config.spatial_step, "angular_step": config.angular_step})
        if progress and pair_index % 500 == 0:
            progress({"phase": "motions", "pair": pair_index, "pairs": len(pairs), "verified": len(motions), "oracle_calls": oracle.calls})
    adjacency = state_adjacency(states, motions)
    if states:
        strong_count, strong = connected_components(adjacency, directed=True, connection="strong")
        weak_count, weak = connected_components(adjacency, directed=True, connection="weak")
        for state, s, w in zip(states, strong, weak):
            state["strong_component"] = int(s); state["weak_component"] = int(w)
    else:
        strong_count = weak_count = 0
    transit_sites = {states[i]["site"] for e in motions for i in (e["source"], e["target"])}
    stopping_faces = {s["surface_id"] for s in sites if s["state_ids"]}
    transit_faces = {sites[i]["surface_id"] for i in transit_sites}
    stats = {"sites": len(sites), "states": len(states), "motions": len(motions),
             "strong_components": int(strong_count), "weak_components": int(weak_count),
             "native_area": float(surface.areas.sum()),
             "area_with_stopping_witness": float(sum(surface.areas[i] for i in stopping_faces)),
             "area_with_transit_witness": float(sum(surface.areas[i] for i in transit_faces)),
             "area_scope": "sampling-domain area with a verified witness, not certified continuous free area",
             "unknown_continuous_area": float(surface.areas.sum()),
             "proven_blocked_continuous_area": 0.,
             "pose_status_counts": dict(Counter(r["status"] for s in sites for r in s["pose_evidence"])),
             "rejected_primitive_reasons": dict(Counter(r["reason"] for r in rejected)),
             "oracle_calls": oracle.calls, "oracle_cache_hits": oracle.cache_hits,
             "support_ray_calls": oracle.support_calls, "elapsed_seconds": time.perf_counter()-began}
    return surface, {"sites": sites, "states": states, "motions": motions,
                     "rejected_primitives": rejected, "statistics": stats,
                     "calibration": oracle.calibration, "unsupported_native_islands": surface.unsupported}
