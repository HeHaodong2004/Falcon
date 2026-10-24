"""Public hierarchy, routing, compilation, persistence and validation services."""
from collections import defaultdict
from dataclasses import asdict
import copy
import heapq
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np

from .config import (ALGORITHM_VERSION, SCHEMA_VERSION, BuildConfig, LocalConfig,
                     RobotConfig, digest, file_digest, read_json, write_json, scene_fingerprint,
                     robot_asset_fingerprint)
from .local import LocalGraph
from .native import HabitatNavigation, path_length, samples
from .surface import Surface
from .topology import build_global_graph, concatenate


class HierarchicalMap:
    def __init__(self, navigation, surface, graph, robot, build_config, local_config,
                 map_id, topology_id, geometry=None, metadata=None):
        self.navigation, self.surface, self.graph = navigation, surface, graph
        self.robot_config, self.build_config, self.local_config = robot, build_config, local_config
        self.map_id, self.topology_id = map_id, topology_id
        self.geometry, self.metadata = geometry, metadata or {}
        self.local = LocalGraph(surface, robot, local_config, topology_id)
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edges = {e["id"]: e for e in graph["edges"]}
        self.adjacency = defaultdict(list)
        for edge in graph["edges"]:
            self.adjacency[edge["source"]].append((edge["target"], edge["id"], edge["length"]))
            self.adjacency[edge["target"]].append((edge["source"], edge["id"], edge["length"]))

    def get_global_graph(self):
        return copy.deepcopy(self.graph)

    def get_local_patch(self, position, radius, floor_context=None):
        return copy.deepcopy(self.local.get_local_graph_patch(position, radius, floor_context))

    def get_local_nodes_for_path(self, path, radius=None):
        return copy.deepcopy(self.local.get_local_nodes_for_path(path, radius))

    def get_associated_local_nodes(self, global_node_id):
        node = self.nodes[global_node_id]
        region_ids = set(node["regions"])
        ids = np.flatnonzero(np.isin(self.surface.region_ids, list(region_ids)))
        h = self.local.resolution
        cells = set()
        for i in ids:
            lo = np.floor(self.surface.low[i, [0, 2]]/h).astype(int)
            hi = np.floor(self.surface.high[i, [0, 2]]/h).astype(int)
            cells.update((x, z) for x in range(lo[0], hi[0]+1) for z in range(lo[1], hi[1]+1))
        if len(cells) > self.local_config.max_patch_cells:
            raise ValueError("Region association exceeds budget; extract ROI patches")
        nodes = []
        for x, z in sorted(cells):
            for local in self.local._cell(x, z):
                weight = sum(local["region_weights"].get(str(r), 0) for r in region_ids)
                if weight:
                    nodes.append({"node": copy.deepcopy(local), "association_weight": weight})
        return nodes

    def get_global_candidates(self, position, constraints=None):
        """Constraints: limit, max_distance, min_distance, types, floor_id, require_native_pose."""
        constraints = constraints or {}
        valid_keys = {"limit", "max_distance", "min_distance", "types", "floor_id", "require_native_pose"}
        if set(constraints)-valid_keys:
            raise ValueError(f"Unknown constraints: {set(constraints)-valid_keys}")
        p = self.navigation.snap(position)
        island = self.navigation.island(p)
        result = []
        for node in self.graph["nodes"]:
            if node["island"] != island:
                continue
            if "floor_id" in constraints and node["floor_id"] != constraints["floor_id"]:
                continue
            if constraints.get("types") and node["type"] not in constraints["types"]:
                continue
            require_pose = constraints.get("require_native_pose", node["pose_validation"] != "navmesh_only")
            if require_pose and not node["visiting_yaws"]:
                continue
            path = self.navigation.shortest_path(p, node["position"])
            if path is None:
                continue
            distance = path_length(path)
            if not constraints.get("min_distance", 0) <= distance <= constraints.get("max_distance", math.inf):
                continue
            result.append({"node_id": node["id"], "geodesic_distance": distance,
                           "type": node["type"], "floor_id": node["floor_id"],
                           "pose_validation": node["pose_validation"]})
        return sorted(result, key=lambda c: (c["geodesic_distance"], c["node_id"]))[:constraints.get("limit", 12)]

    def get_graph_route(self, start, target_node):
        if target_node not in self.nodes:
            raise KeyError(target_node)
        start = self.navigation.snap(start)
        if self.navigation.island(start) != self.nodes[target_node]["island"]:
            raise ValueError("Start and target are on disconnected native islands")
        sid = self.surface.locate(start)
        region = int(self.surface.region_ids[sid])
        # Enter only through actual associated support, not a nearest point
        # across a wall or on the floor above. Multiple portals improve stretch.
        entry_nodes = [n for n in self.graph["nodes"] if region in n["regions"]]
        costs, previous, entries, queue = {}, {}, {}, []
        for node in entry_nodes:
            p = self.navigation.shortest_path(start, node["position"])
            if p is None:
                continue
            if not self.navigation.validate_path(p)["valid"]:
                try:
                    p = self.surface.route_between(start, node["position"])
                except ValueError:
                    continue
            nid, length = node["id"], path_length(p)
            costs[nid], previous[nid], entries[nid] = length, None, p
            heapq.heappush(queue, (length, nid))
        while queue:
            cost, nid = heapq.heappop(queue)
            if cost != costs[nid]:
                continue
            if nid == target_node:
                break
            for other, eid, length in self.adjacency[nid]:
                new_cost = cost + length
                if new_cost < costs.get(other, math.inf) - 1e-9:
                    costs[other], previous[other] = new_cost, (nid, eid)
                    heapq.heappush(queue, (new_cost, other))
        if target_node not in costs:
            raise ValueError("Global graph has no route for a native connected pair")
        node_ids, edge_ids = [target_node], []
        while previous[node_ids[-1]] is not None:
            parent, eid = previous[node_ids[-1]]
            node_ids.append(parent); edge_ids.append(eid)
        node_ids.reverse(); edge_ids.reverse()
        return {"map_id": self.map_id, "target_node": target_node,
                "start": start.tolist(), "node_ids": node_ids, "edge_ids": edge_ids,
                "entry_path": entries[node_ids[0]].tolist(), "length": costs[target_node]}

    def compile_navmesh_path(self, global_route):
        if global_route["map_id"] != self.map_id:
            raise ValueError("Route belongs to a different map")
        ids, eids = global_route["node_ids"], global_route["edge_ids"]
        if len(ids) != len(eids)+1 or not ids or ids[-1] != global_route["target_node"]:
            raise ValueError("Malformed route")
        paths = [np.asarray(global_route["entry_path"])]
        if (np.linalg.norm(paths[0][0]-global_route["start"]) > 0.08 or
                np.linalg.norm(paths[0][-1]-self.nodes[ids[0]]["position"]) > 0.08):
            raise ValueError("Entry path does not connect to selected graph route")
        for a, b, eid in zip(ids[:-1], ids[1:], eids):
            e = self.edges[eid]
            if (a, b) == (e["source"], e["target"]):
                paths.append(np.asarray(e["path"]))
            elif (b, a) == (e["source"], e["target"]):
                paths.append(np.asarray(e["path"])[::-1])
            else:
                raise ValueError("Selected edge does not join the specified nodes")
        path = concatenate(paths)
        if np.linalg.norm(path[-1]-self.nodes[ids[-1]]["position"]) > 0.08:
            raise ValueError("Compiled path misses target")
        verdict = self.navigation.validate_path(path)
        if not verdict["valid"]:
            raise ValueError(f"Compiled path invalid: {verdict}")
        return path

    def make_visit_query(self, start, target_node, start_yaw=0.0, target_yaw=None, speed=None):
        speed = self.robot_config.max_speed if speed is None else speed
        if not 0 < speed <= self.robot_config.max_speed or not math.isfinite(start_yaw):
            raise ValueError("Invalid speed or start heading")
        route = self.get_graph_route(start, target_node)
        path = self.compile_navmesh_path(route)
        yaw, t = float(start_yaw), 0.0
        poses = [[t, *path[0], yaw]]
        for a, b in zip(path[:-1], path[1:]):
            delta = b-a
            new_yaw = math.atan2(-delta[2], delta[0])
            angle = math.atan2(math.sin(new_yaw-yaw), math.cos(new_yaw-yaw))
            if abs(angle) > 1e-8:
                t += abs(angle)/self.robot_config.max_yaw_rate
                yaw += angle
                poses.append([t, *a, yaw])
            t += np.linalg.norm(delta)/speed
            poses.append([t, *b, yaw])
        if target_yaw is None and self.nodes[target_node]["visiting_yaws"]:
            target_yaw = min(self.nodes[target_node]["visiting_yaws"],
                             key=lambda y: abs(math.atan2(math.sin(y-yaw), math.cos(y-yaw))))
        if target_yaw is not None:
            if not math.isfinite(target_yaw):
                raise ValueError("Target yaw must be finite")
            angle = math.atan2(math.sin(target_yaw-yaw), math.cos(target_yaw-yaw))
            if abs(angle) > 1e-8:
                t += abs(angle)/self.robot_config.max_yaw_rate; yaw += angle
                poses.append([t, *path[-1], yaw])
        return {"schema_version": SCHEMA_VERSION, "map_id": self.map_id,
                "foundation": "native_navmesh_baseline", "robot_feasibility": "UNKNOWN",
                "validation_level": "native_navmesh_only", "execution_certified": False,
                "executable_under_stated_contract": False,
                "target_node": target_node, "route": route, "path": path.tolist(),
                "timed_poses": poses, "target_yaw": yaw,
                "motion": {"kind": "stop_turn_translate", "speed": speed,
                           "max_yaw_rate": self.robot_config.max_yaw_rate,
                           "controller_adapter": "reserved_for_L4", "time_units": "seconds"}}

    def validate_query_path(self, path, robot_config=None, spatial_step=0.05, angle_step=0.05):
        robot = robot_config or self.robot_config
        if robot != self.robot_config:
            raise ValueError("Robot configuration differs; load/rebuild a compatible map")
        query = path if isinstance(path, dict) else None
        points = query["path"] if query else path
        verdict = self.navigation.validate_path(points, min(spatial_step, self.local.resolution/2))
        result = {"navmesh": verdict, "native_static": None, "motion_valid": None,
                  "valid": verdict["valid"], "level": "navmesh_only",
                  "physical_safety_margin_certified": False}
        if query:
            if query.get("map_id") != self.map_id or query.get("target_node") not in self.nodes:
                raise ValueError("Query map/target mismatch")
            poses = np.asarray(query["timed_poses"], dtype=float)
            if poses.ndim != 2 or poses.shape[1] != 5 or len(poses) < 1 or not np.isfinite(poses).all():
                raise ValueError("Invalid timed poses")
            target = self.nodes[query["target_node"]]["position"]
            aligned = (np.linalg.norm(poses[0, 1:4]-np.asarray(points)[0]) < 1e-6 and
                       np.linalg.norm(poses[-1, 1:4]-target) < 1e-6 and
                       np.linalg.norm(np.asarray(points)[-1]-target) < 1e-6)
            dt = np.diff(poses[:, 0]); dist = np.linalg.norm(np.diff(poses[:, 1:4], axis=0), axis=1)
            turns = abs(np.diff(poses[:, 4]))
            moving_points = concatenate([poses[:, 1:4]])
            path_alignment = (moving_points.shape == np.asarray(points).shape and
                              np.allclose(moving_points, points, atol=1e-6))
            motion_valid = bool(aligned and path_alignment and np.all(dt > 0) and
                                np.all(dist <= robot.max_speed*dt+1e-6) and
                                np.all(turns <= robot.max_yaw_rate*dt+1e-6))
            result["motion_valid"] = motion_valid
            result["valid"] &= motion_valid
            if self.geometry is not None and self.geometry.robot is not None:
                count, failed = 0, None
                # Includes turns in place. No physics step, no hidden humans.
                pairs = list(zip(poses[:-1], poses[1:])) or [(poses[0], poses[0])]
                for a, b in pairs:
                    steps = max(1, math.ceil(np.linalg.norm(b[1:4]-a[1:4])/spatial_step),
                                math.ceil(abs(b[4]-a[4])/angle_step))
                    for alpha in np.linspace(0, 1, steps+1):
                        pose = a+alpha*(b-a); count += 1
                        ground = self.navigation.project_path_sample(pose[1:4])
                        if not self.geometry.pose_free(ground, pose[4]):
                            failed = pose.tolist(); break
                    if failed is not None:
                        break
                result["native_static"] = {"valid": failed is None, "checked_poses": count,
                                           "failed_pose": failed, "spatial_step": spatial_step,
                                           "angle_step": angle_step,
                                           "scope": "sampled articulated reference posture; ground support filtered",
                                           "support_contract": "distal_feet_within_climb; slope_or_static_ray_clear_column",
                                           "support_probe_half_span": .05,
                                           "support_probe_height_margin": .02}
                result["valid"] &= failed is None
                result["level"] = "native_static_sampled"
        return result

    def get_static_clearance(self, position):
        return self.navigation.clearance(position, self.build_config.clearance_search)

    def get_visibility_geometry(self, position=None, **kwargs):
        if position is None:
            return {"scene": self.metadata.get("scene"), "type": "native_static_collision_mesh",
                    "loaded": self.geometry is not None, "navmesh_is_visibility_geometry": False}
        if self.geometry is None:
            raise RuntimeError("Attach StaticGeometry to perform 3D visibility queries")
        return {"distances": self.geometry.ray_distances(position, **kwargs),
                "source": "native_static_collision_rays", **kwargs}


def build_navmesh_hierarchical_map(scene_id, navmesh, robot_config, cache_dir,
                           build_config=BuildConfig(), local_config=LocalConfig(),
                           scene=None, geometry=None, rebuild=False):
    started = time.perf_counter()
    nav = HabitatNavigation(navmesh, robot_config)
    scene_identity = scene_fingerprint(scene) if scene is not None else scene_id
    inputs = {"schema": SCHEMA_VERSION, "algorithm": ALGORITHM_VERSION,
              "scene_id": scene_id, "scene_identity": scene_identity,
              "navmesh": nav.identity, "navigation": robot_config.navigation,
              "robot_static_geometry": robot_asset_fingerprint(robot_config),
              "build": asdict(build_config), "native_geometry_features": geometry is not None,
              "habitat_sim": nav.hs.__version__}
    topology_id = digest(inputs)
    cache = Path(cache_dir).resolve()
    directory = cache / "topology" / topology_id
    cached = (directory / "manifest.json").exists() and not rebuild
    if cached:
        top = read_json(directory / "manifest.json")
        if top["inputs"] != inputs:
            raise ValueError("Topology cache input mismatch")
        for filename in ("graph.json", "surface.npz"):
            if file_digest(directory / filename) != top["checksums"][filename]:
                raise ValueError(f"Corrupt topology cache: {filename}")
        graph = read_json(directory / "graph.json")
        with np.load(directory / "surface.npz", allow_pickle=False) as archive:
            surface = Surface(nav, build_config, {k: archive[k] for k in archive.files})
        surface.floor_levels = graph["floor_levels"]
    else:
        surface = Surface(nav, build_config)
        graph = build_global_graph(surface, build_config, geometry)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "graph.json", graph)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as f:
            np.savez_compressed(f, **surface.arrays()); name = f.name
        os.replace(name, directory / "surface.npz")
        top = {"schema_version": SCHEMA_VERSION, "topology_id": topology_id, "inputs": inputs,
               "checksums": {f: file_digest(directory / f) for f in ("graph.json", "surface.npz")}}
        write_json(directory / "manifest.json", top)
    map_id = digest({"topology": topology_id, "robot": asdict(robot_config), "local": asdict(local_config)})
    manifest = {"schema_version": SCHEMA_VERSION, "map_id": map_id, "topology_id": topology_id,
                "scene_id": scene_id, "scene": str(Path(scene).resolve()) if scene else None,
                "scene_sha256": scene_identity if scene else None,
                "navmesh": nav.path, "navmesh_sha256": nav.identity,
                "topology": os.path.relpath(directory, cache / "maps"),
                "robot": asdict(robot_config), "build": asdict(build_config), "local": asdict(local_config),
                "coordinates": {"units": "metres", "up": "+Y", "horizontal": "XZ",
                                "yaw_zero": "+X", "positive_yaw": "toward -Z",
                                "position": "NavMesh ground; not articulated root",
                                "floors": "height-mode metadata; native 3D surfaces determine connectivity"}}
    filename = cache / "maps" / f"{map_id}.json"
    write_json(filename, manifest)
    result = HierarchicalMap(nav, surface, graph, robot_config, build_config, local_config,
                             map_id, topology_id, geometry, manifest)
    result.manifest_path = filename
    result.build_metrics = {"cache_hit": cached, "elapsed_seconds": time.perf_counter()-started}
    return result


def load_hierarchical_map(manifest_path, robot_config=None, geometry=None, navmesh=None):
    manifest_path = Path(manifest_path).resolve()
    m = read_json(manifest_path)
    if m.get("schema_version") == 2:
        from .robotmap import load_robot_hierarchical_map
        return load_robot_hierarchical_map(manifest_path, robot_config, geometry, navmesh)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported graph schema; rebuild from source NavMesh (no implicit migration)")
    robot = RobotConfig(**m["robot"])
    if robot_config is not None and robot_config != robot:
        raise ValueError("Robot configuration mismatch; build/load a compatible map")
    build, local = BuildConfig(**m["build"]), LocalConfig(**m["local"])
    navigation = HabitatNavigation(navmesh or m["navmesh"], robot)
    if navigation.identity != m["navmesh_sha256"]:
        raise ValueError("Source NavMesh hash mismatch")
    if m.get("scene") and scene_fingerprint(m["scene"]) != m["scene_sha256"]:
        raise ValueError("Source scene hash mismatch")
    directory = (manifest_path.parent / m["topology"]).resolve()
    top = read_json(directory / "manifest.json")
    if top.get("schema_version") != SCHEMA_VERSION or digest(top["inputs"]) != m["topology_id"]:
        raise ValueError("Incompatible topology artifact")
    if ("robot_static_geometry" in top["inputs"] and
            top["inputs"]["robot_static_geometry"] != robot_asset_fingerprint(robot)):
        raise ValueError("Robot collision assets/reference posture changed; rebuild the map")
    expected_map_id = digest({"topology": m["topology_id"], "robot": m["robot"], "local": m["local"]})
    if expected_map_id != m["map_id"]:
        raise ValueError("Map configuration identity mismatch")
    for f, checksum in top["checksums"].items():
        if file_digest(directory / f) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {f}")
    graph = read_json(directory / "graph.json")
    with np.load(directory / "surface.npz", allow_pickle=False) as archive:
        surface = Surface(navigation, build, {k: archive[k] for k in archive.files})
    surface.floor_levels = graph["floor_levels"]
    result = HierarchicalMap(navigation, surface, graph, robot, build, local, m["map_id"],
                             m["topology_id"], geometry, m)
    result.manifest_path = manifest_path
    return result


def build_hierarchical_map(scene_id, navmesh, robot_config, cache_dir,
                           build_config=BuildConfig(), local_config=LocalConfig(),
                           scene=None, geometry=None, rebuild=False, traversability_config=None):
    """Public construction uses robot feasibility before extracting topology.

    For the historical diagnostic abstraction explicitly use
    build_navmesh_hierarchical_map. Schema-1 artifacts are never promoted.
    """
    from .robotmap import build_robot_hierarchical_map, TraversabilityConfig
    return build_robot_hierarchical_map(scene_id, navmesh, robot_config, cache_dir,
               build_config, local_config, scene, geometry,
               traversability_config or TraversabilityConfig(), rebuild)
