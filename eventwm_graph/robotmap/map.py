"""Persistent robot map and feasibility-aware hierarchy/collector interfaces."""
from collections import defaultdict
from dataclasses import asdict, replace
import copy
import gzip
import heapq
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from ..config import (REPO, BuildConfig, LocalConfig, RobotConfig, digest, file_digest,
                      read_json, write_json, scene_fingerprint, robot_asset_fingerprint)
from ..local import LocalGraph
from ..native import HabitatNavigation, StaticGeometry, path_length
from ..surface import Surface
from .config import (ROBOT_SCHEMA, ROBOT_ALGORITHM, TraversabilityConfig, TRAVERSABLE,
                     NON_TRAVERSABLE, UNKNOWN, NoVerifiedRoute, angle_delta)
from .body import BodyOracle
from .roadmap import (extract_roadmap, state_adjacency, join_poses, motion_duration,
                      heading_connectors)
from .topology import build_robot_topology, predecessor_path


def simplify_verified_turns(poses):
    """Erase angular backtracking at one position using only covered arcs.

    Concatenated primitives may visit their reference heading unnecessarily.
    Every retained angle is already a tested witness, and every resulting arc
    is a subset of the original contiguous angular coverage at that position.
    No translation shortcut or new collision claim is introduced.
    """
    q = np.asarray(poses, dtype=float)
    parts, i = [], 0
    while i < len(q):
        j = i+1
        while j < len(q) and np.linalg.norm(q[j,:3]-q[i,:3]) < 1e-8:
            j += 1
        group = q[i:j]
        angles = np.unwrap(group[:,3])
        start, end = angles[0], angles[-1]
        if abs(end-start)<1e-8:
            parts.append(group[:1])
        else:
            low, high = sorted((start,end))
            retained = sorted(set(float(a) for a in angles if low+1e-8<a<high-1e-8), reverse=end<start)
            parts.append(np.array([[*group[0,:3],a] for a in (start,*retained,end)]))
        i = j
    return join_poses(parts)


def robot_identity(robot):
    # Human collision reference and local k do not alter static body feasibility.
    return digest({"assets": robot_asset_fingerprint(robot), "navigation": robot.navigation,
                   "motion": {"speed": robot.max_speed, "yaw_rate": robot.max_yaw_rate},
                   "safety_margin": robot.safety_margin})


def implementation_identity():
    import inspect
    import hashlib
    result = {name: file_digest(Path(__file__).with_name(name))
              for name in ("config.py", "body.py", "roadmap.py", "topology.py", "reuse.py")}
    result.update({"source/"+name:file_digest(Path(__file__).parents[1]/name)
                   for name in ("native.py", "surface.py", "config.py")})
    result["pose_predicate"] = hashlib.sha256('\n'.join(inspect.getsource(getattr(BodyOracle,n))
                for n in ('__init__','__enter__','pose','support','contact_records')).encode()).hexdigest()
    return result


def _write_compressed(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        # Fixed gzip timestamp and no temporary filename in the header.
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as f:
            f.write(json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode())
        name = raw.name
    os.replace(name, path)


class RobotHierarchicalMap:
    def __init__(self, navigation, surface, roadmap, graph, manifest, manifest_path, geometry=None):
        self.navigation, self.surface, self.roadmap, self.graph = navigation, surface, roadmap, graph
        self.metadata, self.manifest_path = manifest, Path(manifest_path)
        self.robot_config = RobotConfig(**manifest["robot"])
        self.local_config = LocalConfig(**manifest["local"])
        self.build_config = BuildConfig(**manifest["build"])
        self.traversability_config = TraversabilityConfig(**manifest["traversability"])
        self.map_id, self.topology_id = manifest["map_id"], manifest["traversability_id"]
        self.robot_configuration_id = manifest["robot_configuration_id"]
        self.geometry = geometry
        self.artifacts = (self.manifest_path.parent/manifest["artifacts"]).resolve()
        self.local = LocalGraph(surface, self.robot_config, self.local_config, self.topology_id)
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edges = {e["id"]: e for e in graph["edges"]}
        self.states, self.sites, self.motions = roadmap["states"], roadmap["sites"], roadmap["motions"]
        self.state_graph = state_adjacency(self.states, self.motions)
        self.pair_motion = {(m["source"], m["target"]): m["id"] for m in self.motions}
        self.state_paths = {}
        self.positions = np.array([s["position"] for s in self.states]).reshape(-1, 3)
        self.tree = cKDTree(self.positions) if len(self.positions) else None
        self.exact_states = defaultdict(list)
        for s in self.states:
            if s['status']==TRAVERSABLE:
                self.exact_states[tuple(np.round(s["position"], 6))].append(s["id"])
        self.adjacency = defaultdict(list)
        for e in graph["edges"]:
            self.adjacency[e["source"]].append((e["target"], e["id"], e["duration"]))
        self.route_cache = {}
        self.endpoint_cache = {}
        self.entry_cache = {}

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self,value):
        if value is not None:
            if value.robot_config is None or robot_identity(value.robot_config)!=self.robot_configuration_id:
                raise ValueError('Attached collision geometry belongs to a different robot configuration')
            if Path(value.scene).resolve()!=Path(self.metadata['scene']).resolve():
                raise ValueError('Attached collision geometry belongs to a different static scene')
        self._geometry=value

    def get_global_graph(self):
        return copy.deepcopy(self.graph)

    def get_traversability_map(self):
        return {"map_id": self.map_id, "traversability_id": self.topology_id,
                "robot_configuration_id": self.robot_configuration_id,
                "contract": asdict(self.traversability_config),
                "statistics": copy.deepcopy(self.roadmap["statistics"]),
                "unsupported_native_islands": copy.deepcopy(self.surface.unsupported),
                "surface_domain": "full native support retained; untested configurations UNKNOWN",
                "artifact_directory": str(self.artifacts)}

    def get_native_path(self, start, goal):
        path = self.navigation.shortest_path(start, goal)
        return {"path": None if path is None else path.tolist(), "robot_feasibility": UNKNOWN,
                "validation_level": "native_navmesh_only", "executable": False}

    def _state_path(self, start, end):
        if end not in self.state_paths:
            self.state_paths[end] = dijkstra(self.state_graph.T, directed=True, indices=end, return_predecessors=True)
        distances, predecessors = self.state_paths[end]
        if not np.isfinite(distances[start]):
            raise NoVerifiedRoute("no_verified_configuration_path", source_state=start, target_state=end)
        seq = predecessor_path(predecessors, end, start)[::-1]
        return [self.pair_motion[a, b] for a, b in zip(seq[:-1], seq[1:])]

    def _poses_for_motions(self, ids, state=None):
        if ids:
            return join_poses([self.motions[i]["poses"] for i in ids])
        return np.array([[*self.states[state]["position"], self.states[state]["yaw"]]])

    @staticmethod
    def _turn(oracle, p, start, end):
        delta = angle_delta(start, end)
        for angle in (delta, delta-2*math.pi if delta > 0 else delta+2*math.pi):
            count = max(1, math.ceil(abs(angle)/(math.pi/4)))
            poses = [[*p, float(y)] for y in np.linspace(start, start+angle, count+1)]
            proof = oracle.motion(poses)
            if proof["status"] == TRAVERSABLE:
                return np.asarray(proof["swept_poses"])
        return None

    def _connect_pose(self, position, yaw, outgoing=True, search_scale=1):
        if not math.isfinite(yaw):
            raise NoVerifiedRoute("invalid_endpoint_orientation")
        try:
            p = self.navigation.snap(position)
        except ValueError as error:
            raise NoVerifiedRoute("endpoint_outside_native_support_domain", diagnostic=str(error)) from error
        if np.linalg.norm(p-np.asarray(position))>1e-5:
            raise NoVerifiedRoute("endpoint_requires_explicit_surface_projection",
                                  requested_position=list(position), projected_position=p.tolist(),
                                  projection_distance=float(np.linalg.norm(p-np.asarray(position))))
        q = [*p.tolist(), float(angle_delta(0, yaw))]
        radius = self.traversability_config.endpoint_radius*search_scale
        neighbors = self.traversability_config.endpoint_neighbors*search_scale**2
        key = digest({"foundation": self.topology_id, "pose": q, "outgoing": outgoing,
                      "attachment_algorithm": "distinct-site-budget-2",
                      "radius": radius, "neighbors": neighbors})
        if key in self.endpoint_cache:
            return self.endpoint_cache[key]
        filename = self.artifacts/"endpoints"/(key+".json")
        if filename.exists():
            result = read_json(filename)
            if result["identity"] != key or result["content_sha256"] != digest(result["evidence"]):
                raise ValueError("Endpoint feasibility cache checksum mismatch")
            self.endpoint_cache[key] = result["evidence"]
            return result["evidence"]
        exact = [i for i in self.exact_states.get(tuple(np.round(p, 6)), [])
                 if abs(angle_delta(self.states[i]["yaw"], yaw)) < 1e-7 and
                 np.linalg.norm(np.asarray(self.states[i]["position"])-p) < 1e-6]
        if exact:
            result = {"status": TRAVERSABLE, "endpoint_key": key,
                      "connections": [{"state": i, "poses": [q], "duration": 0.} for i in exact],
                      "source": "cached_exact_configuration"}
        else:
            if self.geometry is None:
                raise NoVerifiedRoute("endpoint_not_preverified_attach_geometry_or_use_cached_start", pose=q)
            from .refinement import WitnessOracle, blocking_witness
            witnesses=self.roadmap.get('stronger_contact_witnesses',[])
            with WitnessOracle(self.geometry, self.navigation, self.traversability_config,witnesses=witnesses) as oracle:
                verdict = oracle.pose(p, yaw)
                if verdict["status"] != TRAVERSABLE:
                    raise NoVerifiedRoute("endpoint_pose_not_feasible", verdict["status"], pose=q, evidence=verdict)
                nearby = self.tree.query_ball_point(p, radius) if self.tree else []
                nearby = sorted(nearby, key=lambda i: (np.linalg.norm(self.positions[i]-p), i))
                # A site can have many disconnected heading states. Counting
                # each as a spatial neighbor exhausted the budget on fringe
                # poses and missed valid nearby connections in the y9 scene.
                site_order=list(dict.fromkeys(self.states[i]['site'] for i in nearby))[:neighbors]
                nearby=[i for i in nearby if self.states[i]['site'] in site_order and self.states[i]['status']==TRAVERSABLE]
                connections, failures = [], []
                by_site = defaultdict(list)
                for i in nearby:
                    by_site[self.states[i]["site"]].append(self.states[i])
                for site_id, states in sorted(by_site.items()):
                    site = self.sites[site_id]; s = np.asarray(site["position"])
                    if self.navigation.island(p) != site["island"] or not self.navigation.segment_free(p, s):
                        continue
                    a, b = (p, s) if outgoing else (s, p)
                    if np.linalg.norm(a-b) < 1e-7:
                        for state in states:
                            turn = self._turn(oracle, p, yaw if outgoing else state["yaw"], state["yaw"] if outgoing else yaw)
                            if turn is not None:
                                connections.append({"state": state["id"], "poses": turn.tolist(),
                                                    "duration": motion_duration(turn, self.robot_config)})
                        continue
                    heading = math.atan2(a[2]-b[2], b[0]-a[0])
                    transit = oracle.motion([[*a, heading], [*b, heading]])
                    if transit["status"] != TRAVERSABLE:
                        failures.append({k: v for k, v in transit.items() if k != "swept_poses"}); continue
                    endpoint_turn = self._turn(oracle, p, yaw if outgoing else heading, heading if outgoing else yaw)
                    if endpoint_turn is None:
                        continue
                    for state_id, stop_to_heading in heading_connectors(site, states, heading, oracle):
                        poses = (join_poses([endpoint_turn, transit["swept_poses"], stop_to_heading[::-1]]) if outgoing else
                                 join_poses([stop_to_heading, transit["swept_poses"], endpoint_turn]))
                        if blocking_witness(poses,witnesses) is not None:
                            continue
                        connections.append({"state": state_id, "poses": poses.tolist(),
                                            "duration": motion_duration(poses, self.robot_config)})
                result = {"status": TRAVERSABLE if connections else UNKNOWN, "endpoint_key": key,
                          "connections": connections, "failures": failures,
                          "source": "native_body_support_endpoint_primitives",
                          "reason": "verified_connections" if connections else "no_verified_connector_within_sampling_budget"}
        write_json(filename, {"identity": key, "evidence": result, "content_sha256": digest(result)})
        self.endpoint_cache[key] = result
        return result

    def _entries(self, position, yaw, outgoing=True, search_scale=1):
        evidence = self._connect_pose(position, yaw, outgoing, search_scale)
        cache_key = (evidence["endpoint_key"], outgoing)
        if cache_key in self.entry_cache:
            return self.entry_cache[cache_key]
        result = {}
        for connector in evidence["connections"]:
            state = connector["state"]
            for node in self.graph["nodes"]:
                if node["weak_component"] != self.states[state]["weak_component"]:
                    continue
                anchor = node["state_id"]
                try:
                    ids = self._state_path(state, anchor) if outgoing else self._state_path(anchor, state)
                except NoVerifiedRoute:
                    continue
                duration = connector["duration"]+sum(self.motions[i]["duration"] for i in ids)
                if node["id"] not in result or duration < result[node["id"]]["duration"]:
                    result[node["id"]] = {"connector": connector, "duration": duration,
                                         "endpoint_key": evidence["endpoint_key"], "state": state, "motion_ids": ids}
        # Local attachment should not replace the selected global route with a
        # direct trip to every anchor. Preserve a closest attachment in EACH
        # directed component: a cheap one-way fringe anchor must not erase an
        # accessible core anchor and create a false disconnection.
        if result:
            best = {}
            for key, value in result.items():
                component = self.nodes[key]["strong_component"]
                best[component] = min(best.get(component, math.inf), value["duration"])
            result = {k: v for k, v in result.items()
                      if v["duration"] <= best[self.nodes[k]["strong_component"]]+2.0}
        if not result:
            # An empty bounded attachment search is incomplete evidence. A
            # second finite search recovered an old-valid y9 request without
            # moving either endpoint or weakening any native check. Its
            # radius/budget already participate in the endpoint cache key.
            if search_scale==1:
                return self._entries(position,yaw,outgoing,search_scale=2)
            raise NoVerifiedRoute("no_verified_entry_to_robot_topology", endpoint_evidence=evidence)
        for entry in result.values():
            connector = entry.pop("connector")
            motions = self._poses_for_motions(entry["motion_ids"], entry["state"])
            entry["poses"] = join_poses([connector["poses"], motions] if outgoing else [motions, connector["poses"]]).tolist()
        self.entry_cache[cache_key] = result
        return result

    def _global_search(self, entries):
        costs = {n: e["duration"] for n, e in entries.items()}
        previous = {n: None for n in entries}
        queue = [(cost, node) for node, cost in costs.items()]; heapq.heapify(queue)
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            for other, edge, duration in self.adjacency[node]:
                new = cost+duration
                if new < costs.get(other, math.inf)-1e-8:
                    costs[other] = new; previous[other] = (node, edge)
                    heapq.heappush(queue, (new, other))
        return costs, previous

    def get_global_candidates(self, position, constraints=None, start_yaw=0.):
        constraints = constraints or {}
        allowed = {"limit", "min_distance", "max_distance", "types", "floor_id", "require_native_pose", "returnable"}
        if set(constraints)-allowed:
            raise ValueError(f"Unknown candidate constraints: {set(constraints)-allowed}")
        if len(position) == 4:
            start_yaw, position = position[3], position[:3]
        entries = self._entries(position, start_yaw)
        costs, _ = self._global_search(entries)
        returnable = None
        if constraints.get("returnable"):
            try:
                incoming = self._entries(position, start_yaw, outgoing=False)
            except NoVerifiedRoute:
                return []
            reverse = defaultdict(list)
            for edge in self.edges.values():
                reverse[edge["target"]].append(edge["source"])
            returnable, pending = set(incoming), list(incoming)
            while pending:
                for other in reverse[pending.pop()]:
                    if other not in returnable:
                        returnable.add(other); pending.append(other)
        result = []
        for node_id, cost in costs.items():
            node = self.nodes[node_id]
            if constraints.get("types") and node["type"] not in constraints["types"]:
                continue
            if "floor_id" in constraints and constraints["floor_id"] != node["floor_id"]:
                continue
            if returnable is not None and node_id not in returnable:
                continue
            native = self.navigation.shortest_path(position, node["position"])
            if native is None:
                continue
            distance = path_length(native)
            if not constraints.get("min_distance", 0.) <= distance <= constraints.get("max_distance", math.inf):
                continue
            result.append({"node_id": node_id, "geodesic_distance": distance,
                           "robot_route_duration": cost, "type": node["type"], "floor_id": node["floor_id"],
                           "robot_feasibility": TRAVERSABLE, "reachable_from_supplied_pose": True,
                           "validation_level": self.traversability_config.validation_level})
        return sorted(result, key=lambda x: (x["robot_route_duration"], x["node_id"]))[:constraints.get("limit", 12)]

    def get_graph_route(self, start, target_node, start_yaw=0., target_pose=None):
        if len(start) == 4:
            start_yaw, start = start[3], start[:3]
        entries = self._entries(start, start_yaw)
        exits = None
        if target_pose is not None:
            exits = self._entries(target_pose[:3], target_pose[3], outgoing=False)
        elif target_node not in self.nodes:
            raise KeyError(target_node)
        costs, previous = self._global_search(entries)
        if exits is not None:
            options = [(costs[n]+v["duration"], n) for n, v in exits.items() if n in costs]
            if not options:
                raise NoVerifiedRoute("no_verified_robot_route_between_endpoint_components")
            _, target_node = min(options)
        elif target_node not in costs:
            raise NoVerifiedRoute("target_not_reachable_in_verified_robot_graph", target_node=target_node)
        nodes, edges = [target_node], []
        while previous[nodes[-1]] is not None:
            parent, edge = previous[nodes[-1]]; nodes.append(parent); edges.append(edge)
        nodes.reverse(); edges.reverse()
        entry = entries[nodes[0]]
        mid = [self._poses_for_motions(self.edges[e]["motion_ids"]) for e in edges]
        tail = exits[target_node] if exits is not None else None
        poses = simplify_verified_turns(join_poses([entry["poses"], *mid, *([tail["poses"]] if tail else [])]))
        record = {"schema_version": ROBOT_SCHEMA, "map_id": self.map_id,
                  "robot_configuration_id": self.robot_configuration_id, "traversability_id": self.topology_id,
                  "start": list(start), "start_yaw": float(start_yaw), "target_node": target_node,
                  "target_pose": poses[-1].tolist(), "node_ids": nodes, "edge_ids": edges,
                  "entry": entry, "exit": tail, "configuration_path": poses.tolist(),
                  "length": path_length(poses[:, :3]), "duration": motion_duration(poses, self.robot_config),
                  "robot_feasibility": TRAVERSABLE, "validation_level": self.traversability_config.validation_level,
                  "execution_certified": False}
        route_id = digest(record)
        filename = self.artifacts/"routes"/(route_id+".json")
        if not filename.exists():
            write_json(filename, record)
        route = {**record, "route_id": route_id}
        self.route_cache[route_id] = record
        return route

    def compile_robot_path(self, route):
        route_id = route.get("route_id")
        if not isinstance(route_id, str) or len(route_id) != 64 or any(c not in "0123456789abcdef" for c in route_id):
            raise ValueError("Missing verified route identity")
        record = {k: v for k, v in route.items() if k != "route_id"}
        if digest(record) != route_id:
            raise ValueError("Route content differs from its feasibility certificate")
        cached = self.route_cache.get(route_id)
        if cached is None:
            path = self.artifacts/"routes"/(route_id+".json")
            if not path.exists():
                raise NoVerifiedRoute("route_has_no_cached_feasibility_evidence")
            cached = read_json(path)
        if digest(cached) != route_id or cached != record:
            raise ValueError("Route cache checksum mismatch")
        if route["map_id"] != self.map_id or route["robot_configuration_id"] != self.robot_configuration_id:
            raise ValueError("Route belongs to a different robot map")
        poses = np.asarray(route["configuration_path"])
        path = [poses[0, :3]]+[q[:3] for p, q in zip(poses[:-1], poses[1:]) if np.linalg.norm(q[:3]-p[:3]) > 1e-8]
        return {"path": np.asarray(path).tolist(), "poses": poses.tolist(),
                "robot_feasibility": TRAVERSABLE, "validation_level": route["validation_level"],
                "source": "cached_verified_body_primitives", "execution_certified": False}

    def compile_navmesh_path(self, route):
        """Compatibility name; schema-2 maps compile robot-verified motions."""
        return np.asarray(self.compile_robot_path(route)["path"])

    def make_visit_query(self, start, target_node, start_yaw=0., target_yaw=None, speed=None, target_pose=None):
        if target_yaw is not None:
            if target_pose is None:
                target_pose = [*self.nodes[target_node]["position"], target_yaw]
        route = self.get_graph_route(start, target_node, start_yaw, target_pose)
        return self.query_from_route(route, speed)

    def query_from_route(self, route, speed=None):
        speed = self.robot_config.max_speed if speed is None else float(speed)
        if not 0 < speed <= self.robot_config.max_speed:
            raise ValueError("Speed exceeds configured motion contract")
        compiled = self.compile_robot_path(route)
        poses = np.asarray(compiled["poses"])
        time_value, yaw = 0., poses[0, 3]
        timed = [[time_value, *poses[0, :3], yaw]]
        for a, b in zip(poses[:-1], poses[1:]):
            delta = angle_delta(a[3], b[3])
            time_value += np.linalg.norm(b[:3]-a[:3])/speed+abs(delta)/self.robot_config.max_yaw_rate
            yaw += delta
            timed.append([float(time_value), *b[:3].tolist(), float(yaw)])
        return {"schema_version": ROBOT_SCHEMA, "map_id": self.map_id,
                "robot_configuration_id": self.robot_configuration_id, "target_node": route["target_node"],
                "route": route, "path": compiled["path"], "timed_poses": timed, "target_yaw": float(yaw),
                "robot_feasibility": TRAVERSABLE, "validation_level": compiled["validation_level"],
                "executable_under_stated_contract": True, "execution_certified": False,
                "motion": {"kind": "stop_turn_forward", "speed": speed, "max_yaw_rate": self.robot_config.max_yaw_rate,
                           "articulation": "reference", "controller_adapter": "Falcon bounded execution evaluated separately"}}

    def validate_query_path(self, query, robot_config=None, spatial_step=None, angle_step=None, recheck=False):
        if robot_config is not None and robot_config != self.robot_config:
            raise ValueError("Robot configuration mismatch")
        if not isinstance(query, dict) or "route" not in query:
            return {"valid": False, "robot_feasibility": UNKNOWN, "validation_level": "native_navmesh_only",
                    "reason": "robot_feasibility_requires_orientation_and_cached_motion_route"}
        compiled = self.compile_robot_path(query["route"])
        reference = np.asarray(compiled["poses"])
        timed = np.asarray(query["timed_poses"], dtype=float)
        aligned = (timed.shape == (len(reference), 5) and np.isfinite(timed).all() and
                   np.allclose(timed[:, 1:4], reference[:, :3], atol=1e-7) and
                   np.allclose(np.sin(timed[:, 4]-reference[:, 3]), 0., atol=1e-7) and
                   np.allclose(np.cos(timed[:, 4]-reference[:, 3]), 1., atol=1e-7) and
                   np.asarray(query["path"]).shape == np.asarray(compiled["path"]).shape and
                   np.allclose(query["path"], compiled["path"], atol=1e-7))
        valid = bool(aligned and query["map_id"] == self.map_id and
                     query["robot_configuration_id"] == self.robot_configuration_id and
                     query["target_node"] == query["route"]["target_node"] and
                     abs(angle_delta(query["target_yaw"], reference[-1, 3])) < 1e-7)
        if aligned:
            dt = np.diff(timed[:, 0]); dist = np.linalg.norm(np.diff(timed[:, 1:4], axis=0), axis=1)
            valid &= bool(np.all(dt > 0) and np.all(dist <= self.robot_config.max_speed*dt+1e-6) and
                          np.all(abs(np.diff(timed[:, 4])) <= self.robot_config.max_yaw_rate*dt+1e-6))
        result = {"valid": valid, "motion_valid": valid, "robot_feasibility": TRAVERSABLE if valid else UNKNOWN,
                  "validation_level": self.traversability_config.validation_level,
                  "level": self.traversability_config.validation_level,
                  "evidence": "reused immutable body/support motion samples", "fresh_native": None,
                  "execution_certified": False, "continuous_collision_certified": False,
                  "physical_safety_margin_certified": False}
        if recheck and valid:
            if self.geometry is None:
                raise ValueError("Fresh body validation requires StaticGeometry")
            with BodyOracle(self.geometry, self.navigation, self.traversability_config) as oracle:
                proof = oracle.motion(reference, spatial_step, angle_step, use_cache=False, project_to_surface=False)
            result["fresh_native"] = {k: v for k, v in proof.items() if k != "swept_poses"}
            result["valid"] &= proof["status"] == TRAVERSABLE
            result["robot_feasibility"] = proof["status"] if not result["valid"] else TRAVERSABLE
        return result

    def _annotate_patch(self, patch):
        by_cell = defaultdict(list)
        h = self.local.resolution
        for state in self.states:
            if state['status']!=TRAVERSABLE:continue
            key = tuple(np.floor(np.asarray(state["position"])[[0, 2]]/h).astype(int))
            by_cell[key].append(state)
        included = set()
        for node in patch["nodes"]:
            faces = {s["surface_id"] for s in node["support"]}
            ids = [s["id"] for s in by_cell[tuple(node["cell"])] if s["surface_id"] in faces]
            node["robot_state_witnesses"] = ids
            node["robot_continuous_region_status"] = UNKNOWN
            node["human_occupancy_support_retained"] = True
            included.update(ids)
        for edge in patch["edges"]:
            edge["robot_feasibility"] = UNKNOWN
            edge["scope"] = "native spatial adjacency; not an executable robot transition"
        patch["robot_motions"] = [{"id": m["id"], "source_state": m["source"], "target_state": m["target"],
                                    "robot_feasibility": TRAVERSABLE, "traversability_id": self.topology_id,
                                    "compiler_ref": {"kind": "verified_robot_motion", "motion_id": m["id"]}}
                                   for m in self.motions if m["source"] in included and m["target"] in included]
        patch["robot_states"] = [copy.deepcopy(self.states[i]) for i in sorted(included)]
        patch["robot_map_id"] = self.map_id
        patch["support_semantics"] = "full native occupancy support plus separate cached robot configuration witnesses"
        return patch

    def get_local_patch(self, position, radius, floor_context=None):
        return self._annotate_patch(copy.deepcopy(self.local.get_local_graph_patch(position, radius, floor_context)))

    def get_local_nodes_for_path(self, path, radius=None):
        # Transit height bridges have separately verified body configurations.
        # Occupancy support remains on native surfaces; project only this
        # association query, never the executable robot geometry.
        support_path=[self.navigation.project_path_sample(p).tolist() for p in path]
        patch=self._annotate_patch(copy.deepcopy(self.local.get_local_nodes_for_path(support_path, radius)))
        patch['association_path_scope']='native support projection of unchanged robot transit geometry'
        return patch

    def get_associated_local_nodes(self, node_id):
        faces = set(self.nodes[node_id]["support_surface_ids"])
        cells, result = set(), []
        h = self.local.resolution
        for face in faces:
            low = np.floor(self.surface.low[face, [0, 2]]/h).astype(int)
            high = np.floor(self.surface.high[face, [0, 2]]/h).astype(int)
            cells.update((x, z) for x in range(low[0], high[0]+1) for z in range(low[1], high[1]+1))
        if len(cells) > self.local_config.max_patch_cells:
            raise ValueError("Association exceeds local budget; use an ROI")
        for cell in sorted(cells):
            for node in self.local._cell(*cell):
                area = sum(s["area"] for s in node["support"] if s["surface_id"] in faces)
                if area:
                    result.append({"node": copy.deepcopy(node), "association_weight": area/node["area"]})
        return result

    def get_static_clearance(self, position):
        return self.navigation.clearance(position, self.build_config.clearance_search)

    def get_visibility_geometry(self, position=None, **kwargs):
        if position is None:
            return {"scene": self.metadata["scene"], "loaded": self.geometry is not None,
                    "type": "native_static_collision_mesh"}
        if self.geometry is None:
            raise ValueError("Native visibility requires attached scene geometry")
        return {"distances": self.geometry.ray_distances(position, **kwargs), "source": "static_collision_rays", **kwargs}

    def request_visit(self, start_pose=None, target_node=None, index=0, constraints=None, local_radius=.4):
        """Collector boundary: cached start, reachable target, route and support."""
        if start_pose is None:
            starts = [n for n in self.graph["nodes"] if self.adjacency[n["id"]]]
            if not starts:
                raise NoVerifiedRoute("no_cached_transit_start")
            node = starts[index % len(starts)]
            start_pose = [*node["position"], node["yaw"]]
        if target_node is None:
            options = self.get_global_candidates(start_pose, constraints or {"min_distance": 1., "max_distance": 8., "limit": 12})
            if not options:
                raise NoVerifiedRoute("no_reachable_verified_target")
            target_node = options[index % len(options)]["node_id"]
        query = self.make_visit_query(start_pose, target_node)
        return {"start_pose": list(start_pose), "target_node": target_node, "query": query,
                "start_articulation": {"kind":"reference", "joint_positions":self.roadmap['calibration']['reference_joints'],
                                       "animated_gait_phase":None},
                "feasibility": self.validate_query_path(query),
                "collector_admission": "reference_posture_geometry_only; actual action execution is not certified",
                "local_support": self.get_local_nodes_for_path(query["path"], local_radius)}


def _build_robot_hierarchical_map(scene_id, navmesh, robot_config, cache_dir,
                                 build_config=BuildConfig(), local_config=LocalConfig(),
                                 scene=None, geometry=None, traversability_config=TraversabilityConfig(),
                                 rebuild=False, progress=None):
    began = time.perf_counter()
    if scene is None:
        raise ValueError("Robot traversability requires source static scene geometry")
    if geometry is not None:
        if geometry.robot_config is None or robot_identity(geometry.robot_config)!=robot_identity(robot_config):
            raise ValueError('Construction geometry does not match the requested robot configuration')
        if Path(geometry.scene).resolve()!=Path(scene).resolve():
            raise ValueError('Construction geometry does not match the requested scene')
    navigation = HabitatNavigation(navmesh, robot_config)
    inputs = {"schema": ROBOT_SCHEMA, "algorithm": ROBOT_ALGORITHM, "scene": scene_fingerprint(scene),
              "scene_id": scene_id, "navmesh": navigation.identity, "robot": robot_identity(robot_config),
              "traversability": asdict(traversability_config), "surface": asdict(build_config),
              "implementation": implementation_identity(), "habitat_sim": navigation.hs.__version__}
    from .reuse import find_pose_evidence, seed_exact_poses
    source = find_pose_evidence(cache_dir, inputs)
    inputs["reused_exact_pose_source"] = source.name if source is not None else None
    from .pose_batches import find_batches,seed_batches,batch_identity
    batches=find_batches(cache_dir,inputs)
    if batches is not None:
        inputs['native_pose_cache']={'content_id':batch_identity(read_json(batches))}
    identity = digest(inputs)
    cache = Path(cache_dir).resolve(); folder = cache/"robot_traversability"/identity
    cached = (folder/"manifest.json").exists() and not rebuild
    if not cached:
        if geometry is None:
            with StaticGeometry(scene, robot_config) as native:
                result = _build_robot_hierarchical_map(scene_id, navmesh, robot_config, cache_dir, build_config,
                           local_config, scene, native, traversability_config, rebuild, progress)
                result.geometry = None
                return result
        with BodyOracle(geometry, navigation, traversability_config) as oracle:
            reused = seed_exact_poses(oracle, source) if source is not None else 0
            if batches is not None:
                reused+=seed_batches(oracle,batches)
            if progress and reused:
                progress({"phase":"reuse_exact_pose_evidence", "poses":reused,"source":source.name,
                          "coarse_motion_validity_reused":False})
            surface, roadmap = extract_roadmap(navigation, geometry, robot_config, build_config,
                                                traversability_config, oracle, progress)
            roadmap["statistics"]["reused_exact_pose_count"] = reused
            graph = build_robot_topology(surface, roadmap, traversability_config, geometry)
        folder.mkdir(parents=True, exist_ok=True)
        _write_compressed(folder/"roadmap.json.gz", roadmap)
        write_json(folder/"graph.json", graph)
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as f:
            np.savez_compressed(f, **surface.arrays()); name = f.name
        os.replace(name, folder/"surface.npz")
        write_json(folder/"manifest.json", {"schema_version": ROBOT_SCHEMA, "inputs": inputs,
                   "checksums": {n: file_digest(folder/n) for n in ("roadmap.json.gz", "graph.json", "surface.npz")}})
    map_id = digest({"traversability": identity, "robot": asdict(robot_config), "local": asdict(local_config)})
    manifest_path = cache/"maps"/(map_id+".json")
    metadata = {"schema_version": ROBOT_SCHEMA, "foundation": "robot_traversability", "map_id": map_id,
                "traversability_id": identity, "robot_configuration_id": inputs["robot"],
                "scene_id": scene_id, "scene": str(Path(scene).resolve()), "scene_sha256": inputs["scene"],
                "navmesh": navigation.path, "navmesh_sha256": navigation.identity,
                "robot": asdict(robot_config), "build": asdict(build_config), "local": asdict(local_config),
                "traversability": asdict(traversability_config),
                "artifacts": os.path.relpath(folder, manifest_path.parent),
                "coordinates": {"units": "metres", "up": "+Y", "horizontal": "XZ", "yaw_zero": "+X",
                                "positive_yaw": "toward -Z", "position": "native ground support", "time": "seconds"}}
    write_json(manifest_path, metadata)
    result = load_robot_hierarchical_map(manifest_path, geometry=geometry)
    result.build_metrics = {"cache_hit": cached, "elapsed_seconds": time.perf_counter()-began}
    return result


def load_robot_hierarchical_map(manifest_path, robot_config=None, geometry=None, navmesh=None):
    manifest_path = Path(manifest_path).resolve()
    m = read_json(manifest_path)
    if m.get("schema_version") != ROBOT_SCHEMA or m.get("foundation") != "robot_traversability":
        raise ValueError("Not a schema-2 robot map; rebuild explicitly from source geometry")
    robot = RobotConfig(**m["robot"])
    if robot_config is not None and robot_config != robot:
        raise ValueError("Robot configuration mismatch")
    navigation = HabitatNavigation(navmesh or m["navmesh"], robot)
    if navigation.identity != m["navmesh_sha256"] or scene_fingerprint(m["scene"]) != m["scene_sha256"]:
        raise ValueError("Source navigation or scene changed")
    if robot_identity(robot) != m["robot_configuration_id"]:
        raise ValueError("Robot collision geometry or motion contract changed")
    expected = digest({"traversability": m["traversability_id"], "robot": m["robot"], "local": m["local"]})
    if expected != m["map_id"]:
        raise ValueError("Robot map configuration identity mismatch")
    folder = (manifest_path.parent/m["artifacts"]).resolve()
    artifact = read_json(folder/"manifest.json")
    if digest(artifact["inputs"]) != m["traversability_id"] or artifact["inputs"]["implementation"] != implementation_identity():
        raise ValueError("Traversability implementation changed; explicit rebuild required")
    inputs = artifact["inputs"]
    if 'native_pose_cache' in inputs:
        from .pose_batches import find_batches
        find_batches(folder.parents[1],inputs)
    if 'refinement' in inputs:
        from .refinement import REFINEMENT_VERSION
        if (inputs['refinement'].get('version')!=REFINEMENT_VERSION or
            inputs['refinement'].get('implementation')!=file_digest(Path(__file__).with_name('refinement.py')) or
            inputs['refinement']!=m.get('refinement')):
            raise ValueError('Stronger-evidence revision changed; explicit rebuild required')
    if (artifact.get("schema_version") != ROBOT_SCHEMA or
        inputs["traversability"] != m["traversability"] or inputs["surface"] != m["build"] or
        inputs["navmesh"] != m["navmesh_sha256"] or inputs["scene"] != m["scene_sha256"] or
        inputs["scene_id"] != m["scene_id"] or inputs["robot"] != m["robot_configuration_id"]):
        raise ValueError("Manifest disagrees with the cached traversability contract")
    for filename, checksum in artifact["checksums"].items():
        if file_digest(folder/filename) != checksum:
            raise ValueError(f"Traversability artifact checksum mismatch: {filename}")
    with gzip.open(folder/"roadmap.json.gz", "rt") as f:
        roadmap = json.load(f)
    graph = read_json(folder/"graph.json")
    cfg = replace(BuildConfig(**m["build"]), surface_max_edge=m["traversability"]["surface_max_edge"])
    with np.load(folder/"surface.npz", allow_pickle=False) as data:
        surface = Surface(navigation, cfg, {k: data[k] for k in data.files})
    # Legacy Surface arrays omit unsupported-island diagnostics. Recover them
    # from the checksummed native source instead of hiding degenerate islands
    # on a reload. A missing usable island is an artifact error, not a blocked
    # robot region.
    missing_islands = set(range(navigation.pf.num_islands))-set(map(int,surface.islands))
    if missing_islands:
        _, _, surface.unsupported = navigation.triangles()
        if missing_islands-{int(row['island']) for row in surface.unsupported}:
            raise ValueError('Serialized surface omits a usable native island')
    surface.floor_levels = graph["floor_levels"]
    return RobotHierarchicalMap(navigation, surface, roadmap, graph, m, manifest_path, geometry)


def build_robot_hierarchical_map(*args, stronger_evidence_audit=True, **kwargs):
    """Build the robot roadmap, audit it, then construct the published topology."""
    source=_build_robot_hierarchical_map(*args,**kwargs)
    if not stronger_evidence_audit:
        return source  # Explicit diagnostic checkpoint; never a silent fallback.
    from .refinement import refine_robot_map
    return refine_robot_map(source,progress=kwargs.get('progress'))
