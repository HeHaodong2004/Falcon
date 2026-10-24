"""Independent Habitat navigation and optional native static collision geometry."""
from pathlib import Path
import math
import numpy as np

from .config import REPO, file_digest


def path_length(points):
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def samples(points, step):
    p = np.asarray(points, dtype=float)
    if len(p) == 1:
        yield p[0]
    for a, b in zip(p[:-1], p[1:]):
        count = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0, 1, count + 1):
            yield a + t * (b - a)


class HabitatNavigation:
    def __init__(self, navmesh, robot):
        import habitat_sim
        self.hs = habitat_sim
        self.path = str(Path(navmesh).resolve())
        self.pf = habitat_sim.PathFinder()
        if not self.pf.load_nav_mesh(self.path):
            raise ValueError(f"Cannot load NavMesh: {navmesh}")
        self.identity = file_digest(self.path)
        self.settings = {k: getattr(self.pf.nav_mesh_settings, k) for k in
                         dir(self.pf.nav_mesh_settings) if not k.startswith("_")
                         and isinstance(getattr(self.pf.nav_mesh_settings, k),
                                        (int, float, bool))}
        for key, expected in (("agent_radius", robot.nav_radius),
                              ("agent_height", robot.nav_height),
                              ("agent_max_climb", robot.max_climb),
                              ("agent_max_slope", robot.max_slope)):
            if not np.isclose(self.settings[key], expected, rtol=0, atol=1e-5):
                raise ValueError(f"Incompatible {key}: mesh={self.settings[key]}, "
                                 f"robot={expected}. Rebuild in the Stage 0 cache.")

    def snap(self, p, island=-1, max_distance=0.08):
        p = np.asarray(p, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Expected finite world XYZ")
        q = np.asarray(self.pf.snap_point(p, island), dtype=float)
        if not np.isfinite(q).all() or np.linalg.norm(q - p) > max_distance:
            raise ValueError("Position is outside the requested NavMesh surface")
        return q

    def island(self, p):
        return int(self.pf.get_island(self.snap(p)))

    def shortest_path(self, a, b):
        a, b = self.snap(a), self.snap(b)
        if self.island(a) != self.island(b):
            return None
        if np.linalg.norm(a - b) < 1e-7:
            return np.array([a])
        request = self.hs.ShortestPath()
        request.requested_start, request.requested_end = a, b
        if not self.pf.find_path(request):
            return None
        p = np.asarray(request.points, dtype=float)
        if not np.isfinite(p).all() or len(p) == 0:
            return None
        return p

    def segment_free(self, a, b, tolerance=0.002):
        """Bidirectional native traversal with Falcon's sliding-enabled semantics.

        No-sliding is tried first. Sliding may resolve a polygon-boundary
        numerical trap only if the native geodesic is also straight-length.
        A long detour behind a wall cannot satisfy this test.
        """
        a, b = np.asarray(a), np.asarray(b)
        if not np.isfinite([a, b]).all():
            return False
        def reached(start, target):
            delta = np.asarray(self.pf.try_step_no_sliding(start, target)) - target
            # Detour's moveAlongSurface height and straight-path interpolation
            # differ on detail triangles/stairs. Only native climb-sized vertical
            # correction is allowed; horizontal wall clipping stays strict.
            if (np.linalg.norm(delta[[0, 2]]) < tolerance and
                    abs(delta[1]) <= self.settings["agent_max_climb"] + 0.025):
                return True
            delta = np.asarray(self.pf.try_step(start, target)) - target
            if np.linalg.norm(delta[[0, 2]]) >= tolerance or abs(delta[1]) > self.settings["agent_max_climb"]+.025:
                return False
            request = self.hs.ShortestPath()
            request.requested_start, request.requested_end = start, target
            return bool(self.pf.find_path(request) and
                        request.geodesic_distance <= np.linalg.norm(target-start)*1.01 + .001)
        return reached(a, b) and reached(b, a)

    def validate_path(self, points, step=0.05):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all():
            return {"valid": False, "reason": "invalid_points"}
        try:
            islands = {self.island(x) for x in p}
            if len(islands) != 1:
                return {"valid": False, "reason": "island_change"}
            prev = None
            checked = 0
            for q in samples(p, step):
                snapped = self.snap(q, max_distance=self.settings["agent_max_climb"] + 0.025)
                if np.linalg.norm((snapped-q)[[0, 2]]) > 0.002:
                    return {"valid": False, "reason": "sample_outside_surface"}
                if prev is not None and not self.segment_free(prev, q):
                    return {"valid": False, "reason": "native_step_blocked", "sample": checked,
                            "from": prev.tolist(), "to": q.tolist(),
                            "stepped": np.asarray(self.pf.try_step_no_sliding(prev, q)).tolist()}
                prev = q
                checked += 1
            return {"valid": True, "checked_samples": checked,
                    "scope": "native_navmesh_center_path", "island": islands.pop()}
        except ValueError as e:
            return {"valid": False, "reason": str(e)}

    def clearance(self, p, search=10.0):
        q = self.snap(p)
        d = float(self.pf.distance_to_closest_obstacle(q, search))
        return {"navmesh_boundary_distance": d, "censored": d >= search - 1e-5,
                "search_radius": search, "physical_clearance": None}

    def project_path_sample(self, position):
        """Correct only the height detail of an interpolated native path sample."""
        p = np.asarray(position, dtype=float)
        q = self.snap(p, max_distance=self.settings["agent_max_climb"]+0.025)
        if np.linalg.norm((q-p)[[0, 2]]) > 0.002:
            raise ValueError("Path sample requires a horizontal obstacle crossing")
        return q

    def triangles(self):
        """Extract per island: identical projected coordinates never imply adjacency."""
        triangles, islands, unsupported = [], [], []
        for island in range(self.pf.num_islands):
            before = len(triangles)
            vertices = np.asarray(self.pf.build_navmesh_vertices(island), dtype=float)
            indices = np.asarray(self.pf.build_navmesh_vertex_indices(island), dtype=int)
            if len(indices) == 0:
                unsupported.append({"island": island, "area": self.pf.island_area(island),
                                    "reason": "no_exported_triangles"})
                continue
            for tri in vertices[indices.reshape(-1, 3)]:
                if np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) < 1e-10:
                    continue
                triangles.append(tri)
                islands.append(island)
            if len(triangles) == before:
                unsupported.append({"island": island, "area": self.pf.island_area(island),
                                    "reason": "only_degenerate_exported_triangles"})
        return np.asarray(triangles).reshape(-1, 3, 3), np.asarray(islands, dtype=int), unsupported


class StaticGeometry:
    """Renderer-free static scene. Never loads humans or steps simulation physics."""
    def __init__(self, scene, robot=None, dataset="default"):
        import habitat_sim
        self.hs = habitat_sim
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(Path(scene).resolve()) if scene != "NONE" else "NONE"
        cfg.scene_dataset_config_file = dataset
        cfg.enable_physics = True
        cfg.create_renderer = False
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(
            cfg, [habitat_sim.agent.AgentConfiguration()]))
        self.scene = cfg.scene_id
        self.robot = None
        self.pose_cache = {}
        self.robot_config = robot
        self.last_collision = None
        if robot is not None:
            from types import SimpleNamespace
            from habitat.articulated_agents.robots.spot_robot import SpotRobot
            if "hab_spot_arm" not in robot.robot_urdf:
                self.close()
                raise ValueError("Native pose adapter currently supports Spot; supply another adapter")
            self.robot = SpotRobot(SimpleNamespace(
                articulated_agent_urdf=str((REPO / robot.robot_urdf).resolve())), self.sim)
            self.robot.reconfigure()
            self.robot.reset()
            self.robot.base_rot = 0.0
            self.robot.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    def close(self):
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def ray_distances(self, position, height=1.25, count=32, max_distance=8.0):
        import magnum as mn
        origin = np.asarray(position) + [0, height, 0]
        robot_id = self.robot.sim_obj.object_id if self.robot else None
        result = []
        for angle in np.linspace(0, 2 * math.pi, count, endpoint=False):
            ray = self.hs.geo.Ray(mn.Vector3(origin), mn.Vector3(math.cos(angle), 0, -math.sin(angle)))
            hits = self.sim.cast_ray(ray, max_distance).hits
            result.append(min([float(h.ray_distance) for h in hits if h.object_id != robot_id]
                              + [max_distance]))
        return result

    def pose_free(self, position, yaw):
        """Actual articulated static contacts, ground-support filter explicit."""
        if self.robot is None:
            raise ValueError("Load a robot to query native contacts")
        key = tuple(np.round([*position, float(yaw)], 7))
        if key in self.pose_cache:
            return self.pose_cache[key]
        import magnum as mn
        obj = self.robot.sim_obj
        before = mn.Matrix4(obj.transformation)
        try:
            self.robot.base_rot = float(yaw)
            self.robot.base_pos = np.asarray(position, dtype=float)
            self.sim.perform_discrete_collision_detection()
            for c in self.sim.get_physics_contact_points():
                if obj.object_id not in (c.object_id_a, c.object_id_b):
                    continue
                if c.object_id_a == c.object_id_b:
                    continue
                link = c.link_id_a if c.object_id_a == obj.object_id else c.link_id_b
                foot_links = self.robot.params.leg_joints[2::3]
                # The collector's 0.8 normal cutoff incorrectly classified
                # ordinary foot support on rough scanned floors as an obstacle.
                # Ground support here is restricted to actual distal leg links,
                # the configured climb envelope and configured walkable slope.
                low_foot = (link in foot_links and
                            max(c.position_on_a_in_ws[1], c.position_on_b_in_ws[1]) <= position[1] + self.robot_config.max_climb)
                ground = low_foot and abs(c.contact_normal_on_b_in_ws[1]) >= math.cos(math.radians(self.robot_config.max_slope))
                if low_foot and not ground:
                    # Scanned floors contain small steep facets below Recast's
                    # walkable surface. Verify a clear column above the allowed
                    # climb height using real collision rays. A vertical wall
                    # extending above the step envelope remains an obstacle.
                    contact = np.asarray(c.position_on_b_in_ws if c.object_id_a == obj.object_id else c.position_on_a_in_ws, dtype=float)
                    normal = np.asarray(c.contact_normal_on_b_in_ws, dtype=float).copy()
                    normal[1] = 0
                    normal /= max(np.linalg.norm(normal), 1e-9)
                    contact[1] = position[1] + self.robot_config.max_climb + .02
                    blocked = False
                    for sign in (-1, 1):
                        ray = self.hs.geo.Ray(mn.Vector3(contact + sign*.05*normal), mn.Vector3(-sign*normal))
                        hits = self.sim.cast_ray(ray, .1).hits
                        if any(h.object_id != obj.object_id for h in hits):
                            blocked = True; break
                    ground = not blocked
                if not ground:
                    self.last_collision = {"robot_link": int(link),
                                           "contact_y": float(max(c.position_on_a_in_ws[1], c.position_on_b_in_ws[1])),
                                           "normal_y": float(c.contact_normal_on_b_in_ws[1]),
                                           "contact_distance": float(c.contact_distance)}
                    self.pose_cache[key] = False
                    return False
            self.pose_cache[key] = True
            if len(self.pose_cache) > 50000:
                self.pose_cache.clear()
            return True
        finally:
            obj.transformation = before
            self.sim.perform_discrete_collision_detection()

    def build_navmesh(self, path, robot):
        cfg = self.hs.NavMeshSettings()
        cfg.set_defaults()
        cfg.agent_radius, cfg.agent_height = robot.nav_radius, robot.nav_height
        cfg.agent_max_climb, cfg.agent_max_slope = robot.max_climb, robot.max_slope
        cfg.include_static_objects = True
        # Use a fresh PathFinder; no changes to an existing simulator's navigation.
        pf = self.hs.PathFinder()
        if not self.sim.recompute_navmesh(pf, cfg):
            raise ValueError("Native NavMesh construction failed")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not pf.save_nav_mesh(str(path)):
            raise IOError("NavMesh write failed")
