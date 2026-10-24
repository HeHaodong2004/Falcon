"""Scoped native body/support oracle. Contact rejection matches Stage 0 v1.

Queries do not step physics. The context restores articulation and transform
once, avoiding the expensive unrelated initial-pose collision query per probe.
"""
import itertools
import math
import numpy as np

from .config import TRAVERSABLE, NON_TRAVERSABLE, UNKNOWN, angle_delta


class BodyOracle:
    def __init__(self, geometry, navigation, config, check_support=True):
        if geometry.robot is None:
            raise ValueError("A native articulated robot is required")
        self.geometry, self.nav, self.config = geometry, navigation, config
        self.robot, self.sim = geometry.robot, geometry.sim
        self.robot_config = geometry.robot_config
        self.check_support = check_support
        self.cache = {}
        self.calls = self.cache_hits = 0
        self.support_calls = 0
        self.active = False

    def __enter__(self):
        import magnum as mn
        self.before = mn.Matrix4(self.robot.sim_obj.transformation)
        self.before_joints = np.asarray(self.robot.sim_obj.joint_positions).copy()
        self.before_velocities = np.asarray(self.robot.sim_obj.joint_velocities).copy()
        self.robot.reset()
        self.robot.base_rot = 0.
        self.robot.base_pos = np.zeros(3)
        self.reference_joints = np.asarray(self.robot.sim_obj.joint_positions).copy()
        self.foot_points = []
        all_points = []
        for link in [-1, *self.robot.sim_obj.get_link_ids()]:
            node = self.robot.sim_obj.get_link_scene_node(link)
            bb = node.cumulative_bb
            pts = np.array([node.absolute_transformation().transform_point(mn.Vector3(p))
                            for p in itertools.product(*zip(bb.min, bb.max))])
            all_points.extend(pts)
            if link in self.robot.params.leg_joints[2::3]:
                self.foot_points.append(pts[pts[:, 1] <= pts[:, 1].min()+1e-5].mean(0))
        self.foot_points = np.asarray(self.foot_points)
        pts = np.asarray(all_points)
        self.calibration = {"reference_joints": self.reference_joints.tolist(),
                            "body_bounds_min": pts.min(0).tolist(),
                            "body_bounds_max": pts.max(0).tolist(),
                            "outer_radius": float(np.linalg.norm(pts[:, [0, 2]], axis=1).max()),
                            "foot_support_probes": self.foot_points.tolist(),
                            "method": "transformed matching collision/visual link bounds; native meshes used for contact",
                            "articulation": "installed Spot reset posture; gait not certified"}
        self.active = True
        return self

    def __exit__(self, *args):
        self.robot.sim_obj.joint_positions = self.before_joints
        self.robot.sim_obj.joint_velocities = self.before_velocities
        self.robot.sim_obj.transformation = self.before
        self.sim.perform_discrete_collision_detection()
        self.active = False

    def contact_records(self, position):
        """Classify current native contacts without changing benchmark labels."""
        import magnum as mn
        obj = self.robot.sim_obj
        records = []
        for c in self.sim.get_physics_contact_points():
            if obj.object_id not in (c.object_id_a, c.object_id_b) or c.object_id_a == c.object_id_b:
                continue
            link = c.link_id_a if c.object_id_a == obj.object_id else c.link_id_b
            contact_y = max(c.position_on_a_in_ws[1], c.position_on_b_in_ws[1])
            low_foot = link in self.robot.params.leg_joints[2::3] and contact_y <= position[1]+self.robot_config.max_climb
            ground = low_foot and abs(c.contact_normal_on_b_in_ws[1]) >= math.cos(math.radians(self.robot_config.max_slope))
            if low_foot and not ground:
                contact = np.asarray(c.position_on_b_in_ws if c.object_id_a == obj.object_id else c.position_on_a_in_ws, dtype=float)
                normal = np.asarray(c.contact_normal_on_b_in_ws, dtype=float).copy()
                normal[1] = 0
                normal /= max(np.linalg.norm(normal), 1e-9)
                contact[1] = position[1]+self.robot_config.max_climb+self.config.support_probe_height_margin
                blocked = False
                for sign in (-1, 1):
                    ray = self.geometry.hs.geo.Ray(mn.Vector3(contact+sign*self.config.support_probe_half_span*normal),
                                                   mn.Vector3(-sign*normal))
                    if any(h.object_id != obj.object_id for h in self.sim.cast_ray(ray, 2*self.config.support_probe_half_span).hits):
                        blocked = True
                        break
                ground = not blocked
            records.append({"robot_link": int(link), "link_name": obj.get_link_name(link),
                            "contact_y": float(contact_y), "relative_height": float(contact_y-position[1]),
                            "normal_y": float(c.contact_normal_on_b_in_ws[1]),
                            "contact_distance": float(c.contact_distance),
                            "position": list(c.position_on_b_in_ws if c.object_id_a == obj.object_id else c.position_on_a_in_ws),
                            "support_contact": bool(ground)})
        return records

    def support(self, position, yaw):
        """Native downward collision rays at four calibrated distal-link probes.

        This establishes bounded terrain-column evidence, not force balance.
        Missing/ambiguous support is UNKNOWN, never inferred blocked or safe.
        """
        import magnum as mn
        c, s = math.cos(yaw), math.sin(yaw)
        offsets = self.foot_points @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        points = np.asarray(position)+offsets
        heights = []
        drop = self.robot_config.max_climb+self.nav.settings["cell_height"]+self.config.support_quantization_allowance
        for point in points:
            self.support_calls += 1
            origin = point.copy()
            origin[1] = position[1]+self.config.support_ray_lift
            ray = self.geometry.hs.geo.Ray(mn.Vector3(origin), mn.Vector3(0, -1, 0))
            hits = [h for h in self.sim.cast_ray(ray, self.config.support_ray_lift+drop).hits
                    if h.object_id != self.robot.sim_obj.object_id]
            if not hits:
                return {"status": UNKNOWN, "reason": "missing_foot_column_support"}
            hit = min(hits, key=lambda h: h.ray_distance)
            height = origin[1]-hit.ray_distance
            if height > position[1]+self.robot_config.max_climb:
                return {"status": UNKNOWN, "reason": "terrain_above_climb_envelope"}
            heights.append(float(height))
        return {"status": TRAVERSABLE, "terrain_height_range": [min(heights), max(heights)],
                "scope": "four finite terrain columns; native center support; no balance or gait guarantee"}

    def pose(self, position, yaw, use_cache=True):
        if not self.active:
            raise RuntimeError("Use BodyOracle as a context manager")
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not math.isfinite(yaw):
            return {"status": UNKNOWN, "reason": "invalid_pose"}
        yaw = angle_delta(0, yaw)
        key = tuple(np.round([*p, yaw], 8))
        if use_cache and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        projected = np.asarray(self.nav.pf.snap_point(p))
        if (not np.isfinite(projected).all() or
                np.linalg.norm((projected-p)[[0, 2]]) > .002 or
                abs(projected[1]-p[1]) > self.robot_config.max_climb+.025):
            return {"status": UNKNOWN, "reason": "pose_outside_native_support_domain"}
        self.calls += 1
        self.robot.base_rot = yaw
        self.robot.base_pos = p
        self.sim.perform_discrete_collision_detection()
        contacts = self.contact_records(p)
        blocked = next((c for c in contacts if not c["support_contact"]), None)
        if blocked is not None:
            result = {"status": NON_TRAVERSABLE, "reason": "native_body_contact", "contact": blocked}
        elif self.check_support:
            support = self.support(p, yaw)
            result = {"status": support["status"], "reason": support.get("reason", "body_clear_supported"),
                      "support": support}
        else:
            result = {"status": TRAVERSABLE, "reason": "body_clear_support_not_checked"}
        if use_cache:
            self.cache[key] = result
        return result

    def motion(self, poses, spatial_step=None, angular_step=None, use_cache=True, project_to_surface=True):
        """Validate a specified stop/turn/forward primitive, including endpoints."""
        spatial_step = spatial_step or self.config.spatial_step
        angular_step = angular_step or self.config.angular_step
        q = np.asarray(poses, dtype=float)
        if q.ndim != 2 or q.shape[1] != 4 or not len(q) or not np.isfinite(q).all():
            return {"status": UNKNOWN, "reason": "invalid_motion"}
        checked, previous, swept, bridge_samples = 0, None, [], 0
        for a, b in list(zip(q[:-1], q[1:])) or [(q[0], q[0])]:
            distance = np.linalg.norm(b[:3]-a[:3])
            turn = angle_delta(a[3], b[3])
            if distance > 1e-7 and abs(turn) > 1e-6:
                return {"status": UNKNOWN, "reason": "simultaneous_turn_translation_outside_contract"}
            if np.linalg.norm((b-a)[[0, 2]]) > 1e-7:
                desired = math.atan2(a[2]-b[2], b[0]-a[0])
                horizontal = (b-a)[[0, 2]]
                forward = np.array([math.cos(a[3]), -math.sin(a[3])])
                lateral = abs(horizontal[0]*forward[1]-horizontal[1]*forward[0])
                # Native float32 projection perturbs dense path segments by
                # sub-micrometre amounts. Bound lateral error in metres; do not
                # mistake that quantization for a nonholonomic violation.
                if (np.dot(horizontal, forward) < 0 or
                    (abs(angle_delta(a[3], desired)) > 1e-5 and
                     lateral > self.config.lateral_numerical_tolerance)):
                    return {"status": NON_TRAVERSABLE, "reason": "forward_heading_constraint"}
            count = max(1, math.ceil(distance/spatial_step), math.ceil(abs(turn)/angular_step))
            for t in np.linspace(0, 1, count+1):
                p = a[:3]+t*(b[:3]-a[:3]); yaw = a[3]+t*turn
                if project_to_surface:
                    try:
                        p = self.nav.project_path_sample(p)
                    except ValueError:
                        return {"status": NON_TRAVERSABLE, "reason": "native_surface_sample", "failed_pose": [*p, yaw]}
                if (previous is not None and np.linalg.norm(previous[:3]-p)<1e-12 and
                        abs(angle_delta(previous[3],yaw))<1e-12):
                    continue  # This exact sample already passed in this call.
                if previous is not None and not self.nav.segment_free(previous[:3], p):
                    return {"status": NON_TRAVERSABLE, "reason": "native_transition", "failed_pose": [*p, yaw]}
                current = np.array([*p,yaw])
                if previous is None:
                    candidates=[current]
                else:
                    gap=np.linalg.norm(p-previous[:3]); rotation=angle_delta(previous[3],yaw)
                    subdivisions=max(1,math.ceil(gap/(spatial_step+self.config.lateral_numerical_tolerance)),
                                     math.ceil(abs(rotation)/(angular_step+1e-9)))
                    # Native height projection can jump at detail surfaces.
                    # Check the actual XYZ bridge, not another projection that
                    # would skip these intervening configurations again.
                    bridge_samples+=subdivisions-1
                    candidates=[np.array([*(previous[:3]+f*(p-previous[:3])),previous[3]+f*rotation])
                                for f in np.linspace(0,1,subdivisions+1)[1:]]
                for sample in candidates:
                    result=self.pose(sample[:3],sample[3],use_cache)
                    checked+=1
                    if result['status']!=TRAVERSABLE:
                        return {'status':result['status'],'reason':result['reason'],
                                'failed_pose':sample.tolist(),'pose_evidence':result,'checked_samples':checked}
                    swept.append(sample.tolist())
                previous=current
        return {"status": TRAVERSABLE, "checked_samples": checked,
                "spatial_step": spatial_step, "angular_step": angular_step,
                "validation_level": self.config.validation_level,
                "additional_xyz_bridge_samples": bridge_samples,
                "maximum_xyz_sample_spacing": float(np.linalg.norm(np.diff(np.asarray(swept)[:,:3],axis=0),axis=1).max()) if len(swept)>1 else 0.,
                "swept_poses": swept, "continuous_collision_certified": False,
                "gait_certified": False}
