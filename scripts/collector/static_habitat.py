"""Habitat static-world adapter for the single canonical feasibility checker.

No environment stepping, RNG use, human positions, or human policy calls.
"""
import itertools
import math
import numpy as np
import magnum as mn
import habitat_sim
from collector.static_feasibility import RobotGeometry


class HabitatStaticWorld:
    def __init__(self, planner, sim, robot):
        self.planner,self.pf,self.sim,self.robot=planner,planner.pf,sim,robot
        self.original=mn.Matrix4(robot.sim_obj.transformation)
        # This installed Habitat build exposes per-link scene bounds, not obj.aabb.
        # Spot URDF uses identical visual/collision meshes + origins on every link.
        # Include each transformed local AABB corner in the yaw-invariant disc bound.
        corners=[]
        for link in [-1]+list(robot.sim_obj.get_link_ids()):
            node=robot.sim_obj.get_link_scene_node(link)
            bb=node.cumulative_bb
            transform=node.absolute_transformation()
            corners.extend(transform.transform_point(mn.Vector3(*v)) for v in
                           itertools.product(*zip(bb.min,bb.max)))
        base=np.asarray(robot.base_pos)
        radius=max(float(np.linalg.norm((np.asarray(c)-base)[[0,2]])) for c in corners)
        self.geometry=RobotGeometry(max(.25,radius),source='Spot URDF matching collision/visual link AABB corners about base_pos')
        self.static_ids={habitat_sim.stage_id}
        for manager in (sim.get_rigid_object_manager(),sim.get_articulated_object_manager()):
            for obj in manager.get_objects_by_handle_substring('').values():
                if obj.object_id != robot.sim_obj.object_id and obj.motion_type==habitat_sim.physics.MotionType.STATIC:
                    self.static_ids.add(obj.object_id)
                    if hasattr(obj,'link_object_ids'): self.static_ids.update(obj.link_object_ids)
        self.cache={}

    def path(self,*args): return self.planner.path(*args)

    def clearance(self,position):
        snapped=np.asarray(self.pf.snap_point(position))
        if not np.isfinite(snapped).all() or np.linalg.norm(snapped-np.asarray(position))>.04:
            return -1.
        return float(self.pf.distance_to_closest_obstacle(position,5.))

    def segment_free(self,a,b):
        stepped=np.asarray(self.pf.try_step(a,b))
        return bool(np.linalg.norm(stepped-np.asarray(b))<.005)

    def pose_free(self,position,yaw):
        key=tuple(np.round([*position,yaw],9))
        if key in self.cache:return self.cache[key]
        obj=self.robot.sim_obj
        # The fixed controller is yaw-only with unchanged arm/leg posture. Pose writes are
        # collision queries only; restored even on error, with no physics or sensor step.
        before=mn.Matrix4(obj.transformation)
        try:
            self.robot.base_rot=yaw
            self.robot.base_pos=mn.Vector3(*position)
            self.sim.perform_discrete_collision_detection()
            ok=True
            for cp in self.sim.get_physics_contact_points():
                ids={cp.object_id_a,cp.object_id_b}
                if obj.object_id not in ids or not (ids & self.static_ids):continue
                ground=(abs(cp.contact_normal_on_b_in_ws[1])>.8 and
                        max(cp.position_on_a_in_ws[1],cp.position_on_b_in_ws[1])<position[1]+.12)
                if not ground:ok=False;break
            self.cache[key]=ok
            return ok
        finally:
            obj.transformation=before
            self.sim.perform_discrete_collision_detection()
