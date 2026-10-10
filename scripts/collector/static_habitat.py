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

    def _contacts_free(self,position):
        obj=self.robot.sim_obj
        for cp in self.sim.get_physics_contact_points():
            ids={cp.object_id_a,cp.object_id_b}
            if obj.object_id not in ids or not (ids & self.static_ids):continue
            ground=(abs(cp.contact_normal_on_b_in_ws[1])>.8 and
                    max(cp.position_on_a_in_ws[1],cp.position_on_b_in_ws[1])<position[1]+.12)
            if not ground:return False
        return True

    def native_control_check(self,query):
        """Check native float transforms as well as the nominal swept trajectory.

        Match BaseVelAction.update_base with an independent STATIC pathfinder.
        No environment/physics/sensor step, clock, RNG, or human-policy call.
        Preserve the full matrix: converting accumulated rotation back to yaw
        loses the float rounding which caused the b027 certification miss.
        """
        obj=self.robot.sim_obj
        before=mn.Matrix4(obj.transformation)
        vc=habitat_sim.physics.VelocityControl()
        vc.controlling_lin_vel=vc.lin_vel_is_local=True
        vc.controlling_ang_vel=vc.ang_vel_is_local=True
        checked=0
        try:
            obj.transformation=self.original
            for i,(v,w) in enumerate(query['executable_controls']):
                if v or w:
                    trans=obj.transformation
                    rigid=habitat_sim.RigidState(mn.Quaternion.from_matrix(trans.rotation()),trans.translation)
                    vc.linear_velocity=mn.Vector3(v,0,0)
                    vc.angular_velocity=mn.Vector3(0,w,0)
                    target=vc.integrate_transform(query['integration_dt'],rigid)
                    end=self.pf.try_step(rigid.translation,target.translation)
                    if end!=rigid.translation:end-=self.robot.params.base_offset
                    obj.transformation=mn.Matrix4.from_(target.rotation.to_matrix(),end)
                self.sim.perform_discrete_collision_detection()
                checked+=1
                position=np.asarray(self.robot.base_pos)
                if not self._contacts_free(position):
                    forward=np.asarray(obj.transformation.transform_vector(mn.Vector3(1,0,0)))
                    return dict(static_feasible=False,checked_poses=checked,failed_segment=i,
                                failed_pose=[(i+1)*query['control_dt'],*position.tolist(),
                                             math.atan2(-forward[2],forward[0])],
                                rejection_reason='native_controller_static_geometry_overlap')
            return dict(static_feasible=True,checked_poses=checked)
        finally:
            obj.transformation=before
            self.sim.perform_discrete_collision_detection()

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
            ok=self._contacts_free(position)
            self.cache[key]=ok
            return ok
        finally:
            obj.transformation=before
            self.sim.perform_discrete_collision_detection()
