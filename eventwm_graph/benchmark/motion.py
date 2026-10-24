"""Native kinematic action compiler. No body-safety or gait guarantee is implied.

Prediction uses the installed native VelocityControl and PathFinder, with the
same root/base offset ordering as BaseVelAction.update_base. Independent replay
calls the actual Python action classes and advances unmodified native physics.
"""
import math
import numpy as np
from ..native import path_length


def delta(a, b):
    return math.atan2(math.sin(b-a), math.cos(b-a))


class KinematicModel:
    def __init__(self, navigation, contract):
        import habitat_sim
        self.hs, self.nav, self.contract = habitat_sim, navigation, contract
        self.pf = navigation.pf
        self.offset = np.asarray(contract["base_offset"])
        self.ctrl = habitat_sim.physics.VelocityControl()
        self.ctrl.controlling_lin_vel = self.ctrl.controlling_ang_vel = True
        self.ctrl.lin_vel_is_local = self.ctrl.ang_vel_is_local = True
        self.speeds = {short: contract["actions"][long] for short, long in
            (("forward", "discrete_move_forward"), ("left", "discrete_turn_left"),
             ("right", "discrete_turn_right"), ("stop", "discrete_stop"))}
        self.distance_step = self.speeds["forward"]["lin_speed"]/contract["control_frequency"]
        self.angle_step = abs(self.speeds["left"]["ang_speed"])/contract["control_frequency"]

    def step(self, pose, action):
        import magnum as mn
        p=np.asarray(pose,dtype=float)
        transform=mn.Matrix4.from_(mn.Quaternion.rotation(mn.Rad(p[3]),mn.Vector3(0,1,0)).to_matrix(),mn.Vector3(p[:3]-self.offset))
        return self.pose_from_transform(self.step_transform(transform,action))

    def pose_from_transform(self,transform):
        import magnum as mn
        forward=transform.transform_vector(mn.Vector3(1,0,0))
        return np.array([*(np.asarray(transform.translation)+self.offset),math.atan2(-forward[2],forward[0])])

    def step_transform(self,transform,action):
        import magnum as mn
        if action == "stop":
            return mn.Matrix4(transform)
        if action not in self.speeds:
            raise ValueError("Action is absent from the installed benchmark")
        settings = self.speeds[action]
        self.ctrl.linear_velocity = mn.Vector3(settings["lin_speed"], 0., 0.)
        self.ctrl.angular_velocity = mn.Vector3(0., settings["ang_speed"], 0.)
        start = self.hs.RigidState(mn.Quaternion.from_matrix(transform.rotation()),transform.translation)
        end = self.ctrl.integrate_transform(1./self.contract["control_frequency"], start)
        stepped = np.asarray(self.pf.try_step(start.translation, end.translation), dtype=float)
        # Match native Vector3 exact equality, including the failure case.
        if not np.array_equal(stepped, np.asarray(start.translation)):
            stepped -= self.offset
        return mn.Matrix4.from_(end.rotation.to_matrix(),mn.Vector3(stepped))

    def compile(self, path, start_yaw=0., target_yaw=None, max_actions=500,
                goal_tolerance=.16, waypoint_tolerance=.18):
        """Follow the selected polyline without resetting at graph junctions.

        Intermediate corners can only be shortened by a native straight step.
        Search failure is UNKNOWN, never proof of blocked space.
        """
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all():
            raise ValueError("Expected finite surface path")
        q = np.array([*path[0], start_yaw], dtype=float)
        import magnum as mn
        transform=mn.Matrix4.from_(mn.Quaternion.rotation(mn.Rad(start_yaw),mn.Vector3(0,1,0)).to_matrix(),mn.Vector3(q[:3]-self.offset))
        trace, actions = [q.copy()], []
        goal = path[-1]
        target = 1 if len(path)>1 else 0
        stuck = 0
        visited = {}
        reason = "action_budget"
        for _ in range(max_actions):
            error = np.linalg.norm(q[:3]-goal)
            if target >= len(path)-1 and error <= goal_tolerance:
                if target_yaw is None or abs(delta(q[3], target_yaw)) <= self.angle_step/2+1e-5:
                    return {"status": "VERIFIED_TRAVERSABLE", "reached": True,
                        "validation_level": "native_kinematic_action_prediction", "poses": np.asarray(trace).tolist(),
                        "actions": actions, "goal_error": float(error), "length": path_length(np.asarray(trace)[:,:3]),
                        "full_body_safe": None, "continuous_collision_certified": False}
                action = "left" if delta(q[3],target_yaw)>0 else "right"
            else:
                while target < len(path)-1 and np.linalg.norm(q[:3]-path[target]) <= waypoint_tolerance:
                    target += 1
                # Avoid pathological short segments, while respecting the chosen route.
                look = target
                while look < len(path)-1 and np.linalg.norm(q[:3]-path[look+1]) < .65:
                    if not self.nav.segment_free(q[:3],path[look+1]): break
                    look += 1
                target = look
                d = path[target]-q[:3]
                desired = math.atan2(-d[2],d[0])
                angle = delta(q[3],desired)
                action = ("left" if angle>0 else "right") if abs(angle)>self.angle_step/2+1e-5 else "forward"
            transform = self.step_transform(transform,action)
            nxt = self.pose_from_transform(transform)
            if not np.isfinite(nxt).all(): reason="nonfinite_native_step";break
            if int(self.pf.get_island(nxt[:3])) != int(self.pf.get_island(q[:3])):
                reason="native_island_change";break
            moved = np.linalg.norm(nxt[:3]-q[:3])
            stuck = stuck+1 if action=="forward" and moved<.001 else 0
            actions.append(action); trace.append(nxt.copy()); q=nxt
            key=tuple(np.round(q,3))+(target,)
            visited[key]=visited.get(key,0)+1
            if stuck>=3: reason="blocked_native_motion";break
            if visited[key]>=5: reason="controller_cycle";break
        return {"status":"UNKNOWN", "reached":False, "reason":reason,
            "validation_level":"native_kinematic_action_prediction", "poses":np.asarray(trace).tolist(),
            "actions":actions, "goal_error":float(np.linalg.norm(q[:3]-goal)),
            "full_body_safe":None, "continuous_collision_certified":False}


def replay_native(hmap, prediction, contract, diagnostics=True):
    """Replay actual Falcon actions, measuring parity and benchmark scene contact.

    Scene contact is reported, not made into a new termination rule. Human
    contact requires a social episode and is deliberately not certified here.
    """
    from types import SimpleNamespace
    import magnum as mn
    from omegaconf import OmegaConf
    import falcon.additional_action as actions
    from ..config import REPO
    from ..robotmap.execution import signed_yaw
    from ..robotmap.body import BodyOracle
    from ..robotmap.config import TraversabilityConfig
    from habitat.tasks.rearrange.utils import rearrange_collision
    geometry=hmap.geometry
    if geometry is None: raise ValueError("Native replay needs static scene geometry")
    robot,sim=geometry.robot,geometry.sim
    sim.pathfinder.load_nav_mesh(hmap.navigation.path)
    proxy=SimpleNamespace(agents_mgr=[SimpleNamespace(articulated_agent=robot,grasp_mgr=SimpleNamespace(snap_idx=None))],
        ctrl_freq=contract['control_frequency'],step_filter=sim.step_filter)
    proxy.get_physics_contact_points=sim.get_physics_contact_points
    proxy.get_agent_data=lambda index:proxy.agents_mgr[index]
    proxy.scene_obj_ids=[]
    task=SimpleNamespace(is_stop_called=False,should_end=False)
    instances={}
    for short,cls,long in (("forward",actions.DiscreteMoveForwardAction,"discrete_move_forward"),
            ("left",actions.DiscreteTurnLeftAction,"discrete_turn_left"),
            ("right",actions.DiscreteTurnRightAction,"discrete_turn_right"),
            ("stop",actions.DiscreteStopAction,"discrete_stop")):
        values=dict(contract['actions'][long])
        if 'leg_animation_checkpoint' in values: values['leg_animation_checkpoint']=str((REPO/values['leg_animation_checkpoint']).resolve())
        instances[short]=cls(config=OmegaConf.create(values),sim=proxy,name='agent_0_'+long,task=task)
    before=mn.Matrix4(robot.sim_obj.transformation);joints=np.asarray(robot.sim_obj.joint_positions).copy()
    velocities=np.asarray(robot.sim_obj.joint_velocities).copy();old_frame=actions.play_i
    motors=[(joint,robot.sim_obj.get_joint_motor_settings(mid)) for mid,joint in robot.sim_obj.existing_joint_motor_ids.items()]
    trace=[];contacts=[];scene_penalties=[]
    oracle=BodyOracle(geometry,hmap.navigation,TraversabilityConfig())
    initial=np.asarray(prediction['poses'][0]);t0=sim.get_world_time()
    try:
        robot.reset();robot.base_rot=float(initial[3]);robot.base_pos=initial[:3];actions.play_i=0
        for mid in list(robot.sim_obj.existing_joint_motor_ids):robot.sim_obj.remove_joint_motor(mid)
        robot._update_motor_settings_cache()
        for i in range(len(prediction['actions'])+1):
            p=np.asarray(robot.base_pos,dtype=float);yaw=signed_yaw(robot)
            trace.append({'pose':[*p.tolist(),yaw],'time':float(sim.get_world_time()-t0),
                          'gait_frame':int(actions.play_i),'joints':np.asarray(robot.sim_obj.joint_positions).tolist()})
            if diagnostics:
                sim.perform_discrete_collision_detection()
                c=[x for x in oracle.contact_records(p) if not x['support_contact']]
                if c:contacts.append({'step':i,'contacts':c})
                # Use the actual reward's classification separately from the
                # conservative Stage-0 distal-foot diagnostic.
                collision,details=rearrange_collision(proxy,True,ignore_base=False,agent_idx=0)
                if collision:scene_penalties.append({'step':i,'robot_scene':int(details.robot_scene_colls)})
            if i==len(prediction['actions']) or task.should_end:break
            instances[prediction['actions'][i]].step(task=task)
            sim.step_physics(1./contract['physics_target_sps'])
            if contract['update_articulated_agent']:robot.update()
            if contract['step_physics']:
                for _ in range(contract['physics_substeps']):sim.step_world(-1)
        observed=np.asarray([r['pose'] for r in trace]);expected=np.asarray(prediction['poses'])[:len(trace)]
        pos_error=np.linalg.norm(observed[:,:3]-expected[:,:3],axis=1)
        yaw_error=np.abs(np.arctan2(np.sin(observed[:,3]-expected[:,3]),np.cos(observed[:,3]-expected[:,3])))
        return {'validation_level':'actual_falcon_kinematic_actions_static_scene',
            'pose_parity':bool(len(trace)==len(prediction['poses']) and pos_error.max()<2e-4 and yaw_error.max()<2e-4),
            'maximum_position_error':float(pos_error.max()),'maximum_yaw_error':float(yaw_error.max()),
            'trace':trace,'contact_diagnostics':contacts,'native_scene_collision_events':scene_penalties,
            'scene_contact_reward_penalty_configured':contract['scene_contact_reward_penalty'],
            'benchmark_should_end':task.should_end,'stop_called':task.is_stop_called,
            'full_social_episode':False,'full_body_safe':not bool(contacts),
            'scope':'Exact installed actions and physics; contacts recorded without inventing scene-contact termination'}
    finally:
        robot.sim_obj.joint_positions=joints;robot.sim_obj.joint_velocities=velocities
        robot.sim_obj.transformation=before;actions.play_i=old_frame
        for joint,settings in motors:robot.sim_obj.create_joint_motor(joint,settings)
        robot._update_motor_settings_cache();sim.perform_discrete_collision_detection()
