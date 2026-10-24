"""Bounded closed-loop trials using the installed Falcon action classes.

This is a static-world action adapter, not a full social-navigation episode.
It preserves action parameters, leg animation, kinematic mode and physics
stepping. Results certify only the particular executed trace, never a map.
"""
import math
from types import SimpleNamespace
import numpy as np

from ..config import REPO
from .body import BodyOracle
from .config import angle_delta, TraversabilityConfig


def _distance_to_path(point, path):
    if len(path) == 1:
        return float(np.linalg.norm(point-path[0]))
    a, d = path[:-1], np.diff(path, axis=0)
    t = np.clip(np.sum((point-a)*d, axis=1)/np.maximum(np.sum(d*d, axis=1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(point-(a+t[:, None]*d), axis=1)))


def signed_yaw(robot):
    """Installed base_rot getter returns unsigned quaternion angle, not yaw."""
    import magnum as mn
    forward = robot.sim_obj.transformation.transform_vector(mn.Vector3(1,0,0))
    return math.atan2(-forward[2],forward[0])


def motion_segments(timed_poses):
    """Preserve the query's turning direction/winding and translation bends."""
    q=np.asarray(timed_poses)[:,1:]
    result=[]
    for a,b in zip(q[:-1],q[1:]):
        distance=np.linalg.norm(b[:3]-a[:3]);angle=b[3]-a[3]
        if distance<1e-7 and abs(angle)<1e-8:continue
        kind='turn' if distance<1e-7 else 'forward'
        merge=(bool(result) and result[-1]['kind']==kind and
               ((kind=='forward' and abs(angle_delta(result[-1]['end'][3],a[3]))<1e-6) or
                (kind=='turn' and (result[-1]['end'][3]-result[-1]['start'][3])*angle>0)))
        if merge:result[-1]['end']=b
        else:result.append({'kind':kind,'start':a,'end':b})
    return result


def execute_visit(hmap, query, max_actions=240, goal_tolerance=.25, diagnostic_native_baseline=False):
    """Track the geometric route using native discrete forward/left/right.

    Checks native contacts after each actual action and after physics. It does
    not replace these actions with pose teleportation, change their speeds, or
    disable contacts. Initial placement is the only teleportation.
    """
    import habitat_sim
    import magnum as mn
    import falcon.additional_action as actions
    from habitat_baselines.config.default import get_config
    from omegaconf import OmegaConf

    if hmap.geometry is None:
        raise ValueError("Execution trials require an attached native scene")
    if diagnostic_native_baseline and hmap.metadata.get('schema_version')!=1:
        raise ValueError("Native diagnostic execution is restricted to explicit schema-1 baselines")
    if not diagnostic_native_baseline and not hmap.validate_query_path(query)["valid"]:
        raise ValueError("Refusing an invalid query")
    config = get_config("social_nav_v2/falcon_hm3d.yaml")
    simulator = config.habitat.simulator
    geometry = hmap.geometry
    robot, sim = geometry.robot, geometry.sim
    sim.pathfinder.load_nav_mesh(hmap.navigation.path)
    proxy = SimpleNamespace(agents_mgr=[SimpleNamespace(articulated_agent=robot,
                            grasp_mgr=SimpleNamespace(snap_idx=None))],
                            ctrl_freq=simulator.ctrl_freq, step_filter=sim.step_filter)
    task = SimpleNamespace(is_stop_called=False, should_end=False)
    instances, settings = {}, {}
    for short, cls in (("forward", actions.DiscreteMoveForwardAction),
                       ("left", actions.DiscreteTurnLeftAction),
                       ("right", actions.DiscreteTurnRightAction)):
        key = "agent_0_discrete_"+{"forward":"move_forward", "left":"turn_left", "right":"turn_right"}[short]
        values = OmegaConf.to_container(config.habitat.task.actions[key], resolve=True)
        values["leg_animation_checkpoint"] = str((REPO/values["leg_animation_checkpoint"]).resolve())
        settings[short] = values
        instances[short] = cls(config=OmegaConf.create(values), sim=proxy, name=key, task=task)
    before = mn.Matrix4(robot.sim_obj.transformation)
    joints = np.array(robot.sim_obj.joint_positions)
    velocities = np.array(robot.sim_obj.joint_velocities)
    old_frame = actions.play_i
    path = np.asarray(query["path"])
    poses = np.asarray(query["timed_poses"])
    segments=motion_segments(poses)
    linear_increment=abs(settings['forward']['lin_speed'])/simulator.ctrl_freq
    angular_increment=abs(settings['left']['ang_speed'])/simulator.ctrl_freq
    trace, collisions = [], []
    target_index, stagnant = 0, 0
    turn_target=None
    unwrapped_yaw=float(poses[0,4]);last_yaw=unwrapped_yaw
    reason, reached = "action_budget", False
    diagnosis = None
    traversal = getattr(hmap,'traversability_config',TraversabilityConfig())
    oracle = BodyOracle(geometry, hmap.navigation, traversal)
    motor_state = []
    try:
        robot.reset()
        robot.base_rot = float(poses[0, 4]); robot.base_pos = poses[0, 1:4]
        actions.play_i = 0
        # The benchmark kinematic mode removes motors. No motor parameter or
        # physics setting is changed; the robot is already kinematic here.
        motor_state = [(mid, joint, robot.sim_obj.get_joint_motor_settings(mid))
                       for mid, joint in robot.sim_obj.existing_joint_motor_ids.items()]
        for mid, _, _ in motor_state:
            robot.sim_obj.remove_joint_motor(mid)
        robot._update_motor_settings_cache()
        for step in range(max_actions+1):
            position = np.asarray(robot.base_pos, dtype=float)
            yaw = signed_yaw(robot)
            unwrapped_yaw+=angle_delta(last_yaw,yaw);last_yaw=yaw
            sim.perform_discrete_collision_detection()
            contacts = [c for c in oracle.contact_records(position) if not c["support_contact"]]
            error = float(np.linalg.norm(position-path[-1]))
            trace.append({"step":step, "position":position.tolist(), "yaw":yaw,
                          "goal_error":error, "route_deviation":_distance_to_path(position,path),
                          "gait_frame":int(actions.play_i),"motion_segment":target_index})
            if contacts:
                collisions.append({"step":step,"contacts":contacts}); reason="static_body_contact"; break
            selected=None
            while target_index<len(segments) and selected is None:
                segment=segments[target_index];a,b=segment['start'],segment['end']
                if segment['kind']=='turn':
                    if turn_target is None:
                        turn_target=unwrapped_yaw+angle_delta(yaw,a[3])+(b[3]-a[3])
                    delta=turn_target-unwrapped_yaw
                    if abs(delta)<=angular_increment/2+.001:
                        target_index+=1;turn_target=None;continue
                    selected='left' if delta>0 else 'right'
                else:
                    direction=np.array([math.cos(a[3]),-math.sin(a[3])])
                    delta_position=b[:3]-position
                    remaining=float(np.dot(delta_position[[0,2]],direction))
                    if remaining<=linear_increment/2+.005 and np.linalg.norm(delta_position)<=goal_tolerance:
                        target_index+=1;continue
                    desired=math.atan2(-delta_position[2],delta_position[0])
                    delta=angle_delta(yaw,desired)
                    selected=('left' if delta>0 else 'right') if abs(delta)>angular_increment/2+.005 else 'forward'
            if selected is None:
                terminal_turn=angle_delta(yaw,poses[-1,4])
                if error<=goal_tolerance and abs(terminal_turn)>angular_increment/2+.001:
                    # Endpoint tracking can finish a forward segment at a
                    # slightly different heading. Execute a real final turn;
                    # do not enlarge the orientation acceptance tolerance.
                    selected='left' if terminal_turn>0 else 'right'
                else:
                    reached=error<=goal_tolerance and abs(terminal_turn)<=angular_increment
                    reason='reached_position_and_orientation' if reached else 'terminal_pose_tolerance'
                    break
            if trace[-1]['route_deviation']>.4:
                reason='route_deviation';break
            if step == max_actions:
                break
            trace[-1]["action"] = selected
            instances[selected].step(task=task)
            sim.perform_discrete_collision_detection()
            immediate = [c for c in oracle.contact_records(np.asarray(robot.base_pos)) if not c["support_contact"]]
            if immediate:
                collisions.append({"step":step,"phase":"after_action","contacts":immediate})
                reason="static_body_contact"; break
            # EmbodiedTask.step then RearrangeSim_v2.step ordering.
            sim.step_physics(1./config.habitat.task.physics_target_sps)
            if simulator.update_articulated_agent:
                robot.update()
            if simulator.step_physics:
                for _ in range(simulator.ac_freq_ratio):
                    sim.step_world(-1)
            moved = np.linalg.norm(np.asarray(robot.base_pos)-position)
            stagnant = stagnant+1 if selected=="forward" and moved<.01 else 0
            if stagnant>=5:
                reason="stuck_native_step_filter"; break
    finally:
        final_error = float(np.linalg.norm(np.asarray(robot.base_pos)-path[-1]))
        if collisions:
            current_position = np.asarray(robot.base_pos).copy()
            current_yaw = signed_yaw(robot)
            nearest = int(np.argmin(np.linalg.norm(poses[:,1:4]-current_position,axis=1)))
            with BodyOracle(geometry,hmap.navigation,traversal) as checker:
                reference = checker.pose(current_position,current_yaw,use_cache=False)
            diagnosis = {"actual_pose":[*current_position.tolist(),current_yaw],
                         "reference_posture_at_actual_pose":reference,
                         "nearest_planned_position_distance":float(np.linalg.norm(poses[nearest,1:4]-current_position)),
                         "nearest_planned_heading_difference":abs(angle_delta(poses[nearest,4],current_yaw)),
                         "interpretation":"animated posture differs from clear reference posture" if reference['status']=='TRAVERSABLE'
                                          else "actual tracked pose is outside the verified reference route; tracking/turn quantization or posture may contribute"}
        # Native isolated trial state is restored without changing scene assets.
        robot.sim_obj.joint_positions=joints; robot.sim_obj.joint_velocities=velocities
        robot.sim_obj.transformation=before; actions.play_i=old_frame
        for _, joint, setting in motor_state:
            robot.sim_obj.create_joint_motor(joint, setting)
        robot._update_motor_settings_cache()
        sim.perform_discrete_collision_detection()
    return {"validation_level":"actual_falcon_discrete_actions_static_world",
            "source_foundation":"native_navmesh_diagnostic" if diagnostic_native_baseline else "robot_traversability",
            "tracker":"stop_turn_forward_segments_v3_final_alignment",
            "initial_gait_frame":0,
            "full_social_environment_episode":False,"execution_certified":False,
            "reached":reached,"reason":reason,"actions":sum("action" in x for x in trace),
            "goal_error":final_error,
            "maximum_route_deviation":max(x["route_deviation"] for x in trace),
            "static_contact_events":len(collisions),"contacts":collisions,"trace":trace,
            "failure_diagnosis":diagnosis,
            "action_settings":settings,"control_frequency":simulator.ctrl_freq,
            "physics_target_sps":config.habitat.task.physics_target_sps,
            "physics_substeps":simulator.ac_freq_ratio,
            "scope":"native kinematic Spot, CSV leg animation, real action classes and native physics; no human agents, policy, force-balance or gait certification"}
