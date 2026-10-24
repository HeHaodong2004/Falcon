"""Privileged outcome supervision, kept out of histories and query generation."""
import math
import numpy as np

METRIC_VERSION='falcon_psc_geodesic_fallback_and_mesh_contact_1'


def distances(sim,human_count,use_geo=True):
    """Exact installed PSC center convention, including disconnected fallback."""
    import habitat_sim
    robot=np.asarray(sim.get_agent_state(0).position)
    official=[];euclidean=[]
    for i in range(1,human_count+1):
        human=np.asarray(sim.get_agent_state(i).position)
        # Preserve the native dtype/order at the strict 1m benchmark boundary.
        # Float64 promotion can turn a float32 norm of exactly1 into <1.
        distance=float(np.linalg.norm(human-robot,ord=2))
        eu=float(np.linalg.norm(human.astype(np.float64)-robot.astype(np.float64)))
        if use_geo:
            request=habitat_sim.ShortestPath();request.requested_start=robot;request.requested_end=human
            if sim.pathfinder.find_path(request):distance=float(sim.geodesic_distance(robot,human))
        euclidean.append(eu);official.append(distance)
    return official,euclidean


def contact_labels(rt):
    """Native contact events, with supporting floor separated from obstacles.

    The benchmark's robot-scene contact count is also retained separately: it
    can include support feet and is a reward event, not a termination condition.
    No center-distance heuristic is substituted for articulated mesh contacts.
    """
    from habitat.tasks.rearrange.utils import rearrange_collision
    sim=rt.sim;sim.perform_discrete_collision_detection()
    robot=sim.get_agent_data(0).articulated_agent
    rid=robot.sim_obj.object_id;human_ids={sim.get_agent_data(i).articulated_agent.sim_obj.object_id for i in range(1,rt.count+1)}
    human_ray_ids=set(human_ids)
    for i in range(1,rt.count+1):human_ray_ids.update(sim.get_agent_data(i).articulated_agent.sim_obj.link_object_ids.keys())
    base=np.asarray(robot.base_pos);obstacle=[];human=False
    for cp in sim.get_physics_contact_points():
        ids={cp.object_id_a,cp.object_id_b}
        if rid not in ids:continue
        if ids & human_ids:human=True;continue
        # Ignore the actor's self contact; retain only environmental contacts.
        if len(ids)==1:continue
        link=cp.link_id_a if cp.object_id_a==rid else cp.link_id_b
        contact_y=max(cp.position_on_a_in_ws[1],cp.position_on_b_in_ws[1])
        low_foot=(link in robot.params.leg_joints[2::3] and contact_y<=base[1]+.2)
        support=low_foot and abs(float(cp.contact_normal_on_b_in_ws[1]))>=math.cos(math.radians(45.))
        if low_foot and not support:
            # Reuse the previously validated BodyOracle terrain-edge rule:
            # a low distal-foot side contact is support only when finite
            # columns at the climb envelope are unobstructed on BOTH sides.
            import habitat_sim
            contact=np.asarray(cp.position_on_b_in_ws if cp.object_id_a==rid else cp.position_on_a_in_ws,dtype=float).copy()
            normal=np.asarray(cp.contact_normal_on_b_in_ws,dtype=float).copy();normal[1]=0.
            norm=np.linalg.norm(normal)
            if norm>1e-9:
                normal/=norm;contact[1]=base[1]+.22;blocked=False
                for sign in (-1,1):
                    ray=habitat_sim.geo.Ray(contact+sign*.05*normal,-sign*normal)
                    if any(h.object_id!=rid and h.object_id not in human_ray_ids for h in sim.cast_ray(ray,.1).hits):blocked=True
                support=not blocked
        if not support:obstacle.append({'link_a':int(cp.link_id_a),'link_b':int(cp.link_id_b),
            'distance':float(cp.contact_distance),'point':np.asarray(cp.position_on_a_in_ws).tolist(),
            'normal_y':float(cp.contact_normal_on_b_in_ws[1]),'robot_link':int(link)})
    _,details=rearrange_collision(sim,True,ignore_base=False,agent_idx=0)
    return {'human_mesh_contact':human,'static_obstacle_contact':bool(obstacle),
            'static_obstacle_contact_witnesses':obstacle,
            'benchmark_robot_scene_contact_count':int(details.robot_scene_colls)}


def step_labels(rt,timestamp):
    official,euclidean=distances(rt.sim,rt.count)
    contacts=contact_labels(rt)
    return {'timestamp':float(timestamp),'official_center_distances_m':official,
            'euclidean_center_distances_m':euclidean,
            'official_dmin_m':min(official) if official else None,
            'euclidean_dmin_m':min(euclidean) if euclidean else None,
            'psc_violation':any(d<1. for d in official),
            'euclidean_ps_violation':any(d<1. for d in euclidean),**contacts}


def summarize(labels,poses,root_timestamp,actions,termination,query,pointgoal,prefix_compliant=0,prefix_steps=0):
    n=len(labels)
    times=np.asarray([r['timestamp'] for r in labels],dtype=float)
    if n and (np.any(np.diff(times)<=0) or times[0]<=root_timestamp):raise ValueError('Noncausal outcome labels')
    dt=np.diff(np.r_[root_timestamp,times])
    bad=np.asarray([r['psc_violation'] for r in labels],dtype=bool)
    minima=np.asarray([r['official_dmin_m'] if r['official_dmin_m'] is not None else np.inf for r in labels])
    eu=np.asarray([r['euclidean_dmin_m'] if r['euclidean_dmin_m'] is not None else np.inf for r in labels])
    depth=np.maximum(0.,1.-minima)
    collision=np.asarray([r['human_mesh_contact'] for r in labels],dtype=bool)
    static=np.asarray([r['static_obstacle_contact'] for r in labels],dtype=bool)
    path=np.asarray(poses,dtype=float)
    qgoal=np.asarray(query['waypoints'][-1]);goal=np.asarray(pointgoal)
    position_error=float(np.linalg.norm(path[-1,:3]-qgoal))
    yaw_error=abs(math.atan2(math.sin(path[-1,3]-query['final_yaw']),math.cos(path[-1,3]-query['final_yaw'])))
    completed=(termination=='query_complete' and position_error<=query['compiled']['position_tolerance']+.001 and
               yaw_error<=query['compiled']['yaw_tolerance']+.0001)
    first=lambda mask:float(times[np.flatnonzero(mask)[0]]-root_timestamp) if mask.any() else None
    per_human=[]
    for i in range(len(labels[0]['official_center_distances_m']) if n else 0):
        d=np.asarray([r['official_center_distances_m'][i] for r in labels])
        per_human.append({'supervision_human_index':i,'min_official_distance_m':float(d.min()),
                          'violating_steps':int((d<1.).sum()),'duration_s':float(dt[d<1.].sum())})
    terminated=termination in ('human_collision','episode_action_limit','environment_termination')
    return {'metric_version':METRIC_VERSION,'psc_radius_m':1.,
        'psc_distance_convention':'installed geodesic center distance, XYZ Euclidean fallback',
        'psc_timestep_convention':'post-action endpoints, including terminal; root excluded',
        'human_collision':bool(collision.any()),'static_obstacle_collision':bool(static.any()),
        'first_collision_time_s':first(collision|static),
        'collision_termination':termination=='human_collision',
        'any_psc_violation':bool(bad.any()),'psc':float(1.-bad.mean()) if n else None,
        'psc_valid':bool(n),'psc_denominator_executed_steps':n,
        'episode_prefix_plus_query_psc':float((prefix_compliant+n-int(bad.sum()))/(prefix_steps+n)) if prefix_steps+n else 1.,
        'minimum_official_distance_m':float(minima.min()) if n and np.isfinite(minima.min()) else None,
        'minimum_euclidean_distance_m':float(eu.min()) if n and np.isfinite(eu.min()) else None,
        'first_psc_violation_time_s':first(bad),'violating_steps':int(bad.sum()),
        'violation_duration_s':float(dt[bad].sum()),'maximum_intrusion_depth_m':float(depth.max()) if n else 0.,
        'integrated_intrusion_severity_m_s':float(np.sum(depth*dt)),
        'human_collision_after_psc_violation':bool(collision.any() and bad.any() and first(collision)>=first(bad)),
        'per_human':per_human,'query_complete':bool(completed),'termination':termination,
        'elapsed_s':float(times[-1]-root_timestamp) if n else 0.,'executed_steps':n,
        'robot_path_length_m':float(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1).sum()),
        'final_pose':path[-1].tolist(),'terminal_position_error_m':position_error,'terminal_yaw_error_rad':yaw_error,
        'goal_euclidean_distance_change_m':float(np.linalg.norm(path[0,:3]-goal)-np.linalg.norm(path[-1,:3]-goal)),
        'pointgoal_proximity_reached':bool(np.linalg.norm(path[-1,:3]-goal)<.2),
        'episode_terminated':terminated,'episode_success':False if terminated else None,
        'episode_success_semantics':'official success requires STOP; queries never invoke STOP',
        'spl_stl_inputs':{'success':False if terminated else None,'query_path_length_m':float(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1).sum()),
                          'query_action_count':n,'stop_called':False},
        'executed_actions':list(actions[:n])}
