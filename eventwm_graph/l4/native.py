"""Native environment boundary and OFFLINE evaluation; never imported by policy.

The environment sees private initialization; Executor only receives the public
query, declared static map, exact legal observations and optional held forecasts.
"""
from dataclasses import asdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time
import numpy as np
from eventwm_graph.config import file_digest

CHECKPOINT=Path('/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt')
FALCON_CHECKPOINT=Path('/research-user/workspace/baselines/Falcon/pretrained_model/falcon_noaux_25.pth')


class LegalObserver:
    """Reuse Stage 1's exact frustum and static+human occlusion tests.

    The benchmark supplies full static geometry separately. Human packets do
    not expose hidden IDs, goals, velocities or future trajectories.
    """
    def __init__(self,runtime):
        from eventwm_graph.partial.observer import GeometryObserver
        self.sensor=GeometryObserver(runtime,.1);self.runtime=runtime

    def observe(self):
        from scripts.collector.observation import HumanDetection
        from eventwm_graph.partial.observer import in_frustum
        from eventwm_graph.robotmap.execution import signed_yaw
        rt=self.runtime;s=self.sensor;t=float(rt.sim.get_world_time())
        robot=rt.sim.get_agent_data(0).articulated_agent
        pose=[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)]
        origin,rotation=s.camera.camera_pose();detections=[]
        for i,ids in enumerate(s.human_ids,1):
            p=np.asarray(rt.sim.get_agent_data(i).articulated_agent.base_pos,dtype=float)
            target=p+[0,s.config.torso_height_m,0]
            if in_frustum(target[None],origin,rotation,s.config)[0] and s.clear_ray(origin,target,ids):
                detections.append(HumanDetection(t,tuple(p)))
        return dict(timestamp=t,robot_pose=pose,tracks=s.tracker.update(t,sorted(detections,key=lambda d:d.position)),
                    robot_joint_positions=np.asarray(robot.sim_obj.joint_positions,dtype=float).tolist(),
                    camera_origin=origin.tolist(),camera_rotation=rotation.tolist(),static_updates=[],
                    observation_valid=True,human_sensor_available=True,
                    sensor_protocol='ideal_frustum_observed_cspace_dynamic_occlusion_1')


def install_metrics(rt,hmap):
    from habitat.tasks.nav.nav import DistanceToGoal,Success,SPL
    from falcon.additional_metric import STL,PersonalSpaceCompliance,STLMeasurementConfig,PersonalSpaceComplianceMeasurementConfig
    from habitat_baselines.config.default import get_config
    cfg=get_config(hmap.contract['config_name']).habitat.task.measurements
    for key,cls in (('distance_to_goal',DistanceToGoal),('success',Success),('spl',SPL),('stl',STL),('psc',PersonalSpaceCompliance)):
        config=cfg.get(key,STLMeasurementConfig() if key=='stl' else PersonalSpaceComplianceMeasurementConfig())
        measure=cls(sim=rt.sim,config=config);rt.task.measurements.measures[key]=measure
        measure.reset_metric(episode=rt.episode,task=rt.task,observations=rt.obs)


def make_episode(case,hmap):
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    d=case['original_episode'];info=dict(d['info']);info['graph_navmesh']=hmap.navigation.path
    scene=case.get('environment_overrides',{}).get('scene_id',d['scene_id'])
    return NavigationEpisode(episode_id=str(d['episode_id']),scene_id=scene,start_position=d['start_position'],
        start_rotation=d['start_rotation'],goals=[NavigationGoal(**g) for g in d['goals']],info=info)


def static_snapshot(geometry,timestamp,pose):
    cells=[]
    offset=np.rint(geometry.origin/geometry.resolution).astype(int)
    for i,j in np.argwhere(geometry.free):
        cells.append(dict(cell=[int(i+offset[0]),int(j+offset[1])],state=1,height=float(geometry.heights[i,j])))
    return dict(timestamp=timestamp,resolution=geometry.resolution,height=pose[1],revision=1,cells=cells)


def run_case(case,variant,folder,hmap=None,model=None):
    import habitat_sim
    from eventwm_graph.benchmark.map import load_certified_map
    from eventwm_graph.stage1.runtime import NativeEpisode
    from eventwm_graph.partial.metrics import step_labels
    from eventwm_graph.partial.query import yaw_to
    from .reference import QuerySpec,compile_reference,ActionSpec
    from .geometry import StaticGrid
    from .executor import Executor,RuntimeContext,Status
    from .forecast import ForecastCache
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();record=dict(case_id=case['case_id'],variant=variant,seed=case['seed'],
        scene_id=case['scene_id'],status='RUNNING',case=case,static_access='full certified static map',
        human_access='legal exact jaw-frustum visible centers; no oracle future',
        checkpoint_sha256=file_digest(CHECKPOINT) if variant=='B2' else None)
    (folder/'case.json').write_text(json.dumps(record,indent=2))
    if hmap is None:hmap=load_certified_map(case['manifest'])
    ep=make_episode(case,hmap)
    latencies=[];environment_ms=[];update_ms=[];labels=[];observations=[];decisions=[];forecast_records=[]
    actions=[];poses=[];timestamps=[];compile_error=None;handle=None;forecast_cache=None
    transition_ms=[];online_ms=[];request_ms=[]
    with ExitStack() as resources, NativeEpisode(hmap,ep,case['seed'],visual=variant=='B3') as rt:
        install_metrics(rt,hmap);sensor=LegalObserver(rt);ob=sensor.observe();observations.append(ob)
        poses.append(ob['robot_pose']);timestamps.append(ob['timestamp']);goal=np.asarray(ep.goals[0].position)
        initial_metrics=rt.env.get_metrics();initial_distance=float(initial_metrics['distance_to_goal'])
        executor=Executor();spec=ActionSpec()
        if variant=='B3':
            from .falcon_policy import FalconPolicy
            policy=FalconPolicy(rt)
            record['static_access']='depth only; no explicit map';record['human_access']='depth only'
        else:
            # Fixed high-level reference for all B0/B1/B2 arms. Native shortest
            # path is a supplied static route, never an added certified edge.
            try:
                route=hmap.navigation.shortest_path(ob['robot_pose'][:3],goal)
                if route is None: raise ValueError('Disconnected native static reference')
                route=route.copy();route[0]=ob['robot_pose'][:3]
                geometry=StaticGrid.from_navigation(hmap.navigation,route)
                geometry.predictor=hmap.model
                if case.get('query_prefix_m'):
                    # A predeclared short selected query, not an L4 route choice.
                    remaining=float(case['query_prefix_m']);prefix=[route[0]]
                    for a,b in zip(route[:-1],route[1:]):
                        length=float(np.linalg.norm(b-a))
                        if length>=remaining:
                            prefix.append(a+(b-a)*remaining/max(length,1e-12));break
                        prefix.append(b);remaining-=length
                    route=np.asarray(prefix)
                yaw=ob['robot_pose'][3] if len(route)<2 else yaw_to(route[-2],route[-1])
                query=QuerySpec(case['case_id'],route,ob['robot_pose'],yaw,ob['timestamp'],execution_limit=480,duration_limit=26.88)
                if case.get('query_prefix_m'):
                    from .reference import from_stage1
                    query=from_stage1(dict(query_id=case['case_id'],final_yaw=yaw,
                                          compiled=dict(ok=True,route=route.tolist())),ob,execution_limit=480)
                ref=compile_reference(query,geometry,spec)
                (folder/'reference.json').write_text(json.dumps(asdict(ref),indent=2))
                handle=executor.start(query,ref,RuntimeContext(geometry,profile_id='l4.progress_first.'+variant+'.v2'))
                if variant=='B2':
                    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
                    if model is None:model=IntermittentWorldModel.load(CHECKPOINT)
                    graph=static_snapshot(geometry,ob['timestamp'],ob['robot_pose'])
                    t=time.perf_counter();belief=model.initialize([ob],graph);update_ms.append((time.perf_counter()-t)*1000)
                    forecast_cache=ForecastCache(model)
            except (ValueError,RuntimeError) as exc:compile_error=str(exc);record['status']='REFERENCE_UNSUPPORTED'
        nominal_index=0
        for _ in range(500):
            if rt.env.episode_over: break
            if variant=='B3': action=policy.act(rt)
            elif compile_error: break
            elif variant=='B0':
                if nominal_index>=len(ref.native_action_schedule):break
                t=time.perf_counter()
                action=ref.native_action_schedule[nominal_index];nominal_index+=1
                latencies.append((time.perf_counter()-t)*1000)
            else:
                forecast=None
                if variant=='B2':
                    request_start=time.perf_counter()
                    from .reactive import nominal_action,near_term_risk
                    target=handle.tracker.route[handle.tracker.next_waypoint]
                    nominal=nominal_action(np.asarray(ob['robot_pose']),target,query,
                                           handle.tracker.next_waypoint==len(handle.tracker.route)-1)
                    request=not handle.maneuver and handle.tracker.path_length>=handle.cooldown_distance and near_term_risk(
                        np.asarray(ob['robot_pose']),nominal,executor._people(ob,ob['timestamp']),spec)
                    request_ms.append((time.perf_counter()-request_start)*1000)
                    try:forecast=forecast_cache.get(belief,ob,seed=case['seed']) if request else None
                    except (ValueError,RuntimeError) as exc:record.setdefault('l2_failures',[]).append(str(exc))
                    if forecast_cache.last_failure:
                        record.setdefault('l2_failures',[]).append(forecast_cache.last_failure)
                    if forecast is not None and (not forecast_records or forecast.prediction_start_time!=forecast_records[-1]['timestamp']):
                        f=forecast.joint_future_reference
                        forecast_records.append(dict(timestamp=forecast.prediction_start_time,times=f.times.tolist(),
                            positions=f.positions.tolist(),source_belief_version=forecast.source_belief_version))
                d=executor.act(handle,ob,ob,forecast);decisions.append(asdict(d));latencies.append(d.control_ms)
                action=d.action
                if action is None:break
            before=time.perf_counter();rt.step(action);ob=sensor.observe();environment_ms.append((time.perf_counter()-before)*1000)
            actions.append(action);poses.append(ob['robot_pose']);timestamps.append(ob['timestamp']);observations.append(ob)
            label=step_labels(rt,ob['timestamp']);labels.append(label)
            if variant in ('B1','B2'):
                limit=rt.frames>=hmap.contract['max_episode_steps']
                t=time.perf_counter()
                executor.on_transition(handle,dict(action=action,timestamp=ob['timestamp'],robot_pose=ob['robot_pose'],
                    observation_ref=f'actual:{len(observations)-1}',human_collision=bool(rt.env.get_metrics()['human_collision']),
                    static_collision=label['static_obstacle_contact'],episode_termination=bool(rt.env.episode_over and not limit),episode_truncation=limit))
                transition_ms.append((time.perf_counter()-t)*1000)
                online_ms.append(latencies[-1]+transition_ms[-1]+(request_ms[-1] if variant=='B2' else 0.))
                if variant=='B2' and not rt.env.episode_over:
                    t=time.perf_counter()
                    try:belief=model.observe(belief,ob)
                    except (ValueError,RuntimeError) as exc:record.setdefault('l2_failures',[]).append(str(exc))
                    update_ms.append((time.perf_counter()-t)*1000)
                if handle.status!=Status.RUNNING:break
            if action=='stop':break
        if variant in ('B1','B2') and handle is not None:
            record['feedback']=executor.feedback(handle);record['status']=handle.status.value
            record['query_completed']=handle.status==Status.QUERY_COMPLETED
        elif variant=='B0' and not compile_error:
            record['query_completed']=bool(handle.tracker.complete(poses[-1]))
            # Completion checks must use all actual ordered route arrivals.
            for p,t in zip(poses,timestamps):handle.tracker.update(p,t)
            record['query_completed']=bool(handle.tracker.complete(poses[-1]))
            record['status']='QUERY_COMPLETED' if record['query_completed'] else 'INTERRUPTED'
        # Explicit experiment-driver handoff. STOP is never represented as yield.
        # Native success remains conditional on the unchanged goal threshold.
        if not rt.env.episode_over:
            rt.step('stop');ob=sensor.observe();actions.append('stop');poses.append(ob['robot_pose']);timestamps.append(ob['timestamp'])
            observations.append(ob);labels.append(step_labels(rt,ob['timestamp']))
        metrics={k:float(v) for k,v in rt.env.get_metrics().items() if isinstance(v,(int,float,bool,np.number))}
        if variant=='B3':record['status']='NATIVE_SUCCESS' if metrics.get('success') else 'NATIVE_FAILURE'
        record.update(metrics=metrics,compile_error=compile_error,initial_distance=initial_distance,
            human_collision=bool(metrics.get('human_collision')),static_collision=any(l['static_obstacle_contact'] for l in labels),
            path_length=float(np.linalg.norm(np.diff(np.asarray(poses)[:,:3],axis=0),axis=1).sum()),
            actual_duration=timestamps[-1]-timestamps[0],native_actions=len(actions),
            timeout=rt.frames>=500,blocked=record['status']=='BLOCKED',
            l4_ms=latencies,environment_ms=environment_ms,l2_update_ms=update_ms,
            l4_transition_ms=transition_ms,l4_online_ms=online_ms,l2_request_ms=request_ms,
            l2_refresh_ms=forecast_cache.refresh_ms if forecast_cache else [])
    record['wall_seconds']=time.perf_counter()-began
    (folder/'actual.json').write_text(json.dumps(dict(actions=actions,poses=poses,timestamps=timestamps,
        observations=observations,decisions=decisions,forecasts=forecast_records,offline_labels=labels),indent=2))
    (folder/'report.json').write_text(json.dumps(record,indent=2))
    return record
