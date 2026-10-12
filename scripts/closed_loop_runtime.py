"""Native SocialNav loop; privileged evaluation stays outside policy inputs."""
import math
import signal
import time
from pathlib import Path
import numpy as np

from collector.observation import (HabitatHumanObserver, HumanDetection,
                                   ObservationConfig, legal_tracks)
from eventquerywm.navigation import (NavigationConfig, GoalQueryGenerator,
    WorldModelRisk, RiskAwareSelector, TimedWaypointController)


def register_simulator():
    import habitat_sim
    from habitat.core.registry import registry
    from falcon.rearrange_sim_v2 import RearrangeSim_v2
    from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
    if registry.get_simulator('EventWMResearchSim-v0'):
        return

    @registry.register_simulator(name='EventWMResearchSim-v0')
    class EventWMResearchSim(RearrangeSim_v2):
        def _load_navmesh(self, episode):
            from hashlib import sha256
            cfg = self.habitat_config.agents.agent_0
            settings = habitat_sim.NavMeshSettings(); settings.set_defaults()
            settings.agent_radius = cfg.radius; settings.agent_height = cfg.height
            settings.agent_max_climb = cfg.max_climb; settings.agent_max_slope = cfg.max_slope
            settings.include_static_objects = True
            identity = sha256((episode.scene_id+repr((cfg.radius,cfg.height,cfg.max_climb,cfg.max_slope))).encode()).hexdigest()[:20]
            directory = Path(self.habitat_config.research_navmesh_dir); directory.mkdir(parents=True,exist_ok=True)
            path = directory/(identity+'.navmesh')
            if path.exists():
                assert self.pathfinder.load_nav_mesh(str(path))
            else:
                assert self.recompute_navmesh(self.pathfinder,settings)
                self.pathfinder.save_nav_mesh(str(path))
            self._largest_indoor_island_idx = get_largest_island_index(self.pathfinder,self,allow_outdoor=False)


class StaticGeometry:
    def __init__(self, sim):
        self.sim, self.pf = sim, sim.pathfinder

    def path(self, start, goal, max_projection=None):
        import habitat_sim
        goal = np.asarray(goal,dtype=float)
        end = np.asarray(self.pf.snap_point(goal),dtype=float)
        if not np.isfinite(end).all() or (max_projection is not None and np.linalg.norm(end-goal) > max_projection):
            return None, None
        path = habitat_sim.ShortestPath();path.requested_start = np.asarray(start,dtype=float);path.requested_end = end
        if not self.pf.find_path(path):
            return None, None
        return [np.asarray(p).tolist() for p in path.points],float(path.geodesic_distance)

    def feasible(self, query):
        # Static PF only. try_step neither steps physics nor accesses humans.
        trajectory = query['nominal_timed_trajectory']
        for a,b in zip(trajectory,trajectory[1:]):
            start,end=np.asarray(a[1:4]),np.asarray(b[1:4])
            if np.linalg.norm((end-start)[[0,2]]) < 1e-7:
                continue
            actual=np.asarray(self.pf.try_step(start,end))
            if not np.isfinite(actual).all() or np.linalg.norm((actual-end)[[0,2]]) > .03:
                return False
        return True


def pose(sim):
    robot=sim.get_agent_data(0).articulated_agent
    forward=np.asarray(robot.base_transformation.transform_vector([1.,0.,0.]))
    return np.asarray(robot.base_pos,dtype=float).tolist(),math.atan2(-forward[2],forward[0])


class LegalObservationAdapter(HabitatHumanObserver):
    """Current visible detections only; additional human/dynamic ray occlusion.

    Simulator IDs stay inside the sensor layer, solely to ignore the target's own
    surface. The tracker receives anonymous detections and estimates velocity
    from past visible measurements; it never receives simulator velocities.
    """
    def observe(self,timestamp,active_count):
        import habitat_sim
        origin,rotation=self.camera_pose()
        robot=self.sim.get_agent_data(0).articulated_agent.sim_obj
        robot_ids={robot.object_id,*robot.link_object_ids.keys()}
        detections=[];witnesses=[]
        for i in range(1,active_count+1):
            agent=self.sim.get_agent_data(i).articulated_agent
            obj=agent.sim_obj;ignore=robot_ids|{obj.object_id,*obj.link_object_ids.keys()}
            current=np.asarray(agent.base_pos,dtype=float)
            def ray_clear(a,b):
                delta=b-a;length=float(np.linalg.norm(delta))
                ray=habitat_sim.geo.Ray(a,delta/length)
                for hit in sorted(self.sim.cast_ray(ray,max_distance=length).hits,key=lambda h:h.ray_distance):
                    if hit.object_id not in ignore and hit.ray_distance < length-self.config.los_tolerance_m:
                        return False,dict(object_id=int(hit.object_id),distance_m=float(hit.ray_distance))
                return True,None
            visible,witness=self.detector.visibility.evaluate(origin,rotation,current,True,ray_clear)
            if visible:
                detections.append(HumanDetection(float(timestamp),tuple(current)))
            witnesses.append(dict(slot=i,visible=visible,witness=witness))
        detections.sort(key=lambda d:d.position)
        self.last_tracks=self.tracker.update(timestamp,detections)
        self.last_timestamp=timestamp
        self.diagnostics.append(dict(timestamp=float(timestamp),perception_only=True,humans=witnesses,
                                     detection_count=len(detections)))
        return legal_tracks(self.last_tracks)

    def metadata(self,command):
        p,yaw=pose(self.sim);t=float(self.sim.get_world_time())
        tracks=self.observe(t,self.active_count)
        return dict(timestamp=t,robot_position=p,robot_yaw=yaw,previous_command=list(command),
            human_tracks=tracks,tracking_available=True,
            sensor_keys=getattr(self,'rendered_sensor_keys',[]))


def run_episode(env,args,row,directory):
    import json
    from dataclasses import asdict
    from collector.core import LegalRoot
    from eventwm_closed_loop import save
    import cv2
    cv2.setNumThreads(1)
    obs=env.reset();sim,task=env.sim,env.task
    config=NavigationConfig()
    observation_cfg=ObservationConfig(**args.observation_config)
    observer=LegalObservationAdapter(sim,observation_cfg);observer.active_count=task._human_num
    observer.rendered_sensor_keys=['agent_0_articulated_agent_jaw_rgb','agent_0_articulated_agent_jaw_depth'] if args.render else []
    success_distance=float(task.measurements.measures['success']._config.success_distance)
    save(directory/'policy_configuration.json',dict(navigation=asdict(config),observation=asdict(observation_cfg),
        common_history_initialization_steps=10,goal_stop_distance=success_distance,
        occlusion='Stage, static/dynamic objects and other humans; target/self excluded',
        native_task_class=type(task).__name__,metrics=list(task.measurements.measures)))
    geometry=StaticGeometry(sim);generator=GoalQueryGenerator(geometry,config)
    selector=RiskAwareSelector(config,args.policy=='eventwm');controller=TimedWaypointController()
    wm=WorldModelRisk(args.checkpoint) if args.policy=='eventwm' else None
    human_actions=tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1,7))
    robot_action=task.actions['agent_0_eventwm_velocity']
    goal=np.asarray(env.current_episode.goals[0].position,dtype=float).tolist()
    start_position,start_yaw=pose(sim);_,initial_distance=geometry.path(start_position,goal)
    initial_time=float(sim.get_world_time());history=[observer.metadata([0.,0.])]
    trajectory=[start_position];issued_displacements=[];inferences=[];plans=[];frames=[];clock_deltas=[]
    interrupted=False;failure_tracking=False;failure_candidates=False;oscillations=0;replans=0
    recent_families=[];commands_count=0;position_errors=[];ever_stuck=False
    stop_requested=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.__setitem__(0,True))
    streams={name:(directory/(name+'.jsonl')).open('w') for name in ('observations','decisions','actions')}
    def log(name,body):
        streams[name].write(json.dumps(body,allow_nan=False)+'\n');streams[name].flush()
    log('observations',history[0])
    def official_metrics():
        return {k:float(v) if np.isfinite(v) else None for k,v in env.get_metrics().items() if isinstance(v,(int,float,bool,np.number))}
    def save_frame(observations,label):
        if not args.video or commands_count%2:
            return
        panels=[]
        for name in ('agent_0_articulated_agent_jaw_rgb','agent_0_third_rgb'):
            if name in observations:
                panel=np.asarray(observations[name])[...,:3].astype(np.uint8).copy()
                cv2.putText(panel,label[:35],(3,12),cv2.FONT_HERSHEY_SIMPLEX,.28,(255,255,255),1)
                panels.append(panel)
        if panels:
            frames.append(np.concatenate(panels,axis=1))
        elif not args.render:
            from habitat.utils.visualizations import maps
            top=env.get_metrics().get('top_down_map')
            if not top:
                return
            image=maps.colorize_topdown_map(top['map']).copy()
            shape=image.shape[:2]
            def point(position):
                a,b=maps.to_grid(position[2],position[0],shape,sim=sim)
                return int(b),int(a)
            robot_position,robot_yaw=pose(sim)
            for a,b in zip(trajectory,trajectory[1:]):
                cv2.line(image,point(a),point(b),(10,110,255),2)
            for i in range(1,observer.active_count+1):
                # Privileged evaluation/video overlay only, never returned as
                # policy history, candidate features or selector information.
                cv2.circle(image,point(np.asarray(sim.get_agent_data(i).articulated_agent.base_pos)),5,(230,70,70),-1)
            cv2.circle(image,point(goal),7,(50,190,50),2)
            cv2.circle(image,point(robot_position),5,(10,100,255),-1)
            forward=np.asarray(robot_position)+[.5*math.cos(robot_yaw),0,-.5*math.sin(robot_yaw)]
            cv2.arrowedLine(image,point(robot_position),point(forward),(10,100,255),2)
            image=cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
            banner=np.full((64,512,3),25,np.uint8)
            cv2.putText(banner,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(banner,'Native rollout; privileged top-down evaluation overlay',(5,45),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
            frames.append(np.concatenate([banner,image]))
    def step(command,query_id=None,index=None,tracking=None,stop=False):
        nonlocal obs,commands_count,history
        before=float(sim.get_world_time())
        action_name='agent_0_discrete_stop' if stop else 'agent_0_eventwm_velocity'
        values={name:np.asarray([1.],dtype=np.float32) for name in human_actions}
        if not stop:
            values['agent_0_base_vel']=np.asarray([command[0]/robot_action._lin_speed,
                                                command[1]/robot_action._ang_speed],dtype=np.float32)
            assert np.max(np.abs(values['agent_0_base_vel'])) <= 1+1e-6
        obs=env.step(dict(action=(action_name,*human_actions),action_args=values));commands_count+=1
        issued_displacements.append(abs(command[0])/sim.ctrl_freq)
        meta=observer.metadata(command);history.append(meta)
        clock_deltas.append(meta['timestamp']-before)
        history=[v for v in history if v['timestamp'] >= meta['timestamp']-config.history_window_s]
        p,y=pose(sim);trajectory.append(p)
        _,distance=geometry.path(p,goal)
        log('observations',meta)
        log('actions',dict(step=commands_count,timestamp=meta['timestamp'],action=action_name,
            command_native=list(command),action_args={k:v.tolist() for k,v in values.items()},
            query_id=query_id,control_index=index,robot_position=p,robot_yaw=y,goal_distance=distance,
            tracking=tracking,official_metrics=official_metrics(),episode_over=bool(env.episode_over)))
        save_frame(obs,f'{args.policy} step {commands_count}')
        return distance
    try:
        save_frame(obs,args.policy+' initial')
        # Common measured perception initialization; valid nonterminal native
        # zero controls count toward the official 500-step budget in both arms.
        for _ in range(10):
            if env.episode_over:
                break
            step([0.,0.])
        dt=float(np.median(clock_deltas)) if clock_deltas else 1/task._physics_target_sps+sim.ac_freq_ratio*sim.get_physics_time_step()
        integration_dt=1/sim.ctrl_freq
        save(directory/'clock.json',dict(control_dt=dt,integration_dt=integration_dt,initial_clock_deltas=clock_deltas,
             formula_dt=1/task._physics_target_sps+sim.ac_freq_ratio*sim.get_physics_time_step()))
        while not env.episode_over:
            if stop_requested[0] or (args.output/'STOP_RESEARCH').exists():
                interrupted=True;break
            p,yaw=pose(sim);_,distance=geometry.path(p,goal)
            if distance is not None and distance < success_distance:
                step([0.,0.],stop=True);continue
            begin=time.perf_counter()
            root=LegalRoot(f'{row["tag"]}/step{commands_count}',tuple(p),yaw,dt,integration_dt)
            candidates,generation=generator.generate(root,goal)
            if not candidates:
                failure_candidates=True
                # Failed geometry cannot invent a successful terminal state.
                # Bounded waiting continues to official timeout while recording.
                log('decisions',dict(step=commands_count,generation=generation,candidates=[],reason='no_feasible_progress'))
                step([0.,0.]);continue
            probabilities,latency=wm.evaluate(history,candidates) if wm else ([None]*len(candidates),dict(inference_s=0.,tensorization_and_inference_s=0.,history_length=len(history)))
            chosen,reason=selector.select(candidates,probabilities)
            plans.append(time.perf_counter()-begin);inferences.append(latency['inference_s'])
            selected=candidates[chosen];query=selected['query'];replans+=1
            recent_families.append(selected['family'])
            if len(recent_families)>=4 and recent_families[-4]==recent_families[-2] and recent_families[-3]==recent_families[-1] and recent_families[-1]!=recent_families[-2]:
                oscillations+=1
            log('decisions',dict(step=commands_count,timestamp=float(sim.get_world_time()),goal=goal,
                robot_position=p,robot_yaw=yaw,generation=generation,candidates=candidates,
                safe_probabilities=probabilities,selected=chosen,selected_query_id=query['query_id'],
                reason=reason,latency=latency,planning_s=plans[-1],consecutive_wait_s=selector.wait_s))
            execution_begin=float(sim.get_world_time())
            limit=min(len(query['executable_controls']),max(1,round(config.execution_s/dt)))
            for index in range(limit):
                if env.episode_over or stop_requested[0] or (args.output/'STOP_RESEARCH').exists():
                    break
                p,yaw=pose(sim);command,tracking=controller.command(query,index,p,yaw)
                position_errors.append(tracking['position_error_m'])
                if tracking['position_error_m'] > .2 or abs(tracking['heading_error_rad']) > .4:
                    failure_tracking=True;break
                step(command,query['query_id'],index,tracking)
                if len(trajectory)>80 and sum(issued_displacements[-80:])>.5:
                    moved=sum(np.linalg.norm((np.asarray(b)-a)[[0,2]]) for a,b in zip(trajectory[-81:-1],trajectory[-80:]))
                    if moved<.1:
                        ever_stuck=True
            selector.executed(selected,float(sim.get_world_time())-execution_begin)
            if stop_requested[0]:
                interrupted=True;break
    finally:
        for stream in streams.values():
            stream.close()
    metrics=official_metrics()
    success=float(metrics['success']);collision=float(metrics['human_collision'])
    terminal=bool(env.episode_over) and not interrupted
    termination='success' if success else 'human_collision' if collision else 'timeout' if commands_count>=500 else 'interrupted' if interrupted else 'other_terminal'
    path_length=float(sum(np.linalg.norm((np.asarray(b)-a)[[0,2]]) for a,b in zip(trajectory,trajectory[1:])))
    result=dict(valid_terminal=terminal,tag=row['tag'],scene_id=row['scene_id'],episode_id=row['episode_id'],
        seed=row['seed'],policy=args.policy,start_position=start_position,start_yaw=start_yaw,goal=goal,
        navigation_success=success,spl=float(metrics['spl']),psc=float(metrics['psc']),human_collision=collision,
        timeout=float(termination=='timeout'),stuck=float(ever_stuck),tracking_failure=failure_tracking,
        planner_oscillation=oscillations>5,candidate_failure=failure_candidates,termination_reason=termination,
        navigation_steps=commands_count,navigation_time_s=float(sim.get_world_time())-initial_time,
        path_length_m=path_length,initial_goal_distance_m=initial_distance,
        path_efficiency=min(1.,initial_distance/max(path_length,1e-6)) if initial_distance else 0.,
        wm_inference_ms=1000*float(np.mean(inferences)) if inferences else 0.,
        planning_ms=1000*float(np.mean(plans)) if plans else 0.,replanning_events=replans,
        wm_calls=len(inferences) if wm else 0,oscillation_events=oscillations,
        tracking_error_mean_m=float(np.mean(position_errors)) if position_errors else 0.,
        tracking_error_max_m=max(position_errors) if position_errors else 0.,official_metrics=metrics,
        policy_inputs='Only legal visible-track history, robot pose, goal and static NavMesh; no evaluation measures',
        video_configuration=('Actual jaw RGB and third-person camera' if args.render else 'Native pose trajectory on official top-down map; privileged human overlay; no rendered RGB')+', sampled every two native steps')
    save(directory/'metrics.json' if terminal else directory/'interrupted.json',result)
    save(directory/'robot_trajectory.json',trajectory)
    save(directory/'perception_audit.json',observer.diagnostics)
    if args.video and frames:
        import imageio.v2 as imageio
        with imageio.get_writer(str(directory/'navigation.mp4'),fps=1/(2*dt),codec='libx264',
                                ffmpeg_params=['-threads','1'],macro_block_size=16) as video:
            for frame in frames:
                video.append_data(frame)
        imageio.imwrite(directory/'representative_frame.png',frames[len(frames)//2])
    print(json.dumps(result),flush=True)
    return result
