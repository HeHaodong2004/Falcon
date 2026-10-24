"""Native Falcon episode adapter. The benchmark actions/controllers are unchanged.

The nonvisual mode retains Spot's live jaw SceneNode and the existing EventWM
structured perception protocol. Pixel sensors are optional validation baselines.
"""
import copy
import math
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
from ..config import REPO

_HEADLESS_INSTALLED=False


def install_headless():
    """Geometry-node adapter used by the prior validated nonvisual collector."""
    global _HEADLESS_INSTALLED
    if _HEADLESS_INSTALLED:return
    import habitat_sim
    key='agent_0_articulated_agent_jaw_depth'
    class GeometryNode:
        def __init__(self,node):self._sensor_object=SimpleNamespace(node=node)
        def close(self):pass
    def attach(sim):
        sensors=sim._Simulator__sensors[sim._default_agent_id]
        if key not in sensors:sensors[key]=GeometryNode(sim.agents[sim._default_agent_id].scene_node.create_child())
    original_sanitize=habitat_sim.Simulator._sanitize_config
    def sanitize(config):
        original_sanitize(config)
        config.sim_cfg.create_renderer=False;config.sim_cfg.requires_textures=False;config.sim_cfg.load_semantic_mesh=False
        if not config.sim_cfg.enable_physics:raise ValueError('Physics must remain enabled')
    habitat_sim.Simulator._sanitize_config=staticmethod(sanitize)
    original_post=habitat_sim.Simulator.__attrs_post_init__
    def post(self):original_post(self);attach(self)
    habitat_sim.Simulator.__attrs_post_init__=post
    original_reconfigure=habitat_sim.Simulator.reconfigure
    def reconfigure(self,*args,**kwargs):
        result=original_reconfigure(self,*args,**kwargs);attach(self);return result
    habitat_sim.Simulator.reconfigure=reconfigure
    original_observe=habitat_sim.Simulator.get_sensor_observations
    def observe(self,*args,**kwargs):
        sensors=self._Simulator__sensors[self._default_agent_id];node=sensors.pop(key,None)
        try:return original_observe(self,*args,**kwargs)
        finally:
            if node is not None:sensors[key]=node
    habitat_sim.Simulator.get_sensor_observations=observe
    _HEADLESS_INSTALLED=True


def register_sim():
    import habitat_sim
    from habitat.core.registry import registry
    from falcon.rearrange_sim_v2 import RearrangeSim_v2
    from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
    if registry.get_simulator('GraphV2SurfaceSim-v0') is not None:return
    @registry.register_simulator(name='GraphV2SurfaceSim-v0')
    class GraphV2SurfaceSim(RearrangeSim_v2):
        def _load_navmesh(self,episode):
            path=episode.info['graph_navmesh']
            if not self.pathfinder.load_nav_mesh(path):raise RuntimeError('Certified NavMesh load failed')
            self._largest_indoor_island_idx=get_largest_island_index(self.pathfinder,self,allow_outdoor=False)


def episode_for_map(hmap,start_node,seed,human_count):
    """Static-only initialization. All oracle initialization fields stay private."""
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    rng=np.random.default_rng(seed);pf=hmap.navigation.pf;pf.seed(int(seed))
    start=np.asarray(hmap.nodes[start_node]['position'])
    island=int(pf.get_island(start))
    def sample():return np.asarray(pf.get_random_navigable_point(island_index=island)).tolist()
    goal=sample();info={'human_num':int(human_count),'graph_navmesh':hmap.navigation.path}
    for i in range(human_count):
        points=[sample() for _ in range(3)]
        for k,p in enumerate(points):info[f'human_{i}_waypoint_{k}_position']=p
        info[f'human_{i}_waypoint_0_rotation']=float(rng.uniform(0,2*math.pi))
    return NavigationEpisode(episode_id=str(seed),scene_id=hmap.source.metadata['scene'],
        # This installed task's quaternion_to_rad_angle reads WXYZ, unlike
        # generic Habitat's XYZW episode convention. Record/use its real rule.
        start_position=start.tolist(),start_rotation=[1.,0.,0.,0.],goals=[NavigationGoal(position=goal)],info=info)


class NativeEpisode:
    def __init__(self,hmap,episode,seed,visual=False):
        import habitat_sim  # Required import order on the installed build.
        import habitat
        import falcon.additional_action
        import falcon.additional_metric
        from habitat.config import read_write
        from habitat_baselines.config.default import get_config
        from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
        from scripts.collector.observation import ObservationConfig,HabitatHumanObserver
        import torch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
        register_sim()
        cfg=get_config(hmap.contract['config_name'])
        self.observation_config=ObservationConfig.from_depth_sensor(cfg.habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor)
        with read_write(cfg):
            cfg.habitat.simulator.type='GraphV2SurfaceSim-v0'
            cfg.habitat.simulator.scene_dataset='default'
            cfg.habitat.simulator.concur_render=False
            cfg.habitat.task.lab_sensors={}
            # These are the unchanged benchmark collision/termination measures.
            # Pixel map/SPL/reward display is unnecessary for structured labels.
            cfg.habitat.task.measurements={k:v for k,v in cfg.habitat.task.measurements.items()
                if k in ('did_multi_agents_collide','human_collision','num_steps')}
            cfg.habitat.seed=int(seed)
            if not visual:
                for agent in cfg.habitat.simulator.agents.values():agent.sim_sensors={}
                install_headless()
            else:
                # Preserve exactly the configured jaw depth camera for parity;
                # optional RGB display is not part of this learning interface.
                for key,agent in cfg.habitat.simulator.agents.items():
                    if key=='agent_0':agent.sim_sensors={'jaw_depth_sensor':agent.sim_sensors.jaw_depth_sensor}
                    else:agent.sim_sensors={}
        dataset=PointNavDatasetV1();dataset.episodes=[episode]
        self.env=habitat.Env(config=cfg.habitat,dataset=dataset)
        self.env.seed(seed);self.obs=self.env.reset()
        # Match an independent fresh-process root. Falcon keeps this module
        # global across Env objects; it is part of replay state, not human RNG.
        falcon.additional_action.play_i=0
        self.sim=self.env.sim;self.task=self.env.task;self.count=episode.info['human_num']
        self.observer=HabitatHumanObserver(self.sim,self.observation_config)
        self.seed=seed;self.episode=episode;self.config=cfg
        self.human_actions=tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1,self.count+1))
        self.action_names={a:'agent_0_discrete_'+b for a,b in
            (('forward','move_forward'),('left','turn_left'),('right','turn_right'),('stop','stop'))}
        self.frames=0

    def truth(self):
        return [{'oracle_id':i,'position':np.asarray(self.sim.get_agent_data(i).articulated_agent.base_pos,dtype=float).tolist(),
                 'transform':np.asarray(self.sim.get_agent_data(i).articulated_agent.sim_obj.transformation,dtype=float).tolist(),
                 'offset_transform':np.asarray(self.sim.get_agent_data(i).articulated_agent.offset_transform,dtype=float).tolist(),
                 'joints':np.asarray(self.sim.get_agent_data(i).articulated_agent.sim_obj.joint_positions,dtype=float).tolist()}
                for i in range(1,self.count+1)]

    def observation(self):
        from ..robotmap.execution import signed_yaw
        timestamp=float(self.sim.get_world_time());robot=self.sim.get_agent_data(0).articulated_agent
        tracks=self.observer.observe(timestamp,self.count)
        origin,rotation=self.observer.camera_pose()
        return {'timestamp':timestamp,'robot_pose':[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)],
            'robot_joint_positions':np.asarray(robot.sim_obj.joint_positions,dtype=float).tolist(),
            'tracks':tracks,'observation_valid':True,'camera_origin':origin.tolist(),'camera_rotation':rotation.tolist(),
            'negative_evidence':{'kind':'static_geometric_visibility_reference','evaluated':True},
            'sensor_protocol':'eventwm_ideal_structured_jaw_v1','episode_over':bool(self.env.episode_over),
            'human_collision':bool(self.env.get_metrics().get('human_collision',0))}

    def step(self,action):
        if action not in self.action_names:raise ValueError('No invented WAIT/continuous action is supported')
        if self.env.episode_over:raise RuntimeError('Refusing post-termination observations')
        names=(self.action_names[action],*self.human_actions)
        args={name:np.array([1.]) for name in self.human_actions}
        self.obs=self.env.step({'action':names,'action_args':args});self.frames+=1
        return self.observation()

    def close(self):self.env.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def episode_record(episode):
    return {'episode_id':episode.episode_id,'scene_id':episode.scene_id,
        'start_position':list(episode.start_position),'start_rotation':list(episode.start_rotation),
        'goals':[{'position':list(g.position)} for g in episode.goals],
        'info':copy.deepcopy(episode.info),'rotation_semantics':'installed task WXYZ angle conversion'}


class SharedFuture:
    """One exogenous native-human realization, many counterfactual observers.

    This is a latent continuation. It has no legal observations after a branch
    terminates. Human controller calls and their ordering are untouched; they
    do not consult robot agent 0 under the pinned default configuration.
    Production use requires comparison against full Env.step branches.
    """
    def __init__(self,runtime):
        import magnum as mn
        import falcon.additional_action as actions
        self.rt=runtime;self.sim=runtime.sim;self.task=runtime.task
        robot=self.sim.get_agent_data(0).articulated_agent
        self.robot_transform=mn.Matrix4(robot.sim_obj.transformation)
        self.robot_joints=np.asarray(robot.sim_obj.joint_positions).copy()
        self.robot_velocities=np.asarray(robot.sim_obj.joint_velocities).copy()
        self.gait_frame=actions.play_i;self.root_observation=runtime.observation()
        self.root_observer=copy.deepcopy(runtime.observer.tracker);self.prefix_steps=runtime.frames
        self.truth=[runtime.truth()];self.times=[float(self.sim.get_world_time())]

    def generate(self,steps):
        action={'action':self.rt.human_actions,'action_args':{n:np.array([1.]) for n in self.rt.human_actions}}
        for _ in range(steps):
            # Task/controller + physics continuation; episode termination is
            # evaluated separately per robot branch, never used to alter humans.
            self.task.step(action=action,episode=self.rt.episode)
            self.truth.append(self.rt.truth());self.times.append(float(self.sim.get_world_time()))
        return self

    def _restore_humans(self,index):
        import magnum as mn
        for row in self.truth[index]:
            actor=self.sim.get_agent_data(row['oracle_id']).articulated_agent
            obj=actor.sim_obj
            actor.offset_transform=mn.Matrix4(np.asarray(row['offset_transform']))
            obj.transformation=mn.Matrix4(np.asarray(row['transform']))
            obj.joint_positions=np.asarray(row['joints']);obj.joint_velocities=np.zeros_like(obj.joint_velocities)

    def evaluate(self,query):
        import falcon.additional_action as actions
        from scripts.collector.observation import HabitatHumanObserver
        from ..robotmap.execution import signed_yaw
        robot=self.sim.get_agent_data(0).articulated_agent
        robot.sim_obj.transformation=self.robot_transform;robot.sim_obj.joint_positions=self.robot_joints
        robot.sim_obj.joint_velocities=self.robot_velocities;actions.play_i=self.gait_frame
        self.task.should_end=False;self.task.is_stop_called=False
        collision=self.task.measurements.measures['human_collision'];collision._ever_collide=False
        observer=HabitatHumanObserver(self.sim,self.rt.observation_config)
        observer.tracker=copy.deepcopy(self.root_observer)
        self._restore_humans(0);robot.update()
        rows=[];termination='query_complete'
        action_sequence=query['actions']
        if len(action_sequence)>=len(self.times):raise ValueError('Shared future does not cover the complete query')
        for i,action in enumerate(action_sequence,1):
            # Actual installed action and leg animation. Kinematic physics does
            # not integrate either actor; endpoint state parity is a pilot gate.
            self.task.actions[self.rt.action_names[action]].step(task=self.task)
            self._restore_humans(i);robot.update()
            self.task.measurements.measures['did_multi_agents_collide'].update_metric(task=self.task)
            collision.update_metric(episode=self.rt.episode,task=self.task,observations={})
            timestamp=self.times[i]
            tracks=observer.observe(timestamp,self.rt.count)
            origin,rotation=observer.camera_pose()
            human_collision=bool(collision.get_metric())
            ended=bool(self.task.should_end or self.prefix_steps+i>=self.rt.config.habitat.environment.max_episode_steps)
            rows.append({'timestamp':timestamp,'robot_pose':[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)],
                'robot_joint_positions':np.asarray(robot.sim_obj.joint_positions,dtype=float).tolist(),
                'tracks':tracks,'observation_valid':True,'camera_origin':origin.tolist(),'camera_rotation':rotation.tolist(),
                'negative_evidence':{'kind':'static_geometric_visibility_reference','evaluated':True},
                'sensor_protocol':'eventwm_ideal_structured_jaw_v1','episode_over':ended,'human_collision':human_collision})
            if ended:
                termination='human_collision' if human_collision else 'stop' if self.task.is_stop_called else 'episode_action_limit'
                break
        return {'observations':rows,'termination':termination,'censored':len(rows)<len(action_sequence),
                'executed_actions':len(rows),'planned_actions':len(action_sequence),
                'shared_future_indices':list(range(1,len(rows)+1)),
                'implementation':'shared_exogenous_native_humans_actual_robot_actions_v1',
                'production_equivalence_requires_pilot':True}
