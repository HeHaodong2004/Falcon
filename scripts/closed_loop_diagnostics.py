"""Bounded native actuator checks, kept separate from benchmark episodes."""
import random
import time
import numpy as np
import torch


def timing_smoke(args,guard):
    import habitat
    from eventwm_closed_loop import read,save,config_for
    from closed_loop_runtime import pose,StaticGeometry
    from collector.core import LegalRoot,compile_motion
    from eventquerywm.navigation import NavigationConfig,GoalQueryGenerator,TimedWaypointController
    guard.check()
    torch.set_num_threads(1)
    row=read(args.output/'episodes.json')['rows'][0]
    directory=args.output/'control_timing_audit'
    cfg=config_for(args,row,directory)
    if not args.render:
        from closed_loop_nonvisual import install
        install()
    human_actions=tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1,7))
    controller=TimedWaypointController();results=[];begin=time.time()
    with habitat.Env(config=cfg.habitat) as env:
        def step(command,stop=False):
            guard.check()
            if (args.output/'STOP_RESEARCH').exists() or time.time()-begin>180:
                raise RuntimeError('Stopped optional timing audit only')
            action=env.task.actions['agent_0_eventwm_velocity']
            values={name:np.asarray([1.],dtype=np.float32) for name in human_actions}
            if not stop:
                values['agent_0_base_vel']=np.asarray([command[0]/action._lin_speed,
                    command[1]/action._ang_speed],dtype=np.float32)
            env.step(dict(action=('agent_0_discrete_stop' if stop else 'agent_0_eventwm_velocity',*human_actions),
                          action_args=values))
            assert not env.episode_over or stop, 'Unexpected early terminal during actuator diagnostic'

        for timing in ('normal','slow','wait'):
            random.seed(row['seed']);np.random.seed(row['seed']);env.seed(row['seed']);env.reset()
            sim=env.sim;clock=[]
            for _ in range(10):
                before=float(sim.get_world_time());step([0.,0.]);clock.append(float(sim.get_world_time())-before)
            dt=float(np.median(clock));p,yaw=pose(sim)
            root=LegalRoot('actuator-audit',tuple(p),yaw,dt,1/sim.ctrl_freq)
            if timing=='wait':
                q=compile_motion(root,[p],[0.,0.],'WAIT',10*dt,0.,dict(speed=.7,yaw_speed=1.5)).data
                indices=range(10)
            else:
                candidates,_=GoalQueryGenerator(StaticGeometry(sim),NavigationConfig()).generate(root,env.current_episode.goals[0].position)
                q=next(c['query'] for c in candidates if c['family']=='route/'+timing)
                start=next(i for i,c in enumerate(q['executable_controls']) if abs(c[0])>1e-9)
                for index in range(start):
                    p,yaw=pose(sim);command,_=controller.command(q,index,p,yaw);step(command)
                indices=range(start,start+10)
            p0,y0=pose(sim);t0=float(sim.get_world_time());records=[]
            for index in indices:
                p,yaw=pose(sim);command,tracking=controller.command(q,index,p,yaw)
                step(command);p1,y1=pose(sim)
                records.append(dict(index=index,command_native=command,robot_position=p1,robot_yaw=y1,
                                    timestamp=float(sim.get_world_time()),tracking=tracking))
            elapsed=float(sim.get_world_time())-t0
            length=float(sum(np.linalg.norm((np.asarray(b)-a)[[0,2]]) for a,b in
                       zip([p0]+[r['robot_position'] for r in records[:-1]],[r['robot_position'] for r in records])))
            results.append(dict(timing=timing,steps=10,elapsed_s=elapsed,path_length_m=length,
                observed_speed_mps=length/elapsed,start_position=p0,end_position=p1,
                episode_over_before_stop=bool(env.episode_over),records=records))
        step([0.,0.],stop=True)
        stop_terminal=bool(env.episode_over)
    normal,slow,wait=results
    ratio=slow['path_length_m']/normal['path_length_m']
    assert normal['path_length_m']>.1 and .48<ratio<.52
    assert wait['path_length_m']<1e-6 and not wait['episode_over_before_stop'] and stop_terminal
    result=dict(scope='Native actuator diagnostic; excluded from navigation benchmark totals',
                row=row,results=results,slow_to_normal_distance_ratio=ratio,stop_is_terminal=stop_terminal,
                wait_is_nonterminal=True,passed=True)
    save(directory/'timing_audit.json',result)
    print(dict(stage='native_timing_audit_passed',ratio=ratio,wait_distance_m=wait['path_length_m']),flush=True)
    return result
