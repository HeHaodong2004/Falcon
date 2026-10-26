"""OFFLINE ENVIRONMENT: measure range at recorded branch endpoints.

The simulator/private initialization remain behind the sensing boundary. Each
forecast receives only sensor frames in its already executed ancestor prefix.
No endpoint after that forecast's start may enter its predictor inputs.
"""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import habitat_sim
import numpy as np
from .diagnose import OUT, HF, read, save, sha, source_rows, member
from .current_range_sensor import CurrentRangeSensor
from .range_history import ancestor_frames
from eventwm_graph.stage1.runtime import NativeEpisode, SharedFuture


def run(roots=2110, offset=0, split='dev',sensor_size=128,destination=None):
    import magnum as mn
    import falcon.additional_action as native_actions
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    rows, manifest_path = source_rows(split)
    if split != 'dev':raise ValueError('full preparation is development-only, no confirmation set')
    rows = rows[offset:offset+roots]
    if sensor_size not in (128,256):raise ValueError('supported current sensor sizes128/256')
    if sensor_size!=128 and destination is None:raise ValueError('resolution change requires separate sensor artifacts')
    if destination is not None and Path(destination).name!=destination:raise ValueError('destination must be a simple artifact directory name')
    dest = OUT/(destination or'full_range_branches')
    dest.mkdir(exist_ok=True)
    for row in rows:
        path = dest/(row['root_id']+'.npz')
        receipt = path.with_suffix('.json')
        identity = dict(role='ENVIRONMENT-only past/current range sensor extension',
            sensor_size=sensor_size,
            source_manifest_sha256=sha(manifest_path), files=row['files'],
            sources={p:sha(Path(__file__).with_name(p)) for p in ['range_full_replay.py','range_branch_replay.py','current_range_sensor.py']})
        sources=dest/'sources';sources.mkdir(exist_ok=True)
        for name,digest in identity['sources'].items():
            snapshot=sources/(digest+'_'+name)
            if not snapshot.exists():snapshot.write_bytes(Path(__file__).with_name(name).read_bytes())
        if receipt.exists():
            prior=read(receipt)
            matched_source=prior['identity']==identity
            # A previously measured pilot frame packet is exactly the same
            # legal sensor input. Preserve its ORIGINAL producer receipt.
            equivalent_pilot=(sensor_size==256 and prior['identity'].get('sensor_size')==256 and prior['identity']['files']==row['files']and prior['identity']['sources'].get('current_range_sensor.py')==identity['sources']['current_range_sensor.py']and max(prior['max_errors'].values())<=1.e-6)
            if not(matched_source or equivalent_pilot)or prior['sha256']!=sha(path):
                raise ValueError('range replay cache identity changed')
            print('REUSE',row['root_id'],flush=True)
            continue
        legal=member(row,'legal_root.json.gz'); nodes=member(row,'future_observations.json.gz')
        private=member(row,'privileged_initialization.json.gz'); ep=private['episode']
        config=read(private['source_map_manifest'])['contract']
        seed=read(Path(row['folder'])/'manifest.json')['seed']
        episode=NavigationEpisode(episode_id=ep['episode_id'],scene_id=ep['scene_id'],start_position=ep['start_position'],start_rotation=ep['start_rotation'],goals=[NavigationGoal(**g)for g in ep['goals']],info=copy.deepcopy(ep['info']))
        tick=time.perf_counter(); errors=[]; depth=[]; origins=[]; rotations=[]; timestamps=[]
        with NativeEpisode(SimpleNamespace(contract=config),episode,seed,visual=False)as rt:
            sensor=CurrentRangeSensor(rt,legal['partial_graph']['resolution'],size=sensor_size)
            def capture(expected, timestamp):
                packet=sensor.measure();o,r=packet['camera_origin'],packet['camera_rotation']
                err=dict(origin=float(np.linalg.norm(o-expected['camera_origin'])),rotation=float(np.max(abs(r-np.asarray(expected['camera_rotation'])))),timestamp=abs(timestamp-expected['timestamp']))
                if err['origin']>1.e-4 or err['rotation']>1.e-5 or err['timestamp']>1.e-6:
                    raise ValueError('causal range replay pose/time mismatch '+repr(err))
                errors.append(err);depth.append(packet['range_image']);origins.append(o);rotations.append(r);timestamps.append(timestamp)
            capture(legal['history'][0],rt.sim.get_world_time())
            for action,expected in zip(legal['history_actions'],legal['history'][1:]):
                rt.step(action);capture(expected,rt.sim.get_world_time())
            history_count=len(depth)
            tape=SharedFuture(rt)
            levels=[]
            for node in nodes:levels.append(1 if node['parent']is None else levels[node['parent']]+1)
            tape.generate(max(levels,default=0))
            robot=rt.sim.get_agent_data(0).articulated_agent
            def state():
                return (mn.Matrix4(robot.sim_obj.transformation),np.asarray(robot.sim_obj.joint_positions).copy(),np.asarray(robot.sim_obj.joint_velocities).copy(),native_actions.play_i)
            root_state=(tape.robot_transform,tape.robot_joints,tape.robot_velocities,tape.gait_frame)
            states=[]
            for node,level in zip(nodes,levels):
                transform,joints,velocity,gait=root_state if node['parent']is None else states[node['parent']]
                robot.sim_obj.transformation=transform;robot.sim_obj.joint_positions=joints
                robot.sim_obj.joint_velocities=velocity;native_actions.play_i=gait
                rt.task.should_end=False;rt.task.is_stop_called=False
                tape._restore_humans(level-1);robot.update()
                rt.task.actions[rt.action_names[node['action']]].step(task=rt.task)
                tape._restore_humans(level);robot.update()
                states.append(state());capture(node['observation'],tape.times[level])
        temp=path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,depth=np.array(depth),origins=np.array(origins),rotations=np.array(rotations),timestamps=np.array(timestamps),history_count=np.int32(history_count),parents=np.array([-1 if n['parent']is None else n['parent']for n in nodes],np.int32))
        temp.replace(path)
        report=dict(identity=identity,sha256=sha(path),root_id=row['root_id'],scene_id=row['scene_id'],frames=len(depth),history_frames=history_count,branch_frames=len(nodes),max_errors={k:max(e[k]for e in errors)for k in errors[0]},seconds=time.perf_counter()-tick)
        save(receipt,report);print('COMPLETE',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=2110);p.add_argument('--offset',type=int,default=0);p.add_argument('--split',choices=['train','dev'],default='dev');p.add_argument('--sensor-size',type=int,default=128);p.add_argument('--destination');run(**vars(p.parse_args()))
