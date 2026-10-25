"""Bounded causal replay of PAST camera depth, not privileged WM inputs.

Private initialization is used only by the offline environment reconstruction.
The predictor will receive rendered current/past sensor depth, never the scene
mesh, navigation support, initialization, or future observations/camera poses.
"""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import habitat_sim
import numpy as np
from .diagnose import OUT,save,read,source_rows,member
from eventwm_graph.stage1.runtime import NativeEpisode
from scripts.collector.observation import HabitatHumanObserver


def run(roots=2,split='dev',software=False,range_camera=False,benchmark=False):
    if software:
        # Private process-only software EGL fallback. No system driver or
        # collector configuration is changed. Magnum's -1 CUDA sentinel uses
        # EGL device selection instead of CUDA/EGL interop.
        import os
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
        os.environ['MAGNUM_DEVICE']='0'
        from habitat_baselines.config import default as config_module
        from habitat.config import read_write
        original=config_module.get_config
        def get_config(*args,**kwargs):
            cfg=original(*args,**kwargs)
            with read_write(cfg):cfg.habitat.simulator.habitat_sim_v0.gpu_device_id=-1
            return cfg
        config_module.get_config=get_config
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    rows,_=source_rows(split)
    if benchmark:
        from eventwm_graph.stage2_segment.benchmark import inputs
        ids={r['root_id']for r in inputs('multi')};rows=[r for r in rows if r['root_id']in ids]
    elif split=='dev':
        from .diagnose import HF
        ids=set(read(HF/'protocol.json')['pilot_selection_roots']);rows=[r for r in rows if r['root_id']in ids]
    reports=[]
    for row in rows[:roots]:
        legal=member(row,'legal_root.json.gz');private=member(row,'privileged_initialization.json.gz');ep=private['episode'];source=read(private['source_map_manifest']);manifest=read(Path(row['folder'])/'manifest.json');hmap=SimpleNamespace(contract=source['contract'])
        episode=NavigationEpisode(episode_id=ep['episode_id'],scene_id=ep['scene_id'],start_position=ep['start_position'],start_rotation=ep['start_rotation'],goals=[NavigationGoal(**g)for g in ep['goals']],info=copy.deepcopy(ep['info']));tick=time.perf_counter()
        with NativeEpisode(hmap,episode,manifest['seed'],visual=not range_camera)as rt:
            print('INIT',row['scene_id'],'seconds',time.perf_counter()-tick,'sensor_keys',list(rt.sim.get_sensor_observations()),'history_actions',len(legal['history_actions']),flush=True)
            observer=HabitatHumanObserver(rt.sim,rt.observation_config);depth=[];origins=[];rotations=[];times=[];errors=[]
            range_sensor=None
            if range_camera:
                from .current_range_sensor import CurrentRangeSensor
                range_sensor=CurrentRangeSensor(rt,legal['partial_graph']['resolution'])
            def capture(expected):
                o,r=observer.camera_pose()
                if range_sensor is not None:image=range_sensor.measure()['range_image']
                else:
                    raw=rt.sim.get_sensor_observations()
                    image=np.asarray(raw[rt.observation_config.sensor_uuid],np.float32)
                depth.append(image);origins.append(o);rotations.append(r)
                times.append(float(rt.sim.get_world_time()))
                errors.append(dict(origin=float(np.linalg.norm(o-expected['camera_origin'])),rotation=float(np.max(abs(r-np.asarray(expected['camera_rotation'])))),timestamp=abs(times[-1]-expected['timestamp'])))
            capture(legal['history'][0])
            for action,expected in zip(legal['history_actions'],legal['history'][1:]):
                rt.step(action);capture(expected)
            report=dict(role='Causal PAST replay; IDEAL CURRENT RANGE SENSOR EXTENSION, not rendered RGB-D, not a WM forecast score.'if range_camera else'Causal replay of actually PAST rendered depth. Not a WM forecast score.',root_id=row['root_id'],scene_id=row['scene_id'],human_count=ep['info']['human_num'],frames=len(depth),shape=depth[-1].shape,depth_quantiles=np.nanquantile(depth[-1],[0,.01,.5,.99,1]).tolist(),camera_errors=errors,seconds=time.perf_counter()-tick)
            if max(e['origin']for e in errors)>1.e-4 or max(e['rotation']for e in errors)>1.e-5 or max(e['timestamp']for e in errors)>1.e-6:raise ValueError('past replay camera/time mismatch: '+repr(report))
            dest=OUT/('range_benchmark'if benchmark else'range_replay'if range_camera else'depth_replay');dest.mkdir(exist_ok=True);path=dest/(row['root_id']+'.npz');temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,depth=np.array(depth),origins=np.array(origins),rotations=np.array(rotations),timestamps=np.array(times));temp.replace(path);reports.append(report);save(dest/'probe.json',reports);print('REPLAY',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=2);p.add_argument('--split',default='dev');p.add_argument('--software',action='store_true');p.add_argument('--range-camera',action='store_true');p.add_argument('--benchmark',action='store_true');run(**vars(p.parse_args()))
