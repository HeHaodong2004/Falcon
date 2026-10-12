#!/usr/bin/env python3
"""Isolated native-rendering diagnostics and validated recorded-action replay."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
DEFAULT=PROJECT/'output/eventwm_native_visualization_20261012'
SOURCE=PROJECT/'output/eventwm_closed_loop_20261012'
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'habitat-lab'),str(ROOT/'habitat-baselines'),str(ROOT/'scripts')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')


def diagnose(args):
    egl=C.CDLL('libEGL.so.1')
    egl.eglGetProcAddress.restype=C.c_void_p
    egl.eglGetProcAddress.argtypes=[C.c_char_p]
    def function(name,result,types):
        pointer=egl.eglGetProcAddress(name.encode());assert pointer,name
        return C.CFUNCTYPE(result,*types)(pointer)
    query=function('eglQueryDevicesEXT',C.c_uint,[C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int)])
    strings=function('eglQueryDeviceStringEXT',C.c_char_p,[C.c_void_p,C.c_int])
    attrib=function('eglQueryDeviceAttribEXT',C.c_uint,[C.c_void_p,C.c_int,C.POINTER(C.c_ssize_t)])
    devices=(C.c_void_p*16)();count=C.c_int();assert query(16,devices,C.byref(count))
    result=dict(environment={name:os.environ.get(name) for name in
        ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','__EGL_VENDOR_LIBRARY_FILENAMES','MAGNUM_GPU_VALIDATION')},devices=[])
    for i in range(count.value):
        cuda=C.c_ssize_t(-999);ok=attrib(devices[i],0x323A,C.byref(cuda))
        result['devices'].append(dict(egl_index=i,extensions=(strings(devices[i],0x3055) or b'').decode(),
            cuda_attribute_success=bool(ok),cuda_device=cuda.value,
            drm_device=(strings(devices[i],0x3233) or b'').decode(),drm_render_node=(strings(devices[i],0x3377) or b'').decode()))
    try:
        cuda=C.CDLL('libcuda.so.1');init=cuda.cuInit(0);number=C.c_int();status=cuda.cuDeviceGetCount(C.byref(number))
        result['cuda']=dict(init_status=init,count_status=status,count=number.value)
    except OSError as e:
        result['cuda']=dict(error=str(e))
    result['loaded_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(name in line for name in ('libEGL','libcuda','libnvidia','libGLX')) and '/' in line})
    args.output.mkdir(parents=True,exist_ok=True)
    mode='system' if args.system_egl else 'private'
    (args.output/f'egl_devices_cvd{os.environ.get("CUDA_VISIBLE_DEVICES","all").replace(",","_")}_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def probe(args):
    import habitat_sim
    import magnum as mn
    import numpy as np
    import imageio.v2 as imageio
    cfg=habitat_sim.SimulatorConfiguration();cfg.gpu_device_id=args.gpu_id
    cfg.scene_id='/home/haodong/workspace/baselines/Falcon/data/scene_datasets/hm3d/val/00889-HMkoS756sz6/HMkoS756sz6.basis.glb'
    cfg.enable_physics=True;cfg.load_semantic_mesh=False
    agent=habitat_sim.agent.AgentConfiguration()
    sensor=habitat_sim.CameraSensorSpec();sensor.uuid='rgb';sensor.sensor_type=habitat_sim.SensorType.COLOR
    sensor.resolution=[240,320];sensor.position=mn.Vector3(0,1.5,0);agent.sensor_specifications=[sensor]
    print(json.dumps(dict(stage='create_native_renderer',gpu_id=args.gpu_id,cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent])) as sim:
        state=habitat_sim.AgentState();state.position=[2.064392,.171484,-3.817842]
        sim.initialize_agent(0,state);rgb=sim.get_sensor_observations()['rgb'][...,:3]
        args.output.mkdir(parents=True,exist_ok=True);path=args.output/f'native_probe_gpu{args.gpu_id}.png'
        imageio.imwrite(path,rgb)
        print(json.dumps(dict(stage='native_rgb_ok',path=str(path),shape=list(rgb.shape),std=float(np.std(rgb)))),flush=True)


def prepare(args):
    """Extract cached matching GL libraries into research output; never install."""
    package=Path('/var/cache/apt/archives/libnvidia-gl-595_595.91.07-0ubuntu0.22.04.1_amd64.deb')
    assert package.exists()
    assert '595.91.07' in Path('/proc/driver/nvidia/version').read_text(),'Cached graphics runtime does not match the running kernel'
    directory=args.output/'graphics_runtime'
    directory.mkdir(parents=True,exist_ok=True)
    subprocess.run(['dpkg-deb','--extract',str(package),str(directory)],check=True)
    lib=directory/'usr/lib/x86_64-linux-gnu'
    assert (lib/'libEGL_nvidia.so.595.91.07').exists()
    vendor=directory/'research_nvidia_egl.json'
    vendor.write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(lib/'libEGL_nvidia.so.595.91.07'))))+'\n')
    import hashlib
    manifest=dict(package=str(package),package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        library_directory=str(lib),vendor_json=str(vendor),kernel=Path('/proc/driver/nvidia/version').read_text(),
        scope='Process-private graphics libraries extracted from existing cache; system drivers/CUDA/Conda untouched')
    (args.output/'graphics_runtime.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def replay(args):
    """Native action replay, checked against original robot/human/metric logs."""
    import random
    import shutil
    import numpy as np
    import torch
    import habitat
    import falcon
    import cv2
    from omegaconf import OmegaConf,open_dict
    from habitat.config import read_write
    from habitat.utils.visualizations.utils import observations_to_image,images_to_video
    from eventwm_closed_loop import read,save,sha,research_guard,research_lock
    from closed_loop_runtime import register_simulator,pose,LegalObservationAdapter
    from collector.observation import ObservationConfig
    torch.set_num_threads(1);cv2.setNumThreads(1)
    lock=research_lock(args.output);guard=research_guard(args.output);guard.check()
    source=args.source/'runs'/args.policy/args.tag
    original=read(source/'metrics.json');assert original['valid_terminal']
    provenance=read(source/'provenance.json');row=provenance['row']
    assert sha(row['episode_file'])==row['episode_sha256']
    actions=[json.loads(line) for line in (source/'actions.jsonl').read_text().splitlines()]
    human_truth={round(v['timestamp'],8):v for v in read(source/'perception_audit.json')}
    legal=[json.loads(line) for line in (source/'observations.jsonl').read_text().splitlines()]
    assert len(actions)==original['navigation_steps']
    directory=args.output/'replays'/args.policy/args.tag
    assert not (directory/'validation.json').exists(),'Completed replay preserved'
    directory.mkdir(parents=True,exist_ok=True);frames_dir=directory/'frames';frames_dir.mkdir(exist_ok=True)
    # Private immutable NavMesh reuse gives rendering the same static geometry.
    navmeshes=args.output/'navmeshes';navmeshes.mkdir(exist_ok=True)
    for mesh in (args.source/'navmeshes').glob('*.navmesh'):
        if not (navmeshes/mesh.name).exists():shutil.copyfile(mesh,navmeshes/mesh.name)
    register_simulator();cfg=OmegaConf.load(source/'resolved_config.yaml')
    with read_write(cfg):
        cfg.habitat.simulator.research_navmesh_dir=str(navmeshes)
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id=args.gpu_id
        cfg.habitat.simulator.concur_render=False
        sensors=cfg.habitat.simulator.agents.agent_0.sim_sensors
        # Keep the actual jaw-depth calibration/FoV unchanged. Only diagnostic
        # RGB resolution and Falcon's native third-person RGB view are added.
        sensors.jaw_rgb_sensor.height=sensors.jaw_rgb_sensor.width=512
        with open_dict(sensors):
            sensors.third_rgb_sensor=OmegaConf.create(OmegaConf.to_container(cfg.habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor,resolve=True))
            sensors.third_rgb_sensor.height=sensors.third_rgb_sensor.width=640
    OmegaConf.save(cfg,directory/'render_config.yaml',resolve=True)
    random.seed(row['seed']);np.random.seed(row['seed']);torch.manual_seed(row['seed'])
    metrics_max_error=0.;robot_max_error=0.;human_max_error=0.;yaw_max_error=0.;clock_max_error=0.
    observer_error=0.;frames=[];frame_rows=[];begin=time.time()
    with habitat.Env(config=cfg.habitat) as env:
        env.seed(row['seed']);observations=env.reset();sim=env.sim
        assert sim.renderer is not None,'Native renderer required'
        p,yaw=pose(sim);assert np.linalg.norm(np.asarray(p)-original['start_position'])<1e-5
        observer=LegalObservationAdapter(sim,ObservationConfig(**read(source/'policy_configuration.json')['observation']))
        observer.active_count=env.task._human_num
        initial_meta=observer.metadata([0.,0.])
        def check_legal(expected,actual):
            nonlocal observer_error
            a,b=expected['human_tracks'],actual['human_tracks']
            assert len(a)==len(b),'Rendering changed legal track count'
            for left,right in zip(a,b):
                for key in ('track_id','visible_now','valid','velocity_valid','num_observations'):
                    assert left[key]==right[key],('Legal observation mismatch',key,left,right)
                for key in ('position','estimated_velocity','first_seen_time','last_seen_time','observation_age'):
                    if left[key] is None or right[key] is None:
                        assert left[key] is None and right[key] is None
                        continue
                    observer_error=max(observer_error,float(np.max(np.abs(np.asarray(left[key])-right[key]))))
        check_legal(legal[0],initial_meta)
        def capture(step,meta):
            third=observations_to_image({'agent_0_third_rgb':observations['agent_0_third_rgb']},{})
            jaw=observations_to_image({'agent_0_articulated_agent_jaw_rgb':observations['agent_0_articulated_agent_jaw_rgb']},{})
            if step==0:
                import imageio.v2 as imageio
                imageio.imwrite(directory/'native_initial.png',third)
                save(directory/'native_sensor_keys.json',list(observations))
            cv2.imwrite(str(frames_dir/f'{step:04d}_third.jpg'),cv2.cvtColor(third,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(frames_dir/f'{step:04d}_jaw.jpg'),cv2.cvtColor(jaw,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
            if args.native_only:frames.append(third)
            p,yaw=pose(sim);origin,rotation=observer.camera_pose()
            forward=-np.asarray(rotation)[:,2]
            frame_rows.append(dict(step=step,timestamp=float(sim.get_world_time()),robot_position=p,robot_yaw=yaw,
                camera_position=np.asarray(origin).tolist(),camera_yaw=float(np.arctan2(-forward[2],forward[0])),
                human_tracks=meta['human_tracks']))
        capture(0,initial_meta)
        for index,record in enumerate(actions):
            guard.check()
            if (args.output/'STOP_RESEARCH').exists() or time.time()-begin>600:
                raise RuntimeError('Stopped visualization replay only')
            action_names=(record['action'],*tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1,7)))
            values={k:np.asarray(v,dtype=np.float32) for k,v in record['action_args'].items()}
            observations=env.step(dict(action=action_names,action_args=values))
            p,yaw=pose(sim);t=float(sim.get_world_time())
            robot_max_error=max(robot_max_error,float(np.linalg.norm(np.asarray(p)-record['robot_position'])))
            yaw_max_error=max(yaw_max_error,abs((yaw-record['robot_yaw']+np.pi)%(2*np.pi)-np.pi))
            clock_max_error=max(clock_max_error,abs(t-record['timestamp']))
            truth=human_truth[round(record['timestamp'],8)]
            for witness in truth['humans']:
                expected=np.asarray(witness['witness']['torso'])-[0.,1.,0.]
                actual=np.asarray(sim.get_agent_data(witness['slot']).articulated_agent.base_pos)
                human_max_error=max(human_max_error,float(np.linalg.norm(actual-expected)))
            measured={k:float(v) for k,v in env.get_metrics().items() if isinstance(v,(int,float,bool,np.number))}
            for key,target in record['official_metrics'].items():
                if target is not None:metrics_max_error=max(metrics_max_error,abs(measured[key]-target))
            assert bool(env.episode_over)==record['episode_over'],('Terminal mismatch',index)
            meta=observer.metadata(record['command_native']);check_legal(legal[index+1],meta)
            capture(index+1,meta)
        assert env.episode_over
        final_metrics={k:float(v) for k,v in env.get_metrics().items() if isinstance(v,(int,float,bool,np.number))}
    passed=max(robot_max_error,human_max_error,yaw_max_error,clock_max_error,observer_error)<1e-4 and metrics_max_error<1e-8
    validation=dict(passed=passed,policy=args.policy,tag=args.tag,seed=row['seed'],native_rendering=True,
        robot_max_position_error_m=robot_max_error,human_max_position_error_m=human_max_error,
        yaw_max_error_rad=yaw_max_error,clock_max_error_s=clock_max_error,legal_observation_max_error=observer_error,
        official_metric_max_error=metrics_max_error,action_count=len(actions),original_metrics=original['official_metrics'],
        replay_metrics=final_metrics,source_actions_sha256=sha(source/'actions.jsonl'),
        source_provenance=provenance,graphics_runtime=read(args.output/'graphics_runtime.json'),
        scope='Recorded original actions; native human actors; exact initialization and seed; no new policy/model decisions')
    save(directory/'validation.json',validation);save(directory/'frames.json',frame_rows)
    assert passed,validation
    if args.native_only:
        images_to_video(frames,str(directory),'native_falcon_rgb',fps=12,verbose=False,ffmpeg_params=['-threads','1'])
    print(json.dumps({k:v for k,v in validation.items() if k not in ('source_provenance','graphics_runtime','original_metrics')}),flush=True)


def compose(args):
    from native_planning_video import compose as run
    return run(args)


def verify(args):
    from eventwm_closed_loop import read,save,sha
    result={}
    for file in sorted((args.output/'videos').glob('*.mp4')):
        metadata=subprocess.check_output([str(Path(sys.prefix)/'bin/ffprobe'),'-v','error',
            '-show_entries','stream=codec_name,width,height,nb_frames,pix_fmt,avg_frame_rate,duration','-of','json',str(file)],text=True)
        subprocess.run([str(Path(sys.prefix)/'bin/ffmpeg'),'-v','error','-threads','1','-filter_threads','1',
            '-i',str(file),'-f','null','-'],check=True,capture_output=True)
        result[file.name]=dict(metadata=json.loads(metadata),full_decode_passed=True,sha256=sha(file),bytes=file.stat().st_size)
    assert len(result)==3
    save(args.output/'playability.json',result);print(json.dumps(result,indent=2))


def status(args):
    result=dict(replays=[dict(policy=p.parent.parent.name,tag=p.parent.name,passed=json.loads(p.read_text())['passed'])
                        for p in args.output.glob('replays/*/*/validation.json')],stop_requested=(args.output/'STOP_RESEARCH').exists())
    file=args.output/'visualization_process.json'
    if file.exists():
        process=json.loads(file.read_text());pid=process['pid']
        try:
            process['alive']=b'eventwm_native_render.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError,PermissionError):process['alive']=False
        result['process']=process
    print(json.dumps(result,indent=2))


def stop(args):
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'STOP_RESEARCH').write_text('Stop visualization only.\n')
    print('Visualization stop requested; collector untouched.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','diagnose','probe','replay','compose','verify','status','stop'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--gpu-id',type=int,default=0)
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--policy',choices=('static','eventwm'),default='eventwm')
    parser.add_argument('--tag',default='ep001_YRUkbU5xsYj_6');parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--system-egl',action='store_true',help='Read-only diagnosis of installed graphics stack, without private selection')
    parser.add_argument('--cuda-visible',default='1');parser.add_argument('--show-ground-truth',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    args.source=args.source.resolve()
    if args.output==args.source or args.source in args.output.parents or '/production/' in str(args.output) or str(args.output).startswith('/tmp/eventwm_v4_300k'):
        raise ValueError('Visualization output must be separate from original rollouts and protected production data')
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_visible
    current=os.getpriority(os.PRIO_PROCESS,0)
    if current<19:os.nice(19-current)
    if args.command in ('diagnose','probe','replay') and not args.system_egl and (args.output/'graphics_runtime.json').exists() and not os.environ.get('EVENTWM_PRIVATE_EGL'):
        runtime=json.loads((args.output/'graphics_runtime.json').read_text())
        os.environ['LD_LIBRARY_PATH']=runtime['library_directory']+':'+os.environ.get('LD_LIBRARY_PATH','')
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=runtime['vendor_json']
        os.environ['__GL_SHADER_DISK_CACHE_PATH']=str(args.output/'shader_cache')
        os.environ['EVENTWM_PRIVATE_EGL']='1'
        os.execv(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
    if args.command in ('replay','compose'):
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'visualization_process.json').write_text(json.dumps(dict(pid=os.getpid(),command=args.command,started=time.time(),worktree=str(ROOT)))+'\n')
    globals()[args.command](args)


if __name__=='__main__':main()
