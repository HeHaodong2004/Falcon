#!/usr/bin/env python3
"""Isolated, sequential EventWM closed-loop SocialNav experiments."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/haodong/workspace/projects/eventwm-falcon')
ASSETS = Path('/home/haodong/workspace/baselines/Falcon/data')
DEFAULT = PROJECT/'output/eventwm_closed_loop_20261012'
CHECKPOINT = PROJECT/'output/eventwm_v4_full_scaling_20261012/runs/all_full_seed20261013/best.pt'
EXPECTED_SHA = '6478820b4fb1a109d5ba62282064257be98dd3fb6f1a3db0195eb1eff03ec09e'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT/'habitat-lab'), str(ROOT/'habitat-baselines'), str(ROOT/'scripts')]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MAGNUM_LOG', 'quiet')
os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(body, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(args):
    """Small official validation subset, selected before any policy rollout."""
    from collections import Counter
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'episodes.json').exists(), 'Existing frozen membership preserved'
    if args.from_manifest:
        frozen=read(args.from_manifest)
        rows=[]
        for original in frozen['rows']:
            source_file=Path(original['source_file'])
            assert sha(source_file)==original['source_sha256'], 'Official source changed'
            with gzip.open(source_file,'rt') as stream:
                source_episodes=json.load(stream)['episodes']
            episode=next(e for e in source_episodes if str(e['episode_id'])==str(original['episode_id']))
            body=json.dumps(dict(episodes=[episode]),sort_keys=True,separators=(',',':')).encode()
            content_sha=hashlib.sha256(body).hexdigest()
            if original.get('episode_content_sha256'):
                assert content_sha==original['episode_content_sha256']
            path=args.output/'episodes'/(original['tag']+'.json.gz')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(gzip.compress(body,mtime=0))
            rows.append(dict(original,episode_file=str(path),episode_sha256=sha(path),
                             episode_content_sha256=content_sha))
        save(args.output/'episodes.json',dict(frozen,rows=rows,
             replayed_frozen_manifest_sha256=sha(args.from_manifest)))
        print(json.dumps(dict(stage='frozen_membership_replayed',episodes=len(rows))),flush=True)
        return
    source = ASSETS/'datasets/pointnav/social-hm3d/val/content'
    pool, missing = [], 0
    for file in sorted(source.glob('*.json.gz')):
        with gzip.open(file, 'rt') as f:
            episodes = json.load(f)['episodes']
        for episode in episodes:
            scene = ASSETS/'scene_datasets'/episode['scene_id']
            if not scene.exists():
                missing += 1; continue
            distance = episode['info']['geodesic_distance']
            if args.min_distance <= distance <= args.max_distance:
                pool.append((file, episode, scene))
    # Deterministic round-robin across scenes, short-distance integration scope.
    pool.sort(key=lambda v:(v[1]['info']['human_num'],v[1]['info']['geodesic_distance'],str(v[0]),v[1]['episode_id']))
    if args.balanced_humans:
        buckets={n:[v for v in pool if v[1]['info']['human_num']==n] for n in sorted({v[1]['info']['human_num'] for v in pool})}
        pool=[]
        while any(buckets.values()):
            for bucket in buckets.values():
                if bucket:
                    pool.append(bucket.pop(0))
    selected, used_scenes, selected_keys = [], set(), set()
    while len(selected)<args.episodes:
        added=0
        for file,episode,scene in pool:
            key=(str(file),episode['episode_id'])
            if str(scene) in used_scenes or key in selected_keys:
                continue
            selected.append((file,episode,scene));used_scenes.add(str(scene));selected_keys.add(key);added+=1
            if len(selected)==args.episodes:
                break
        if not added:
            break
        used_scenes.clear()
    assert len(selected) == args.episodes, (len(selected), 'Insufficient available eligible episodes')
    rows = []
    for i, (source_file, episode, scene) in enumerate(selected):
        tag = f'ep{i:03d}_{scene.name.split(".")[0]}_{episode["episode_id"]}'
        path = args.output/'episodes'/f'{tag}.json.gz'
        path.parent.mkdir(exist_ok=True)
        # Preserve official contents exactly; scene paths resolve in local config.
        with gzip.open(path, 'wt') as f:
            json.dump(dict(episodes=[episode]), f)
        rows.append(dict(tag=tag, episode_id=episode['episode_id'], scene_id=episode['scene_id'],
            scene_path=str(scene), source_file=str(source_file), source_sha256=sha(source_file),
            episode_file=str(path), episode_sha256=sha(path), seed=args.seed+i,
            humans=episode['info']['human_num'], declared_geodesic_distance=episode['info']['geodesic_distance']))
    save(args.output/'episodes.json', dict(version='eventwm-closed-loop-socialnav-v1',
        selection='Before outcomes: shortest eligible episode per distinct available scene, ordered by human count and declared distance',
        dataset='Official Social-HM3D validation', distance_range_m=[args.min_distance,args.max_distance],
        balanced_human_counts=args.balanced_humans,
        missing_scene_episodes=missing, available_eligible_episodes=len(pool), rows=rows))
    print(json.dumps(dict(stage='episodes_frozen',rows=rows)), flush=True)


def config_for(args, row, directory):
    import copy
    import habitat_sim
    import falcon
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config
    from omegaconf import OmegaConf, open_dict
    from closed_loop_runtime import register_simulator
    register_simulator()
    cfg = get_config('social_nav_v2/falcon_hm3d.yaml')
    with read_write(cfg), open_dict(cfg.habitat.task.actions):
        cfg.habitat.simulator.type = 'EventWMResearchSim-v0'
        with open_dict(cfg.habitat.simulator):
            cfg.habitat.simulator.research_navmesh_dir = str(args.output/'navmeshes')
        cfg.habitat.simulator.concur_render = False
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        cfg.habitat.dataset.data_path = row['episode_file']
        cfg.habitat.dataset.scenes_dir = str(ASSETS/'scene_datasets')
        cfg.habitat.dataset.content_scenes = ['*']
        cfg.habitat.dataset.split = 'val'
        cfg.habitat.environment.max_episode_steps = 500
        cfg.habitat.seed = row['seed']
        cfg.habitat.task.should_save_to_cache = False
        # Goal/odometry are read through an explicit legal boundary. No oracle
        # human sensors or auxiliary predictions are requested by this policy.
        cfg.habitat.task.lab_sensors = {}
        # Preserve every official task measurement and termination implementation.
        action = copy.deepcopy(cfg.habitat.task.actions.agent_0_discrete_move_forward)
        action.type = 'BaseVelAction'; action.lin_speed = 25.; action.ang_speed = 10.
        cfg.habitat.task.actions.agent_0_eventwm_velocity = action
        sensors = cfg.habitat.simulator.agents.agent_0.sim_sensors
        depth = copy.deepcopy(sensors.jaw_depth_sensor)
        depth.height = depth.width = 128
        rgb = OmegaConf.create({k:v for k,v in OmegaConf.to_container(depth,resolve=True).items()
                               if k not in ('min_depth','max_depth','normalize_depth')})
        rgb.type = 'HabitatSimRGBSensor'; rgb.uuid = 'articulated_agent_jaw_rgb'
        third = copy.deepcopy(cfg.habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor)
        third.height = third.width = 128
        sensors.clear()
        sensors.jaw_depth_sensor = depth
        sensors.jaw_rgb_sensor = rgb
        if args.video:
            sensors.third_rgb_sensor = third
    directory.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,directory/'resolved_config.yaml',resolve=True)
    return cfg


def episode(args):
    import random
    import habitat_sim
    import numpy as np
    import torch
    import habitat
    from closed_loop_runtime import run_episode
    torch.set_num_threads(1)
    manifest = read(args.output/'episodes.json')
    row = next(r for r in manifest['rows'] if r['tag'] == args.tag)
    assert sha(row['episode_file']) == row['episode_sha256']
    assert sha(args.checkpoint) == EXPECTED_SHA, 'Unexpected checkpoint; select via explicit reviewed configuration'
    random.seed(row['seed']); np.random.seed(row['seed']); torch.manual_seed(row['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    directory = args.output/'runs'/args.policy/args.tag
    if (directory/'metrics.json').exists():
        raise ValueError('Completed episode preserved; reproduce in a separate output directory')
    if (directory/'actions.jsonl').exists():
        archive=args.output/'interrupted_attempts'/args.policy/(args.tag+'_'+str(time.time_ns()))
        archive.parent.mkdir(parents=True,exist_ok=True)
        directory.rename(archive)
    cfg = config_for(args,row,directory)
    from collector.observation import ObservationConfig
    args.observation_config=asdict(ObservationConfig.from_depth_sensor(cfg.habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor))
    if not args.render:
        from closed_loop_nonvisual import install
        install()
    save(directory/'provenance.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        model='Unchanged EventQueryWM-M; SR=1 safe query execution',policy=args.policy,
        episode_manifest_sha256=sha(args.output/'episodes.json'),row=row,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        research_source_sha256={name:sha(ROOT/name) for name in ('scripts/eventwm_closed_loop.py','scripts/closed_loop_runtime.py',
            'scripts/closed_loop_nonvisual.py','scripts/eventquerywm/navigation.py')},
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,habitat_sim=habitat_sim.__version__,
        perception='Simulator-backed current-state FoV + ray occlusion; anonymous legal tracking; no GT velocity or future trajectory',
        action_interface='Native continuous BaseVelAction extension; official task metrics/termination unchanged',
        benchmark_action_space_extension=True))
    with habitat.Env(config=cfg.habitat) as env:
        env.seed(row['seed'])
        return run_episode(env,args,row,directory)


def research_guard(output):
    from closed_loop_production_guard import ProductionGuard
    guard = ProductionGuard(output)
    reference=output/'production_reference.json'
    if not reference.exists():
        save(reference,dict(baseline_qph=guard.baseline,production=guard.start))
    else:
        guard.baseline=read(reference)['baseline_qph']
    guard_spec=read(output/'production_guard.json')
    guard_spec.update(baseline_qph=guard.baseline,baseline_policy='Original pre-research reference retained across resumes')
    # Resume must not introduce another ten-minute monitoring blind spot.
    # Seed the rolling window from the collector's existing read-only log.
    progress_log=PROJECT/'output/eventwm_300k_logs/progress.jsonl'
    if progress_log.exists():
        with progress_log.open('rb') as stream:
            stream.seek(max(0,progress_log.stat().st_size-8*1024*1024))
            for line in stream.read().splitlines():
                try:
                    record=json.loads(line)
                    if (record.get('runner',{}).get('pid')==guard.start['runner_pid']
                        and record.get('source_commit')==guard.start['source_commit']
                        and 0 <= guard.start['timestamp']-record['timestamp'] <= 1800):
                        guard.samples.append(dict(timestamp=record['timestamp'],
                            progress=dict(primary_queries=record['new_retained'])))
                except (ValueError,KeyError):
                    continue
    guard_spec['resume_window']='Seeded from existing collector progress log; original baseline retained'
    save(output/'production_guard.json',guard_spec)
    return guard


def research_lock(output):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'research.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A research suite already owns this output; refusing concurrent workloads')
    return lock


def suite(args):
    lock=research_lock(args.output)
    if args.command=='resume' and (args.output/'STOP_RESEARCH').exists():
        (args.output/'STOP_RESEARCH').unlink()
    save(args.output/'research_process.json',dict(pid=os.getpid(),worktree=str(ROOT),started=time.time()))
    guard=research_guard(args.output)
    rows = read(args.output/'episodes.json')['rows']
    if args.limit:
        rows = rows[:args.limit]
    for policy in args.policies.split(','):
        assert policy in ('static','eventwm')
        for row in rows:
            completed = args.output/'runs'/policy/row['tag']/'metrics.json'
            if completed.exists() and read(completed).get('valid_terminal'):
                continue
            if (args.output/'STOP_RESEARCH').exists():
                print('Research stop requested; collector unaffected',flush=True);return
            guard.check()
            command = [sys.executable,str(Path(__file__).resolve()),'episode','--output',str(args.output),
                       '--tag',row['tag'],'--policy',policy,'--checkpoint',str(args.checkpoint)]
            if args.video:
                command.append('--video')
            if args.render:
                command.append('--render')
            begin = time.time()
            with (args.output/'episode_worker.log').open('a') as log:
                worker = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(args.output/'active_worker.json',dict(pid=worker.pid,parent=os.getpid(),policy=policy,tag=row['tag']))
                while worker.poll() is None:
                    time.sleep(1)
                    if time.time()-begin > args.worker_wall_limit:
                        terminate_worker(worker)
                        raise RuntimeError('Bounded episode wall-time exceeded; stopped research worker only')
                    try:
                        guard.check()
                    except BaseException:
                        terminate_worker(worker);raise
                if worker.returncode:
                    raise RuntimeError(f'Research episode failed: {policy}/{row["tag"]}; see episode_worker.log')
            print(json.dumps(dict(stage='episode_worker_finished',policy=policy,tag=row['tag'],
                valid_terminal=completed.exists() and read(completed).get('valid_terminal'),wall_s=time.time()-begin)),flush=True)
    summarize(args)


def terminate_worker(worker):
    """Signal only our directly owned child; bounded fallback prevents orphans."""
    worker.terminate()
    try:
        worker.wait(timeout=30)
    except subprocess.TimeoutExpired:
        worker.kill();worker.wait(timeout=10)


def timing_smoke(args):
    from closed_loop_diagnostics import timing_smoke as run
    lock=research_lock(args.output)
    save(args.output/'research_process.json',dict(pid=os.getpid(),worktree=str(ROOT),started=time.time()))
    return run(args,research_guard(args.output))


def summarize(args):
    import numpy as np
    groups = {};all_rows={}
    for policy in ('static','eventwm'):
        rows = [read(p) for p in sorted((args.output/'runs'/policy).glob('*/metrics.json'))]
        all_rows[policy]={r['tag']:r for r in rows if r['valid_terminal']}
    common=set(all_rows['static'])&set(all_rows['eventwm'])
    for policy in ('static','eventwm'):
        rows=[all_rows[policy][tag] for tag in sorted(common)]
        if not rows:
            continue
        groups[policy] = dict(episodes=len(rows),
            metrics={k:float(np.mean([r[k] for r in rows])) for k in
                     ('navigation_success','spl','psc','human_collision','timeout','stuck','navigation_time_s',
                      'wm_inference_ms','planning_ms')},
            failure_counts={k:sum(bool(r[k]) for r in rows) for k in
                            ('human_collision','timeout','stuck','tracking_failure','planner_oscillation','candidate_failure')},
            rows=rows)
        successful=[r for r in rows if r['navigation_success']]
        groups[policy]['metrics']['successful_path_efficiency']=float(np.mean([r['path_efficiency'] for r in successful])) if successful else None
        groups[policy]['metrics']['successful_navigation_time_s']=float(np.mean([r['navigation_time_s'] for r in successful])) if successful else None
    save(args.output/'comparison.json',dict(scope='Small, frozen official Social-HM3D subset; extended continuous robot controls; no SOTA claim',
        episode_manifest_sha256=sha(args.output/'episodes.json'),paired_episode_tags=sorted(common),
        completed_by_policy={k:len(v) for k,v in all_rows.items()},policies=groups))
    print(json.dumps({k:dict(episodes=v['episodes'],metrics=v['metrics']) for k,v in groups.items()}),flush=True)


def status(args):
    files=('research_process.json','active_worker.json','comparison.json')
    result={p:read(args.output/p) for p in files if (args.output/p).exists() and p!='comparison.json'}
    for value in result.values():
        pid=value['pid'];proc=Path(f'/proc/{pid}')
        try:
            cmd=(proc/'cmdline').read_bytes()
            value['alive']=b'eventwm_closed_loop.py' in cmd and (proc/'cwd').resolve()==ROOT
        except (FileNotFoundError,PermissionError):
            value['alive']=False
    result['completed']={policy:sum(read(p).get('valid_terminal',False) for p in (args.output/'runs'/policy).glob('*/metrics.json'))
                         for policy in ('static','eventwm')}
    result['stop_requested']=(args.output/'STOP_RESEARCH').exists()
    print(json.dumps(result,indent=2))


def stop(args):
    # Cooperative stop file is read by this research process only. Never signal
    # a PID from a shared/production registry or change any collector controls.
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'STOP_RESEARCH').write_text('Stop this research at the next safe boundary.\n')
    print('Research stop requested; active episode stops cooperatively and is excluded until resumed.')


def resume(args):
    suite(args)


def verify_isolation(args):
    from closed_loop_production_guard import production_sample
    fingerprints=read(PROJECT/'output/eventwm_v4_200k_research_20261012/production_source_fingerprints.json')
    science=Path(fingerprints['science'])
    actual={name:sha(science/name) for name in fingerprints['hashes']}
    sample=production_sample()
    initial=read(args.output/'production_reference.json')['production']
    stable={key:sample[key]==initial[key] for key in ('runner_pid','native_limit','source_commit','pipeline_settings')}
    source_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=science,text=True).strip()
    result=dict(production=sample,unchanged=stable,source_hashes=actual,
        source_hashes_match=actual==fingerprints['hashes'],source_tracked_status=source_status,
        checkpoint_sha256=sha(args.checkpoint),scope='Read-only production sample and source hashes; research output only')
    save(args.output/'production_final.json',result)
    assert all(stable.values()) and result['source_hashes_match'] and not source_status
    assert result['checkpoint_sha256']==EXPECTED_SHA and sample['runner_alive']
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','episode','suite','summarize','status','stop','resume','timing-smoke','verify-isolation'))
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--tag');p.add_argument('--policy',choices=('static','eventwm'),default='eventwm')
    p.add_argument('--policies',default='static,eventwm');p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261012);p.add_argument('--min-distance',type=float,default=2.)
    p.add_argument('--max-distance',type=float,default=8.);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--balanced-humans',action='store_true')
    p.add_argument('--from-manifest',type=Path,help='Replay frozen membership and seeds, verifying official source hashes')
    p.add_argument('--video',action='store_true');p.add_argument('--render',action='store_true',help='Use RGB/EGL instead of validated renderer-free geometry perception')
    p.add_argument('--worker-wall-limit',type=float,default=600)
    a=p.parse_args();a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve()
    # Process-local priority only. Production processes and scheduling stay intact.
    current_nice=os.getpriority(os.PRIO_PROCESS,0)
    if current_nice<19:
        os.nice(19-current_nice)
    if '/production/' in str(a.output) or str(a.output).startswith('/tmp/eventwm_v4_300k'):
        raise ValueError('Protected production output')
    # Process-local asset link. Simulator navmesh/cache writes are redirected.
    if not (ROOT/'data').exists():
        (ROOT/'data').symlink_to(ASSETS,target_is_directory=True)
    os.chdir(ROOT)
    globals()[a.command.replace('-','_')](a)


if __name__=='__main__':
    main()
