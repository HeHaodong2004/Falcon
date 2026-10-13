#!/usr/bin/env python3
"""Resumable scene-first orchestration around the unchanged EventWM v3 collector."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from collector.core import read,write,digest,load_wm_input,canonical
from collector.v2 import QueryConfig
from collector.v3 import PROTOCOL,social_outcome_labels
from collector.static_feasibility import CHECKER_VERSION,StaticConfig
from collector.wm_sampling import legal_root_state,SAMPLER_VERSION
from eventwm_scale import seals,verify_root
from eventwm_wm_train_pilot import allocation

REPO=Path(__file__).resolve().parents[1]
REPORT=REPO/'reports/eventwm_mass_collection'
DEFAULT_OUTPUT=REPO/'datasets/raw/eventwm_social_outcome_v3_train_scale_01'
COUNTS=(0,1,2,4,6)
SAMPLING_CONFIG=read(REPO/'configs/eventwm_wm_train_pilot.json')
CATEGORIES=allocation(SAMPLING_CONFIG['root_target_percent'],20,random.Random(20261007))
VISIBLE_CYCLE=allocation(SAMPLING_CONFIG['visible_target_percent'],10,random.Random(20261008))
LOST_CYCLE=allocation(SAMPLING_CONFIG['lost_target_percent'],20,random.Random(20261009))
class NoRetainedValidQueries(ValueError):
    """A correctly audited root whose non-collision outcomes are all unknown."""


BANNED_SUFFIXES={'.png','.jpg','.jpeg','.mp4','.avi','.webm','.npz','.zip'}


def hashes():
    paths=[REPO/'scripts/eventwm_collect.py',REPO/'scripts/eventwm_mass_collection.py',REPO/'scripts/eventwm_mass_assets.py',
           *sorted((REPO/'scripts/collector').glob('*.py')),*sorted((REPO/'scripts/controlled').glob('*.py')),
           REPO/'falcon/additional_metric.py',REPO/'configs/eventwm_wm_train_pilot.json',REPO/'scripts/eventwm_wm_train_pilot.py',
           REPO/'scripts/eventwm_wm_train_index.py',REPO/'scripts/eventwm_v3_sampling_pilot.py',REPO/'scripts/eventwm_scale.py']
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def git_head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def clean():return not subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=REPO,text=True).strip()


def region(position):return '_'.join(str(math.floor(float(x)/s)) for x,s in zip(position,(3.,2.,3.)))


def episode_for(scene,seed,human_count,covered):
    """Static-navmesh episode recipe; native policy/goals execution is unchanged.

    Uses the validated controlled.generate geometry constraints. Start candidates
    favor least-covered 3m x 2m x 3m cells, based only on static positions.
    """
    import habitat_sim
    import numpy as np
    pf=habitat_sim.PathFinder()
    if not pf.load_nav_mesh(scene['scene']['navmesh']):raise RuntimeError('prepared_navmesh_unavailable')
    pf.seed(seed);rng=np.random.RandomState(seed)
    base_cases=read(scene['prepared_cases']);base_root=Path(scene['prepared_cases']).parent
    template=read(base_root/base_cases[0]['episode'])
    island=template['info']['island_index']
    def point():return np.asarray(pf.get_random_navigable_point(island_index=island))
    def distance(a,b):
        path=habitat_sim.ShortestPath();path.requested_start=a;path.requested_end=b
        return path.geodesic_distance if pf.find_path(path) else float('inf')
    candidates=[point() for _ in range(128)]
    candidates.sort(key=lambda p:covered.get(region(p),0))
    for start in candidates:
        for _ in range(256):
            goal=point();d=distance(start,goal)
            if abs(float(start[1]-goal[1]))<.25 and 4<d<10:break
        else:continue
        break
    else:raise RuntimeError('static_episode_unavailable: no connected 4-10m goal')
    def sample(avoid):
        for _ in range(30000):
            q=point()
            if abs(float(q[1]-start[1]))>.3 or np.linalg.norm((q-start)[[0,2]])<1.2:continue
            if any(np.linalg.norm((q-x)[[0,2]])<.8 for x in avoid):continue
            if np.isfinite(distance(start,q)):return q
        raise RuntimeError('static_episode_unavailable: no separated human waypoint')
    info=dict(geodesic_distance=float(d),sampling_seed=seed,policy_seed=seed,scene_dataset=scene['scene_dataset'],
              scene_name=scene['name'],island_index=island,human_num=human_count)
    chosen=[]
    for h in range(6):
        q=sample(chosen);chosen.append(q)
        for k in range(3):
            pos=q if k==0 else sample([q]);info[f'human_{h}_waypoint_{k}_position']=pos.tolist()
            info[f'human_{h}_waypoint_{k}_rotation']=float(rng.uniform(0,2*np.pi))
    yaw=math.atan2(-(goal[0]-start[0]),-(goal[2]-start[2]))
    return dict(episode_id=str(seed),scene_id=scene['scene_id'],start_position=start.tolist(),
                start_rotation=[0,math.sin(yaw/2),0,math.cos(yaw/2)],goals=[dict(position=goal.tolist(),radius=.2)],info=info)


def schedule_spec(index,scenes,master_seed=410000):
    s=scenes[index%len(scenes)];round_id=index//len(scenes)
    phase=round_id+index%len(scenes)
    natural=(phase%8==7)
    # Natural slots do not consume targeted category/substratum allocation.
    # Count previous natural slots over complete scene rounds and this prefix.
    n=len(scenes);j=index%n
    natural_before=sum((r+n)//8-r//8 for r in range(round_id))+(round_id+j)//8-round_id//8
    target_index=index-natural_before
    cycle,offset=divmod(target_index,len(CATEGORIES))
    category=CATEGORIES[offset]
    category_ordinal=cycle*CATEGORIES.count(category)+CATEGORIES[:offset].count(category)
    visible_ordinal=cycle*sum(c in ('visible','visible_lost') for c in CATEGORIES)+sum(c in ('visible','visible_lost') for c in CATEGORIES[:offset])
    lost_ordinal=cycle*sum(c in ('lost','visible_lost') for c in CATEGORIES)+sum(c in ('lost','visible_lost') for c in CATEGORIES[:offset])
    allowed=tuple(c for c in SAMPLING_CONFIG['human_counts'] if c>=2) if category=='visible_lost' else tuple(SAMPLING_CONFIG['human_counts'])
    count=COUNTS[(phase//8)%5] if natural else 0 if category=='zero' else allowed[category_ordinal%len(allowed)]
    seed=master_seed+index
    cfg=SAMPLING_CONFIG
    if natural:
        sampling=dict(profile='natural',sampler_seed=seed+1000000,window_start_step=20,window_end_step=180,prefix_movement_queries=3)
    else:
        sampling=dict(profile='wm_train',sampler_version=SAMPLER_VERSION,human_count=count,target_category=category,
            target_visible_bucket=VISIBLE_CYCLE[visible_ordinal%len(VISIBLE_CYCLE)] if category in ('visible','visible_lost') else None,
            target_lost_bucket=LOST_CYCLE[lost_ordinal%len(LOST_CYCLE)] if category in ('lost','visible_lost') else None,
            min_prefix_steps=random.Random(seed).randint(20,60) if category in ('zero','unseen') else 10,
            max_prefix_steps=cfg['max_prefix_steps'],scan_angular_velocity=6. if phase%2 else -6.,
            visible_boundaries_m=cfg['visible_boundaries_m'],lost_boundaries_s=cfg['lost_boundaries_s'],allow_visible_inside_1m=False)
    profile='natural' if natural else 'wm_train'
    role='unseen_evaluation' if s['scene_role']=='unseen_evaluation' else 'natural_reference' if natural else 'train'
    recipe=dict(protocol_version=PROTOCOL,case=f"mass_{index:08d}_{s['name']}_h{count}",seed=seed,
        sampling=sampling,sampling_profile=profile,sampling_profile_family='natural' if natural else 'targeted',
        mechanism='fresh_process_seeded_episode_plus_exact_prefix',query_config=asdict(QueryConfig(movement_queries=8)),
        static_config=asdict(StaticConfig()),debug_static_reference=False,sampling_audit=True)
    # Freeze one JSON-native recipe before sharing it with the manifest and
    # root_job. Dataclass tuples (notably QueryConfig.radii) otherwise differ
    # from the lists read back at the strict pre-launch equality guard.
    recipe=json.loads(canonical(recipe))
    rid='v3m_'+digest(dict(recipe=recipe,physical_scene_id=s['physical_scene_id']))[:20];recipe['root_id']=rid
    return dict(root_id=rid,sequence=index,round=round_id,seed=seed,scene=s['name'],physical_scene_id=s['physical_scene_id'],
        family=s['family'],scene_role=s['scene_role'],role=role,split_group=rid,human_count=count,
        root_profile=profile,sampling_profile=profile,requested_category=None if natural else category,
        recipe=recipe,state='pending',attempts=[])


def catalog_link(output):
    d=REPO/'evaluation';d.mkdir(exist_ok=True)
    link=d/'controlled50';target=output/'catalogue'
    target.mkdir(parents=True,exist_ok=True)
    if link.is_symlink() and link.resolve()!=target.resolve():link.unlink()
    if not link.exists():link.symlink_to(target,target_is_directory=True)
    if link.resolve()!=target.resolve():raise RuntimeError('Refuse to replace non-production controlled catalog')


def init(args):
    out=args.output.resolve();file=out/'manifest.json'
    if file.exists():
        m=read(file)
        if m['protocol']!=PROTOCOL or m['checker']!=CHECKER_VERSION or m['storage_profile']!='structured_production':raise RuntimeError('Resume protocol mismatch')
        if m['source_hashes']!=hashes():raise RuntimeError('Resume source hash mismatch')
        if args.target<m['target_primary_queries']:raise ValueError('Resume target cannot shrink')
        m['target_primary_queries']=args.target;m['max_completed_roots']=args.max_roots or math.ceil(args.target/33)
        return m
    out.mkdir(parents=True,exist_ok=True)
    if any(p.name not in ('.lock',) for p in out.iterdir()):raise ValueError('New output is not empty')
    inv=read(REPORT/'scene_inventory.json');scenes=[s for s in inv['scenes'] if s.get('compatible')]
    if not scenes:raise RuntimeError('No compatible scenes')
    # Family-interleaved round robin; every compatible scene retained in one role.
    families={f:iter(sorted([s for s in scenes if s['family']==f],key=lambda s:s['physical_scene_id'])) for f in sorted({s['family'] for s in scenes})}
    ordered=[]
    while families:
        for f,it in list(families.items()):
            s=next(it,None)
            if s is None:del families[f]
            else:ordered.append(s)
    write(out/'scene_inventory.json',inv)
    m=dict(schema_version=4,prefix_truth_encoding='json-gzip-v1',branch_truth_encoding='jsonl-gzip-v1',sr_contract_version='binary-execution-safety-v1',contact_serialization='sorted-exact-multiset-v1',protocol_version=PROTOCOL,protocol=PROTOCOL,checker=CHECKER_VERSION,storage_profile='structured_production',
           collector_storage_profile='production',query_config=asdict(QueryConfig(movement_queries=8)),static_config=asdict(StaticConfig()),
           frozen_commit=git_head(),source_hashes=hashes(),target_primary_queries=args.target,
           max_completed_roots=args.max_roots or math.ceil(args.target/33),min_free_gb=args.min_free_gb,
           roots=[],next_sequence=0,scenes=ordered,state='prepared_not_started',sessions=[],master_seed=410000,
           max_attempts=2,root_timeout_seconds=12000,collection_seconds=0.,catalogue='catalogue',
           role_policy='12.5% scenes held out per large family; 1/8 recipe opportunities natural independent of outcomes; never split a root')
    save(out,m);return m


def root_job(output,m,spec):
    catalog=output/'catalogue';scene=next(s for s in m['scenes'] if s['name']==spec['scene'])
    if hashlib.sha256(Path(scene['scene']['navmesh']).read_bytes()).hexdigest()!=scene['navmesh_sha256']:raise RuntimeError('prepared_navmesh_hash_mismatch')
    covered=Counter(r.get('start_region') for r in m['roots'] if r['scene']==spec['scene'] and r.get('start_region'))
    episode=episode_for(scene,spec['seed'],spec['human_count'],covered)
    epfile=catalog/'episodes'/f"{spec['root_id']}.json";write(epfile,episode)
    spec['start_region']=region(episode['start_position']);spec['episode_sha256']=hashlib.sha256(epfile.read_bytes()).hexdigest()
    case=dict(tag=spec['recipe']['case'],scene=spec['scene'],humans=spec['human_count'],seed=spec['seed'],
        episode=str(epfile.relative_to(catalog)),episode_sha256=spec['episode_sha256'])
    cases=read(catalog/'cases.json') if (catalog/'cases.json').exists() else []
    assert not any(c['tag']==case['tag'] for c in cases)
    cases.append(case);write(catalog/'cases.json',cases)
    write(catalog/'scenes.json',[s['scene'] for s in m['scenes']])
    write(output/'recipes'/f"{spec['root_id']}.json",spec['recipe'])


def inspect_root(path,root):
    from collector.branch_storage import branch_path
    if branch_path(path).name!='branches.jsonl.gz':raise RuntimeError('global_integrity: mixed branch storage encoding')
    if any(p.suffix.lower() in BANNED_SUFFIXES for p in path.rglob('*') if p.is_file()):raise RuntimeError('global_integrity: image/video/archive file written')
    audits=read(path/'audits/audits.json')
    if any(v['status']!='PASS' for k,v in audits.items() if k!='human_reactivity'):raise RuntimeError('global_integrity: collector audit failed')
    qs=read(path/'queries/queries.json');labels=read(path/'labels/labels.json')
    primary=[l for l in labels if l['branch_id'].startswith('b')]
    if len(primary)!=33 or len(qs)!=33 or len({l['query_id'] for l in primary})!=33:raise RuntimeError('global_integrity: query count/identity')
    if any(q['sha256']!=digest(q['query']) or q['query']['static_precheck']['checker_version']!=CHECKER_VERSION or q['query']['static_precheck']['static_feasible'] is not True for q in qs):raise RuntimeError('global_integrity: query hash/static certificate')
    if any(l['static_feasibility_violation'] for l in labels):raise RuntimeError('global_integrity: static violation')
    excluded=[l for l in primary if not l['social_outcome_valid']]
    if any(l['invalid_reason']!='incomplete_or_invalid_nonhuman_outcome' or l['sr_valid'] or l['sr_label'] is not None for l in excluded):
        raise RuntimeError('global_integrity: invalid primary evidence')
    primary=[l for l in primary if l['social_outcome_valid'] and l['sr_valid']]
    if not primary:raise NoRetainedValidQueries('no_retained_valid_queries')
    history=load_wm_input(path,primary[0]['query_id'])['history']
    state=legal_root_state(history,root['human_count'],read(REPO/'configs/eventwm_wm_train_pilot.json'))
    position=history[-1]['robot_position'];last=history[-1]['previous_command']
    retained_ids={l['query_id'] for l in primary}
    retained_queries=[q for q in qs if q['query']['query_id'] in retained_ids]
    variants=Counter('WAIT' if q['query']['direction']=='WAIT' else
         'half_speed' if q['query']['speed_scale']==.5 else f"wait_{q['query']['wait_duration']}" if q['query']['wait_duration'] else 'immediate_normal' for q in retained_queries)
    stats=dict(primary_queries=len(primary),class_counts=dict(Counter(l['social_outcome_name'] for l in primary)),
        binary_targets=dict(Counter(str(int(l['social_outcome_class']==2)) for l in primary)),
        sr_targets=dict(Counter(str(l['sr_label']) for l in primary)),static_violations=0,invalid=len(excluded),
        invalid_reasons=dict(Counter(l['invalid_reason'] for l in excluded)),
        legal_state=state,robot_state='moving' if any(last) else 'stationary',spatial_region_id=root['physical_scene_id']+'/'+region(position),
        spatial_paths=8,variants=dict(variants),bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()))
    return stats,primary


def aggregate(m):
    done=[r for r in m['roots'] if r['state']=='completed'];counts=Counter(r['scene'] for r in done)
    report=dict(completed_roots=len(done),failed_roots=sum(r['state']=='failed' for r in m['roots']),
        running_roots=sum(r['state']=='running' for r in m['roots']),target_roots=m['max_completed_roots'],
        primary_queries=sum(r['stats']['primary_queries'] for r in done),target_primary_queries=m['target_primary_queries'],
        compatible_scenes=len(m['scenes']),completed_scenes=len(counts),remaining_scenes=len(m['scenes'])-len(counts),
        regions=len({r['stats']['spatial_region_id'] for r in done}),scene_counts=dict(counts),families=dict(Counter(r['family'] for r in done)),
        human_counts=dict(Counter(str(r['human_count']) for r in done)),roles=dict(Counter(r['role'] for r in done)),
        root_categories=dict(Counter(r['stats']['legal_state']['category'] for r in done)),
        visible=dict(Counter(r['stats']['legal_state']['visible_distance_bucket'] for r in done if r['stats']['legal_state']['visible_distance_bucket'])),
        lost=dict(Counter(r['stats']['legal_state']['lost_age_bucket'] for r in done if r['stats']['legal_state']['lost_age_bucket'])),
        robot_state=dict(Counter(r['stats']['robot_state'] for r in done)),
        failed_reasons=dict(Counter(a.get('reason','unknown') for r in m['roots'] for a in r['attempts'] if a['state']=='failed')),
        completed_root_bytes=sum(r['stats']['bytes'] for r in done),collection_seconds=m['collection_seconds'])
    for key in ('class_counts','binary_targets','sr_targets','invalid_reasons','variants'):
        c=Counter()
        for r in done:c.update(r['stats'].get(key,{}))
        report[key]=dict(c)
    report['static_violations']=sum(r.get('stats',{}).get('static_violations',0) for r in m['roots'])+sum(a.get('reason')=='static_violation' for r in m['roots'] for a in r['attempts'])
    report['invalid']=sum(r['stats']['invalid'] for r in done)
    report['queries_per_hour']=report['primary_queries']/m['collection_seconds']*3600 if m['collection_seconds'] else 0.
    return report


def save(output,m):
    from collector.storage import write_compact_json
    write_compact_json(output/'manifest.json',m);write_compact_json(output/'aggregate.json',aggregate(m))


def hardware(previous=None):
    cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));t=time.monotonic()
    memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    # Whole physical NVMe counters; exclude partitions to avoid double counting.
    disks=[l.split() for l in Path('/proc/diskstats').read_text().splitlines()]
    writes=sum(int(r[9])*512 for r in disks if r[2].startswith('nvme') and 'p' not in r[2])
    result=dict(timestamp=time.time(),monotonic=t,cpu_counters=cpu,write_bytes=writes,load=list(os.getloadavg()),
                ram_used_gb=(memory['MemTotal']-memory['MemAvailable'])/1e9,ram_available_gb=memory['MemAvailable']/1e9)
    if previous:
        delta=[a-b for a,b in zip(cpu,previous['cpu_counters'])];total=sum(delta[:8])
        result.update(cpu_percent=100*(1-(delta[3]+delta[4])/total) if total else 0.,
                      io_wait_percent=100*delta[4]/total if total else 0.,disk_write_mb_s=(writes-previous['write_bytes'])/(t-previous['monotonic'])/1e6)
    try:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5,stderr=subprocess.DEVNULL)
        result['gpus']=[dict(zip(('index','utilization','memory_used_mib','memory_total_mib'),map(float,line.split(',')))) for line in text.splitlines()]
    except Exception:result['gpus']=[]
    return result


def failure_reason(path,log):
    text=log.read_text(errors='replace')[-16000:] if log.exists() else ''
    for f in path.glob('audits/workers/*/run.log'):text+='\n'+f.read_text(errors='replace')[-5000:]
    if (path/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists() or 'STATIC_FEASIBILITY_VIOLATION' in text:return 'static_violation',True,text[-6000:]
    for marker in ('Root replay mismatch','Collector integrity audit failed','Canonical/worker mismatch','global_integrity'):
        if marker in text:return 'integrity_failure',True,text[-6000:]
    for marker in ('out of memory','CUDA_ERROR_OUT_OF_MEMORY','std::bad_alloc'):
        if marker in text:return 'oom',True,text[-6000:]
    if 'wm_train_stratum_unavailable' in text:return 'requested_legal_stratum_unavailable',False,text[-6000:]
    if 'insufficient_static_feasible_queries' in text:return 'insufficient_static_feasible_queries',False,text[-6000:]
    if 'Prefix terminated before root' in text:return 'prefix_terminated_before_root',False,text[-6000:]
    if 'TimeoutExpired' in text:return 'worker_timeout',False,text[-6000:]
    return 'collection_failure_unresolved',False,text[-6000:]


def recover_completed(output,root):
    """Recover a completion published just before an interrupted manifest write."""
    attempt=root['attempts'][-1];path=output/attempt['path']
    if not (path/'COMPLETE.json').exists():return False
    seal_file=f"root_seals/{root['root_id']}.json"
    verify_root(output,dict(root,path=attempt['path'],seals=read(output/seal_file)),deep=True)
    stats,labels=inspect_root(path,root)
    root.update(state='completed',stats=stats,path=attempt['path'],seal_file=seal_file,
                query_ids=sorted(l['query_id'] for l in labels))
    attempt.update(state='completed',recovered_completion=True)
    write(output/'primary_indices'/f"{root['root_id']}.json",dict(root_id=root['root_id'],role=root['role'],scene_role=root['scene_role'],
         split_group=root['root_id'],physical_scene_id=root['physical_scene_id'],path=root['path'],samples=labels))
    return True


def external_roots(args):
    """Share the proven machine budget with an unchanged predecessor draining."""
    path=getattr(args,'drain_from',None)
    if path is None:return 0
    if path.resolve()==args.output.resolve():raise ValueError('Drain predecessor must be a different dataset')
    previous=read(path/'manifest.json')
    if not previous['state'].startswith(('draining_','stopped_')):
        raise RuntimeError('Predecessor must be draining or stopped before shared-budget launch')
    return sum(r['state']=='running' for r in previous['roots'])


def run(args):
    if args.concurrency<1 or args.concurrency>36:raise ValueError('Concurrency must be 1..36')
    total_capacity=getattr(args,'total_concurrency',None) or args.concurrency
    if not args.concurrency<=total_capacity<=36:raise ValueError('Total concurrency must cover workers and remain at most 36')
    if args.benchmark and (not args.stop_after_attempts or args.stop_after_attempts>240):raise ValueError('Benchmark must be explicitly bounded to at most 240 attempts')
    if not 0<=getattr(args,'startup_stagger',3.)<=60:raise ValueError('Startup stagger must be 0..60 seconds')
    if args.target<=0 or (args.max_roots is not None and args.max_roots<=0):raise ValueError('Positive target/root budget required')
    if args.storage_profile!='structured_production':raise ValueError('Only structured_production is authorized')
    if args.min_free_gb<150:raise ValueError('Disk guard may not be below 150 GB')
    if not args.benchmark:
        release=read(REPORT/'release.json')
        if not release['benchmark_pass'] or git_head()!=release['commit'] or not clean():raise RuntimeError('Production requires clean frozen commit and passing benchmark')
        if hashes()!=release['source_hashes']:raise RuntimeError('Production source/static checker hash mismatch')
        if args.concurrency!=release['selected_concurrency'] or args.gpus!=release['gpu_assignment']:raise RuntimeError('Use benchmark-approved concurrency/GPU assignment')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    with (REPO/'output/mass_catalog.lock').open('a') as catalog_lock,(args.output/'.lock').open('a') as lock:
        fcntl.flock(catalog_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=init(args);catalog_link(args.output)
        if not args.benchmark and m['frozen_commit']!=git_head():raise RuntimeError('Manifest frozen commit mismatch')
        # Verify all completed outputs before skipping; no canonical record appended twice.
        for root in m['roots']:
            if root['state']=='completed':verify_root(args.output,dict(root,seals=read(args.output/root['seal_file'])),deep=True)
            elif root['state']=='running':
                if recover_completed(args.output,root):continue
                if root.get('pid') and Path(f"/proc/{root['pid']}/cmdline").exists():
                    command=Path(f"/proc/{root['pid']}/cmdline").read_bytes()
                    if str(args.output).encode() in command:raise RuntimeError('Previous worker still alive; do not resume until it stops')
                root['state']='failed';root['attempts'][-1].update(state='failed',reason='interrupted_before_commit')
        m['state']='running';m['runner']=dict(pid=os.getpid(),host=socket.gethostname(),started=time.time())
        session=dict(started=time.time(),concurrency=args.concurrency,gpus=args.gpus,threads=1,startup_stagger_seconds=getattr(args,'startup_stagger',3.),benchmark=args.benchmark)
        session.update(drain_from=str(args.drain_from.resolve()) if getattr(args,'drain_from',None) else None,total_concurrency=total_capacity)
        external_roots(args)
        m['sessions'].append(session);save(args.output,m)
        stopping=False;fatal=False;active={};launched=0;previous=hardware();last_save=time.monotonic();began=last_save;base_seconds=m['collection_seconds'];next_launch=began
        def stop(signum,frame):
            nonlocal stopping
            stopping=True;m['state']='draining_user_stop'
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        telemetry=(args.output/'resource_samples.jsonl').open('a',buffering=1)
        try:
            while True:
                now=time.monotonic();m['collection_seconds']=base_seconds+now-began
                total=aggregate(m)
                if not args.max_roots and total['completed_roots']>=m['max_completed_roots'] and total['primary_queries']<args.target:
                    m['max_completed_roots']+=math.ceil((args.target-total['primary_queries'])/33)
                if not stopping and (total['primary_queries']>=args.target or (args.max_roots and total['completed_roots']>=m['max_completed_roots'])):
                    stopping=True;m['state']='target_reached' if total['primary_queries']>=args.target else 'root_budget_exhausted'
                if not stopping and args.stop_after_attempts and launched>=args.stop_after_attempts:
                    stopping=True;m['state']='bounded_benchmark_draining'
                free=shutil.disk_usage(args.output).free
                # Reserve 2 GB per live/root slot above the hard guard; never delete data.
                if free<(args.min_free_gb+max(5,2*args.concurrency))*1e9:
                    stopping=fatal=True;m['state']='stopped_disk_guard'
                if now-last_save>=5:
                    current=hardware(previous);previous=current;telemetry.write(json.dumps(current)+'\n')
                    if any(g['memory_used_mib']>g['memory_total_mib']-2048 for g in current['gpus']):
                        stopping=fatal=True;m['state']='stopped_gpu_memory_guard'
                    if current['ram_available_gb']<16:
                        stopping=fatal=True;m['state']='stopped_ram_guard'
                    save(args.output,m);last_save=now
                if fatal:
                    fatal_since=locals().get('fatal_since',now)
                    for proc,log,root,t0 in active.values():
                        if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL if now-fatal_since>10 else signal.SIGTERM)
                for slot,(proc,log,root,t0) in list(active.items()):
                    if proc.poll() is None and now-t0>m['root_timeout_seconds']:
                        os.killpg(proc.pid,signal.SIGKILL);root['attempts'][-1]['timeout']=True
                    code=proc.poll()
                    if code is None:continue
                    log.close();attempt=root['attempts'][-1];path=args.output/attempt['path']
                    attempt.update(seconds=now-t0,returncode=code)
                    try:
                        if code:
                            reason,global_failure,detail=failure_reason(path,Path(attempt['log']))
                            if attempt.get('timeout'):reason='root_timeout'
                            elif fatal and code<0:reason='cancelled_after_global_failure'
                            attempt.update(state='failed',reason=reason,detail=detail);root['state']='failed'
                            if global_failure:stopping=fatal=True;m['state']='global_'+reason
                        else:
                            stats,labels=inspect_root(path,root)
                            new_ids={l['query_id'] for l in labels}
                            existing_ids={q for r in m['roots'] if r['state']=='completed' for q in r.get('query_ids',[])}
                            if new_ids & existing_ids:raise RuntimeError('global_integrity: duplicate primary query IDs')
                            root['query_ids']=sorted(new_ids)
                            root_seals=seals(path)
                            seal_file=f"root_seals/{root['root_id']}.json"
                            write(args.output/seal_file,root_seals)
                            root.update(state='completed',stats=stats,path=attempt['path'],seal_file=seal_file)
                            write(path/'COMPLETE.json',dict(root_id=root['root_id'],seal_digest=digest(root_seals)))
                            attempt['state']='completed'
                            write(args.output/'primary_indices'/f"{root['root_id']}.json",dict(root_id=root['root_id'],role=root['role'],scene_role=root['scene_role'],
                                split_group=root['root_id'],physical_scene_id=root['physical_scene_id'],path=root['path'],samples=labels))
                    except NoRetainedValidQueries as e:
                        root['state']='failed';attempt.update(state='failed',reason='no_retained_valid_queries',detail=str(e))
                    except Exception as e:
                        reason='static_violation' if 'static violation' in str(e) else 'integrity_failure'
                        root['state']='failed';attempt.update(state='failed',reason=reason,detail=repr(e));stopping=fatal=True;m['state']='global_'+reason
                    del active[slot];save(args.output,m)
                    print(root['state'].upper(),root['root_id'],root['scene'],attempt.get('reason'),round(attempt['seconds'],1),flush=True)
                # Recount after completions so MAX_ROOTS is a hard bound.
                total=aggregate(m)
                # The predecessor only drains, so this count cannot grow between dispatches.
                external_count=external_roots(args)
                # Dispatch only a bounded number of workers; never hidden-label selection.
                for slot in range(args.concurrency):
                    if stopping or slot in active:continue
                    if external_count+len(active)>=total_capacity:break
                    if time.monotonic()<next_launch:break
                    if total['completed_roots']+len(active)>=m['max_completed_roots']:continue
                    if args.stop_after_attempts and launched>=args.stop_after_attempts:break
                    root=next((r for r in m['roots'] if r['state']=='pending'),None)
                    if root is None:root=next((r for r in m['roots'] if r['state']=='failed' and len(r['attempts'])<m['max_attempts'] and
                               r['attempts'][-1]['reason'] in ('interrupted_before_commit','root_timeout','worker_timeout')),None)
                    if root is None:
                        root=schedule_spec(m['next_sequence'],m['scenes'],m['master_seed']);m['next_sequence']+=1;m['roots'].append(root)
                        try:root_job(args.output,m,root)
                        except Exception as e:
                            root['state']='failed';root['attempts'].append(dict(state='failed',reason='episode_preparation_unavailable',detail=repr(e)))
                            launched+=1;save(args.output,m);continue
                    recipe_file=args.output/'recipes'/f"{root['root_id']}.json"
                    if read(recipe_file)!=root['recipe']:raise RuntimeError('Frozen recipe file mismatch')
                    if root.get('episode_sha256'):
                        epfile=args.output/'catalogue/episodes'/f"{root['root_id']}.json"
                        if hashlib.sha256(epfile.read_bytes()).hexdigest()!=root['episode_sha256']:raise RuntimeError('Frozen episode file mismatch')
                    number=len(root['attempts'])+1;rel=f"roots/{root['root_id']}/attempt_{number:02d}"
                    path=args.output/rel;logpath=args.output/'logs'/f"{root['root_id']}_{number}.log";logpath.parent.mkdir(exist_ok=True)
                    attempt=dict(number=number,path=rel,state='running',started=time.time(),log=str(logpath));root['attempts'].append(attempt)
                    gpu=args.gpus[slot%len(args.gpus)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                    cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_collect.py'),'collect','--output',str(path),'--recipe',str(args.output/'recipes'/f"{root['root_id']}.json"),
                         '--protocol','v3','--movement-queries','8','--max-branches','36','--storage-profile','production','--no-render']
                    log=logpath.open('w');proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    root.update(state='running',pid=proc.pid,gpu=gpu);next_launch=time.monotonic()+getattr(args,'startup_stagger',3.);active[slot]=(proc,log,root,time.monotonic());launched+=1;save(args.output,m)
                    print('START',root['root_id'],root['scene'],root['human_count'],root['requested_category'],'GPU',gpu,flush=True)
                if stopping and not active:break
                time.sleep(.5)
        finally:
            # Unexpected Python errors must not leave unowned collector processes.
            for proc,log,root,t0 in active.values():
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                log.close()
            telemetry.close();m['collection_seconds']=base_seconds+time.monotonic()-began
            if m['state']=='draining_user_stop':m['state']='stopped_by_user'
            if m['state']=='bounded_benchmark_draining':m['state']='bounded_benchmark_finished'
            if m['state']=='running':m['state']='interrupted_error'
            session.update(seconds=time.monotonic()-began,state=m['state']);save(args.output,m)
        return 1 if fatal else 0


def status(args):
    out=args.output.resolve()
    if not (out/'manifest.json').exists():print('RUN STATUS: NOT INITIALIZED');return
    m=read(out/'manifest.json');a=read(out/'aggregate.json');hw=hardware();time.sleep(.2);hw=hardware(hw)
    print('RUN STATUS:',m['state'],'PID',m.get('runner',{}).get('pid'))
    print('SCENE COVERAGE:',{k:a[k] for k in ('compatible_scenes','completed_scenes','remaining_scenes')})
    print('REGION COVERAGE:',a['regions'])
    print('ROOTS:',{k:a[k] for k in ('completed_roots','running_roots','failed_roots','target_roots')})
    q=a['primary_queries'];target=a['target_primary_queries'];rate=a['queries_per_hour']
    print('QUERIES:',q,'/',target,f'{q/target:.2%}',f'{rate:.2f}/hour','ETA hours:',(target-q)/rate if rate else 'unknown')
    for label,key in [('HUMAN COUNT DISTRIBUTION','human_counts'),('ROOT CATEGORY DISTRIBUTION','root_categories'),('VISIBLE','visible'),('LOST','lost'),('OUTCOME COUNTS','class_counts'),('STATIC VIOLATIONS','static_violations')]:print(label+':',a[key])
    print('DATASET SIZE (completed roots):',a['completed_root_bytes']/1e9,'GB; FREE DISK:',shutil.disk_usage(out).free/1e9,'GB')
    print('CPU / RAM:',{k:v for k,v in hw.items() if k in ('cpu_percent','load','ram_used_gb','ram_available_gb','io_wait_percent','disk_write_mb_s')})
    for g in hw['gpus']:print('GPU'+str(int(g['index']))+':',g)
    print('CURRENT WORKERS:')
    for r in m['roots']:
        if r['state']=='running':
            print(r['root_id'],r['scene'],'PID',r['pid'],'GPU',r['gpu'])
            log=Path(r['attempts'][-1]['log']);print('LATEST LOG:',log.read_text(errors='replace').splitlines()[-3:] if log.exists() else 'not yet created')


def request_stop(args):
    output=args.output.resolve();m=read(output/'manifest.json');runner=m.get('runner',{})
    pid=runner.get('pid')
    if runner.get('host')!=socket.gethostname() or not pid:raise RuntimeError('No local runner recorded')
    cmdfile=Path(f'/proc/{pid}/cmdline')
    if not cmdfile.exists():print('Runner already stopped');return
    cmd=cmdfile.read_bytes()
    parts=cmd.split(b'\0')
    supplied=Path(parts[parts.index(b'--output')+1].decode()) if b'--output' in parts else DEFAULT_OUTPUT
    actual=(Path(os.readlink(f'/proc/{pid}/cwd'))/supplied).resolve()
    if b'eventwm_mass_collection.py' not in cmd or actual!=output:raise RuntimeError('PID identity mismatch; refusing to signal')
    os.kill(pid,signal.SIGTERM);print('Drain requested: no new roots; current roots finish or reach bounded timeout')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','run','status','dry-run','stop'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--target',type=int,default=300000)
    p.add_argument('--max-roots',type=int);p.add_argument('--concurrency',type=int,default=1);p.add_argument('--gpus',default='0,1')
    p.add_argument('--startup-stagger',type=float,default=3.)
    p.add_argument('--drain-from',type=Path)
    p.add_argument('--total-concurrency',type=int)
    p.add_argument('--min-free-gb',type=float,default=150);p.add_argument('--storage-profile',default='structured_production')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--stop-after-attempts',type=int)
    return p

if __name__=='__main__':
    a=parser().parse_args();a.gpus=[int(x) for x in a.gpus.split(',')]
    if a.action=='status':status(a)
    elif a.action=='stop':request_stop(a)
    elif a.action=='run':sys.exit(run(a))
    elif a.action=='init':init(a)
    else:
        inv=read(REPORT/'scene_inventory.json');scenes=[s for s in inv['scenes'] if s.get('compatible')]
        print(json.dumps(dict(action='dry-run',target=a.target,roots=a.max_roots or math.ceil(a.target/33),concurrency=a.concurrency,
              scenes=len(scenes),free_gb=shutil.disk_usage(a.output.parent if a.output.parent.exists() else REPO).free/1e9,
              source_hashes=hashes(),storage_profile=a.storage_profile),indent=2))
