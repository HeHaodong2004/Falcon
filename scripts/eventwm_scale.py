#!/usr/bin/env python3
"""Sequential, bounded orchestration of the validated one-root collector."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from collector.core import digest, read, write

SOURCE = REPO/'evaluation/controlled50'
class GlobalIntegrityError(RuntimeError):
    pass


BRANCH_RESERVATION = 36  # <=33 primary queries + 2 repeats + censored WAIT


def provenance():
    paths = [Path(__file__), REPO/'scripts/eventwm_collect.py',
             *sorted((REPO/'scripts/collector').glob('*.py')),
             *sorted((REPO/'scripts/controlled').glob('*.py')),
             SOURCE/'scenes.json', SOURCE/'cases.json']
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def plan(args):
    if not args.scenes or not args.human_counts or not args.roots_per_config:
        raise ValueError('Explicit scenes, human counts and roots/config are required')
    if len(set(args.scenes))!=len(args.scenes) or len(set(args.human_counts))!=len(args.human_counts):
        raise ValueError('Duplicate scene/count selection')
    if min(args.roots_per_config,args.max_total_roots,args.max_total_branches,args.max_attempts)<1:
        raise ValueError('Bounds must be positive')
    if len(args.scenes)*len(args.human_counts)*args.roots_per_config>args.max_total_roots:
        raise ValueError('Plan exceeds root bound before materializing recipes')
    cases = read(SOURCE/'cases.json'); scenes = read(SOURCE/'scenes.json')
    result=[]
    for scene_index,scene in enumerate(args.scenes):
        spec=next(s for s in scenes if s['name']==scene)
        for count in args.human_counts:
            case=next(c for c in cases if c['scene']==scene and c['humans']==count)
            for index in range(args.roots_per_config):
                seed=args.seed+scene_index*1000+count*100+index
                profile=args.profiles[(scene_index*len(args.human_counts)+args.human_counts.index(count)+index)%len(args.profiles)]
                recipe=dict(case=case['tag'],seed=seed,sampling=dict(profile=profile,wait_steps=20+index*20),
                            mechanism='fresh_process_seeded_episode_plus_exact_prefix')
                rid='r_'+digest(recipe)[:16]; recipe['root_id']=rid
                result.append(dict(root_id=rid,scene=scene,human_count=count,episode=case['episode'],
                    episode_id=read(SOURCE/case['episode'])['episode_id'],config=case['tag'],seed=seed,
                    root_family=f"{scene}/{case['episode']}",recipe=recipe,
                    episode_sha256=hashlib.sha256((SOURCE/case['episode']).read_bytes()).hexdigest(),
                    navmesh_sha256=hashlib.sha256((SOURCE/spec['navmesh']).read_bytes()).hexdigest(),
                    state='pending',attempts=[]))
    if len(result)>args.max_total_roots or len(result)*BRANCH_RESERVATION>args.max_total_branches:
        raise ValueError('Plan exceeds root or branch bound (36 reserved branches per root)')
    return dict(schema_version=1,storage_profile=getattr(args,'storage_profile',None) or 'validated',observation_protocol='eventwm-structured-observation-v1',
                provenance=provenance(),bounds=dict(max_total_roots=args.max_total_roots,
                max_total_branches=args.max_total_branches,max_attempts=args.max_attempts,concurrency=1,
                min_free_gb=getattr(args,'min_free_gb',0.)),
                selection=dict(scenes=args.scenes,human_counts=args.human_counts,roots_per_config=args.roots_per_config,
                               seed=args.seed,profiles=args.profiles),reserved_branches=0,roots=result)


def distribution(values):
    if not values: return dict(count=0,min=None,max=None,mean=None)
    return dict(count=len(values),min=min(values),max=max(values),mean=sum(values)/len(values),
                histogram=dict(Counter(str(round(x,3)) for x in values)))


def root_stats(path):
    history=read(path/'legal_observations/history.json'); last=history[-1]['human_tracks']
    qs=read(path/'queries/queries.json'); labels=read(path/'labels/labels.json')
    primary=[x for x in labels if x['branch_id'].startswith('b')]
    return dict(queries=len(qs),branches=len(labels),primary_branches=len(primary),
        legal_detection_observations=sum(t['visible_now'] for h in history for t in h['human_tracks']),
        visible_history=any(t['visible_now'] for h in history for t in h['human_tracks']),
        visible_at_root=any(t['visible_now'] for t in last),lost_at_root=any(not t['visible_now'] for t in last),
        lost_history=any(not t['visible_now'] for h in history for t in h['human_tracks']),
        moving_at_root=any(history[-1]['previous_command']),
        collisions=sum(x['human_collision_event'] for x in primary),
        completion=dict(Counter('completed' if x['completed'] else 'censored' for x in primary)),
        termination=dict(Counter(x['termination_reason'] for x in primary)),
        wait=[q['query']['wait_duration'] for q in qs],speed=[q['query']['speed_scale'] for q in qs],
        duration=[q['query']['planned_duration'] for q in qs])


def aggregate(manifest):
    done=[r for r in manifest['roots'] if r['state']=='completed']; stats=[r['stats'] for r in done]
    branches=sum(s['primary_branches'] for s in stats); collisions=sum(s['collisions'] for s in stats)
    report=dict(root_states=dict(Counter(r['state'] for r in manifest['roots'])),
        completed_roots=len(done),failed_roots=sum(r['state']=='failed' for r in manifest['roots']),
        failed_attempts=sum(a['state']=='failed' for r in manifest['roots'] for a in r['attempts']),
        total_queries=sum(s['queries'] for s in stats),total_branches=sum(s['branches'] for s in stats),
        primary_branches=branches,human_count_distribution=dict(Counter(str(r['human_count']) for r in done)),
        roots_with_zero_legal_detections=sum(not s['visible_history'] for s in stats),
        roots_with_visible_human_history=sum(s['visible_history'] for s in stats),
        roots_with_visible_tracks_at_root=sum(s['visible_at_root'] for s in stats),
        roots_with_lost_tracks_at_root=sum(s['lost_at_root'] for s in stats),
        roots_containing_lost_memory=sum(s['lost_history'] for s in stats),
        moving_roots=sum(s['moving_at_root'] for s in stats),collision_event_count=collisions,
        collision_event_rate=collisions/branches if branches else None,
        reserved_branches=manifest['reserved_branches'],bounds=manifest['bounds'])
    for key in ('completion','termination'):
        counts=Counter()
        for s in stats: counts.update(s[key])
        report[key]=dict(counts)
    for key in ('wait','speed','duration'): report[key]=distribution([v for s in stats for v in s[key]])
    report['completed_root_seconds']=distribution([r['attempts'][-1]['seconds'] for r in done])
    report['completed_root_bytes']=sum(r['bytes'] for r in done)
    return report


def seals(path):
    # All grouped data and diagnostics are sealed, including worker traces/images.
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file()}


def save(output, manifest):
    write(output/'manifest.json',manifest)
    write(output/'aggregate.json',aggregate(manifest))


def verify_root(output, root, deep=False):
    path=output/root['path']
    marker=read(path/'COMPLETE.json')
    if marker['root_id']!=root['root_id'] or marker['seal_digest']!=digest(root['seals']):
        raise RuntimeError('Global corruption: completion marker mismatch')
    if deep and seals_without_marker(path)!=root['seals']:
        raise RuntimeError(f'Global corruption: modified completed root {path}')


def seals_without_marker(path):
    return {k:v for k,v in seals(path).items() if k!='COMPLETE.json'}


def run(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        file=output/'manifest.json'
        if file.exists():
            if not args.resume: raise ValueError('Existing manifest requires --resume')
            manifest=read(file)
            if manifest['provenance']!=provenance():
                raise RuntimeError('Global corruption/config drift: source or commit changed; use a new run')
            if getattr(args,'storage_profile',None) and args.storage_profile!=manifest.get('storage_profile','validated'):
                raise ValueError('Resume storage profile differs from manifest')
            if args.scenes and plan(args)['selection']!=manifest['selection']:
                raise ValueError('Resume selection differs from frozen manifest')
        else:
            manifest=plan(args); save(output,manifest)
        manifest['runner']=dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),state='running')
        save(output,manifest)
        stopping=False
        child=None
        def stop(signum,frame):
            nonlocal stopping
            stopping=True
            if child and child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        attempted=0
        for root in manifest['roots']:
            if root['state']=='completed':
                verify_root(output,root,deep=True)
                print('SKIP',root['root_id'],flush=True); continue
            if stopping or (args.stop_after_roots is not None and attempted>=args.stop_after_roots): break
            # Any interrupted attempt is retained, never overwritten or published.
            for attempt in root['attempts']:
                if attempt['state']=='running':
                    attempt.update(state='failed',error='Interrupted before atomic completion')
                    root['state']='failed'
            if len(root['attempts'])>=manifest['bounds']['max_attempts']: continue
            if manifest['reserved_branches']+BRANCH_RESERVATION>manifest['bounds']['max_total_branches']:
                print('BRANCH BUDGET EXHAUSTED',flush=True); break
            free=shutil.disk_usage(output).free
            if free < manifest['bounds'].get('min_free_gb',0.)*1_000_000_000:
                manifest['runner'].update(state='stopped_low_disk',free_bytes=free)
                print('STOP LOW DISK',free,'bytes; no new root started',flush=True)
                break
            attempt=dict(number=len(root['attempts'])+1,state='running',started=time.time())
            attempt['path']=f"roots/{root['root_id']}/attempt_{attempt['number']:02d}"
            root['attempts'].append(attempt); root['state']='running'
            manifest['reserved_branches']+=BRANCH_RESERVATION
            save(output,manifest)
            recipe_file=output/'recipes'/f"{root['root_id']}.json"; write(recipe_file,root['recipe'])
            path=output/attempt['path']; logpath=output/'logs'/f"{root['root_id']}_{attempt['number']}.log"
            logpath.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_collect.py'),'collect','--output',str(path),
                 '--recipe',str(recipe_file),'--max-branches','36','--no-render',
                 '--storage-profile',manifest.get('storage_profile','validated')]
            print('START',root['root_id'],root['scene'],root['human_count'],flush=True)
            try:
                with logpath.open('w') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,start_new_session=True)
                    deadline=time.monotonic()+args.root_timeout
                    while True:
                        try:
                            rc=child.wait(timeout=min(30,max(.01,deadline-time.monotonic())))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic()>=deadline:
                                os.killpg(child.pid,signal.SIGKILL); child.wait()
                                raise RuntimeError('Root timeout; entire process group terminated')
                            lines=logpath.read_text(errors='replace').splitlines()
                            print('RUNNING',root['scene'],root['human_count'],root['root_id'],
                                  lines[-1] if lines else 'initializing',flush=True)
                if rc:
                    messages=[logpath.read_text(errors='replace')]
                    messages += [p.read_text(errors='replace')[-16384:] for p in path.glob('audits/workers/*/run.log')]
                    if any(marker in message for message in messages for marker in
                           ('Collector integrity audit failed','Root replay mismatch','Canonical/worker mismatch')):
                        raise GlobalIntegrityError(f'Global integrity failure; see {logpath}')
                    raise RuntimeError(f'Collector exit {rc}; see {logpath}')
                root['stats']=root_stats(path)
                root['observation_config']=read(path/'legal_observations/observation_config.json')
                root['query_generation_config']=read(path/'queries/queries.json')[0]['query']['generation_config']
                root['recipe']=read(path/'roots/replay.json')
                root['seals']=seals(path)
                write(path/'COMPLETE.json',dict(root_id=root['root_id'],seal_digest=digest(root['seals'])))
                root.update(state='completed',path=attempt['path'],bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()))
                attempt['state']='completed'
            except Exception:
                root['state']=attempt['state']='failed'; attempt['error']=traceback.format_exc()
                write(output/'failures'/f"{root['root_id']}_{attempt['number']}.json",attempt)
                # Disk failures cannot be isolated safely; do not continue writing.
                if isinstance(sys.exc_info()[1],GlobalIntegrityError):
                    manifest['runner']['state']='global_integrity_failure'
                    raise
                if isinstance(sys.exc_info()[1],OSError) and getattr(sys.exc_info()[1],'errno',None) in (28,5): raise
            finally:
                child=None; attempt['seconds']=time.time()-attempt['started']; save(output,manifest)
            attempted+=1
            print(root['state'].upper(),root['root_id'],flush=True)
        if manifest['runner']['state']=='running':
            manifest['runner']['state']='stopped' if stopping else 'finished_pass'
        save(output,manifest)
        print(json.dumps(aggregate(manifest),indent=2))
        if any(r['state']=='failed' for r in manifest['roots']): return 1
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','run','monitor','validate'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--scenes',nargs='+'); p.add_argument('--human-counts',type=int,nargs='+',choices=[0,1,2,4,6])
    p.add_argument('--roots-per-config',type=int)
    p.add_argument('--max-total-roots',type=int,default=0); p.add_argument('--max-total-branches',type=int,default=0)
    p.add_argument('--seed',type=int,default=2301); p.add_argument('--concurrency',type=int,choices=[1],default=1)
    p.add_argument('--profiles',nargs='+',choices=['wait','seek_visible','seek_move','seek_lost'],
                   default=['wait','seek_visible','seek_move','seek_lost'])
    p.add_argument('--max-attempts',type=int,default=2); p.add_argument('--root-timeout',type=int,default=1800)
    p.add_argument('--min-free-gb',type=float,default=0.)
    p.add_argument('--storage-profile',choices=['validated','production'])
    p.add_argument('--resume',action='store_true'); p.add_argument('--stop-after-roots',type=int)
    args=p.parse_args(); args.output=args.output.absolute()
    if args.command=='plan':
        m=plan(args); print(json.dumps(dict(roots=len(m['roots']),maximum_primary_queries=33*len(m['roots']),
            branches_including_audits=36*len(m['roots']),bounds=m['bounds'],selection=m['selection']),indent=2))
    elif args.command=='run': sys.exit(run(args))
    else:
        m=read(args.output/'manifest.json')
        if args.command=='validate':
            if any(r['state']!='completed' for r in m['roots']): raise RuntimeError('Dataset has incomplete/failed roots')
            for r in m['roots']:
                verify_root(args.output,r,deep=True)
                subprocess.run([sys.executable,str(REPO/'scripts/eventwm_collect.py'),'validate','--output',str(args.output/r['path'])],check=True)
                from eventwm_collect import branches
                bs=branches(args.output/r['path'])
                assert len({b['branch_id'] for b in bs})==len(bs)
                assert all(b['root_id']==r['root_id'] for b in bs)
            assert aggregate(m)==read(args.output/'aggregate.json')
            print('SCALE VALIDATION PASS')
        print(json.dumps(aggregate(m),indent=2))


if __name__=='__main__': main()
