#!/usr/bin/env python3
"""Candidate pipeline for 4b01a7a. Science functions are imported unchanged.

Separate preparation, native execution, validation and durable publication.
New output only: this candidate never migrates a live/original production run.
"""
import argparse
from collections import Counter, deque
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import sys
import time
import traceback
from types import SimpleNamespace

import eventwm_combined_runner as old
import eventwm_mass_collection as mass
from collector.core import read, digest, write
from eventwm_execution_pool import stop_checkpoint
from eventwm_pipeline_pool import PipelineWorker, PipelineValidator as Validator
from eventwm_pipeline_io import StateStore, prepare_episode, reserve_attempt, immutable_json, read_status
from eventwm_pipeline_index import ActiveIndex
from eventwm_resource_monitor import Monitor
from eventwm_300k_storage import compact_validated, pack_rejected, spill_root, recover_spills, capacity
from eventwm_validate_staged import durable_json

REPO = Path(__file__).resolve().parents[1]
BASE = '4b01a7a699462967172c9d61485dea4b40a41f78'
RELEASE = REPO/'configs/eventwm_pipeline_release.json'
RETRY = {'interrupted_before_commit', 'root_timeout', 'worker_timeout'}


def preflight():
    release = read(RELEASE)
    if Path.cwd().resolve() != REPO:
        raise RuntimeError('Run from the candidate checkout')
    if old.git('diff', '--name-only') or old.git('diff', '--cached', '--name-only'):
        raise RuntimeError('Commit the candidate before native testing/collection')
    old.subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
    for path, expected in release['implementation_hashes'].items():
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Candidate source mismatch: '+path)
    frozen = Path(release['frozen_reference']['checkout'])
    if old.git('rev-parse', 'HEAD', repo=frozen) != old.FROZEN_COMMIT or old.git('status', '--porcelain', repo=frozen):
        raise RuntimeError('Frozen scientific reference changed')
    for path, expected in release['protected_scientific_hashes'].items():
        if hashlib.sha256((frozen/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source mismatch: '+path)
    if Path(sys.executable).resolve() != Path(release['python']).resolve():
        raise RuntimeError('Use the eventwm-falcon environment')
    for asset in ('data/robots/hab_spot_arm', 'data/humanoids/humanoid_data',
                  'pretrained_model/falcon_noaux_25.pth'):
        if not (REPO/asset).exists():
            raise RuntimeError('Missing asset: '+asset)
    return release


def retryable(root, manifest):
    return (root['state'] == 'failed' and len(root['attempts']) < manifest['max_attempts']
            and root['attempts'][-1].get('reason') in RETRY)


def executing_capacity(executing, preparation_count, waiting_count, concurrency, backlog):
    # Reserve staging capacity for every active native root before admission.
    return len(executing) < concurrency and waiting_count+len(executing) < backlog


def reject_pipeline_collectors():
    names={'eventwm_pipeline_runner.py','eventwm_pipeline_worker.py','eventwm_pipeline_validator.py'}
    for entry in Path('/proc').glob('[0-9]*'):
        if entry.name==str(os.getpid()): continue
        try:
            parts=[os.fsdecode(x) for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
            if not parts or 'python' not in Path(parts[0]).name: continue
            pipeline=any(Path(x).name in names for x in parts[1:4])
            benchmark=(any(Path(x).name=='eventwm_pipeline_benchmark.py' for x in parts[1:4]) and
                       '--stage' in parts and parts[parts.index('--stage')+1] in ('baseline','candidate'))
            if pipeline or benchmark: raise RuntimeError('Another pipeline process is alive: PID '+entry.name)
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass


def publish_one(output, root):
    value = copy.deepcopy(root)
    path=output/value["attempts"][-1]["path"]
    if not (path/"COMPLETE.json").exists():
        compact_validated(path)
        if output==Path("/tmp/eventwm_v4_300k"):
            spill_root(output,value)
    old.commit_root(output, value)
    return value


def populate_fixture(output, manifest, fixture):
    """Replay saved logical proposals, with their exact saved episode bytes."""
    spec = read(fixture)
    if manifest['roots']:
        if manifest.get('pipeline_fixture_digest') != digest(spec):
            raise RuntimeError('Fixture changed on resume')
        return
    source = Path(spec['source']).resolve()
    previous = read(source/'manifest.json')
    roots = {r['root_id']:r for r in previous['roots']}
    if len(spec['root_ids']) != len(set(spec['root_ids'])):
        raise ValueError('Duplicate fixture IDs')
    for rid in spec['root_ids']:
        root = copy.deepcopy(roots[rid])
        root.update(state='pending', attempts=[], _pipeline_ordinal=len(manifest['roots']))
        for key in ('stats','path','seal_file','query_ids','pid','gpu'):
            root.pop(key, None)
        recipe = read(source/'recipes'/f'{rid}.json')
        root['recipe'] = recipe
        dest = output/'catalogue/episodes'/f'{rid}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/'catalogue/episodes'/f'{rid}.json', dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != root['episode_sha256']:
            raise RuntimeError('Fixture episode hash mismatch')
        case = dict(tag=recipe['case'], scene=root['scene'], humans=root['human_count'],
                    seed=root['seed'], episode=str(dest.relative_to(output/'catalogue')),
                    episode_sha256=root['episode_sha256'])
        immutable_json(output/'catalogue/cases'/f'{rid}.json', case)
        immutable_json(output/'recipes'/f'{rid}.json', recipe)
        manifest['roots'].append(root)
    manifest['pipeline_fixture_digest'] = digest(spec)


def run(args, release):
    output = args.output.resolve()
    if output == Path(read(old.RELEASE)['configuration']['output']).resolve():
        raise RuntimeError('Candidate must use a NEW output, not the original production dataset')
    old.reject_parallel_collectors(output)
    reject_pipeline_collectors()
    output.mkdir(parents=True, exist_ok=True)
    (REPO/'output').mkdir(exist_ok=True)
    # All runners share this lock; a stopped old dataset remains untouched.
    with (REPO/'output/mass_catalog.lock').open('a') as catalog_lock, (output/'.lock').open('a') as output_lock:
        fcntl.flock(catalog_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(output_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior_hashes = mass.hashes
        mass.hashes = lambda: release['implementation_hashes']
        store = None
        try:
            fresh = not (output/'pipeline.sqlite3').exists() and not (output/'manifest.json').exists()
            if (output/'pipeline.sqlite3').exists():
                store = StateStore(output)
                manifest = store.load()
                if manifest is None:
                    manifest = mass.init(SimpleNamespace(output=output, target=args.target, max_roots=None, min_free_gb=150.))
            else:
                if (output/'manifest.json').exists():
                    existing = read(output/'manifest.json')
                    if existing.get('pipeline_version') != 1 and existing.get('source_hashes') != release['implementation_hashes']:
                        raise RuntimeError('Refuse legacy manifest migration')
                manifest = mass.init(SimpleNamespace(output=output, target=args.target, max_roots=None, min_free_gb=150.))
                store = StateStore(output)
            expected_seed=getattr(args,'master_seed',None) or release['configuration']['master_seed']
            if not manifest['roots'] and manifest['next_sequence']==0:
                manifest['master_seed']=expected_seed
            if manifest['master_seed']!=expected_seed:
                raise RuntimeError('Master seed differs from immutable release')
            if fresh:
                manifest['scientific_reference']=release['frozen_reference']
                manifest['protected_scientific_hashes']=release['protected_scientific_hashes']
                manifest['audit_storage_encoding']='lossless-worker-json-gzip-v1'
            settings = dict(concurrency=args.concurrency, preparers=args.preparers, validators=args.validators,
                            prefetch=args.prefetch, backlog=args.backlog, max_proposals=args.max_proposals,
                            target=args.target, fixture_digest=digest(read(args.fixture)) if args.fixture else None)
            if manifest.get('pipeline_settings', settings) != settings:
                raise RuntimeError('Resume settings differ')
            if manifest['source_hashes'] != release['implementation_hashes'] or manifest['frozen_commit'] != old.git('rev-parse','HEAD'):
                raise RuntimeError('Resume source/commit differs')
            if manifest.get('pipeline_release', digest(release)) != digest(release):
                raise RuntimeError('Resume release differs')
            manifest.update(pipeline_version=1, pipeline_settings=settings, pipeline_release=digest(release))
            if args.fixture:
                populate_fixture(output, manifest, args.fixture)
            for index, root in enumerate(manifest['roots']):
                root['_pipeline_ordinal'] = index
            # Completed/sealed native results use the existing recovery verifier.
            recover_spills(output)
            old.recover(output, manifest)
            write(output/'catalogue/scenes.json', [s['scene'] for s in manifest['scenes']])
            mass.catalog_link(output)
            store.flush(manifest, all_roots=True)
            return coordinate(args, output, manifest, store)
        finally:
            if store is not None:
                store.close()
            mass.hashes = prior_hashes


def coordinate(args, output, manifest, store):
    started = time.perf_counter()
    base_seconds = manifest['collection_seconds']
    session_dir = output/'logs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    session_dir.mkdir(parents=True)
    manifest.update(state='running', runner=dict(pid=os.getpid(), host=socket.gethostname()))
    session = dict(started=time.time(), settings=manifest['pipeline_settings'], architecture='PIPELINE_V1',
                   initial_queries=sum(r['stats']['primary_queries'] for r in manifest['roots'] if r['state']=='completed'))
    manifest['sessions'].append(session)
    roots = {r['root_id']:r for r in manifest['roots']}
    scenes = {s['name']:s for s in manifest['scenes']}
    coverage = {name:Counter() for name in scenes}
    for root in roots.values():
        if root.get('start_region'): coverage[root['scene']][root['start_region']] += 1
    index = ActiveIndex(roots, lambda r:retryable(r, manifest))
    running, checkpoints, pids = set(), {}, set()
    native, preparing, validating, packing = {}, {}, {}, {}
    waiting, states, publishing = set(), {}, {}
    pending = deque()
    workers, validators, idle, validator_idle = [], [], [], []
    to_validate = deque()
    retained_ids = {q for r in roots.values() if r['state']=='completed' for q in r.get('query_ids',[])}
    completed = sum(r['state']=='completed' for r in roots.values())
    committed_queries = sum(r['stats']['primary_queries'] for r in roots.values() if r['state']=='completed')
    cursor = 0
    stopping, fatal = committed_queries>=args.target, None
    last_metrics = 0.
    last_guard = 0.
    native_limit = args.concurrency
    last_control = 0.
    max_native_workers = 60 if args.concurrency==36 else args.concurrency
    free_gb = shutil.disk_usage(output).free/1e9
    mem_ok = True
    exec_pool = cf.ThreadPoolExecutor(max_workers=max_native_workers)
    val_pool = cf.ThreadPoolExecutor(max_workers=args.validators)
    pub_pool = cf.ThreadPoolExecutor(max_workers=1)
    pack_pool = cf.ThreadPoolExecutor(max_workers=2)
    prep_pool = cf.ProcessPoolExecutor(max_workers=args.preparers, mp_context=multiprocessing.get_context('spawn'))
    monitor = Monitor(session_dir/'telemetry.jsonl', pids)
    monitor.start()
    events = (session_dir/'scheduler_events.jsonl').open('a', buffering=1)

    def event(kind, **kw):
        events.write(json.dumps(dict(time=time.perf_counter()-started, kind=kind, **kw))+'\n')

    def persist(root):
        index.transition(root)
        store.touch(root)
        store.flush(manifest)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        manifest['state'] = 'draining_user_stop'
        print('DRAIN: finish reserved preparation/native/validation/publication work', flush=True)
    handlers = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}

    def job(root, wid='root', qid=None, limit=None):
        return dict(root_id=root['root_id'], scene=root['scene'], output=str(output/root['attempts'][-1]['path']),
                    worker_id=wid, query_id=qid, limit_steps=limit)

    try:
        unfinished=any(r['state'] in ('pending','preparing','running') or retryable(r,manifest) for r in roots.values())
        if committed_queries < args.target or unfinished:
            with cf.ThreadPoolExecutor(max_workers=args.concurrency+args.validators) as startup:
                ws = [startup.submit(PipelineWorker,i,REPO,output,session_dir,pids,checkpoints,started) for i in range(args.concurrency)]
                vs = [startup.submit(Validator,REPO,output,session_dir/f'validator_{i}',pids) for i in range(args.validators)]
                errors = []
                for f, dest in [(f,workers) for f in ws]+[(f,validators) for f in vs]:
                    try: dest.append(f.result())
                    except Exception as exc: errors.append(exc)
                if errors: raise RuntimeError('Startup failed: '+repr(errors[0]))
            idle, validator_idle = list(workers), list(validators)
        else:
            stopping = True
            manifest['state'] = 'target_reached'
        while True:
            manifest['collection_seconds'] = base_seconds+time.perf_counter()-started
            # Process receipts BEFORE preparing/dispatching. No batch-fill barrier.
            for f in list(preparing):
                root, began = preparing[f]
                if not f.done():
                    if time.monotonic()-began > 300:
                        raise TimeoutError('Episode preparation exceeded 300 seconds: '+root['root_id'])
                    continue
                preparing.pop(f)
                result = f.result()  # Integrity/programming errors are fatal.
                if result['error']:
                    root.update(state='failed')
                    root['attempts'].append(dict(state='failed',reason='episode_preparation_unavailable',detail=result['error']))
                else:
                    root.update(state='pending', start_region=result['start_region'], episode_sha256=result['episode_sha256'])
                    coverage[root['scene']][root['start_region']] += 1
                    root.pop('_preparation_covered',None)
                persist(root)
                event('preparation_end',root_id=root['root_id'],**result)
            for f in list(packing):
                if f.done():
                    root=packing.pop(f)
                    root['storage_pack']=f.result()
                    persist(root)
            for f in list(validating):
                if not f.done(): continue
                validator, root = validating.pop(f)
                validator_idle.append(validator)
                reply = f.result()
                states[root['root_id']]['validation'] = reply
                waiting.add(root['root_id'])
                root['attempts'][-1]['seconds'] = time.time()-root['attempts'][-1]['started']
                persist(root)
                event('validation_end',root_id=root['root_id'],seconds=reply['seconds'])
            for f in list(publishing):
                if not f.done(): continue
                root = publishing.pop(f)
                try:
                    value = f.result()
                except mass.NoRetainedValidQueries:
                    root.update(state='failed')
                    root['attempts'][-1].update(state='failed',reason='no_retained_valid_queries')
                else:
                    if retained_ids.intersection(value['query_ids']):
                        raise RuntimeError('Duplicate retained query IDs')
                    retained_ids.update(value['query_ids'])
                    root.update(value)
                    committed_queries += root['stats']['primary_queries']
                    completed += 1
                    print('COMMITTED', root['root_id'], 'TOTAL', committed_queries, flush=True)
                waiting.remove(root['root_id'])
                states.pop(root['root_id'],None)
                persist(root)
                cursor += 1
            for f in list(native):
                if not f.done(): continue
                worker, j = native.pop(f)
                idle.append(worker)
                rid, wid = j['root_id'], j['worker_id']
                root = roots[rid]
                if root['state']=='failed' or str(output/root['attempts'][-1]['path']) != j['output']:
                    try: f.result()
                    except Exception: pass
                    continue
                try:
                    result = f.result()
                except (TimeoutError,queue.Empty):
                    old.mark_timeout(root,Path(j['output']),wid,time.time()-root['attempts'][-1]['started'])
                    cp = checkpoints.pop(rid,None)
                    receipt = Path(j['output'])/'checkpoints/manifest.json'
                    if cp is None and receipt.exists(): cp=read(receipt)
                    if cp:
                        try: os.killpg(cp['checkpoint_pid'],signal.SIGTERM)
                        except ProcessLookupError: pass
                        pids.discard(cp['checkpoint_pid'])
                    pending = deque(x for x in pending if x['root_id']!=rid)
                    running.discard(rid)
                    states.pop(rid,None)
                    if wid=='root':
                        try: os.killpg(worker.proc.pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                        worker.close()
                        idle.remove(worker)
                        replacement=PipelineWorker(worker.index,REPO,output,session_dir,pids,checkpoints,started)
                        workers[workers.index(worker)]=replacement
                        idle.append(replacement)
                    persist(root)
                    continue
                with Path(root['attempts'][-1]['log']).open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                event('result',root_id=rid,worker_id=wid,failed=bool(result['error']),slot_seconds=result.get('dispatch_wall',0),native_wall_seconds=result.get('wall_seconds',0),native_cpu_seconds=result.get('cpu_seconds',0),metrics=result.get('metrics',{}))
                if result.get('checkpoint'):
                    checkpoints[rid]=result['checkpoint']
                    pids.add(result['checkpoint']['checkpoint_pid'])
                if wid=='root':
                    if result['error']:
                        reason=old.failure_reason(result)
                        root['state']='failed'
                        root['attempts'][-1].update(state='failed',reason=reason,seconds=result['dispatch_wall'],detail=result['error'])
                        if rid in checkpoints: stop_checkpoint(checkpoints.pop(rid),pids)
                        running.discard(rid)
                        persist(root)
                        if output==Path('/tmp/eventwm_v4_300k'):
                            packing[pack_pool.submit(pack_rejected,output,copy.deepcopy(root))]=root
                    else:
                        queries=read(Path(j['output'])/'queries/queries.json')
                        if len(queries)!=33 or rid not in checkpoints:
                            raise RuntimeError('Accepted root requires 33 queries and checkpoint')
                        states[rid]=dict(primary_done=0,queries=queries)
                        pending.extend(job(root,f'b{i:03}',q['query']['query_id']) for i,q in enumerate(queries))
                else:
                    if result['error'] or result['label']['static_feasibility_violation']:
                        raise RuntimeError('Branch/static integrity failure: '+repr(result['error'] or result['label']))
                    st=states[rid]
                    if wid.startswith('b'):
                        st['primary_done']+=1
                        if st['primary_done']==33: pending.append(job(root,'reverse_1',st['queries'][1]['query']['query_id']))
                    elif wid=='reverse_1': pending.append(job(root,'reverse_0',st['queries'][0]['query']['query_id']))
                    elif wid=='reverse_0': pending.append(job(root,'truncated_wait',st['queries'][-1]['query']['query_id'],10))
                    elif wid=='truncated_wait':
                        stop_checkpoint(checkpoints.pop(rid),pids)
                        running.discard(rid)  # Native capacity released immediately.
                        to_validate.append(root)
            while validator_idle and to_validate:
                validator=validator_idle.pop()
                root=to_validate.popleft()
                validating[val_pool.submit(validator.execute,output/root['attempts'][-1]['path'])]=(validator,root)
            while not publishing and cursor<len(manifest['roots']):
                root=manifest['roots'][cursor]
                if root['state']=='completed' or (root['state']=='failed' and not retryable(root,manifest)):
                    cursor+=1
                    continue
                if root['root_id'] in waiting:
                    publishing[pub_pool.submit(publish_one,output,root)]=root
                break
            staged = len(to_validate)+len(validating)+len(waiting)
            if committed_queries>=args.target:
                stopping=True
                manifest['state']='target_reached'
            # Enforce guards at ADMISSION; drain already reserved work on low disk/RAM.
            if time.monotonic()-last_guard >= 1:
                free_gb=capacity(output)/1e9 if output==Path('/tmp/eventwm_v4_300k') else shutil.disk_usage(output).free/1e9
                mem_ok=True
                if monitor.samples:
                    mem=monitor.samples[-1]['meminfo']
                    available=int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))
                    mem_ok=available>=16*1024*1024
                last_guard=time.monotonic()
            if not stopping and (free_gb<40+2*(args.backlog+args.prefetch) or not mem_ok):
                stopping=True
                manifest['state']='draining_resource_guard'
                event('resource_drain',free_gb=free_gb,mem_ok=mem_ok)
            # Retry already reserved scientific proposals, even while draining.
            busy_rids={j['root_id'] for _,j in native.values()}
            for rid in list(index.members['retry']):
                root = roots[rid]
                if rid not in busy_rids:
                    root['state']='pending'
                    persist(root)
                    cursor=min(cursor,root['_pipeline_ordinal'])
            while index.count('pending') and executing_capacity(running,len(preparing),staged,args.concurrency,args.backlog):
                root=index.first('pending')
                reserve_attempt(output,root,session_dir,persist)
                root.update(state='running',pid=os.getpid(),gpu=None)
                # Durable dispatched identity precedes submission to native pool.
                root['attempts'][-1]['state']='running'
                persist(root)
                running.add(root['root_id'])
                pending.append(job(root))
            # Audits then branches shorten checkpoint lifetime; ready roots follow.
            if time.monotonic()-last_control>=1:
                control=output/'native_limit.json'
                limit=args.concurrency
                if control.exists():
                    request=read(control);limit=int(request['limit'])
                    if limit>36 and (not request.get('until') or request['until']>time.time()+600):
                        raise RuntimeError('C48/C60 profile must be bounded to at most ten minutes')
                    if request.get('until') and time.time()>=request['until']:limit=args.concurrency
                    if limit not in (24,36,48,60) or limit>max_native_workers:
                        raise RuntimeError('Unsupported native limit')
                if limit!=native_limit:
                    native_limit=limit
                    event('native_limit',limit=native_limit)
                    print('NATIVE_LIMIT',native_limit,flush=True)
                manifest['native_limit']=native_limit
                last_control=time.monotonic()
            while len(workers)<native_limit:
                worker=PipelineWorker(len(workers),REPO,output,session_dir,pids,checkpoints,started)
                workers.append(worker);idle.append(worker)
            for worker in list(idle):
                if not pending or len(native)>=native_limit: break
                selected=min(pending,key=lambda j:(2 if j['worker_id']=='root' else 1 if j['worker_id'].startswith('b') else 0,
                                                   j['scene']!=worker.last_scene))
                pending.remove(selected)
                idle.remove(worker)
                native[exec_pool.submit(worker.execute,selected)]=(worker,selected)
                event('dispatch',root_id=selected['root_id'],worker_id=selected['worker_id'],slot=worker.index)
            # Reserve future episodes in deterministic sequence; never prepare two
            # proposals from the same scene concurrently (coverage dependency).
            preparing_scenes={r['scene'] for r,_ in preparing.values()}
            pending_roots=index.count('pending')
            while not args.fixture and len(preparing)+pending_roots<args.prefetch and len(packing)<72:
                # Recover reservations before generating a new spec.
                active_preps={r['root_id'] for r,_ in preparing.values()}
                root=index.reserved_preparation(active_preps)
                if root is None:
                    if stopping: break
                    reserved=len(running)+staged+len(preparing)+pending_roots
                    if completed+reserved>=manifest['max_completed_roots']:
                        if reserved==0 and committed_queries<args.target:
                            manifest['max_completed_roots']+=math.ceil((args.target-committed_queries)/33)
                        else: break
                    if args.max_proposals is not None and manifest['next_sequence']>=args.max_proposals: break
                    root=mass.schedule_spec(manifest['next_sequence'],manifest['scenes'],manifest['master_seed'])
                    if root['scene'] in preparing_scenes: break
                    root.update(state='preparing',_pipeline_ordinal=len(manifest['roots']),
                                _preparation_covered=dict(coverage[root['scene']]))
                    manifest['next_sequence']+=1
                    manifest['roots'].append(root)
                    roots[root['root_id']]=root
                    persist(root)
                if root['scene'] in preparing_scenes: break
                preparing_scenes.add(root['scene'])
                future=prep_pool.submit(prepare_episode,output,scenes[root['scene']],copy.deepcopy(root),root['_preparation_covered'])
                preparing[future]=(root,time.monotonic())
                event('preparation_start',root_id=root['root_id'])
            # Stop drains preparations already submitted even if admission closed.
            work=bool(native or preparing or pending or validating or to_validate or publishing or waiting or running or packing)
            unlaunched=bool(index.count('pending') or index.count('preparing'))
            if not work and not unlaunched: break
            if time.perf_counter()-last_metrics>=1:
                event('queue',busy=len(native),idle=len(idle),native_limit=native_limit,native_roots=len(running),preparing=len(preparing),
                      ready=len(pending),staged=staged,packing=len(packing),committed_queries=committed_queries)
                store.flush(manifest)
                store.export(manifest)
                last_metrics=time.perf_counter()
            futures=[*native,*preparing,*validating,*publishing,*packing]
            if futures: cf.wait(futures,timeout=.01,return_when=cf.FIRST_COMPLETED)
            else: time.sleep(.01)
    except BaseException as exc:
        fatal=repr(exc)
        manifest['state']='interrupted_error'
        durable_json(session_dir/'STOP.json',dict(reason=fatal))
        traceback.print_exc()
    finally:
        # Capture checkpoint ownership that appeared just before a worker failed.
        if fatal:
            for root in roots.values():
                if root['state']!='running' or not root['attempts']: continue
                receipt=output/root['attempts'][-1]['path']/'checkpoints/manifest.json'
                if receipt.exists():
                    cp=read(receipt)
                    cmd=Path(f"/proc/{cp['checkpoint_pid']}/cmdline")
                    try:
                        parts=cmd.read_bytes().split(b'\0')
                        if b'--collection-output' in parts and os.fsencode(str(output)) in parts:
                            checkpoints.setdefault(root['root_id'],cp)
                    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
        for cp in list(checkpoints.values()):
            try:
                if fatal: os.killpg(cp['checkpoint_pid'],signal.SIGTERM)
                else: stop_checkpoint(cp,pids)
            except (ProcessLookupError,OSError,RuntimeError): pass
        if fatal:
            for worker in workers+validators:
                if worker.proc.poll() is None:
                    try: os.killpg(worker.proc.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
            # Native preparation must not keep a failed coordinator alive forever.
            for proc in list(getattr(prep_pool,'_processes',{}).values()):
                proc.terminate()
        prep_pool.shutdown(wait=True,cancel_futures=True)
        exec_pool.shutdown(wait=True)
        val_pool.shutdown(wait=True)
        pub_pool.shutdown(wait=True)
        pack_pool.shutdown(wait=True)
        for worker in workers+validators: worker.close()
        monitor.stop()
        events.close()
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        manifest['collection_seconds']=base_seconds+time.perf_counter()-started
        if not fatal:
            manifest['state']=('target_reached' if committed_queries>=args.target else
                               'stopped_by_user' if manifest['state']=='draining_user_stop' else
                               'stopped_resource_guard' if manifest['state']=='draining_resource_guard' else 'bounded_finished')
        session.update(seconds=time.perf_counter()-started,state=manifest['state'],io_metrics=dict(store.metrics))
        store.flush(manifest)
        store.export(manifest,force=True)
        durable_json(session_dir/'SUMMARY.json',dict(state=manifest['state'],fatal=fatal,
                     seconds=session['seconds'],retained=committed_queries,
                     retained_qph=(committed_queries-session.get('initial_queries',0))/session['seconds']*3600,
                     io_metrics=dict(store.metrics)))
    print('SUMMARY',json.dumps(mass.aggregate(manifest)),flush=True)
    return 2 if fatal else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','run','status','stop'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--concurrency',type=int,default=36)
    p.add_argument('--preparers',type=int,default=4)
    p.add_argument('--validators',type=int,default=2)
    p.add_argument('--prefetch',type=int,default=8)
    p.add_argument('--backlog',type=int,default=72)
    p.add_argument('--target',type=int,default=300000)
    p.add_argument('--max-proposals',type=int)
    p.add_argument('--master-seed',type=int)
    p.add_argument('--fixture',type=Path)
    args=p.parse_args()
    args.output=args.output.resolve()
    if min(args.concurrency,args.preparers,args.validators,args.prefetch,args.backlog,args.target)<1:
        p.error('All capacities/target must be positive')
    if args.action=='status':
        if (args.output/'pipeline.sqlite3').exists():
            print(json.dumps(read_status(args.output),indent=2))
        else: print('NOT INITIALIZED')
        return 0
    if args.action=='stop':
        runner=read_status(args.output)['runner']
        if runner['host']!=socket.gethostname(): raise RuntimeError('Host differs')
        pid=runner['pid']
        file=Path(f'/proc/{pid}/cmdline')
        if not file.exists(): print('Already stopped');return 0
        parts=[os.fsdecode(x) for x in file.read_bytes().split(b'\0') if x]
        if str(Path(__file__).resolve()) not in parts or '--output' not in parts or Path(parts[parts.index('--output')+1]).resolve()!=args.output:
            raise RuntimeError('PID identity differs; refusing to signal')
        os.kill(pid,signal.SIGTERM)
        print('Drain requested')
        return 0
    release=preflight()
    if args.action=='check': print('PASS candidate source and frozen reference');return 0
    return run(args,release)


if __name__=='__main__':
    raise SystemExit(main())
