#!/usr/bin/env python3
"""Resumable C36 production integration of the measured COMBINED execution DAG.

The sampler, episode preparation, query generation, labels and all audits remain
the existing V3 functions. This module owns dispatch, durable receipts and seals.
In-memory COW checkpoints are ephemeral; resume replays an uncommitted root using
its saved episode and seed, and never regenerates a completed, verified root.
"""
import argparse
from collections import deque
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from collector.core import read, write, digest
import eventwm_mass_collection as mass
from eventwm_scale import seals, verify_root
from eventwm_validate_staged import durable_json
from eventwm_execution_pool import Worker, Validator, stop_checkpoint
from eventwm_resource_monitor import Monitor

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO/'configs/eventwm_30k_release.json'
BRANCH = 'research/eventwm-v3-root-admission-exploration'
BASE = 'f37ea215cb355ed031a56392942372c7a0089a59'
FROZEN_COMMIT = 'dd1400e0121a8353d79be12d18f98c15bb17e43f'


def git(*args, repo=REPO):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def source_preflight(allow_dirty=False):
    if Path.cwd().resolve() != REPO:
        raise RuntimeError('Run from the root-admission worktree, not the parent checkout')
    if git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Wrong branch; expected ' + BRANCH)
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
    if not allow_dirty and (git('diff', '--name-only') or git('diff', '--cached', '--name-only')):
        raise RuntimeError('Tracked collector/configuration changes are uncommitted')
    release = read(RELEASE)
    if len(release['protected_scientific_hashes']) != 29:
        raise RuntimeError('Release must identify all 29 protected scientific files')
    frozen = Path(release['frozen_reference']['checkout'])
    if git('rev-parse', 'HEAD', repo=frozen) != FROZEN_COMMIT or git('status', '--porcelain', repo=frozen):
        raise RuntimeError('Frozen reference changed or is dirty')
    for path, expected in release['protected_scientific_hashes'].items():
        if hashlib.sha256((frozen/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Protected source hash mismatch: ' + path)
    for path, expected in release['implementation_hashes'].items():
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Validated implementation hash mismatch: ' + path)
    cfg = release['configuration']
    if (cfg['concurrency'], cfg['target_retained_queries'], cfg['exact_root_gate'],
        cfg['mode'], cfg['checkpoint'], cfg['movement_paths'], cfg['primary_queries'],
        cfg['audit_jobs'], cfg['native_cache_max_jobs'], cfg['native_cache_max_scenes'],
        cfg['threads_per_worker']) != (36, 30000, True, 'COMBO_NO_RENDERER', True, 8, 33, 3, 40, 3, 1):
        raise RuntimeError('Release is not the validated C36 configuration')
    if Path(sys.executable).resolve() != Path(release['python']).resolve():
        raise RuntimeError('Use the validated eventwm-falcon Python environment')
    for asset in ('data/robots/hab_spot_arm', 'data/humanoids/humanoid_data',
                  'pretrained_model/falcon_noaux_25.pth'):
        if not (REPO/asset).exists():
            raise RuntimeError('Required local asset missing: ' + asset)
    if not (mass.REPORT/'scene_inventory.json').exists():
        raise RuntimeError('Prepared scene inventory is missing')
    return release


def reject_parallel_collectors(output):
    names = {'eventwm_collect.py', 'eventwm_fast_runner.py', 'eventwm_mass_collection.py',
             'eventwm_combined_runner.py', 'run_exact_gate_panel.py'}
    for entry in Path('/proc').glob('[0-9]*'):
        if entry.name == str(os.getpid()):
            continue
        try:
            parts = (entry/'cmdline').read_bytes().split(b'\0')
            if not parts or b'python' not in Path(os.fsdecode(parts[0])).name.encode():
                continue
            if any(Path(os.fsdecode(x)).name in names for x in parts[1:3]):
                raise RuntimeError('Another collection/benchmark runner is alive: PID ' + entry.name)
            if any(Path(os.fsdecode(x)).name == 'dag_pool_worker.py' for x in parts[1:3]):
                raise RuntimeError('An earlier execution worker is alive: PID ' + entry.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def save_manifest(output, manifest):
    # Preserve the existing JSON schema and aggregate definition; add durability.
    mass.save(output, manifest)
    for name in ('manifest.json', 'aggregate.json'):
        with (output/name).open('rb') as stream:
            os.fsync(stream.fileno())
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit_root(output, root):
    attempt = root['attempts'][-1]
    path = output/attempt['path']
    if not (path/'VALIDATED.json').exists():
        raise RuntimeError('Cannot publish an unvalidated result')
    stats, labels = mass.inspect_root(path, root)
    marker = path/'COMPLETE.json'
    seal_file = f"root_seals/{root['root_id']}.json"
    if marker.exists():
        root_seals = read(output/seal_file)
        verify_root(output, dict(root, path=attempt['path'], seals=root_seals), deep=True)
    else:
        # All root data must reach disk before the completion receipt.
        root_seals = seals(path)
        for name in root_seals:
            with (path/name).open('rb') as stream:
                os.fsync(stream.fileno())
        for directory in sorted([path, *[p for p in path.rglob('*') if p.is_dir()]],
                                key=lambda p: len(p.parts), reverse=True):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        durable_json(output/seal_file, root_seals)
        durable_json(marker, dict(root_id=root['root_id'], seal_digest=digest(root_seals)))
    new_ids = {l['query_id'] for l in labels}
    root.update(state='completed', path=attempt['path'], seal_file=seal_file,
                stats=stats, query_ids=sorted(new_ids))
    attempt['state'] = 'completed'
    durable_json(output/'primary_indices'/f"{root['root_id']}.json",
                 dict(root_id=root['root_id'], role=root['role'], scene_role=root['scene_role'],
                      split_group=root['root_id'], physical_scene_id=root['physical_scene_id'],
                      path=attempt['path'], samples=labels))


def recover(output, manifest):
    """Use the existing completion-marker/seal recovery before considering retry."""
    for root in manifest['roots']:
        if root['state'] == 'completed':
            verify_root(output, dict(root, seals=read(output/root['seal_file'])), deep=True)
        elif root['state'] == 'running':
            if mass.recover_completed(output, root):
                continue
            path = output/root['attempts'][-1]['path']
            if (path/'VALIDATED.json').exists():
                try:
                    commit_root(output, root)
                    root['attempts'][-1]['recovered_validation'] = True
                except mass.NoRetainedValidQueries:
                    root['state'] = 'failed'
                    root['attempts'][-1].update(state='failed', reason='no_retained_valid_queries')
                continue
            # A COW owner cannot survive a host restart. Keep this attempt intact
            # and use the same persisted recipe/episode on the existing retry law.
            root['state'] = 'failed'
            root['attempts'][-1].update(state='failed', reason='interrupted_before_commit')
    ids = [q for root in manifest['roots'] if root['state']=='completed'
           for q in root.get('query_ids', [])]
    if len(ids) != len(set(ids)):
        raise RuntimeError('global_integrity: duplicate retained query IDs on resume')


def ordered_commit(output, manifest, validation_ready):
    cursor = manifest.get('execution_cursor', 0)
    while cursor < len(manifest['roots']):
        root = manifest['roots'][cursor]
        if root['state'] in ('completed', 'failed'):
            cursor += 1
            continue
        if root['root_id'] not in validation_ready:
            break
        retained_ids = {q for r in manifest['roots'] if r['state']=='completed'
                        for q in r.get('query_ids', [])}
        try:
            _, labels = mass.inspect_root(output/root['attempts'][-1]['path'], root)
        except mass.NoRetainedValidQueries:
            root['state'] = 'failed'
            root['attempts'][-1].update(state='failed', reason='no_retained_valid_queries')
            validation_ready.remove(root['root_id'])
            cursor += 1
            continue
        if retained_ids & {l['query_id'] for l in labels}:
            raise RuntimeError('global_integrity: duplicate retained query IDs')
        commit_root(output, root)
        validation_ready.remove(root['root_id'])
        cursor += 1
    manifest['execution_cursor'] = cursor
    save_manifest(output, manifest)


def failure_reason(result):
    error = result['error']
    message = error['message']
    if error['type'] == 'RootStaticPreflightRejected':
        return 'root_static_clearance_preflight'
    if 'insufficient_static_feasible_queries' in message:
        return 'insufficient_static_feasible_queries'
    if 'wm_train_stratum_unavailable' in message:
        return 'requested_legal_stratum_unavailable'
    if 'Prefix terminated before root' in message:
        return 'prefix_terminated_before_root'
    # Only known scientific rejections are nonfatal. Never silently discard a bug.
    raise RuntimeError('Unexpected root failure: ' + repr(error))


def mark_timeout(root, path, worker_id, seconds):
    """Same retryable worker_timeout verdict and preserved evidence as V3 launch."""
    root['state'] = 'failed'
    root['attempts'][-1].update(state='failed', reason='worker_timeout', seconds=seconds,
                               detail='Worker did not acknowledge within the existing 300 s deadline')
    durable_json(path/'audits/workers'/worker_id/'collection_error.json',
                 dict(termination_reason='worker_timeout', timeout_seconds=300))


def populate_validation_panel(output, manifest, panel):
    """Only the explicit <=8-proposal integration-test command uses saved jobs."""
    entries = read(panel)
    if not 1 <= len(entries) <= 8:
        raise ValueError('Integration panel must contain 1..8 saved proposals')
    dataset = Path('/home/haodong/workspace/projects/eventwm-falcon/datasets/raw/eventwm_social_outcome_v3_train_scale_04')
    old = read(dataset/'manifest.json')
    ids = [row['root_id'] for row in entries]
    if manifest['roots']:
        if manifest.get('validation_panel_ids') != ids:
            raise RuntimeError('Resume panel identity changed')
        return
    catalogue = output/'catalogue'
    cases = []
    for rid in ids:
        root = copy.deepcopy(next(r for r in old['roots'] if r['root_id']==rid))
        root.update(state='pending', attempts=[], recipe=read(dataset/'recipes'/f'{rid}.json'))
        for key in ('stats', 'path', 'seal_file', 'query_ids', 'pid', 'gpu'):
            root.pop(key, None)
        episode = dataset/'catalogue/episodes'/f'{rid}.json'
        dest = catalogue/'episodes'/f'{rid}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(episode, dest)
        root['episode_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        cases.append(dict(tag=root['recipe']['case'], scene=root['scene'], humans=root['human_count'],
                          seed=root['seed'], episode=str(dest.relative_to(catalogue)),
                          episode_sha256=root['episode_sha256']))
        write(output/'recipes'/f'{rid}.json', root['recipe'])
        manifest['roots'].append(root)
    write(catalogue/'cases.json', cases)
    write(catalogue/'scenes.json', [s['scene'] for s in manifest['scenes']])
    manifest['validation_panel_ids'] = ids


def run(args, release):
    output = args.output.resolve()
    config = release['configuration']
    if args.action == 'run' and output != Path(config['output']).resolve():
        raise RuntimeError('Production output must match the frozen release configuration')
    if args.action == 'validate-panel' and output == Path(config['output']).resolve():
        raise RuntimeError('A diagnostic panel must not use the production output')
    reject_parallel_collectors(output)
    output.mkdir(parents=True, exist_ok=True)
    (REPO/'output').mkdir(exist_ok=True)
    original_hashes = mass.hashes
    mass.hashes = lambda: release['implementation_hashes']
    init_args = SimpleNamespace(output=output, target=30000, max_roots=None, min_free_gb=150.)
    with (REPO/'output/mass_catalog.lock').open('a') as catalogue_lock, (output/'.lock').open('a') as lock:
        fcntl.flock(catalogue_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = mass.init(init_args)
        if manifest.get('execution_release') not in (None, digest(release)):
            raise RuntimeError('Resume release configuration differs')
        if manifest['frozen_commit'] != git('rev-parse', 'HEAD'):
            raise RuntimeError('Resume commit differs; use the frozen production commit')
        manifest['execution_release'] = digest(release)
        manifest['execution_configuration'] = config
        manifest['scientific_reference'] = release['frozen_reference']
        manifest['protected_scientific_hashes'] = release['protected_scientific_hashes']
        recover(output, manifest)
        if args.action == 'validate-panel':
            populate_validation_panel(output, manifest, args.panel)
        elif manifest.get('validation_panel_ids'):
            raise RuntimeError('A diagnostic dataset cannot become production')
        mass.catalog_link(output)
        manifest['runner'] = dict(pid=os.getpid(), host=socket.gethostname(), started=time.time())
        manifest['state'] = 'running'
        session_id = time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        session_dir = output/'logs'/session_id
        session_dir.mkdir(parents=True)
        session = dict(started=time.time(), concurrency=36, architecture='COMBINED',
                       exact_root_gate=True, renderer=False, checkpoint=True, threads=1,
                       diagnostic=args.action=='validate-panel', logs=str(session_dir))
        manifest['sessions'].append(session)
        save_manifest(output, manifest)
        started = time.perf_counter()
        base_seconds = manifest['collection_seconds']
        pids, checkpoints, validation_ready = set(), {}, set()
        monitor = Monitor(session_dir/'telemetry.jsonl', pids)
        monitor.start()
        pending, idle, busy, validating = deque(), [], {}, {}
        states, active_roots, workers = {}, set(), []
        validator = None
        execution = concurrent.futures.ThreadPoolExecutor(max_workers=36)
        validation = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        stopping = False
        fatal = None
        last_save = time.monotonic()
        events = (session_dir/'scheduler_events.jsonl').open('a', buffering=1)

        def event(kind, **values):
            events.write(json.dumps(dict(time=time.perf_counter()-started, kind=kind, **values))+'\n')

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            manifest['state'] = 'draining_user_stop'
            print('DRAIN: no new proposals; queued proposals, branches and audits will finish', flush=True)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def job(root, wid='root', qid=None, limit=None):
            return dict(root_id=root['root_id'], scene=root['scene'],
                        output=str(output/root['attempts'][-1]['path']),
                        worker_id=wid, query_id=qid, limit_steps=limit)

        def prepare(root):
            if read(output/'recipes'/f"{root['root_id']}.json") != root['recipe']:
                raise RuntimeError('Frozen proposal recipe mismatch')
            epfile = output/'catalogue/episodes'/f"{root['root_id']}.json"
            if hashlib.sha256(epfile.read_bytes()).hexdigest() != root['episode_sha256']:
                raise RuntimeError('Frozen proposal episode mismatch')
            number = len(root['attempts'])+1
            rel = f"roots/{root['root_id']}/attempt_{number:02d}"
            attempt = dict(number=number, path=rel, state='running', started=time.time(),
                           log=str(session_dir/f"{root['root_id']}_{number}.jsonl"))
            root['attempts'].append(attempt)
            root.update(state='running', pid=os.getpid(), gpu=None)
            # A retry is still the same logical proposal, not a new sample.
            manifest['execution_cursor'] = min(manifest.get('execution_cursor', 0),
                                               manifest['roots'].index(root))
            path = output/rel
            (path/'roots').mkdir(parents=True)
            shutil.copyfile(output/'recipes'/f"{root['root_id']}.json", path/'roots/replay.json')
            active_roots.add(root['root_id'])
            pending.append(job(root))
            save_manifest(output, manifest)

        def next_root():
            root = next((r for r in manifest['roots'] if r['state']=='pending'), None)
            if root is None:
                root = next((r for r in manifest['roots'] if r['state']=='failed' and
                             len(r['attempts']) < manifest['max_attempts'] and
                             r['root_id'] not in {j['root_id'] for _, j in busy.values()} and
                             r['attempts'][-1]['reason'] in
                             ('interrupted_before_commit', 'root_timeout', 'worker_timeout')), None)
            if root is None and args.action == 'run':
                root = mass.schedule_spec(manifest['next_sequence'], manifest['scenes'], manifest['master_seed'])
                manifest['next_sequence'] += 1
                manifest['roots'].append(root)
                try:
                    mass.root_job(output, manifest, root)
                except Exception as exc:
                    root['state'] = 'failed'
                    root['attempts'].append(dict(state='failed', reason='episode_preparation_unavailable', detail=repr(exc)))
                    save_manifest(output, manifest)
                    return False
            return root

        try:
            # Resume can be a no-op: verify completed seals, then do not create workers.
            total = mass.aggregate(manifest)
            panel_finished = args.action=='validate-panel' and all(r['state'] in ('completed','failed') for r in manifest['roots'])
            if total['primary_queries'] >= 30000 or panel_finished:
                stopping = True
            else:
                validator = Validator(REPO, output, session_dir, pids)
                with concurrent.futures.ThreadPoolExecutor(max_workers=36) as startup:
                    launches = [startup.submit(Worker, i, REPO, output, session_dir, pids, checkpoints, started) for i in range(36)]
                    startup_errors = []
                    for f in launches:
                        try:
                            workers.append(f.result())
                        except Exception as exc:
                            startup_errors.append(exc)
                    if startup_errors:
                        raise RuntimeError('Pool initialization failure: '+repr(startup_errors[0]))
                idle = list(workers)
            while True:
                manifest['collection_seconds'] = base_seconds + time.perf_counter()-started
                total = mass.aggregate(manifest)
                if not stopping and total['primary_queries'] >= 30000:
                    stopping = True
                    manifest['state'] = 'target_reached'
                if total['completed_roots'] >= manifest['max_completed_roots'] and total['primary_queries'] < 30000:
                    manifest['max_completed_roots'] += math.ceil((30000-total['primary_queries'])/33)
                if shutil.disk_usage(output).free < (150+72)*1e9:
                    raise RuntimeError('Disk guard: C36 requires 222 GB free at dispatch')
                if monitor.samples:
                    mem = monitor.samples[-1]['meminfo']
                    available = int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))
                    if available < 16*1024*1024:
                        raise RuntimeError('RAM guard: less than 16 GiB available')
                while not stopping and len(active_roots) < 36 and total['completed_roots']+len(active_roots) < manifest['max_completed_roots']:
                    root = next_root()
                    if root is False:
                        continue
                    if root is None:
                        stopping = True
                        manifest['state'] = 'bounded_validation_draining'
                        break
                    prepare(root)
                event('queue', ready=len(pending), busy=len(busy), idle=len(idle),
                      validation_backlog=len(validating),
                      root_ready=sum(j['worker_id']=='root' for j in pending),
                      branch_ready=sum(j['worker_id']!='root' for j in pending))
                for worker in list(idle):
                    if not pending:
                        break
                    selected = next((j for j in pending if j['worker_id']!='root' and j['scene']==worker.last_scene), pending[0])
                    pending.remove(selected)
                    idle.remove(worker)
                    busy[execution.submit(worker.execute, selected)] = (worker, selected)
                    event('dispatch', root_id=selected['root_id'], worker_id=selected['worker_id'], slot=worker.index)
                if not pending and not busy and not validating:
                    break
                done, _ = concurrent.futures.wait([*busy, *validating], timeout=.2,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    if f in validating:
                        root = validating.pop(f)
                        reply = f.result()
                        states[root['root_id']]['validation'] = reply
                        validation_ready.add(root['root_id'])
                        root['attempts'][-1]['seconds'] = time.time()-root['attempts'][-1]['started']
                        ordered_commit(output, manifest, validation_ready)
                        active_roots.difference_update(r['root_id'] for r in manifest['roots']
                                                       if r['state'] in ('completed', 'failed'))
                        event('validation_end', root_id=root['root_id'], seconds=reply['seconds'])
                        print('VALIDATED', root['root_id'], reply['result']['retained'], flush=True)
                        continue
                    worker, j = busy.pop(f)
                    idle.append(worker)
                    rid, wid = j['root_id'], j['worker_id']
                    root = next(x for x in manifest['roots'] if x['root_id']==rid)
                    if root['state']=='failed' or Path(j['output']) != output/root['attempts'][-1]['path']:
                        # A timed-out sibling's whole attempt has already been
                        # retired. Preserve its files; never mix it into a retry.
                        try:
                            f.result()
                        except Exception:
                            pass
                        continue
                    try:
                        r = f.result()
                    except (TimeoutError, queue.Empty):
                        mark_timeout(root, Path(j['output']), wid,
                                     time.time()-root['attempts'][-1]['started'])
                        cp = checkpoints.pop(rid, None)
                        receipt = Path(j['output'])/'checkpoints/manifest.json'
                        if cp is None and receipt.exists():
                            cp = read(receipt)
                        if cp is not None:
                            try:
                                os.killpg(cp['checkpoint_pid'], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            pids.discard(cp['checkpoint_pid'])
                        pending = deque(x for x in pending if x['root_id']!=rid)
                        active_roots.discard(rid)
                        if wid=='root':
                            os.killpg(worker.proc.pid, signal.SIGTERM)
                            worker.close()
                            idle.remove(worker)
                            replacement = Worker(worker.index, REPO, output, session_dir,
                                                 pids, checkpoints, started)
                            workers[workers.index(worker)] = replacement
                            idle.append(replacement)
                        ordered_commit(output, manifest, validation_ready)
                        event('worker_timeout', root_id=rid, worker_id=wid)
                        print('FAILED', rid, 'worker_timeout (same saved proposal eligible for retry)', flush=True)
                        continue
                    with Path(root['attempts'][-1]['log']).open('a') as stream:
                        stream.write(json.dumps(r)+'\n')
                    event('result', root_id=rid, worker_id=wid, failed=bool(r['error']))
                    if r.get('checkpoint'):
                        checkpoints[rid] = r['checkpoint']
                        pids.add(r['checkpoint']['checkpoint_pid'])
                    if wid == 'root':
                        if r['error']:
                            reason = failure_reason(r)
                            root['state'] = 'failed'
                            root['attempts'][-1].update(state='failed', reason=reason,
                                                       seconds=r['dispatch_wall'], detail=r['error'])
                            if rid in checkpoints:
                                stop_checkpoint(checkpoints.pop(rid), pids)
                            active_roots.discard(rid)
                            ordered_commit(output, manifest, validation_ready)
                            active_roots.difference_update(x['root_id'] for x in manifest['roots']
                                                           if x['state'] in ('completed', 'failed'))
                            print('FAILED', rid, reason, flush=True)
                        else:
                            queries = read(Path(j['output'])/'queries/queries.json')
                            if len(queries) != 33 or rid not in checkpoints:
                                raise RuntimeError('Accepted root must have 33 queries and exact checkpoint')
                            states[rid] = dict(primary_done=0, queries=queries)
                            pending.extend(job(root, f'b{i:03}', q['query']['query_id']) for i, q in enumerate(queries))
                    else:
                        if r['error'] or r['label']['static_feasibility_violation']:
                            raise RuntimeError('Branch/static integrity failure: ' + repr(r['error'] or r['label']))
                        st, queries = states[rid], states[rid]['queries']
                        if wid.startswith('b'):
                            st['primary_done'] += 1
                            if st['primary_done'] == 33:
                                pending.append(job(root, 'reverse_1', queries[1]['query']['query_id']))
                        elif wid == 'reverse_1':
                            pending.append(job(root, 'reverse_0', queries[0]['query']['query_id']))
                        elif wid == 'reverse_0':
                            pending.append(job(root, 'truncated_wait', queries[-1]['query']['query_id'], 10))
                        elif wid == 'truncated_wait':
                            stop_checkpoint(checkpoints.pop(rid), pids)
                            validating[validation.submit(validator.execute, Path(j['output']))] = root
                            event('validation_submit', root_id=rid)
                if time.monotonic()-last_save >= 5:
                    save_manifest(output, manifest)
                    last_save = time.monotonic()
        except BaseException as exc:
            fatal = repr(exc)
            manifest['state'] = 'interrupted_error'
            durable_json(session_dir/'STOP.json', dict(reason=fatal))
            traceback.print_exc()
        finally:
            # No unowned COW world may remain after coordinator exit.
            if fatal:
                # A root can create its anchor before returning a job receipt.
                # Recover those ownership records as well before killing workers.
                for root in manifest['roots']:
                    if root['state'] != 'running':
                        continue
                    receipt = output/root['attempts'][-1]['path']/'checkpoints/manifest.json'
                    if receipt.exists():
                        cp = read(receipt)
                        cmd = Path(f"/proc/{cp['checkpoint_pid']}/cmdline")
                        try:
                            parts = [os.fsdecode(x) for x in cmd.read_bytes().split(b'\0') if x]
                            if str(REPO/'scripts/dag_pool_worker.py') in parts and '--collection-output' in parts and Path(parts[parts.index('--collection-output')+1]).resolve()==output:
                                checkpoints.setdefault(root['root_id'], cp)
                        except (FileNotFoundError, ProcessLookupError, PermissionError):
                            pass
            for cp in list(checkpoints.values()):
                try:
                    if fatal:
                        os.killpg(cp['checkpoint_pid'], signal.SIGTERM)
                    else:
                        stop_checkpoint(cp, pids)
                except (ProcessLookupError, OSError, RuntimeError):
                    try:
                        os.killpg(cp['checkpoint_pid'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            if fatal:
                for worker in workers:
                    if worker.proc.poll() is None:
                        os.killpg(worker.proc.pid, signal.SIGTERM)
            execution.shutdown(wait=True)
            validation.shutdown(wait=True)
            if validator:
                validator.close()
            for worker in workers:
                worker.close()
            monitor.stop()
            events.close()
            manifest['collection_seconds'] = base_seconds+time.perf_counter()-started
            if not fatal:
                if manifest['state'] == 'draining_user_stop':
                    manifest['state'] = 'stopped_by_user'
                elif args.action=='validate-panel':
                    manifest['state'] = 'bounded_validation_finished'
                elif mass.aggregate(manifest)['primary_queries'] >= 30000:
                    manifest['state'] = 'target_reached'
            session.update(seconds=time.perf_counter()-started, state=manifest['state'])
            save_manifest(output, manifest)
            mass.hashes = original_hashes
        print('SUMMARY', json.dumps(mass.aggregate(manifest)), flush=True)
        return 2 if fatal else 0


def show_status(output):
    if not (output/'manifest.json').exists():
        print('RUN STATUS: NOT INITIALIZED')
        return
    manifest = read(output/'manifest.json')
    total = mass.aggregate(manifest)
    print('RUN STATUS:', manifest['state'], 'PID', manifest.get('runner', {}).get('pid'))
    print('CONFIG: COMBINED C36; exact state-dependent gate ON; renderer OFF')
    print('RETAINED:', total['primary_queries'], '/', total['target_primary_queries'])
    print('CURRENT RUN AVERAGE Q/H:', round(total['queries_per_hour'], 2))
    print('ROOTS:', {k:total[k] for k in ('completed_roots','running_roots','failed_roots')})
    print('FAILURE REASONS:', total['failed_reasons'])
    print('STATIC VIOLATIONS:', total['static_violations'])
    rate = total['queries_per_hour']
    print('ETA HOURS:', max(0, total['target_primary_queries']-total['primary_queries'])/rate if rate else 'unknown')
    print('OUTPUT:', output)


def request_stop(output):
    runner = read(output/'manifest.json').get('runner', {})
    if runner.get('host') != socket.gethostname():
        raise RuntimeError('Recorded runner is not on this host')
    pid = runner.get('pid')
    file = Path(f'/proc/{pid}/cmdline')
    if not file.exists():
        print('Runner already stopped')
        return
    parts = [os.fsdecode(x) for x in file.read_bytes().split(b'\0') if x]
    if str(REPO/'scripts/eventwm_combined_runner.py') not in parts or '--output' not in parts:
        raise RuntimeError('Runner PID identity changed; refusing to signal')
    if Path(parts[parts.index('--output')+1]).resolve() != output:
        raise RuntimeError('Runner output identity changed; refusing to signal')
    os.kill(pid, signal.SIGTERM)
    print('Drain requested; do not resume until RUN STATUS is stopped_by_user')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run', 'status', 'stop', 'validate-panel'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--panel', type=Path)
    args = parser.parse_args()
    release = source_preflight(allow_dirty=args.action=='validate-panel')
    args.output = (args.output or Path(release['configuration']['output'])).resolve()
    if args.action == 'check':
        print('PASS:', BRANCH, git('rev-parse', 'HEAD'))
        print('Protected scientific hashes: 29/29; C36; exact gate ON; renderer OFF; target 30000')
    elif args.action == 'status':
        show_status(args.output)
    elif args.action == 'stop':
        request_stop(args.output)
    else:
        if args.action=='validate-panel' and args.panel is None:
            raise ValueError('A bounded saved panel is required')
        return run(args, release)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
