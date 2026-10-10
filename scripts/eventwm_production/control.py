"""Production operations only; executes the immutable scientific checkout.

Read-only actions never create directories, open writable SQLite, or signal a
process. Native physics, samplers, validators and publication remain unchanged.
"""
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from settings import CONFIG as C, CONFIG_PATH, PROJECT, SCIENCE, DATA, LOGS, TARGET, RELEASE_REPO
from settings import publication_epoch, MEASUREMENT_PREFIX


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def metadata(output=DATA, roots=False):
    path = output / 'pipeline.sqlite3'
    if not path.exists():
        return None, None, []
    # A consistent read transaction; never StateStore (which is writable).
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        m = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        p = json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
        rows = [json.loads(v) for v, in db.execute('SELECT value FROM roots ORDER BY ordinal')] if roots else []
    return m, p, rows


def identity(m, output=DATA, proc_root=Path('/proc')):
    if not m or not m.get('runner'):
        return None
    r = m['runner']
    require(r['host'] == socket.gethostname(), 'Dataset belongs to a different host')
    proc = proc_root / str(r['pid'])
    try:
        args = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        cwd = (proc / 'cwd').resolve(strict=True)
    except FileNotFoundError:
        return None
    if not args or 'python' not in Path(args[0]).name:
        return None  # A reused PID is never signalled.
    scripts = [x for x in args[1:4] if Path(x).name == 'eventwm_pipeline_runner.py']
    if len(scripts) != 1 or '--output' not in args:
        return None
    if Path(args[args.index('--output') + 1]).resolve() != output.resolve():
        return None
    script = (cwd / scripts[0]).resolve()
    require(script == cwd / 'scripts/eventwm_pipeline_runner.py', 'Unexpected runner source path')
    require(git(cwd, 'rev-parse', 'HEAD') == m['frozen_commit'], 'Active worktree commit differs from checkpoint')
    return dict(pid=r['pid'], worktree=str(cwd), branch=git(cwd, 'branch', '--show-current'), source_commit=m['frozen_commit'])


def session_exists(name):
    return subprocess.run(['tmux', 'has-session', '-t', '=' + name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def lock_busy(path):
    if not path.exists():
        return False
    with path.open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def reject_active(m=None):
    if m is None:
        m, _, _ = metadata()
    live = identity(m)
    require(not live, 'Collector already running against this dataset: PID ' + str(live and live['pid']))
    require(not session_exists(C['session']), 'Collector tmux session already exists; inspect it before resume')
    require(not lock_busy(DATA / '.lock'), 'Dataset execution lock is held')
    # The frozen runner disallows simultaneous native pipelines on this host.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and 'python' in Path(args[0]).name:
            names = {'eventwm_pipeline_runner.py', 'eventwm_pipeline_worker.py', 'eventwm_pipeline_validator.py'}
            require(not any(Path(x).name in names for x in args[1:4]),
                    'Another native pipeline process is alive: PID ' + proc.name)


def source_check():
    require(socket.gethostname() == C['host'], 'Wrong workstation for this configuration')
    require(Path(sys.executable).resolve() == Path(C['python']).resolve(), 'Wrong Python environment')
    require(C['native_limit'] == 24, 'The frozen production configuration uses C24')
    require(TARGET > 0 and TARGET <= C['pipeline_settings']['target'] + 30030, 'Invalid combined retained target')
    require(git(SCIENCE, 'rev-parse', 'HEAD') == C['scientific_commit'], 'Scientific source pin differs')
    path = SCIENCE / 'configs/eventwm_pipeline_release.json'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == C['scientific_release_sha256'], 'Scientific release file differs')
    # Existing native preflight covers all 478 source hashes, all 29 protected
    # reference hashes, environment, assets and clean scientific worktrees.
    subprocess.run([C['python'], str(SCIENCE / 'scripts/eventwm_pipeline_runner.py'),
                    'check', '--output', str(DATA)], cwd=SCIENCE, check=True,
                   stdout=subprocess.DEVNULL)
    release = read(path)
    require(DATA.resolve() == Path(release['configuration']['output']).resolve(), 'Output differs from the pinned workstation scientific configuration')
    require(Path(C['archive']).resolve() == Path(release['configuration']['archive']).resolve(), 'Archive differs from the pinned scientific configuration')
    gate = read(PROJECT / 'output/eventwm_72h_v4_branch_gate/VERDICT.json')
    require(gate['passed'] and gate['retained_native_primary_queries'] >= 100 and
            gate['source_commit'] == C.get('parent_scientific_commit',C['scientific_commit']), 'Native V4 gate provenance differs')
    if C.get('recovery_gate'):
        recovery=read(RELEASE_REPO / C['recovery_gate'])
        require(recovery['passed'] and recovery['source_commit']==C['scientific_commit'] and
                recovery['retained_native_primary_queries']>=100 and
                recovery['static_feasibility_violations']==0 and
                recovery['fresh_cow_query_pairs']>=8 and recovery['original_failure_rejected'],
                'Native static recovery gate failed or provenance differs')
    original = read(PROJECT / 'output/eventwm_72h_native_ab/VERDICT.json')
    require(original['scientific_matches'] and original['retained'] >= 100, 'Original native scientific gate failed')
    return release


def legacy():
    proof = read(PROJECT / 'output/eventwm_72h_legacy_verified.json')
    require(proof['all_seals_verified'] and not proof['errors'], 'Legacy seal verification failed')
    require(hashlib.sha256((Path(proof['source']) / 'manifest.json').read_bytes()).hexdigest() ==
            proof['manifest_sha256'], 'Legacy manifest changed')
    return proof


def checkpoint_check(release, m, p, roots, output=None):
    output = output or DATA
    require(m is not None, 'No saved pipeline checkpoint; resume will not create a new dataset')
    require(m['frozen_commit'] == C['scientific_commit'], 'Checkpoint source differs; no implicit migration')
    require(m['source_hashes'] == release['implementation_hashes'], 'Checkpoint implementation hashes differ')
    require(m['pipeline_release'] == canonical_digest(release), 'Checkpoint scientific release differs')
    require(m['pipeline_settings'] == C['pipeline_settings'], 'Resume settings differ from the saved checkpoint')
    require(m['master_seed'] == release['configuration']['master_seed'], 'Checkpoint seed differs')
    require(m['protected_scientific_hashes'] == release['protected_scientific_hashes'], 'Protected hash metadata differs')
    proof = legacy()
    ids = set(proof['accepted_query_ids'])
    root_ids = set(proof['accepted_root_ids'])
    epochs = {e['commit'] for e in m.get('production_source_epochs', [])} | {m['frozen_commit']}
    total = 0
    for r in roots:
        require(r['root_id'] not in root_ids, 'Duplicate root ID')
        root_ids.add(r['root_id'])
        if r['state'] != 'completed':
            continue
        require(r.get('source_commit') in epochs, 'Completed root source epoch is missing')
        require((output / r['seal_file']).is_file() and (output / r['path'] / 'COMPLETE.json').is_file(), 'Completed root lacks seal/COMPLETE')
        queries = r['query_ids']
        require(len(queries) == len(set(queries)) == r['stats']['primary_queries'] == 33, 'Invalid completed query IDs')
        require(not ids.intersection(queries), 'Duplicate query IDs across the existing datasets')
        ids.update(queries)
        total += len(queries)
    require(total == p['primary_queries'], 'Published-root count differs from checkpoint progress')
    return dict(published_new_queries=total, legacy_queries=proof['retained'],
                total_retained=total + proof['retained'], duplicate_query_ids=0,
                source_epochs=sorted(epochs), next_sequence=m['next_sequence'], master_seed=m['master_seed'])


def disk_check(retained):
    output_parent = DATA if DATA.exists() else DATA.parent
    archive = Path(C['archive']).resolve()
    archive_parent = archive if archive.exists() else archive.parent
    a = shutil.disk_usage(output_parent).free
    b = shutil.disk_usage(archive_parent).free
    free = a + b if os.stat(output_parent).st_dev != os.stat(archive_parent).st_dev else a
    required = max(0, TARGET - retained) * C['capacity_bytes_per_query'] + 1e9 * (
        C['rejected_prefix_reserve_gb'] + C['working_reserve_gb'] + C['headroom_gb'])
    require(a >= C['minimum_output_free_gb'] * 1e9, 'Insufficient native output filesystem space')
    require(free >= max(required, C['minimum_aggregate_free_gb'] * 1e9), 'Insufficient aggregate capacity for the remaining target')
    return dict(output_free_gb=a / 1e9, aggregate_free_gb=free / 1e9, conservative_required_gb=required / 1e9)


def check():
    operations_check()
    release = source_check()
    m, p, roots = metadata(roots=True)
    result = checkpoint_check(release, m, p, roots) if m else dict(total_retained=legacy()['retained'], initialized=False)
    result.update(disk_check(result['total_retained']))
    result.update(passed=True, release_version=C['release_version'], scientific_commit=C['scientific_commit'],
                  protected_hashes=29, implementation_hashes=len(release['implementation_hashes']),
                  active=identity(m), config=str(CONFIG_PATH), native_limit=C['native_limit'])
    return result


def status():
    m, p, roots = metadata(roots=True)
    if not m:
        return dict(state='not_initialized', dataset=str(DATA))
    old = legacy()
    roles = collections.Counter(old['role_queries'])
    sr = collections.Counter()
    retained = v4 = geometry = 0
    for r in roots:
        if r['state'] != 'completed':
            continue
        require((DATA / r['seal_file']).is_file() and (DATA / r['path'] / 'COMPLETE.json').is_file(), 'Unsealed completed root')
        n = r['stats']['primary_queries']; retained += n; roles[r['role']] += n
        sr.update(r['stats']['sr_targets'])
        v4 += n if r.get('source_commit') == C['scientific_commit'] else 0
        geometry += n if r.get('recipe', {}).get('episode_sampler', {}).get('train_only') else 0
    # Legacy label totals are already sealed in the immutable manifest.
    old_m = read(Path(old['source']) / 'manifest.json')
    for r in old_m['roots']:
        if r['state'] == 'completed': sr.update(r['stats']['sr_targets'])
    now = time.time(); total = old['retained'] + retained; windows = {}
    log = LOGS / 'progress.jsonl'; history = []
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 8 * 1024 * 1024))
            for line in f.read().decode(errors='replace').splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get('timestamp'), (int, float)) and isinstance(row.get('new_retained'), int):
                        history.append(row)
                except ValueError: continue
    epoch = publication_epoch()
    recovering=bool(C.get('recovery_gate'))
    if recovering:
        # Rates for this collector must not include crash downtime or old-epoch
        # staged publication. Keep the original source's historical evidence.
        history=[r for r in history if r.get('source_commit')==C['scientific_commit'] and
                 r.get('state')=='running' and r.get('runner',{}).get('pid')==m.get('runner',{}).get('pid')]
    for minutes in ((10,15,30,60) if recovering else (15,30,60)):
        candidates = [r for r in history if r['timestamp'] >= epoch['time'] and
                      minutes * 60 <= now - r['timestamp'] <= minutes * 60 + 180]
        if candidates:
            r = min(candidates, key=lambda r: abs(now - r['timestamp'] - minutes * 60))
            seconds = now - r['timestamp']; q = retained - r['new_retained']
            windows[str(minutes)] = dict(seconds=seconds, retained=q, qph=q / seconds * 3600)
    deadline = datetime.fromisoformat('2026-10-12T09:20:00+01:00').timestamp()
    remaining = max(0, TARGET - total)
    preferred = windows.get('60') or windows.get('30') or (windows.get('15') or windows.get('10') if recovering else None)
    measurement_file = PROJECT / 'output' / f'{MEASUREMENT_PREFIX}_30min.json'
    measured = read(measurement_file) if measurement_file.exists() else None
    if recovering and measured and measured.get('source_commit')!=C['scientific_commit']:measured=None
    rate = preferred['qph'] if preferred else measured['actual_publication_qph_including_resume_startup'] if measured else None
    control = read(DATA / 'native_limit.json')
    metrics_file = PROJECT / 'output' / f'{MEASUREMENT_PREFIX}_native_metrics.json'
    metrics = read(metrics_file) if metrics_file.exists() else {}
    integrity_file = PROJECT / 'output/eventwm_300k_combined_index/integrity.json'
    integrity = read(integrity_file) if integrity_file.exists() else {}
    devices = {}
    for name, path in [('output', DATA), ('archive', Path(C['archive']).resolve())]:
        usage = shutil.disk_usage(path)
        devices[name] = dict(path=str(path), free_gb=usage.free / 1e9, device=os.stat(path).st_dev)
    free = devices['output']['free_gb'] + (devices['archive']['free_gb'] if devices['output']['device'] != devices['archive']['device'] else 0)
    return dict(timestamp=now, now_london=datetime.fromtimestamp(now, ZoneInfo('Europe/London')).isoformat(),
                active=identity(m), checkpoint_state=m['state'], dataset=str(DATA), archive=str(Path(C['archive']).resolve()),
                total_retained=total, legacy_retained=old['retained'], new_retained=retained,
                v4_epoch_retained=v4, geometry_train_retained=geometry, roles=dict(roles), sr=dict(sr),
                worker_pool=m['pipeline_settings']['concurrency'], native_limit=control['limit'],
                native_metrics=metrics, windows=windows, startup_inclusive_30_min=measured,
                preferred_qph=rate, rate_basis='current source publication window' if preferred else 'startup-inclusive saved 30-minute measurement',
                required_qph=remaining / ((deadline - now) / 3600) if deadline > now else None,
                estimated_completion_london=datetime.fromtimestamp(now + remaining / rate * 3600, ZoneInfo('Europe/London')).isoformat() if rate and rate > 0 else None,
                storage=devices, aggregate_free_gb=free, deeply_verified_retained=integrity.get('total_verified'),
                integrity_passed=integrity.get('passed'))


@contextlib.contextmanager
def launch_lock():
    LOGS.mkdir(parents=True, exist_ok=True)
    # Outside the dataset so the lock survives a planned reboot relocation.
    with (LOGS / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare_launch(action):
    # Perform duplicate checks BEFORE any directory creation or metadata write.
    reject_active()
    operations_check(require_committed=True)
    saved = (DATA / 'pipeline.sqlite3').exists()
    require(saved if action == 'resume' else not saved,
            'Use resume for the existing dataset' if action == 'start' else 'No saved dataset; resume refuses to create one')
    if action == 'start':
        require(not DATA.exists() or not any(DATA.iterdir()), 'Start refuses a nonempty output directory')
        index = PROJECT / 'output/eventwm_300k_combined_index/verified.sqlite3'
        if index.exists():
            with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
                known = db.execute('SELECT COUNT(*) FROM roots WHERE dataset=?', (str(DATA),)).fetchone()[0]
            require(known == 0, 'Combined index already references this dataset; restore its checkpoint and use resume')
    result = check()
    require(result['total_retained'] < TARGET, 'Retained target already reached; no collector needed')
    require(not (PROJECT / 'output/eventwm_300k_combined_index/FAILURE.json').exists(),
            'An integrity failure receipt exists; investigate it before resuming')
    return result


def operations_check(require_committed=False):
    require(git(RELEASE_REPO, 'branch', '--show-current') == C['release_branch'], 'Wrong production release branch')
    if require_committed:
        require(not git(RELEASE_REPO, 'diff', '--name-only') and not git(RELEASE_REPO, 'diff', '--cached', '--name-only'),
                'Commit production operational changes before launching')
    path = RELEASE_REPO / 'configs/eventwm_300k_operations.json'
    require(path.exists(), 'Operational release hash manifest is missing')
    operation_manifest = read(path)
    require(operation_manifest['release_version'] == C['release_version'] and
            operation_manifest['scientific_commit'] == C['scientific_commit'], 'Operational release pin differs')
    for name, expected in operation_manifest['hashes'].items():
        require(hashlib.sha256((RELEASE_REPO / name).read_bytes()).hexdigest() == expected, 'Operational release file differs: ' + name)


def durable(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def launch(action):
    prepare_launch(action)
    # Serializes simultaneous user starts until the durable tmux session exists.
    with launch_lock():
        prepare_launch_after_lock(action)
        DATA.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        durable(DATA / 'native_limit.json', dict(limit=C['native_limit'], controller='engineering-freeze', reason=C['release_version']))
        command = shlex.join([str(RELEASE_REPO / 'scripts/run_eventwm_300k.sh'), '_run'])
        command = 'EVENTWM_PRODUCTION_CONFIG=' + shlex.quote(str(CONFIG_PATH)) + ' ' + command
        subprocess.run(['tmux', 'new-session', '-d', '-s', C['session'], '-c', str(SCIENCE), command], check=True)
    helpers()
    print('Started durable collector session ' + C['session'] + '; dataset ' + str(DATA))


def prepare_launch_after_lock(action):
    reject_active()
    saved = (DATA / 'pipeline.sqlite3').exists()
    require(saved == (action == 'resume'), 'Dataset initialization changed during launch')
    if action == 'start':
        require(not DATA.exists() or not any(DATA.iterdir()), 'Output became nonempty during launch')


def run_collector():
    # Internal tmux entry: never exposed as a way around duplicate protection.
    m, _, _ = metadata()
    require(not identity(m), 'Collector already running')
    require(not lock_busy(DATA / '.lock'), 'Dataset execution lock is held')
    source_check()
    settings = C['pipeline_settings']
    args = [C['python'], '-u', str(SCIENCE / 'scripts/eventwm_pipeline_runner.py'), 'run', '--output', str(DATA)]
    for name in ('target', 'concurrency', 'preparers', 'validators', 'prefetch', 'backlog'):
        args += ['--' + name, str(settings[name])]
    durable(LOGS / 'last_launch.json', dict(time=time.time(), release_commit=git(RELEASE_REPO, 'rev-parse', 'HEAD'),
                 release_version=C['release_version'], scientific_commit=C['scientific_commit'], config=str(CONFIG_PATH), args=args))
    with (LOGS / 'collector.log').open('ab', buffering=0) as log:
        # A single log writer and no pipeline-dependent exit-code masking.
        result = subprocess.run(['nice', '-n', '5', 'ionice', '-c', '2', '-n', '6', *args], cwd=SCIENCE,
                                stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


def stop():
    m, _, _ = metadata(); live = identity(m)
    if not live:
        print('Collector is already stopped'); return
    # Exact host/PID/command/output/commit identity was checked above. This
    # invokes the existing SIGTERM drain handler; never kills native workers.
    os.kill(live['pid'], signal.SIGTERM)
    print('Graceful drain requested for PID ' + str(live['pid']) + '; poll status until stopped')


def reboot_storage(action):
    """Operator-only atomic relocation while stopped; never overwrite data.

    cv-w06 clears /tmp at boot. /var/tmp is the same persistent filesystem and
    has no active cleanup rule. Restore the literal /tmp pathname before resume
    to preserve the frozen runner's path-specific storage behavior.
    """
    operations_check(require_committed=True)
    with launch_lock():
        relocate_stopped_dataset(action)


def relocate_stopped_dataset(action):
    backup = Path(C['reboot_backup'])
    require(backup.parent.is_dir(), 'Persistent reboot-backup parent is missing')
    if action == 'preserve-for-reboot':
        require(DATA.is_dir() and not backup.exists(), 'Dataset missing or reboot backup already exists; refusing overwrite')
        m, p, roots = metadata(roots=True)
        require(m is not None, 'No saved checkpoint; refusing to preserve an unrecognized directory')
        reject_active(m)
        require(m['state'] in ('stopped_by_user', 'target_reached', 'stopped_resource_guard', 'interrupted_error'),
                'Wait for a stopped checkpoint before preserving data')
        checkpoint_check(source_check(), m, p, roots)
        require(os.stat(DATA).st_dev == os.stat(backup.parent).st_dev, 'Reboot preservation requires the same filesystem')
        stop_support_sessions()
        os.rename(DATA, backup)
        print('Preserved complete dataset at ' + str(backup) + '; restore-after-reboot before resume')
    else:
        require(not DATA.exists() and not DATA.is_symlink(), 'Output already exists; refusing overwrite')
        require((backup / 'pipeline.sqlite3').is_file(), 'No preserved checkpoint; refusing fresh initialization')
        m, p, roots = metadata(output=backup, roots=True)
        reject_active(m)
        checkpoint_check(source_check(), m, p, roots, output=backup)
        require(os.stat(backup).st_dev == os.stat(DATA.parent).st_dev, 'Restore requires the original filesystem')
        os.rename(backup, DATA)
        print('Restored complete dataset at ' + str(DATA) + '; run check, verify, then resume')
    for parent in {DATA.parent, backup.parent}:
        fd = os.open(parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


HELPERS = {
    'monitor': ('monitor.py', 'eventwm_300k_monitor', 'monitor.log'),
    'deadline': ('deadline_watchdog.py', 'eventwm_300k_deadline', 'deadline.log'),
    'integrity': ('integrity_index.py', 'eventwm_300k_integrity', 'integrity_index.log'),
    'phases': ('observe_production.py', 'eventwm_300k_phases', 'phase_observer.log'),
    'supervision': ('successor_masks.py', 'eventwm_300k_supervision', 'successor_masks.log'),
    'sampler': ('sampler_stats.py', 'eventwm_300k_sampler_stats', 'sampler_stats.log'),
    'v4_measurement': ('measure_v4.py', 'eventwm_300k_v4_measurement', 'v4_measurement.log'),
    'native_metrics': ('measure_native_v4.py', 'eventwm_300k_v4_native_metrics', 'v4_native_metrics.log'),
    'legacy_supervision': ('legacy_supervision.py', 'eventwm_300k_legacy_supervision', 'legacy_supervision.log'),
    'finalize': ('finalize_mission.py', 'eventwm_300k_finalize', 'finalize_mission.log'),
}


def stop_support_sessions():
    # Only invoked for an operator-requested preservation after the collector
    # has stopped. Avoid missing-/tmp alarms while intentionally relocated.
    for _, session, _ in HELPERS.values():
        require(session != C['session'], 'Support session must differ from collector')
        if session_exists(session):
            subprocess.run(['tmux', 'kill-session', '-t', '=' + session], check=True)


def helpers():
    operations_check(require_committed=True)
    require((DATA / 'pipeline.sqlite3').exists() or session_exists(C['session']), 'No initialized dataset or starting collector')
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, (_, session, _) in HELPERS.items():
        if session_exists(session):
            print('Helper already present: ' + session); continue
        cmd = 'EVENTWM_PRODUCTION_CONFIG=' + shlex.quote(str(CONFIG_PATH)) + ' ' + shlex.join([
            str(RELEASE_REPO / 'scripts/run_eventwm_300k.sh'), '_helper', '--name', name])
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(PROJECT), cmd], check=True)
        print('Started helper: ' + session)


def run_helper(name):
    script, _, log_name = HELPERS[name]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / (name + '.helper.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with (LOGS / log_name).open('ab', buffering=0) as log:
                code = subprocess.run(['nice', '-n', '15', 'ionice', '-c', '3', C['python'], '-u',
                                      str(Path(__file__).parent / script)], cwd=PROJECT,
                                     stdout=log, stderr=subprocess.STDOUT).returncode
            if code == 0: return
            durable(LOGS / (name + '.helper_failure.json'), dict(time=time.time(), exit_code=code, retry_seconds=30))
            time.sleep(30)


def verify_index():
    file = PROJECT / 'output/eventwm_300k_combined_index/verified.sqlite3'
    with sqlite3.connect(file.resolve().as_uri() + '?mode=ro', uri=True) as db:
        require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Combined index is corrupt')
        n, unique = db.execute('SELECT COUNT(*),COUNT(DISTINCT query_id) FROM examples').fetchone()
        require(n == unique, 'Duplicate indexed query IDs')
    report = read(file.parent / 'integrity.json')
    require(report['passed'], 'Integrity index reports failure')
    require(not (file.parent / 'FAILURE.json').exists(), 'Integrity failure receipt exists')
    return dict(passed=True, indexed_queries=n, report=report, scope='Already deeply verified sealed index; may lag the live collector')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='status', choices=['check', 'status', 'start', 'resume',
                        'logs', 'attach', 'stop', 'drain', 'verify', 'helpers', 'preserve-for-reboot',
                        'restore-after-reboot', '_run', '_helper'])
    parser.add_argument('--follow', action='store_true', help='Follow persistent collector log')
    parser.add_argument('--name', choices=HELPERS)
    args = parser.parse_args()
    if args.action == 'check': print(json.dumps(check(), indent=2))
    elif args.action == 'status': print(json.dumps(status(), indent=2))
    elif args.action in ('start', 'resume'): launch(args.action)
    elif args.action == '_run': run_collector()
    elif args.action == '_helper': require(args.name, '--name required'); run_helper(args.name)
    elif args.action in ('stop', 'drain'): stop()
    elif args.action in ('preserve-for-reboot', 'restore-after-reboot'): reboot_storage(args.action)
    elif args.action == 'helpers': helpers()
    elif args.action == 'verify': print(json.dumps(verify_index(), indent=2))
    elif args.action == 'attach': os.execvp('tmux', ['tmux', 'attach-session', '-t', '=' + C['session']])
    elif args.action == 'logs':
        file = LOGS / 'collector.log'; require(file.exists(), 'Persistent log is missing')
        cmd = ['tail', '-n', '60'] + (['-F'] if args.follow else []) + [str(file)]
        os.execvp('tail', cmd)


if __name__ == '__main__':
    try: main()
    except (RuntimeError, BlockingIOError, subprocess.CalledProcessError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr); raise SystemExit(2)
