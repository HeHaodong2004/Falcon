"""Bounded operational recovery; never changes scientific sources or data."""
import fcntl
import json
import os
import shlex
import subprocess
import time

import control as ctl

SESSION = 'eventwm_300k_recovery'
MAX_RESTARTS = 3
WINDOW = 3600
POLL = 15


def state_path():
    return ctl.LOGS / 'recovery_state.json'


def events(event, **fields):
    ctl.LOGS.mkdir(parents=True, exist_ok=True)
    with (ctl.LOGS / 'recovery.jsonl').open('a') as log:
        log.write(json.dumps(dict(time=time.time(), event=event, **fields)) + '\n')
        log.flush()
        os.fsync(log.fileno())


def state():
    return ctl.read(state_path()) if state_path().exists() else dict(enabled=False, restarts=[])


def save(value):
    ctl.durable(state_path(), value)


def disable(reason='operator_stop'):
    # Same lock as the final restart admission: stop cannot race a restart.
    ctl.LOGS.mkdir(parents=True, exist_ok=True)
    with (ctl.LOGS / 'recovery_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = state()
        value.update(enabled=False, status='disabled', reason=reason, time=time.time())
        save(value)
        events('disabled', reason=reason)


def start():
    ctl.operations_check(require_committed=True)
    m, _, _ = ctl.metadata()
    ctl.require(m is not None, 'Recovery requires the existing checkpoint')
    ctl.require(ctl.identity(m) or ctl.session_exists(ctl.C['session']),
                'Resume the collector before enabling recovery')
    with (ctl.LOGS / 'recovery_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = state()
        value.update(enabled=True, status='watching', reason=None, time=time.time())
        save(value)
        if not ctl.session_exists(SESSION):
            command = 'EVENTWM_PRODUCTION_CONFIG=' + shlex.quote(str(ctl.CONFIG_PATH)) + ' ' + shlex.join([
                str(ctl.RELEASE_REPO / 'scripts/run_eventwm_300k.sh'), '_watch'])
            subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION,
                            '-c', str(ctl.RELEASE_REPO), command], check=True)
        events('enabled', collector=m.get('runner'))
    print('Guarded collector recovery enabled: ' + SESSION)


def stop_reason(pid):
    # Read only the failed collector's own receipt, never an older epoch.
    files = sorted((ctl.DATA / 'logs').glob('*_' + str(pid) + '/STOP.json'))
    return ctl.read(files[-1]).get('reason', '') if files else ''


def refusal(m, reason, restarts, now):
    text = reason.lower()
    if any(word in text for word in ('integrity', 'static', 'duplicate', 'seal',
                                    'hash', 'source', 'corrupt', 'mismatch', 'release differs')):
        return 'Scientific/data integrity failure: ' + reason
    if m['state'] not in ('running', 'interrupted_error'):
        return 'Checkpoint intentionally stopped or guarded: ' + m['state']
    if m['state'] == 'interrupted_error' and not reason:
        return 'Interrupted checkpoint lacks a failure receipt; operator inspection required'
    if len([t for t in restarts if now - t < WINDOW]) >= MAX_RESTARTS:
        return 'Restart budget exhausted: three attempts per hour'
    return None


def block(value, reason):
    with (ctl.LOGS / 'recovery_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = state()
        value.update(enabled=False, status='blocked', reason=reason, time=time.time())
        save(value)
    events('blocked', reason=reason)


def heartbeat(**fields):
    with (ctl.LOGS / 'recovery_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = state()
        if value.get('enabled'):
            value.update(time=time.time(), **fields)
            save(value)


def watch():
    # One watchdog per dataset, independent of monitor helper lifetimes.
    ctl.LOGS.mkdir(parents=True, exist_ok=True)
    with (ctl.LOGS / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dead_pid = None
        while True:
            value = state()
            if not value.get('enabled'):
                # Remain available for a later explicit resume/re-enable.
                time.sleep(POLL)
                continue
            try:
                m, _, _ = ctl.metadata()
                ctl.require(m is not None, 'Existing checkpoint disappeared')
                live = ctl.identity(m)
                failure = ctl.PROJECT / 'output/eventwm_300k_combined_index/FAILURE.json'
                if failure.exists():
                    block(value, 'Integrity failure receipt: ' + str(failure))
                    ctl.stop(disable_recovery=False)
                    continue
                if live:
                    dead_pid = None
                    heartbeat(status='watching', collector_pid=live['pid'])
                elif ctl.session_exists(ctl.C['session']):
                    # Includes deep seal verification before runner identity is
                    # published, and worker cleanup after a fatal coordinator.
                    heartbeat(status='starting_or_exiting')
                else:
                    pid = m.get('runner', {}).get('pid')
                    reason = stop_reason(pid)
                    if dead_pid != pid:
                        events('collector_dead', collector_pid=pid,
                               checkpoint_state=m['state'], reason=reason)
                        dead_pid = pid
                    why = refusal(m, reason, value.get('restarts', []), time.time())
                    if why:
                        block(value, why)
                        continue
                    # Full existing source/checkpoint/ID/capacity and verified
                    # index checks; failures disable recovery, never loop.
                    ctl.check()
                    ctl.verify_index()
                    with (ctl.LOGS / 'recovery_admission.lock').open('a') as admission:
                        fcntl.flock(admission, fcntl.LOCK_EX)
                        value = state()
                        if not value.get('enabled'):
                            continue
                        ctl.reject_active()
                        value['restarts'] = [t for t in value.get('restarts', [])
                                             if time.time() - t < WINDOW] + [time.time()]
                        value.update(status='restarting', time=time.time())
                        save(value)  # Count before launching, including failures.
                        events('restart_attempt', previous_pid=pid,
                               checkpoint=str(ctl.DATA / 'pipeline.sqlite3'))
                        # Use the same guarded launcher and dataset lock. Do not
                        # call start(): automatic retries cannot re-enable a block.
                        ctl.launch('resume', enable_recovery=False)
                        events('restart_launched', previous_pid=pid)
            except Exception as error:
                block(state(), repr(error))
            time.sleep(POLL)
