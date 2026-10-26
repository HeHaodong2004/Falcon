"""Run a command with the session's OWN background jobs paused (SIGSTOP), always
resuming them (SIGCONT) afterwards. Only processes whose command line matches
one of the given patterns AND belongs to this user are touched; their PIDs,
start times and command lines are recorded in a receipt.

Usage: python -m eventwm_graph.stage2_claude.quiet --receipt R.json --pattern P [--pattern P2] -- cmd args...
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def owned(patterns):
    me = os.getuid(); mine = os.getpid(); parent = os.getppid(); found = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) in (mine, parent):
            continue
        try:
            if os.stat('/proc/%s' % pid).st_uid != me:
                continue
            cmd = open('/proc/%s/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            start = open('/proc/%s/stat' % pid).read().split()[21]
        except OSError:
            continue
        if 'quiet' in cmd or not any(p in cmd for p in patterns):
            continue
        found.append(dict(pid=int(pid), start=start, cmd=cmd[:300]))
    return found


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    p = argparse.ArgumentParser(); p.add_argument('--receipt', required=True); p.add_argument('--pattern', action='append', required=True)
    a = p.parse_args(argv[:split]); cmd = argv[split+1:]
    jobs = owned(a.pattern)
    receipt = dict(paused=jobs, paused_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), command=cmd)
    try:
        for j in jobs:
            os.kill(j['pid'], signal.SIGSTOP)
        time.sleep(2)
        receipt['load_after_pause'] = os.getloadavg()
        receipt['returncode'] = subprocess.call(cmd)
    finally:
        for j in jobs:
            try:
                if open('/proc/%d/stat' % j['pid']).read().split()[21] == j['start']:
                    os.kill(j['pid'], signal.SIGCONT)
            except OSError:
                pass
        receipt['resumed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with open(a.receipt, 'w') as f:
            json.dump(receipt, f, indent=1)
    sys.exit(receipt.get('returncode', 1))


if __name__ == '__main__':
    main()
