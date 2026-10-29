"""Process orchestration for Layer 3: Sandbox servers (GPU) + actor workers (CPU) + optional learner (GPU).

    python scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/X --servers 3 --server-devices 0,0,1 \
        --workers 24 --actor-args "--configs orca-native-k10:1 --suite train --selector policy" \
        [--learner-args "--device cuda:1 ..."] [--seconds 3600] [--sequential]

Actors are restarted when they die (a CUDA fault in the environment's range sensor ends that process only); in
``--sequential`` mode (evaluation / Oracle replays over a fixed job list) a clean exit means the worker's share is done.
Throughput (episodes, decisions, decisions/s over the last window) is appended to ``fleet.jsonl`` every minute.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def spawn(module, args, log, env=None, nice=0):
    cmd = [PY, str(ROOT/'scripts/l3_run.py'), module]+list(args)
    if nice:
        cmd = ['nice', '-n', str(nice)]+cmd
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=open(log, 'a'), stderr=subprocess.STDOUT, env=env)


def count_decisions(run, since):
    n_ep = n_dec = 0
    for p in Path(run).glob('actor_*.log'):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('t', 0) >= since and 'decisions' in r:
                n_ep += 1; n_dec += r['decisions']
    return n_ep, n_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--servers', type=int, default=3)
    ap.add_argument('--server-devices', default='0,0,1')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--actor-args', default='')
    ap.add_argument('--learner-args', default=None)
    ap.add_argument('--seconds', type=float, default=1e12)
    ap.add_argument('--sequential', action='store_true')
    ap.add_argument('--worker-gpus', default='0,1', help='GPUs for the actors\' range sensors (round robin)')
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--eval-workers', type=int, default=0, help='extra actors running the greedy policy on dev (eval_spool)')
    ap.add_argument('--labelers', type=int, default=0, help='on-policy counterfactual labellers (cfonline.py)')
    ap.add_argument('--labeler-args', default='')
    ap.add_argument('--eval-args', default='--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool')
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    (run/'fleet_args.json').write_text(json.dumps(vars(a)))
    tag = a.tag or run.name
    procs = {}
    stopping = [False]

    def shutdown(*_):
        stopping[0] = True
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    from eventwm_graph.sandbox.service import start_server, wait_ready
    devs = [int(x) for x in a.server_devices.split(',')]
    servers, ready = [], []
    base_env = dict(os.environ)
    base_env['CUDA_VISIBLE_DEVICES'] = '0,1'
    for i in range(a.servers):
        f = run/('server%d.ready' % i)
        if f.exists():
            f.unlink()
        addr = 'l3.%s.s%d' % (tag, i)
        p = start_server(addr, f, log=run/('server%d.log' % i), map_device='cuda:%d' % devs[i % len(devs)], env=base_env)
        servers.append((addr, p)); ready.append(f)
    if servers:
        wait_ready(ready, [p for _, p in servers], timeout=900)
        print('servers ready', flush=True)
    learner = None
    learner_restarts = [0]
    if a.learner_args is not None:
        learner = spawn('eventwm_graph.l3.learner', ['--run', str(run)]+shlex.split(a.learner_args), run/'learner.log',
                        env=base_env)
    wg = [x for x in a.worker_gpus.split(',')]
    done = set()

    def start_actor(i, restarts=0):
        env = dict(base_env)
        g = wg[i % len(wg)]
        env['CUDA_VISIBLE_DEVICES'] = g+',' + ','.join(x for x in ('0', '1') if x != g)
        env['L3_NWORKERS'] = str(a.workers)
        args = ['--run', str(run), '--worker', str(i), '--seed', str(restarts)]
        if servers:
            args += ['--server', servers[i % len(servers)][0]]
        if i >= a.workers:            # evaluation actor (greedy policy on dev; its own spool and configuration)
            args += shlex.split(a.actor_args)+shlex.split(a.eval_args)
        else:
            args += shlex.split(a.actor_args)
        procs[i] = (spawn('eventwm_graph.l3.actor', args, run/('actor_%02d.out' % i), env=env, nice=a.nice), restarts)
    for i in range(a.workers+a.eval_workers):
        start_actor(i)
        time.sleep(0.5)
    labelers = {}

    def start_labeler(j, restarts=0):
        args = ['--run', str(run), '--worker', str(j), '--workers', str(a.labelers)]+shlex.split(a.labeler_args)
        labelers[j] = (spawn('eventwm_graph.l3.cfonline', args, run/('labeler_%02d.out' % j), env=base_env), restarts)
    for j in range(a.labelers):
        start_labeler(j)
    t0 = time.time()
    last = time.time()
    flog = open(run/'fleet.jsonl', 'a')
    while not stopping[0] and time.time()-t0 < a.seconds:
        time.sleep(5)
        for i, (p, restarts) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if a.sequential and rc == 0:
                done.add(i); del procs[i]
                continue
            if restarts < 50:
                start_actor(i, restarts+1)
            else:
                del procs[i]
        for j, (p, restarts) in list(labelers.items()):
            if p.poll() is not None and restarts < 50:
                start_labeler(j, restarts+1)
        if a.sequential and not procs:
            break
        for addr, p in servers:
            if p.poll() is not None:
                print('server exited', addr, p.returncode, flush=True)
                stopping[0] = True
        if learner is not None and learner.poll() is not None:
            print('learner exited', learner.returncode, flush=True)
            state = run/'learner_state.pt'
            if learner_restarts[0] < 5:
                # resume from the last full state if there is one (the spool is re-ingested; no pre-training),
                # otherwise start the learner again from its original arguments
                largs = [x for x in shlex.split(a.learner_args)]
                if state.exists():
                    if '--pretrain' in largs:
                        i = largs.index('--pretrain'); largs[i+1] = '0'
                    if '--init' in largs:
                        i = largs.index('--init'); largs[i+1] = str(state)
                    else:
                        largs += ['--init', str(state)]
                learner = spawn('eventwm_graph.l3.learner', ['--run', str(run)]+largs, run/'learner.log', env=base_env)
                learner_restarts[0] += 1
                print('learner restarted (%d)' % learner_restarts[0], flush=True)
            else:
                stopping[0] = True
        if time.time()-last > 60:
            ne, nd = count_decisions(run, last)
            rec = dict(t=round(time.time()-t0), window_s=round(time.time()-last), episodes=ne, decisions=nd,
                       decisions_per_s=round(nd/max(time.time()-last, 1), 3), actors=len(procs),
                       restarts=sum(r for _, r in procs.values()))
            flog.write(json.dumps(rec)+'\n'); flog.flush()
            last = time.time()
    # shutdown
    for j, (p, _) in labelers.items():
        p.terminate()
    for i, (p, _) in procs.items():
        p.terminate()
    if learner is not None:
        learner.terminate()
    time.sleep(3)
    for i, (p, _) in procs.items():
        if p.poll() is None:
            p.kill()
    if learner is not None:
        try:
            learner.wait(60)
        except Exception:
            learner.kill()
    from eventwm_graph.sandbox.service import RemoteSandbox
    from eventwm_graph.sandbox.core import SandboxConfig
    for addr, p in servers:
        try:
            RemoteSandbox(addr, SandboxConfig()).shutdown_server()
        except Exception:
            pass
        try:
            p.wait(20)
        except Exception:
            p.kill()
    print('fleet done', flush=True)


if __name__ == '__main__':
    main()
