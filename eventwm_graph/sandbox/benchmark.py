"""World Model Sandbox benchmarks on real episodes (unchanged habitat.Env).

    python scripts/sandbox_run.py eventwm_graph.sandbox.benchmark episodes --out DIR --config k10 \
        --episodes dev:0,dev:3,unseen:2 --decisions 24

Each episode runs the legal default policy (the querygen executor-only goal query, continued every decision interval)
while, at every decision, the frozen querygen generator proposes its K candidates and the sandbox evaluates all of
them.  The executed actions never depend on the sandbox, so the episodes are identical for every sandbox setting.
Per decision the record holds the timings of every component (environment steps, legal sensor, range sensor, stream
update, decision-state construction, generator, Stage 1, imagined rollouts, Stage 2, reductions) and the context size
(frames, graph cells, tracks, candidates).
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def parse_items(text):
    out = []
    for it in text.split(','):
        suite, idx = it.split(':')
        out.append((suite, int(idx)))
    return out


def run_episode(sb, spec, config, planner, dynamics, decisions, range_sensor, interval=16, gc_freeze=True, env='real'):
    from .episode import RealEpisode, BranchEpisode, generator, default_query
    from .core import Candidate
    gen = generator(config, dynamics)
    t0 = time.perf_counter()
    ep = (BranchEpisode if env == 'branch' else RealEpisode)(spec, planner, dynamics, range_sensor=range_sensor, gc_freeze=gc_freeze)
    setup_s = time.perf_counter()-t0
    q = default_query()
    rows = []
    first = True
    for d in range(int(decisions)):
        if ep.over:
            break
        p0 = dict(ep.perf)
        t = time.perf_counter()
        r = ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, interval)
        step_s = time.perf_counter()-t
        if ep.over or r['steps'] == 0:
            break
        p1 = dict(ep.perf)
        t = time.perf_counter()
        state = ep.decision_state()
        state_s = time.perf_counter()-t
        t = time.perf_counter()
        queries, _ = gen.propose(ep.legal, ep.execution)
        gen_s = time.perf_counter()-t
        cands = [Candidate(None, planner, dynamics, execution=ep.execution, level=ep.level, query_id='continue')
                 if qq.kind == 'continue' else Candidate(qq, planner, dynamics, level=ep.level, query_id=qq.label)
                 for qq in queries]
        t = time.perf_counter()
        res = sb.evaluate(state, ep.legal, cands)
        eval_s = time.perf_counter()-t
        prof = res.profile
        s2 = (sb.l2.last_profile or {}).get('stage2', {}) if sb.l2 is not None else {}
        rows.append(dict(
            decision=d, steps=ep.steps, frames=len(ep.stream), cells=len(state.partial_graph['cells']),
            tracks=len(ep.legal.tracks), candidates=len(cands), first=first,
            env_step_ms=1000*(p1['env_s']-p0['env_s']), sensor_ms=1000*(p1['sensor_s']-p0['sensor_s']),
            range_ms=1000*(p1['range_s']-p0['range_s']), stream_ms=1000*(p1['stream_s']-p0['stream_s']),
            legal_ms=1000*(p1['legal_s']-p0['legal_s']), planner_ms=1000*(p1['planner_s']-p0['planner_s']),
            interval_steps=r['steps'], interval_ms=1000*step_s, decision_state_ms=1000*state_s, generator_ms=1000*gen_s,
            sandbox_ms=1000*eval_s, stage1_ms=prof['stage1_ms'], imagine_ms=prof['imagine_ms'], layer2_ms=prof['layer2_ms'],
            stage2_ms=prof.get('stage2_ms'), interaction_ms=prof.get('interaction_ms'), imagined_steps=prof['imagined_steps'],
            cache=dict(sb.last_cache), stage2_calls=s2.get('calls'),
            stage2_parts={k: 1000*float(v) for k, v in s2.items() if k.endswith('_seconds')},
            summaries=[{k: v for k, v in c.items() if k in ('label', 'steps', 'ended_by', 'progress', 'p_collision',
                                                             'p_social', 'predicted_free', 'predicted_blocked')}
                       for c in res.summaries()]))
        first = False
    out = dict(key=spec.key, config=config, planner=planner, dynamics=dynamics, range_sensor=range_sensor, env=env,
               setup_s=setup_s, steps=ep.steps, decisions=rows, range_stats=getattr(ep.ranges, 'stats', None),
               metrics={k: float(v) for k, v in ep.metrics().items() if isinstance(v, (int, float))})
    ep.close()
    return out


def stats(values):
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return None
    return dict(n=int(len(v)), p50=float(np.percentile(v, 50)), mean=float(v.mean()), p95=float(np.percentile(v, 95)),
                max=float(v.max()))


def summarize(records):
    rows = [r for rec in records for r in rec['decisions']]
    warm = [r for r in rows if not r['first']]
    firsts = [r for r in rows if r['first']]
    keys = ('sandbox_ms', 'stage1_ms', 'imagine_ms', 'layer2_ms', 'stage2_ms', 'interaction_ms', 'decision_state_ms',
            'generator_ms', 'interval_ms', 'range_ms', 'env_step_ms', 'sensor_ms', 'stream_ms')
    out = dict(episodes=len(records), decisions=len(rows), warm={k: stats([r[k] for r in warm]) for k in keys},
               first={k: stats([r[k] for r in firsts]) for k in ('sandbox_ms', 'stage1_ms', 'imagine_ms', 'layer2_ms')})
    e2e = [r['interval_ms']+r['decision_state_ms']+r['generator_ms']+r['sandbox_ms'] for r in warm]
    out['end_to_end_decision_ms'] = stats(e2e)
    out['per_step_ms'] = stats([r['interval_ms']/max(1, r['interval_steps']) for r in warm])
    by = {}
    for r in warm:
        b = '%d-%d' % (r['frames']//100*100, r['frames']//100*100+99)
        by.setdefault(b, []).append(r)
    out['by_frames'] = {b: dict(n=len(v), stage2_p50=stats([x['stage2_ms'] for x in v])['p50'],
                                sandbox_p50=stats([x['sandbox_ms'] for x in v])['p50'],
                                state_p50=stats([x['decision_state_ms'] for x in v])['p50']) for b, v in sorted(by.items())}
    return out


def worker(a):
    """One throughput worker: its own sandbox (shared GPU), its own episodes; per-decision wall-clock stamps.

    ``--work full``: environment steps + sensing + decision state + generator + sandbox (the RL inner loop);
    ``--work sandbox``: contexts are collected first (untimed), then the sandbox alone re-evaluates them."""
    from .core import WorldModelSandbox, SandboxConfig, Candidate
    from .episode import RealEpisode, BranchEpisode, generator, default_query, freeze_gc
    from eventwm_graph.querygen import suites
    out = Path(a.out)
    if a.server:                                   # split mode: Layer 2 in a server process (service.py)
        from .service import RemoteSandbox
        sb = RemoteSandbox(a.server, SandboxConfig(**json.loads(a.sandbox)))
    else:
        sb = WorldModelSandbox.from_release(SandboxConfig(**json.loads(a.sandbox)), map_device='cuda:0')
    items = parse_items(a.episodes)
    stamps, comp = [], dict(sandbox=0., interval=0., state=0., generator=0.)
    t_start = time.time()
    ctxs = []
    k = 0
    while time.time()-t_start < a.seconds:
        suite, idx = items[k % len(items)]; k += 1
        spec = suites.load(suite)[idx]
        gen = generator(a.config, a.dynamics)
        ep = (BranchEpisode if a.env == 'branch' else RealEpisode)(spec, a.planner, a.dynamics, range_sensor=a.range, gc_freeze=True,
                                                                 stream=sb.open_stream(spec.key) if a.server else None)
        q = default_query()
        while not ep.over and time.time()-t_start < a.seconds:
            t0 = time.perf_counter()
            r = ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16)
            t1 = time.perf_counter()
            if ep.over or r['steps'] == 0:
                break
            state = ep.decision_state()
            t2 = time.perf_counter()
            queries, _ = gen.propose(ep.legal, ep.execution)
            t3 = time.perf_counter()
            cands = [Candidate(None, a.planner, a.dynamics, execution=ep.execution, level=ep.level) if qq.kind == 'continue'
                     else Candidate(qq, a.planner, a.dynamics, level=ep.level) for qq in queries]
            if a.work == 'sandbox':
                frozen = [Candidate(c.query, c.planner, c.dynamics, execution=None if c.execution is None else c.execution.copy(),
                                    level=c.level) for c in cands]      # the episode keeps running: freeze its state
                ctxs.append((state, ep.legal.copy(), frozen))
                if len(ctxs) >= a.contexts:
                    break
                continue
            sb.evaluate(state, ep.legal, cands)
            t4 = time.perf_counter()
            comp['interval'] += t1-t0; comp['state'] += t2-t1; comp['generator'] += t3-t2; comp['sandbox'] += t4-t3
            stamps.append(time.time())
        ep.close()
        if a.work == 'sandbox' and len(ctxs) >= a.contexts:
            break
    if a.work == 'sandbox':
        freeze_gc()
        sb.evaluate(*ctxs[0])                                    # warm
        t_start = time.time()
        i = 0
        while time.time()-t_start < a.seconds:
            st, legal, cands = ctxs[i % len(ctxs)]; i += 1
            t0 = time.perf_counter()
            sb.evaluate(st, legal, cands)
            comp['sandbox'] += time.perf_counter()-t0
            stamps.append(time.time())
    server_wait = getattr(getattr(sb, 'conn', None), 'wait_s', None)      # split mode: time blocked on server calls
    (out/('worker_%02d.json' % a.id)).write_text(json.dumps(dict(id=a.id, stamps=stamps, components=comp, contexts=len(ctxs),
                                                                  server=a.server, server_wait_s=server_wait)))
    sb.close()


def gpu_free_gb(device=0):
    """Free memory of physical GPU ``device`` (a short-lived probe process; NVML is unavailable on this host)."""
    import subprocess
    import sys
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    try:
        r = subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.mem_get_info(0)[0])'], env=env,
                           capture_output=True, text=True, timeout=120)
        return int(r.stdout.strip().splitlines()[-1])/1e9
    except Exception:
        return None


def throughput(a):
    import subprocess
    import sys
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = parse_items(a.episodes)
    memory = dict(start=gpu_free_gb())
    servers, addresses = [], []
    if a.servers:
        from .service import start_server, wait_ready
        ready = []
        for j in range(a.servers):
            addresses.append('tp-%d-%d' % (os.getpid(), j))
            ready.append(out/('server_%02d.ready' % j))
            if ready[-1].exists():
                ready[-1].unlink()
            servers.append(start_server(addresses[-1], ready[-1], log=out/('server_%02d.log' % j)))
        wait_ready(ready, servers)
        memory['servers_ready'] = gpu_free_gb()
    procs = []
    for i in range(a.workers):
        eps = ','.join('%s:%d' % items[(i+j) % len(items)] for j in range(len(items)))
        cmd = [sys.executable, 'scripts/sandbox_run.py', 'eventwm_graph.sandbox.benchmark', 'worker', '--out', str(out), '--id', str(i),
               '--episodes', eps, '--config', a.config, '--planner', a.planner, '--dynamics', a.dynamics, '--env', a.env,
               '--range', a.range, '--seconds', str(a.seconds), '--work', a.work, '--contexts', str(a.contexts), '--sandbox', a.sandbox]
        if addresses:
            cmd += ['--server', addresses[i % len(addresses)]]
        procs.append(subprocess.Popen(cmd, stdout=open(out/('worker_%02d.log' % i), 'w'), stderr=subprocess.STDOUT))
    time.sleep(min(120., a.seconds/2))
    memory['running'] = gpu_free_gb()
    for p in procs:
        p.wait()
    server_stats = []
    if servers:
        from .service import RemoteSandbox
        for addr, p in zip(addresses, servers):
            try:
                c = RemoteSandbox(addr, timeout=30)
                server_stats.append(c.server_stats())
                c.shutdown_server()
            except Exception as exc:
                server_stats.append(dict(error=repr(exc)))
            try:
                p.wait(timeout=120)
            except Exception:
                p.kill()
    recs = [json.loads((out/('worker_%02d.json' % i)).read_text()) for i in range(a.workers) if (out/('worker_%02d.json' % i)).exists()]
    stamps = sorted(t for r in recs for t in r['stamps'])
    if len(stamps) < 4:
        print('too few decisions', len(stamps)); return
    # steady window: from the time every worker has produced its first decision to the first worker's last one
    lo = max(min(r['stamps']) for r in recs if r['stamps']); hi = min(max(r['stamps']) for r in recs if r['stamps'])
    n = sum(1 for t in stamps if lo <= t <= hi)
    comp = {k: sum(r['components'][k] for r in recs) for k in recs[0]['components']}
    summ = dict(workers=a.workers, servers=a.servers, work=a.work, env=a.env, config=a.config, planner=a.planner, dynamics=a.dynamics,
                decisions=len(stamps), window_s=hi-lo, decisions_per_s=n/max(hi-lo, 1e-9),
                per_worker=[len(r['stamps']) for r in recs], component_seconds=comp, load=os.getloadavg(),
                gpu_free_gb=memory, server_wait_s=sum(r.get('server_wait_s') or 0. for r in recs),
                server_stats=[{k: v for k, v in st.items() if k != 'started'} for st in server_stats])
    for st in summ['server_stats']:
        if 'busy_s' in st and 'now' in st:
            st['busy_share'] = st['busy_s']/max(1e-9, a.seconds)
    (out/'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['episodes', 'worker', 'throughput'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--id', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=300.)
    ap.add_argument('--work', default='full', choices=['full', 'sandbox'])
    ap.add_argument('--contexts', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='k10')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--episodes', default='dev:0')
    ap.add_argument('--decisions', type=int, default=24)
    ap.add_argument('--range', default='fast')
    ap.add_argument('--env', default='real', choices=['real', 'branch'])
    ap.add_argument('--sandbox', default='{}', help='SandboxConfig overrides (JSON)')
    ap.add_argument('--servers', type=int, default=0, help='throughput: split mode with this many Layer 2 server processes')
    ap.add_argument('--server', default='', help='worker: the server address (split mode)')
    a = ap.parse_args()
    if a.mode == 'worker':
        return worker(a)
    if a.mode == 'throughput':
        return throughput(a)
    from .core import WorldModelSandbox, SandboxConfig
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = SandboxConfig(**json.loads(a.sandbox))
    t = time.perf_counter()
    sb = WorldModelSandbox.from_release(cfg, map_device='cuda:0')
    build_s = time.perf_counter()-t
    recs = []
    from eventwm_graph.querygen import suites
    for suite, idx in parse_items(a.episodes):
        spec = suites.load(suite)[idx]
        rec = run_episode(sb, spec, a.config, a.planner, a.dynamics, a.decisions, a.range, env=a.env)
        recs.append(rec)
        (out/(spec.key.replace('/', '_')+'.json')).write_text(json.dumps(rec, indent=1, default=float))
        print(spec.key, 'decisions', len(rec['decisions']), 'steps', rec['steps'], flush=True)
    summ = dict(summarize(recs), build_s=build_s, config=a.config, planner=a.planner, dynamics=a.dynamics,
                range=a.range, env=a.env, sandbox=json.loads(a.sandbox), load=os.getloadavg())
    (out/'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ('decisions', 'end_to_end_decision_ms', 'per_step_ms', 'by_frames')}, indent=1, default=float))
    print('warm', json.dumps({k: (v and {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in summ['warm'].items()}, default=float))
    sb.close()


if __name__ == '__main__':
    main()
