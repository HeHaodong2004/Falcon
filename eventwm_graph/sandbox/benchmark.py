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


def run_episode(sb, spec, config, planner, dynamics, decisions, range_sensor, interval=16, gc_freeze=True):
    from .episode import RealEpisode, generator, default_query
    from .core import Candidate
    gen = generator(config, dynamics)
    t0 = time.perf_counter()
    ep = RealEpisode(spec, planner, dynamics, range_sensor=range_sensor, gc_freeze=gc_freeze)
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
            decision=d, steps=ep.task.steps, frames=len(ep.stream), cells=len(state.partial_graph['cells']),
            tracks=len(ep.legal.tracks), candidates=len(cands), first=first,
            env_step_ms=1000*(p1['env_s']-p0['env_s']), sensor_ms=1000*(p1['sensor_s']-p0['sensor_s']),
            range_ms=1000*(p1['range_s']-p0['range_s']), stream_ms=1000*(p1['stream_s']-p0['stream_s']),
            legal_ms=1000*(p1['legal_s']-p0['legal_s']), planner_ms=1000*(p1['planner_s']-p0['planner_s']),
            interval_steps=r['steps'], interval_ms=1000*step_s, decision_state_ms=1000*state_s, generator_ms=1000*gen_s,
            sandbox_ms=1000*eval_s, stage1_ms=prof['stage1_ms'], imagine_ms=prof['imagine_ms'], layer2_ms=prof['layer2_ms'],
            stage2_ms=prof.get('stage2_ms'), interaction_ms=prof.get('interaction_ms'), imagined_steps=prof['imagined_steps'],
            cache=dict(sb.last_cache), stage2_calls=s2.get('calls'),
            summaries=[{k: v for k, v in c.items() if k in ('label', 'steps', 'ended_by', 'progress', 'p_collision',
                                                             'p_social', 'predicted_free', 'predicted_blocked')}
                       for c in res.summaries()]))
        first = False
    out = dict(key=spec.key, config=config, planner=planner, dynamics=dynamics, range_sensor=range_sensor,
               setup_s=setup_s, steps=ep.task.steps, decisions=rows, range_stats=getattr(ep.ranges, 'stats', None),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['episodes'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='k10')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--episodes', default='dev:0')
    ap.add_argument('--decisions', type=int, default=24)
    ap.add_argument('--range', default='fast')
    ap.add_argument('--sandbox', default='{}', help='SandboxConfig overrides (JSON)')
    a = ap.parse_args()
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
        rec = run_episode(sb, spec, a.config, a.planner, a.dynamics, a.decisions, a.range)
        recs.append(rec)
        (out/(spec.key.replace('/', '_')+'.json')).write_text(json.dumps(rec, indent=1, default=float))
        print(spec.key, 'decisions', len(rec['decisions']), 'steps', rec['steps'], flush=True)
    summ = dict(summarize(recs), build_s=build_s, config=a.config, planner=a.planner, dynamics=a.dynamics,
                range=a.range, sandbox=json.loads(a.sandbox), load=os.getloadavg())
    (out/'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ('decisions', 'end_to_end_decision_ms', 'per_step_ms', 'by_frames')}, indent=1, default=float))
    print('warm', json.dumps({k: (v and {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in summ['warm'].items()}, default=float))
    sb.close()


if __name__ == '__main__':
    main()
