"""Reference experiment (ANALYSIS ONLY, privileged): the ROLLOUT POLICY over the default.

At every decision, every candidate is executed in querygen's exact branch simulator (recorded human tape) for its
interval and then continued by the default policy (keep the running query, else the goal query) to the episode end
or ``cap`` steps; the candidate with the best exact outcome is taken (success, fewer steps first; then still running
at the cap, more legal goal progress first; then timeout; collisions / STOP failures last).  This is one step of
policy iteration from the floor with a PERFECT evaluator: its success rate says how far exact one-step evaluation of
"candidate, then default" can carry Layer 3 (vs. deeper sequential planning).

    python scripts/l3_run.py eventwm_graph.l3.rolloutpolicy --out output/rollout_policy --configs orca-native-k10 \
        --suite dev --limit 24 [--cap 320] [--shard i --shards n]
"""
import argparse
import json
import math
import os
import time
from pathlib import Path


def outcome_key(fin, st0, goal_xz):
    o = fin.outcome
    if o == 'success':
        return (3, -int(fin.index-st0.index))
    if o is None:
        p = fin.legal.pose
        return (2, -math.hypot(p[0]-goal_xz[0], p[2]-goal_xz[1]))
    if o == 'timeout':
        return (1, 0)
    return (0, int(fin.index-st0.index))                 # collision / STOP failure: later is (slightly) better


def run_episode(spec, cfg, cap, scan=False):
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.sandbox.episode import generator, default_query
    from .cfonline import default_continue
    from . import sensorcache
    t0 = time.perf_counter()
    gen = generator(cfg.gen, cfg.dynamics)           # process-wide executor options before the runtime
    rt = EpisodeRuntime(EpisodeSpec(**spec), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
    rt.sim.check_static = False
    rt.prefix_cache = 48
    sensorcache.attach(rt)
    goal = default_query()
    H = type('H', (), {'horizon': 16})()
    st = rt.root
    gx, gz = float(st.legal.goal[0]), float(st.legal.goal[2])
    log = []
    try:
        while not st.terminal and st.index < MAX_EPISODE_STEPS:
            queries, _ = gen.propose(st.legal, st.execution)
            if scan:
                from .env import scan_queries
                queries = list(queries)+scan_queries(st.legal, gen.dyn, [q.label for q in queries])
            best = None
            scores = []
            for q in queries:
                cont = q.kind == 'continue'
                h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
                child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
                fin = default_continue(rt, child, goal, min(MAX_EPISODE_STEPS, st.index+cap))
                key = outcome_key(fin, st, (gx, gz))
                scores.append((q.label, key))
                if best is None or key > best[0]:
                    best = (key, q, child, cont)
            default = 'continue' if 'continue' in [s[0] for s in scores] else 'goal'
            log.append(dict(step=int(st.index), chosen=best[1].label, default=default,
                            n_success=sum(k[0] == 3 for _, k in scores), n=len(scores)))
            st = best[2]                                   # the chosen candidate's interval (already simulated)
        out = dict(key=spec['key'] if 'key' in spec else None, outcome=st.outcome or 'running', steps=int(st.index),
                   decisions=len(log), deviations=sum(d['chosen'] != d['default'] for d in log),
                   seconds=round(time.perf_counter()-t0, 1), log=log)
        return out
    finally:
        try:
            sensorcache.save(rt)
        except Exception:
            pass
        rt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--cap', type=int, default=320)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--scan', action='store_true', help='also evaluate the Layer 3 scan queries')
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from . import suites
    from .env import ConfigSpec
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = suites.load(a.suite)[:a.limit]
    jobs = [(s, c) for s in specs for c in a.configs.split(',')]
    for i, (s, c) in enumerate(jobs):
        if i % a.shards != a.shard:
            continue
        name = '%s__%s.json' % (s.key.replace('/', '_'), c)
        if (out/name).exists():
            continue
        try:
            r = run_episode(dict(s.asdict()), ConfigSpec.parse(c), a.cap, scan=a.scan)
        except Exception as exc:
            import traceback
            r = dict(error=repr(exc), tb=traceback.format_exc()[-1500:])
        r.update(config=c, key=s.key)
        (out/name).write_text(json.dumps(r))
        print(json.dumps({k: v for k, v in r.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
