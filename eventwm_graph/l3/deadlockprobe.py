"""Deadlock probe (ANALYSIS ONLY, privileged): do short multi-step escapes exist in the query pool?

For timeout episodes of an evaluation spool (the floor by default), the episode is replayed in querygen's exact branch
simulator to stalled decision states (fractions ``--at`` of its decisions); from each, every candidate a1 runs for its
interval, then every candidate a2 of the resulting state for its interval, then the default policy to the end (cap
steps).  Reports, per state, how many single candidates (a1 + default) and how many pairs (a1, a2 + default) reach the
goal -- whether one-step improvement can escape (it could not in the rollout reference's timeouts) and whether two
steps can.

    python scripts/l3_run.py eventwm_graph.l3.deadlockprobe --dir output/eval/floor_dev24/spool --configs orca-native-k10 \
        --out output/deadlock_probe [--limit 6 --at 0.5,0.8 --cap 300 --shard i --shards n]
"""
import argparse
import json
import os
import time
from pathlib import Path


def probe_state(rt, gen, st, goal, H, cap, scan=False):
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from eventwm_graph.querygen.oracle import horizon_for
    from .cfonline import default_continue

    def cands(state):
        qs, _ = gen.propose(state.legal, state.execution)
        if scan:
            from .env import scan_queries
            qs = list(qs)+scan_queries(state.legal, gen.dyn, [q.label for q in qs])
        return qs

    def step(state, q):
        cont = q.kind == 'continue'
        h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-state.index)
        return rt.run(state, None if cont else q, horizon=h, cont=cont, geodesic=False)

    end = min(MAX_EPISODE_STEPS, st.index+cap)
    singles, pairs, found = 0, 0, []
    q1s = cands(st)
    for q1 in q1s:
        c1 = step(st, q1)
        if c1.terminal:
            singles += int(c1.outcome == 'success')
            continue
        f1 = default_continue(rt, c1, goal, end)
        singles += int(f1.outcome == 'success')
        for q2 in cands(c1):
            c2 = step(c1, q2)
            f2 = c2 if c2.terminal else default_continue(rt, c2, goal, end)
            if f2.outcome == 'success':
                pairs += 1
                if len(found) < 8:
                    found.append([q1.label, q2.label])
    return dict(step=int(st.index), n1=len(q1s), singles=singles, pairs=pairs, examples=found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--limit', type=int, default=6)
    ap.add_argument('--at', default='0.5,0.8')
    ap.add_argument('--cap', type=int, default=300)
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.episode import generator, default_query
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = set(a.configs.split(','))
    files = []
    for p in sorted(Path(a.dir).glob('*.ep')):
        d = read_episode(p)
        if d['config'] in keep and d['metrics']['outcome'] == 'timeout':
            files.append(p)
    files = files[:a.limit]
    fracs = [float(x) for x in a.at.split(',')]
    H = type('H', (), {'horizon': 16})()
    for i, p in enumerate(files):
        if i % a.shards != a.shard or (out/(p.name+'.json')).exists():
            continue
        d = read_episode(p)
        cfg = ConfigSpec.parse(d['config'])
        gen = generator(cfg.gen, cfg.dynamics)
        rt = EpisodeRuntime(EpisodeSpec(**d['spec']), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
        rt.sim.check_static = False
        rt.prefix_cache = 48
        sensorcache.attach(rt)
        goal = default_query()
        decs = d['metrics']['decision_list']
        targets = sorted({int(f*(len(decs)-1)) for f in fracs})
        rec = dict(episode=p.name, config=d['config'], decisions=len(decs), states=[])
        t0 = time.time()
        st = rt.root
        try:
            for j, dec in enumerate(decs):
                if st.terminal or j > max(targets):
                    break
                if j in targets:
                    r = probe_state(rt, gen, st, goal, H, a.cap, scan=a.scan)
                    r['decision'] = j
                    rec['states'].append(r)
                    print(json.dumps(dict(episode=p.name, **r)), flush=True)
                st = rt.run(st, None if dec['cont'] else _query(dec), horizon=int(dec['horizon']),
                            cont=bool(dec['cont']), geodesic=False)
        finally:
            try:
                sensorcache.save(rt)
            except Exception:
                pass
            rt.close()
        rec['seconds'] = round(time.time()-t0)
        (out/(p.name+'.json')).write_text(json.dumps(rec))


if __name__ == '__main__':
    main()
