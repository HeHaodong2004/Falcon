"""Counterfactual candidate labels along v14 Oracle solutions (privileged; training supervision / diagnostics only).

At every decision state of a replay-verified Oracle solution, EVERY candidate the frozen generator offers is executed
for one decision interval in the exact branch simulator and then handed to the default (no-selection) policy -- keep
the running query, else the goal query -- until the official episode end.  The outcome of "candidate, then default"
is exact (deterministic simulator, recorded human tape) and is a lower bound on the candidate's optimal value: the
Oracle's own continuation (its later decisions) is recorded separately.  Labels never enter the policy input.

Per candidate: interval outcome / steps / path / social steps / geodesic progress, and the final outcome, remaining
steps, path length and social steps of the default continuation.

  python scripts/l3_run.py eventwm_graph.l3.cflabel run --out DIR [--shard i --shards n] RESULT_JSON...
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

ORACLE_RUNS = '/home/haodong/workspace/projects/eventwm-falcon-querygen-claude-20261026/output/runs/official'
ORACLE_DIRS = {'v14/unseen_c1k10': 'k10', 'v14/unseen2_c1k10': 'k10', 'v14/unseen3_c1k10': 'k10',
               'v14/unseen4_c1k10': 'k10', 'test_v14/unseen6_c1k10': 'k10', 'test_v14/unseen6_c1k6': 'k6',
               'test_v14/unseen6_c2k10': 'k10'}


def oracle_results(train_keys=None):
    """(path, generator config) of every PASS Oracle solution whose episode is in ``train_keys`` (L3 training)."""
    out = []
    for d, cfg in sorted(ORACLE_DIRS.items()):
        for p in sorted(Path(ORACLE_RUNS, d).glob('*.json')):
            if p.name == 'summary.json':
                continue
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if r.get('verdict') != 'PASS':
                continue
            if train_keys is not None and r['key'] not in train_keys:
                continue
            out.append((str(p), cfg))
    return out


def _query(x):
    from eventwm_graph.querygen.executor import Query
    return Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                 speed=x['speed'], label=x['label'], meta=x['meta'] or {})


def check_generator(r, cfg):
    """The run's generator configuration must equal the frozen sandbox configuration ``cfg``."""
    from dataclasses import asdict
    from eventwm_graph.sandbox.episode import CONFIGS
    from eventwm_graph.querygen.generator import GeneratorConfig
    kw = dict(CONFIGS[cfg])
    for k in ('tc_variants', 'th_variants', 'vc_variants', 'xc_variants', 'kturns'):
        if k in kw:
            kw[k] = tuple(kw[k])
    mine = json.loads(json.dumps(asdict(GeneratorConfig(**kw))))
    theirs = {k: v for k, v in r['generator'].items() if k in mine}
    diff = {k: (mine[k], theirs.get(k)) for k in mine if mine[k] != theirs.get(k) and k not in ('short_human_m', 'short_steps', 'maneuver_thru')}
    if diff:
        raise ValueError('generator configuration differs from %s: %s' % (cfg, diff))


def default_continue(rt, st, goal):
    while not st.terminal:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else goal, horizon=16, cont=cont, geodesic=False)
    return st


def _delta(a, b):
    return dict(steps=int(b.index-a.index), path=float(b.path_length-a.path_length), social=int(b.psc_steps-a.psc_steps))


def label_episode(path, cfg, out_dir):
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.sandbox.episode import generator, default_query
    r = json.loads(Path(path).read_text())
    check_generator(r, cfg)
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[r['dynamics']]
    t0 = time.perf_counter()
    rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False)
    rt.sim.check_static = False
    rt.prefix_cache = 48
    gen = generator(cfg, r['dynamics'])
    goal = default_query()
    st = rt.root
    st.geodesic = float(rt.start_geodesic)
    states = []
    H = type('H', (), {'horizon': 16})()
    for i, d in enumerate(r['decisions']):
        if st.terminal:
            break
        queries, diag = gen.propose(st.legal, st.execution)
        labels = [q.label for q in queries]
        chosen = 'continue' if d['cont'] else d['label']
        oracle = labels.index(chosen) if chosen in labels else -1
        cands = []
        for k, q in enumerate(queries):
            tq = time.perf_counter()
            cont = q.kind == 'continue'
            h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
            child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=True)
            g1 = float(child.geodesic) if child.outcome is None else (0. if child.outcome == 'success' else float('nan'))
            interval = dict(outcome=child.outcome, progress=float(st.geodesic)-g1 if g1 == g1 else None, **_delta(st, child))
            fin = default_continue(rt, child, goal)
            cands.append(dict(label=q.label, signature=str(q.signature()), interval=interval, outcome=fin.outcome,
                              **_delta(st, fin), seconds=round(time.perf_counter()-tq, 2)))
        states.append(dict(decision=i, step=int(st.index), geodesic=float(st.geodesic), labels=labels, oracle=oracle,
                           oracle_label=chosen, candidates=cands, path_at=float(st.path_length), social_at=int(st.psc_steps)))
        nxt = rt.run(st, None if d['cont'] else _query(d), horizon=int(d['horizon']), cont=bool(d['cont']), geodesic=True)
        st = nxt
    # the Oracle's own continuation from every state (its solution's remaining cost)
    final = st
    for s in states:
        s['oracle_rest'] = dict(outcome=final.outcome, steps=int(final.index-s['step']),
                                path=float(final.path_length-s['path_at']), social=int(final.psc_steps-s['social_at']))
    rt.close()
    rec = dict(key=spec.key, spec=spec.asdict(), planner=r['planner'], dynamics=r['dynamics'], config=cfg,
               source=str(path), start_geodesic=float(r['start_geodesic']), oracle_outcome=final.outcome,
               oracle_steps=int(final.index), oracle_replay=r.get('replay', {}).get('metrics'), states=states,
               seconds=round(time.perf_counter()-t0, 1))
    name = '%s__%s_%s_%s.json' % (spec.key.replace('/', '_'), r['planner'], r['dynamics'], cfg)
    (Path(out_dir)/name).write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'list'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    from . import suites
    keys = {s.key for s in suites.load('train_querygen')}
    items = oracle_results(keys)
    if a.mode == 'list':
        print(len(items))
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = [it for it in items if zlib.crc32(it[0].encode()) % a.shards == a.shard]
    for path, cfg in items:
        r = json.loads(Path(path).read_text())
        name = '%s__%s_%s_%s.json' % (r['key'].replace('/', '_'), r['planner'], r['dynamics'], cfg)
        if (out/name).exists():
            continue
        try:
            rec = label_episode(path, cfg, out)
            print(json.dumps(dict(key=rec['key'], cfg=cfg, planner=rec['planner'], states=len(rec['states']),
                                  seconds=rec['seconds'])), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(json.dumps(dict(key=r['key'], error=repr(exc))), flush=True)


if __name__ == '__main__':
    main()
