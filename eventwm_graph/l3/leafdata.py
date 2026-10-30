"""TRAINING / ANALYSIS DATA (privileged targets): exact candidate LEAF states for a learned long-horizon value.

Every labelled decision of a run's counterfactual labels (``<run>/cf_online``: the exact outcome of each candidate's
interval followed by the default policy to the episode end) is reached again in querygen's exact branch simulator by
replaying the episode's recorded decisions (actions checked).  At the labelled decision each labelled candidate is
executed for its interval exactly as the labeller did, and the decision state and the LEAF state it reaches are
described by ``leafvalue.legal_features`` (what a selector can compute for an imagined leaf) and
``leafvalue.privileged_features`` (true people, analysis only); a second leaf 32 steps from the decision (the
interval + 16 steps of the default policy: the branch the label's continuation took) is described the same way
(Xl32 / Pl32; t32 = outcome code if the branch ended by then).  Targets: the labels.

Output ``<out>/<episode>.npz``: Xs / Xl (legal features at the decision / at the leaf), Ps / Pl (privileged), y
(outcome code: 0 collision, 1 STOP failure, 2 timeout, 3 alive at the cap, 4 success), steps (decision -> end), the
candidate label, the decision index and whether the candidate is the default.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.leafdata --run output/runs/R10 --out output/leafdata/R10 \
        [--shard i --shards n]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

OUTCOME = {'human_collision': 0, 'stop_fail': 1, 'timeout': 2, None: 3, 'success': 4}


def episode_rows(run, rec):
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.sandbox.episode import generator
    from .actor import read_episode
    from .env import ConfigSpec, default_index
    from .cfonline import _query
    from .compiler import LegalField
    from .leafvalue import legal_features, privileged_features
    from . import sensorcache
    d = read_episode(Path(run, 'spool', rec['episode']))
    m, rows = d['metrics'], d['rows']
    targets = {int(k): v for k, v in rec['states'].items() if 'candidates' in v}
    cfg = ConfigSpec.parse(d['config'])
    dyn = FAMILIES[cfg.dynamics]
    gen = generator(cfg.gen, cfg.dynamics)
    rt = EpisodeRuntime(EpisodeSpec(**d['spec']), cfg.planner, dyn, strict=False)
    rt.sim.check_static = False
    rt.prefix_cache = 48
    sensorcache.attach(rt)
    H = type('H', (), {'horizon': 16})()
    tape = np.asarray(rt.tape.positions, np.float64)
    acts = m.get('actions')
    out = dict(Xs=[], Xl=[], Ps=[], Pl=[], Xl32=[], Pl32=[], t32=[], y=[], steps=[], label=[], decision=[],
               is_default=[], interval=[])
    from .cfonline import default_continue
    from eventwm_graph.sandbox.episode import default_query
    goal = default_query()
    st = rt.root
    try:
        for i, dec in enumerate(m['decision_list']):
            if not targets or i > max(targets) or st.terminal:
                break
            if i in targets:
                lab = targets[i]
                queries, _ = gen.propose(st.legal, st.execution)
                if d.get('scan'):
                    from .env import scan_queries
                    queries = list(queries)+scan_queries(st.legal, gen.dyn, [q.label for q in queries])
                labels = [q.label for q in queries]
                if labels != lab['labels']:
                    break
                dflt = labels[default_index(labels)]
                field = LegalField(st.legal)
                L = st.legal
                xs = legal_features(L.pose, L.poses[-8:], L.times[-8:], L.time, L.map, field, L.goal, L.tracks,
                                    st.execution, st.dyn_state, dyn, cfg.planner, st.index, 0, None)
                ps = privileged_features(L.pose, L.poses[-8:], L.times[-8:], tape, st.index)
                for q, c in zip(queries, lab['candidates']):
                    if c.get('skipped') or 'outcome' not in c:
                        continue
                    cont = q.kind == 'continue'
                    h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
                    child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
                    C = child.legal
                    xl = legal_features(C.pose, C.poses[-8:], C.times[-8:], C.time, L.map, field, L.goal, C.tracks,
                                        child.execution, child.dyn_state, dyn, cfg.planner, child.index,
                                        child.index-st.index, child.outcome)
                    pl = privileged_features(C.pose, C.poses[-8:], C.times[-8:], tape, child.index)
                    # second leaf: 16 more steps of the default policy (32 steps from the decision), the same branch
                    # the label's default continuation took
                    l32 = child if child.terminal else default_continue(rt, child, goal, min(MAX_EPISODE_STEPS, st.index+32))
                    C2 = l32.legal
                    xl32 = legal_features(C2.pose, C2.poses[-8:], C2.times[-8:], C2.time, L.map, field, L.goal, C2.tracks,
                                          l32.execution, l32.dyn_state, dyn, cfg.planner, l32.index,
                                          l32.index-st.index, l32.outcome)
                    pl32 = privileged_features(C2.pose, C2.poses[-8:], C2.times[-8:], tape, l32.index)
                    out['Xl32'].append(xl32); out['Pl32'].append(pl32); out['t32'].append(OUTCOME.get(l32.outcome, 3))
                    out['Xs'].append(xs); out['Xl'].append(xl); out['Ps'].append(ps); out['Pl'].append(pl)
                    out['y'].append(OUTCOME.get(c['outcome'], 3)); out['steps'].append(int(c['steps']))
                    out['label'].append(q.label); out['decision'].append(i); out['is_default'].append(q.label == dflt)
                    out['interval'].append(OUTCOME.get(c.get('interval'), 3))
            st = rt.run(st, None if dec['cont'] else _query(dec), horizon=int(dec['horizon']), cont=bool(dec['cont']),
                        geodesic=False)
            if acts is not None:
                mine = [x if isinstance(x, str) else [float(v) for v in x] for x in st.actions]
                if mine != acts[:len(mine)]:
                    break
    finally:
        try:
            sensorcache.save(rt)
        except Exception:
            pass
        rt.close()
    return out, d['config']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=100000)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.run, 'cf_online').glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    for f in files:
        name = f.name[:-len('.ep.json')] if f.name.endswith('.ep.json') else f.stem
        target = out/(name+'.npz')
        if target.exists():
            continue
        t0 = time.perf_counter()
        try:
            rec = json.loads(f.read_text())
            rows, config = episode_rows(a.run, rec)
        except Exception as exc:
            print(json.dumps(dict(episode=f.name, error=repr(exc))), flush=True)
            continue
        np.savez_compressed(target, Xs=np.asarray(rows['Xs'], np.float32).reshape(-1, 0 if not rows['Xs'] else len(rows['Xs'][0])),
                            Xl=np.asarray(rows['Xl'], np.float32), Ps=np.asarray(rows['Ps'], np.float32),
                            Pl=np.asarray(rows['Pl'], np.float32), Xl32=np.asarray(rows['Xl32'], np.float32),
                            Pl32=np.asarray(rows['Pl32'], np.float32), t32=np.asarray(rows['t32'], np.int8),
                            y=np.asarray(rows['y'], np.int8),
                            steps=np.asarray(rows['steps'], np.int16), label=np.asarray(rows['label']),
                            decision=np.asarray(rows['decision'], np.int16),
                            is_default=np.asarray(rows['is_default'], bool), interval=np.asarray(rows['interval'], np.int8),
                            config=np.asarray(config), episode=np.asarray(rec['episode']))
        print(json.dumps(dict(episode=f.name, rows=len(rows['y']), seconds=round(time.perf_counter()-t0, 1))), flush=True)


if __name__ == '__main__':
    main()
