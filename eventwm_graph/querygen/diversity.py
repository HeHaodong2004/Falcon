"""Planner/dynamics diversity: do executors prefer different queries?

At matched decision states (reconstructed by deterministically re-running a
reference decision sequence), the *same* legal candidate set is executed
natively (branch simulator == Env.step) under every planner x dynamics
configuration, from rest.  Every branch is compared at an equal horizon H:
the query runs until done (or H) and the legal ``goal`` query continues under
the same configuration until H.  Privileged value (metres):
    V = -geodesic_H                       (no success by H)
    V = speed_ref * (H - t_success)       (success before H)
    V = FAIL_VALUE                        (collision / overlap / static / stop_fail)
Aggregates (``summarize``): feasibility differences, top-choice disagreement
and the regret V_c(q*_c) - V_c(q*_c') of executing another configuration's
preferred query.  A rank flip alone is not taken as evidence of diversity;
regret measures its consequence.

  python scripts/querygen_run.py eventwm_graph.querygen.diversity run RESULT_JSON --out DIR
  python scripts/querygen_run.py eventwm_graph.querygen.diversity summarize --out DIR
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np

FAIL_VALUE = -1000.
PLANNERS = ('orca', 'sfm', 'dwa')
DYNAMICS = ('native', 'slowturn', 'inertia', 'slowturn_inertia')


def reconstruct_states(rt, decisions, every=2, max_states=6):
    """Re-run the reference decisions; return decision states (non-terminal)."""
    from .executor import Query
    st = rt.root
    states = [st]
    for k, d in enumerate(decisions):
        q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'],
                  speed=d['speed'], label=d['label'], meta=d.get('meta', {}))
        st = rt.run(st, None if d['cont'] else q, horizon=d['end']-d['start'], cont=d['cont'])
        if st.terminal:
            break
        if (k+1) % every == 0:
            states.append(st)
    return states[:max_states]


def evaluate_state(rt, state, queries, horizon=48, speed_ref=0.17):
    from .dynamics import FAMILIES
    from .executor import Query
    goal = Query('goal', label='goal')
    out = {}
    for pl, dn in itertools.product(PLANNERS, DYNAMICS):
        dyn = FAMILIES[dn]
        row = []
        for q in queries:
            if q.kind == 'continue':
                row.append(None)
                continue
            base = state.copy()
            base.dyn_state = 0
            base.execution = None
            t = time.perf_counter()
            ch = rt.run(base, q, horizon=horizon, planner_name=pl, dyn=dyn)
            q_steps = ch.index-state.index
            while not ch.terminal and ch.index-state.index < horizon:
                rem = horizon-(ch.index-state.index)
                cont = ch.execution is not None and ch.execution.status == 'running' and ch.execution.query.kind == 'goal'
                ch = rt.run(ch, None if cont else goal, horizon=rem, cont=cont, planner_name=pl, dyn=dyn)
            dt = time.perf_counter()-t
            steps = ch.index-state.index
            if ch.outcome == 'success':
                v = speed_ref*(horizon-steps)
            elif ch.outcome is None:
                v = -float(ch.geodesic)
            else:
                v = FAIL_VALUE
            row.append(dict(outcome=ch.outcome, steps=steps, query_steps=q_steps, geodesic=float(ch.geodesic),
                            value=v, min_rect=float(ch.min_rect_dist), seconds=dt))
        out[f'{pl}/{dn}'] = row
    return out


def run(result_json, out_dir, horizon=48):
    from .native import EpisodeSpec
    from .episode import EpisodeRuntime
    from .generator import QueryGenerator
    from .dynamics import NATIVE
    r = json.loads(Path(result_json).read_text())
    spec = EpisodeSpec(**r['spec'])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rt = EpisodeRuntime(spec, r['planner'], NATIVE, strict=True, velocity_action=True)
    gen = QueryGenerator(NATIVE)
    states = reconstruct_states(rt, r['decisions'])
    rows = []
    for k, st in enumerate(states):
        queries, diag = gen.propose(st.legal, None)
        res = evaluate_state(rt, st, queries, horizon)
        rows.append(dict(state_index=st.index, geodesic=float(st.geodesic),
                         queries=[q.describe() for q in queries], results=res))
    rt.close()
    rec = dict(key=spec.key, reference=str(result_json), horizon=horizon, states=rows)
    (out_dir/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def summarize(out_dir):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    configs = [f'{p}/{d}' for p, d in itertools.product(PLANNERS, DYNAMICS)]
    n_states = 0
    feas = {c: [] for c in configs}
    top = {c: [] for c in configs}
    vals = []
    for rec in recs:
        for st in rec['states']:
            V = {}
            for c in configs:
                row = st['results'][c]
                V[c] = np.asarray([FAIL_VALUE if x is None else x['value'] for x in row])
                feas[c].append(float(np.mean([x is not None and x['value'] > FAIL_VALUE for x in row])))
            if all(np.max(V[c]) <= FAIL_VALUE for c in configs):
                continue
            n_states += 1
            vals.append(V)
            for c in configs:
                top[c].append(int(np.argmax(V[c])))
    regret = {}
    disagree = {}
    for a in configs:
        for b in configs:
            if a == b:
                continue
            rg = []; dg = []
            for V in vals:
                qa, qb = int(np.argmax(V[a])), int(np.argmax(V[b]))
                if V[a][qa] <= FAIL_VALUE:
                    continue
                dg.append(qa != qb)
                rg.append(min(V[a][qa]-V[a][qb], -FAIL_VALUE))   # metres (or FAIL)
            regret[f'{a}<-{b}'] = dict(mean=float(np.mean(rg)) if rg else None,
                                       p_fail=float(np.mean([x >= -FAIL_VALUE-1e-9 for x in rg])) if rg else None,
                                       n=len(rg))
            disagree[f'{a}|{b}'] = float(np.mean(dg)) if dg else None
    summ = dict(n_states=n_states, feasibility={c: float(np.mean(v)) for c, v in feas.items()},
                regret=regret, top_disagreement=disagree)
    (Path(out_dir)/'summary.json').write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--horizon', type=int, default=48)
    a = ap.parse_args()
    if a.mode == 'run':
        rec = run(a.args[0], a.out, a.horizon)
        print(json.dumps(dict(key=rec['key'], states=len(rec['states']))))
    else:
        print(json.dumps(summarize(a.out), indent=1))


if __name__ == '__main__':
    main()
