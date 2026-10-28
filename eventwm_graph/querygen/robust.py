"""Robustness of an Oracle solution (privileged evaluation only; never used online).

A query-level solution is *robust* if the same decisions still succeed when the timing is perturbed: the queries
are closed-loop (the executor re-plans from its own legal observations every step), so a sensible choice survives
a lost step or two, while a knife-edge sequence that only works with exact timing does not.

Perturbations (``PERTURBATIONS``): the robot holds (native hold = in-place turn pair) for 1 or 2 steps before the
first decision (equivalent to the humans being 1-2 steps ahead of the plan), or for 1 step before the middle / the
second-to-last decision.  The recorded decisions are then re-executed in order in the exact branch simulator
(recorded human tape; official contact rules): a 'continue' whose query has already finished re-issues that query;
an explicit STOP is executed only within the generator's legal stop radius (otherwise the goal query continues);
after the last recorded decision the default policy (goal query with the run's foundation meta) continues to the
official episode end.  Robust success rate = share of perturbed runs that end in official success.

  python scripts/querygen_run.py eventwm_graph.querygen.robust run RESULT_JSON [...] --out DIR
  python scripts/querygen_run.py eventwm_graph.querygen.robust summarize --out DIR
"""
import argparse
import json
import math
import time
from pathlib import Path

PERTURBATIONS = (('start', 1), ('start', 2), ('mid', 1), ('late', 1))


def _query(x):
    from .executor import Query
    return Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                 speed=x['speed'], label=x['label'], meta=x['meta'] or {})


def default_goal_meta(gen_cfg):
    m = {}
    g = gen_cfg or {}
    if g.get('goal_unstick'):
        m['unstick'] = int(g['goal_unstick'])
        if g.get('unstick_keep'):
            m['ukeep'] = 1
        if g.get('static_humans'):
            m['shum'] = 1
        if g.get('unstick_progress'):
            m['uprog'] = 1
    if g.get('evade'):
        m['evade'] = 1
    if m and g.get('bump_front'):
        m['bumpf'] = 1
    if g.get('stop_strict'):
        m['stop19'] = 1
    return m


def perturbed_run(rt, decisions, where, n, goal_meta, stop_radius=0.40):
    from .executor import Query
    st = rt.root
    k_insert = {'start': 0, 'mid': len(decisions)//2, 'late': max(0, len(decisions)-2)}[where]
    last = None
    for i, d in enumerate(decisions):
        if st.terminal:
            break
        if i == k_insert:
            st = rt.run(st, Query('wait', duration=n, label='perturb_hold'), horizon=n)
            if st.terminal:
                break
        q = _query(d)
        cont = bool(d['cont'])
        if q.kind == 'stop':
            p = st.legal.pose
            if math.hypot(float(rt.task.goal[0])-p[0], float(rt.task.goal[2])-p[2]) >= stop_radius:
                continue                      # not at the goal under this timing: keep going (default below)
        if cont and (st.execution is None or st.execution.status != 'running'):
            cont, q = False, (last or q)
        st = rt.run(st, None if cont else q, horizon=int(d['horizon']), cont=cont)
        if not cont:
            last = q
    goal = Query('goal', label='goal', meta=dict(goal_meta))
    while not st.terminal:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else goal, horizon=16, cont=cont)
    return dict(outcome=st.outcome, steps=int(st.index))


def run_one(result_json, out_dir):
    from .native import EpisodeSpec
    from .episode import EpisodeRuntime
    from .dynamics import FAMILIES
    from .generator import QueryGenerator, GeneratorConfig
    r = json.loads(Path(result_json).read_text())
    if r.get('verdict') != 'PASS':
        return None
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[r['dynamics']]
    gen_cfg = r.get('generator') or r.get('gen') or {}
    QueryGenerator(dyn, GeneratorConfig(**{k: v for k, v in gen_cfg.items()}))   # process-wide legal-map options
    meta = default_goal_meta(gen_cfg)
    t0 = time.perf_counter()
    rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False)
    rt.sim.check_static = False
    try:
        runs = []
        for where, n in PERTURBATIONS:
            res = perturbed_run(rt, r['decisions'], where, n, meta)
            runs.append(dict(where=where, hold=n, **res))
    finally:
        rt.close()
    rep = r.get('replay', {})
    m = rep.get('metrics', r.get('official', {}))
    rec = dict(key=spec.key, planner=r['planner'], robust=sum(x['outcome'] == 'success' for x in runs)/len(runs),
               runs=runs, steps=rep.get('steps'), spl=m.get('spl'), psc=m.get('psc'),
               min_rect=rep.get('min_rect_distance'), labels=[d['label'] for d in r['decisions'] if not d['cont']],
               seconds=round(time.perf_counter()-t0, 1))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def summarize(out_dir):
    import numpy as np
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    if not recs:
        return {}
    by = {}
    for where, n in PERTURBATIONS:
        k = f'{where}{n}'
        by[k] = float(np.mean([any(x['where'] == where and x['hold'] == n and x['outcome'] == 'success'
                                   for x in r['runs']) for r in recs]))
    s = dict(episodes=len(recs), robust_mean=float(np.mean([r['robust'] for r in recs])),
             fully_robust=float(np.mean([r['robust'] == 1.0 for r in recs])),
             fragile=sorted((r['key'].split('/')[2], r['robust']) for r in recs if r['robust'] < 0.5),
             by_perturbation=by,
             failure_outcomes=dict(__import__('collections').Counter(x['outcome'] for r in recs for x in r['runs']
                                                                     if x['outcome'] != 'success')))
    (Path(out_dir)/'summary.json').write_text(json.dumps(s, indent=1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('results', nargs='*')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.mode == 'summarize':
        print(json.dumps(summarize(a.out), indent=1))
        return
    for f in a.results:
        if (Path(a.out)/Path(f).name).exists():
            continue
        rec = run_one(f, a.out)
        if rec:
            print(json.dumps(dict(key=rec['key'], robust=rec['robust'], runs=[(x['where'], x['hold'], x['outcome'])
                                                                              for x in rec['runs']],
                                  seconds=rec['seconds'])), flush=True)


if __name__ == '__main__':
    main()
