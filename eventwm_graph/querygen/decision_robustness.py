"""Decision robustness / query-space learnability (privileged evaluation only; never used online).

Question: does Layer 3 need to find the single right query at every decision, or do several meaningfully different
queries lead to success with good SPL, and are suboptimal choices recoverable without clever follow-ups?

At decision states along a replay-verified Oracle solution (its non-default decisions first, then evenly spaced
others), every candidate query the generator offers there is executed for one decision interval in the exact
branch simulator (official rules, recorded human tape).  Then:

1. near-duplicates are merged: candidates whose one-decision executions nearly coincide (same terminal outcome, end
   time within ``END_STEPS``, end pose within ``END_M`` and ``END_DEG``, and every intermediate pose within ``PATH_M``
   of the other's, the shorter execution held at its end) form one *choice*; only choices count as distinct options;
2. each choice (one representative) is continued:
   * ``easy``  -- the foundation's default policy alone (keep the running query, else the goal query) reaches an
     official success: recoverable without further clever decisions;
   * ``clever`` -- otherwise a bounded privileged Oracle search from that state (minimum-intervention cost, so it
     prefers few non-default decisions) finds an official success; the number of non-default decisions it needs is
     recorded;
   * ``dead`` -- neither within the budget;
3. SPL is the episode SPL of the full trajectory (solution prefix + choice + continuation).  A viable choice has
   *good SPL* if it reaches >= GOOD_SPL_FRAC of the best SPL found at that state.

Reported per state: distinct choices, viable choices (easy / any), good-SPL viable choices; share of states with
exactly one viable choice; recovery of non-best choices (easy / clever / dead) and their SPL cost.

  python scripts/querygen_run.py eventwm_graph.querygen.decision_robustness run RESULT_JSON [...] --out DIR
         [--states 4] [--oracle-edges 200]
  python scripts/querygen_run.py eventwm_graph.querygen.decision_robustness summarize --out DIR
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

END_M = 0.25
END_DEG = 20.
PATH_M = 0.25
GOOD_SPL_FRAC = 0.9


def _query(x):
    from .executor import Query
    return Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                 speed=x['speed'], label=x['label'], meta=x['meta'] or {})


def _goal_meta(gen_cfg):
    from .robust import default_goal_meta
    return default_goal_meta(gen_cfg)


def _path(parent, child):
    """Poses (x, z, yaw) executed by the child's decision, starting at the parent's pose."""
    n0 = len(parent.legal.poses)-1
    return np.asarray([[p[0], p[2], p[3]] for p in child.legal.poses[n0:]], dtype=float)


END_STEPS = 3


def _same_choice(a, b):
    """Near-duplicate test between two one-decision executions (dicts with outcome, path, end step): same outcome,
    end time within END_STEPS (waiting longer changes the timing against the humans), end pose and path close."""
    if a['outcome'] != b['outcome'] or abs(a['end'] - b['end']) > END_STEPS:
        return False
    pa, pb = a['path'], b['path']
    ea, eb = pa[-1], pb[-1]
    if math.hypot(ea[0]-eb[0], ea[1]-eb[1]) > END_M:
        return False
    if abs(math.degrees(math.atan2(math.sin(ea[2]-eb[2]), math.cos(ea[2]-eb[2])))) > END_DEG:
        return False
    n = max(len(pa), len(pb))
    A = np.concatenate([pa, np.repeat(pa[-1:], n-len(pa), 0)]) if len(pa) < n else pa
    B = np.concatenate([pb, np.repeat(pb[-1:], n-len(pb), 0)]) if len(pb) < n else pb
    return float(np.hypot(A[:, 0]-B[:, 0], A[:, 1]-B[:, 1]).max()) <= PATH_M


def _spl(rt, st):
    if st.outcome != 'success':
        return 0.
    L, Ls = float(st.path_length), float(rt.start_geodesic)
    return Ls/max(L, Ls) if L > 0 else 1.


def default_continue(rt, st, goal_meta):
    from .executor import Query
    goal = Query('goal', label='goal', meta=dict(goal_meta))
    while not st.terminal:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else goal, horizon=16, cont=cont)
    return st


def oracle_continue(rt, gen, st, edges, intervention_cost=60.):
    """Bounded privileged search from ``st`` (the runtime's root is swapped temporarily)."""
    from .oracle import Oracle, OracleConfig
    root = rt.root
    rt.root = st
    try:
        cfg = OracleConfig(heading_h=True, escape_penalty=80, prefix_cache=48, max_edges=int(edges),
                           max_seconds=7200, weight=3.5, intervention_cost=float(intervention_cost))
        o = Oracle(rt, gen, cfg)
        res = o.search()
    finally:
        rt.root = root
    end = res['state']
    extra = end.decisions[len(st.decisions):]
    nondefault = sum(1 for d in extra if not (d['cont'] or d['query'].kind == 'stop' or d['query'].label == 'goal'))
    return end, res['success'], nondefault, res['stats']['children']


def run_one(result_json, out_dir, n_states=4, oracle_edges=200):
    from .native import EpisodeSpec, MAX_EPISODE_STEPS
    from .episode import EpisodeRuntime
    from .dynamics import FAMILIES
    from .generator import QueryGenerator, GeneratorConfig
    from .oracle import horizon_for, OracleConfig
    r = json.loads(Path(result_json).read_text())
    if r.get('verdict') != 'PASS':
        return None
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[r['dynamics']]
    gen_cfg = r.get('generator') or {}
    gen = QueryGenerator(dyn, GeneratorConfig(**gen_cfg))
    meta = _goal_meta(gen_cfg)
    ocfg = OracleConfig()
    t0 = time.perf_counter()
    rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False)
    rt.sim.check_static = False
    try:
        # decision states along the solution
        dec = r['decisions']
        states, st = [rt.root], rt.root
        for d in dec:
            if st.terminal:
                break
            st = rt.run(st, None if d['cont'] else _query(d), horizon=int(d['horizon']), cont=bool(d['cont']))
            states.append(st)
        crit = [i for i, d in enumerate(dec) if not (d['cont'] or d['label'] == 'goal' or d['kind'] == 'stop')]
        others = [i for i in range(len(dec)) if i not in crit]
        pick = crit[:n_states]
        if len(pick) < n_states and others:
            extra = np.linspace(0, len(others)-1, min(n_states-len(pick), len(others))).round().astype(int)
            pick += [others[k] for k in sorted(set(extra.tolist()))]
        rows = []
        for i in sorted(set(pick)):
            st = states[i]
            if st.terminal:
                continue
            ts = time.perf_counter()
            qs, diag = gen.propose(st.legal, st.execution)
            execs = []
            for q in qs:
                ch = rt.run(st, q, horizon=max(1, min(horizon_for(q, ocfg), MAX_EPISODE_STEPS-st.index)))
                execs.append(dict(label=q.label, kind=q.kind, child=ch, outcome=ch.outcome, path=_path(st, ch),
                                  end=int(ch.index)))
            choices = []                      # list of lists of candidate indices
            for k, e in enumerate(execs):
                for c in choices:
                    if _same_choice(execs[c[0]], e):
                        c.append(k)
                        break
                else:
                    choices.append([k])
            res = []
            for c in choices:
                rep = execs[c[0]]
                ch = rep['child']
                if ch.terminal:
                    kind, end, nd, ed = ('easy' if ch.outcome == 'success' else 'dead'), ch, 0, 0
                else:
                    end = default_continue(rt, ch, meta)
                    kind, nd, ed = ('easy' if end.outcome == 'success' else None), 0, 0
                    if kind is None:
                        end, ok, nd, ed = oracle_continue(rt, gen, ch, oracle_edges)
                        kind = 'clever' if ok else 'dead'
                res.append(dict(members=[execs[k]['label'] for k in c], recovery=kind, spl=_spl(rt, end),
                                steps=int(end.index), nondefault_after=int(nd), oracle_edges=int(ed)))
            best = max([x['spl'] for x in res] + [0.])
            for x in res:
                x['good'] = x['recovery'] != 'dead' and x['spl'] >= GOOD_SPL_FRAC*best and best > 0
            chosen = dec[i]['label'] if not dec[i]['cont'] else 'continue'
            rows.append(dict(decision=i, index=int(st.index), critical=i in crit, chosen=chosen,
                             chosen_choice=next((j for j, x in enumerate(res) if chosen in x['members']), None),
                             candidates=len(qs), choices=res, tracks=len(st.legal.tracks),
                             stuck=bool(diag.get('stuck')), seconds=round(time.perf_counter()-ts, 1)))
        rec = dict(key=spec.key, planner=r['planner'], solution_spl=r['replay']['metrics'].get('spl'),
                   n_decisions=len(dec), critical=crit, states=rows, seconds=round(time.perf_counter()-t0, 1),
                   generator=gen_cfg, oracle_edges=oracle_edges)
    finally:
        rt.close()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def summarize(out_dir):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    recs = [r for r in recs if r.get('states')]
    S = [s for r in recs for s in r['states']]
    if not S:
        return {}
    n_choice = np.asarray([len(s['choices']) for s in S])
    n_cand = np.asarray([s['candidates'] for s in S])
    viable_easy = np.asarray([sum(x['recovery'] == 'easy' for x in s['choices']) for s in S])
    viable_any = np.asarray([sum(x['recovery'] != 'dead' for x in s['choices']) for s in S])
    good = np.asarray([sum(x['good'] for x in s['choices']) for s in S])
    crit = np.asarray([s['critical'] for s in S])
    # suboptimal choices: every non-best viable-or-not choice at a state
    sub = []
    for s in S:
        best = max(x['spl'] for x in s['choices'])
        for x in s['choices']:
            if x['spl'] < best or x['recovery'] == 'dead':
                sub.append(dict(recovery=x['recovery'], spl_loss=best-x['spl'], nd=x['nondefault_after']))

    def hist(v):
        return {str(k): int((v == k).sum()) for k in range(0, int(v.max())+1)}

    def frac(m):
        return float(np.mean(m)) if len(m) else None
    s = dict(episodes=len(recs), states=len(S), critical_states=int(crit.sum()),
             candidates_mean=float(n_cand.mean()), distinct_choices_mean=float(n_choice.mean()),
             viable_any_mean=float(viable_any.mean()), viable_easy_mean=float(viable_easy.mean()),
             good_spl_choices_mean=float(good.mean()),
             good_spl_choices_hist=hist(good),
             single_viable_share=frac(viable_any == 1), single_good_share=frac(good == 1),
             no_easy_viable_share=frac(viable_easy == 0),
             critical=dict(states=int(crit.sum()), good_mean=float(good[crit].mean()) if crit.any() else None,
                           single_good_share=frac(good[crit] == 1) if crit.any() else None),
             suboptimal=dict(n=len(sub), easy=frac(np.asarray([x['recovery'] == 'easy' for x in sub])),
                             clever=frac(np.asarray([x['recovery'] == 'clever' for x in sub])),
                             dead=frac(np.asarray([x['recovery'] == 'dead' for x in sub])),
                             spl_loss_mean=float(np.mean([x['spl_loss'] for x in sub if x['recovery'] != 'dead']))
                             if any(x['recovery'] != 'dead' for x in sub) else None,
                             clever_nondefault_mean=float(np.mean([x['nd'] for x in sub if x['recovery'] == 'clever']))
                             if any(x['recovery'] == 'clever' for x in sub) else None),
             seconds_per_state=float(np.mean([s_['seconds'] for s_ in S])))
    (Path(out_dir)/'summary.json').write_text(json.dumps(s, indent=1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('results', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--states', type=int, default=4)
    ap.add_argument('--oracle-edges', type=int, default=200)
    a = ap.parse_args()
    if a.mode == 'summarize':
        print(json.dumps(summarize(a.out), indent=1))
        return
    for f in a.results:
        if (Path(a.out)/Path(f).name).exists():
            continue
        try:
            rec = run_one(f, a.out, a.states, a.oracle_edges)
        except Exception as e:                       # one episode must not end the study
            import traceback
            traceback.print_exc()
            print(json.dumps(dict(file=f, error=repr(e))), flush=True)
            continue
        if rec:
            print(json.dumps(dict(key=rec['key'], states=[(s['decision'], len(s['choices']),
                                                            sum(x['good'] for x in s['choices'])) for s in rec['states']],
                                  seconds=rec['seconds'])), flush=True)


if __name__ == '__main__':
    main()
