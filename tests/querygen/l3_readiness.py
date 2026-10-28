"""Layer-3 readiness study (official Falcon rules; privileged labels, legal features).

For decision states sampled along a replay-verified Oracle solution (official run JSON) and every candidate query
the generator offers there:
  * legal consequence features (``l3_interface.consequence_features``: kinematic sandbox rollout of the query on
    the legal map against M constant-velocity predicted futures) -- what a Layer-3 policy can see;
  * label: the candidate executed for one decision interval, then the *default policy* (keep the running query,
    else a fresh goal query; with the configured goal meta) until the official episode end -- does the system
    still succeed after this choice (recovery without further help), and how many steps does it take;
  * which candidate the Oracle chose.

  python scripts/querygen_run.py tests.querygen.l3_readiness run RESULT_JSON [...] --out DIR [--states 4]
         [--gen JSON] [--default-meta JSON]
  python scripts/querygen_run.py tests.querygen.l3_readiness summarize --out DIR
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def reconstruct(rt, decisions):
    """Decision states along a recorded branch (replaying its queries in the branch simulator)."""
    from eventwm_graph.querygen.executor import Query
    st = rt.root
    out = [st]
    for d in decisions:
        q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'],
                  speed=d['speed'], label=d['label'], meta=d['meta'] or {})
        if st.terminal:
            break
        st = rt.run(st, None if d['cont'] else q, horizon=d['horizon'], cont=bool(d['cont']))
        out.append(st)
    return out


def default_rollout(rt, st, goal_meta, horizon=16):
    from eventwm_graph.querygen.executor import Query
    q = Query('goal', label='goal', meta=dict(goal_meta))
    while not st.terminal:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else q, horizon=horizon, cont=cont)
    return st


def run_one(result_json, out_dir, n_states=4, gen_kw=None, goal_meta=None, M=16, steps=48):
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for, OracleConfig
    from eventwm_graph.querygen.l3_interface import consequence_features, FEATURE_NAMES
    from eventwm_graph.querygen.sandbox import cv_futures
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    r = json.loads(Path(result_json).read_text())
    if r.get('verdict') != 'PASS':
        return None
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[r['dynamics']]
    gen = QueryGenerator(dyn, GeneratorConfig(**(gen_kw if gen_kw is not None else r['generator'])))
    rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False)
    rt.sim.check_static = False
    ocfg = OracleConfig()
    try:
        states = reconstruct(rt, r['decisions'])
        dec = r['decisions']
        # decision states: those where the Oracle took a non-default decision first, then evenly spaced others
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
            t = time.perf_counter()
            queries, diag = gen.propose(st.legal, st.execution)
            gen_ms = 1000*(time.perf_counter()-t)
            t = time.perf_counter()
            F = cv_futures(st.legal.tracks, st.legal.time, steps, M=M, seed=int(st.index))
            X, _ = consequence_features(st.legal, queries, r['planner'], dyn, F, steps)
            feat_ms = 1000*(time.perf_counter()-t)
            chosen = dec[i]['label'] if not dec[i]['cont'] else 'continue'
            labels = []
            for q in queries:
                t = time.perf_counter()
                ch = rt.run(st, q, horizon=min(horizon_for(q, ocfg), MAX_EPISODE_STEPS-st.index))
                end = default_rollout(rt, ch, goal_meta or {}) if not ch.terminal else ch
                labels.append(dict(label=q.label, kind=q.kind, outcome=end.outcome, steps=int(end.index),
                                   query_outcome=ch.outcome, seconds=round(time.perf_counter()-t, 2)))
            rows.append(dict(decision=i, critical=i in crit, index=int(st.index), geodesic=float(st.geodesic),
                             tracks=len(st.legal.tracks), stuck=bool(diag.get('stuck')), chosen=chosen,
                             candidates=labels, features=X.tolist(), generator_ms=round(gen_ms, 1),
                             features_ms=round(feat_ms, 1)))
        rec = dict(key=spec.key, planner=r['planner'], K=len(queries) if rows else None, goal_meta=goal_meta or {},
                   solution_steps=r['replay']['steps'], n_decisions=len(dec), critical=crit,
                   feature_names=FEATURE_NAMES, states=rows)
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
        return rec
    finally:
        rt.close()


def summarize(out_dir):
    from eventwm_graph.querygen.l3_interface import auc
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    n_states = n_good = 0
    good_counts, good_frac, crit_good, default_good, chosen_good = [], [], [], [], []
    feats, lab = [], []
    gen_ms, feat_ms = [], []
    for rec in recs:
        names = rec['feature_names']
        ix = {n: k for k, n in enumerate(names)}
        for st in rec['states']:
            ok = [c['outcome'] == 'success' for c in st['candidates']]
            n_states += 1
            good_counts.append(sum(ok)); good_frac.append(sum(ok)/max(1, len(ok)))
            if st['critical']:
                crit_good.append(sum(ok))
            labs = [c['label'] for c in st['candidates']]
            dflt = next((k for k, c in enumerate(st['candidates']) if c['kind'] == 'continue'), None)
            if dflt is None and 'goal' in labs:
                dflt = labs.index('goal')
            if dflt is not None:
                default_good.append(ok[dflt])
            if st['chosen'] in labs:
                chosen_good.append(ok[labs.index(st['chosen'])])
            X = np.asarray(st['features'])
            for k, c in enumerate(st['candidates']):
                feats.append(X[k]); lab.append(ok[k])
            gen_ms.append(st['generator_ms']); feat_ms.append(st['features_ms'])
    X = np.asarray(feats); y = np.asarray(lab, dtype=bool)
    ix = {n: k for k, n in enumerate(recs[0]['feature_names'])} if recs else {}
    summ = dict(episodes=len(recs), states=n_states,
                good_candidates_mean=float(np.mean(good_counts)) if good_counts else None,
                good_candidates_hist={str(k): int(sum(1 for g in good_counts if g == k)) for k in range(0, 11)},
                states_with_ge2_good=float(np.mean([g >= 2 for g in good_counts])) if good_counts else None,
                states_with_0_good=float(np.mean([g == 0 for g in good_counts])) if good_counts else None,
                good_fraction_mean=float(np.mean(good_frac)) if good_frac else None,
                critical_states=len(crit_good),
                critical_good_mean=float(np.mean(crit_good)) if crit_good else None,
                default_choice_recovers=float(np.mean(default_good)) if default_good else None,
                oracle_choice_recovers=float(np.mean(chosen_good)) if chosen_good else None,
                auc_progress=auc(X[:, ix['progress_m']], y) if len(X) else None,
                auc_neg_p_overlap=auc(-X[:, ix['p_overlap']], y) if len(X) else None,
                auc_dmin_p05=auc(X[:, ix['dmin_p05']], y) if len(X) else None,
                generator_ms_p50=float(np.median(gen_ms)) if gen_ms else None,
                features_ms_p50=float(np.median(feat_ms)) if feat_ms else None)
    (Path(out_dir)/'summary.json').write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('results', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--states', type=int, default=4)
    ap.add_argument('--gen', default='')
    ap.add_argument('--default-meta', default='{}')
    a = ap.parse_args()
    if a.mode == 'summarize':
        print(json.dumps(summarize(a.out), indent=1))
        return
    for f in a.results:
        name = Path(f).name
        if (Path(a.out)/name).exists():
            continue
        rec = run_one(f, a.out, a.states, json.loads(a.gen) if a.gen else None, json.loads(a.default_meta))
        if rec:
            print(json.dumps(dict(key=rec['key'], states=len(rec['states']), good=[sum(c['outcome'] == 'success'
                                  for c in s['candidates']) for s in rec['states']])), flush=True)


if __name__ == '__main__':
    main()
