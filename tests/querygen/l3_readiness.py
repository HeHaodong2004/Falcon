"""Layer-3 readiness study (official Falcon rules; privileged labels, legal features).

For decision states sampled along a replay-verified Oracle solution (official run JSON) and every candidate query
the generator offers there:
  * legal consequence features (``l3_interface.consequence_features``: kinematic sandbox rollout of the query on
    the legal map against M constant-velocity predicted futures) -- what a Layer-3 policy can see;
  * label: the candidate executed for one decision interval, then the *default policy* (keep the running query,
    else a fresh goal query; with the configured goal meta) until the official episode end -- does the system
    still succeed after this choice (recovery without further help), and how many steps does it take;
  * which candidate the Oracle chose;
  * optional one-decision recovery (``--recovery N``): for up to N candidates per state whose default continuation
    fails, can ONE later decision still save the episode -- at the next decision state, does any candidate
    followed by the default policy succeed?
The default policy's goal meta follows the result's generator configuration (``--default-meta auto``: unstick /
evade as in ``evaluate.run_executor_only``), so the labels use the same foundation as the Oracle solution.

  python scripts/querygen_run.py tests.querygen.l3_readiness run RESULT_JSON [...] --out DIR [--states 4]
         [--gen JSON] [--default-meta auto|JSON] [--recovery N]
  python scripts/querygen_run.py tests.querygen.l3_readiness watch SUITE_DIR [...] --out DIR --shard i --shards n
         (process the PASS results of running Oracle suites as they appear)
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


def default_meta_of(gen_cfg):
    """The default policy's goal-query meta for a generator configuration (as ``evaluate.run_executor_only``)."""
    m = {}
    if (gen_cfg or {}).get('goal_unstick'):
        m['unstick'] = int(gen_cfg['goal_unstick'])
    if (gen_cfg or {}).get('evade'):
        m['evade'] = 1
    if m.get('unstick') and (gen_cfg or {}).get('unstick_keep'):
        m['ukeep'] = 1
    if m.get('unstick') and (gen_cfg or {}).get('static_humans'):
        m['shum'] = 1
    if m.get('unstick') and (gen_cfg or {}).get('unstick_progress'):
        m['uprog'] = 1
    if m and (gen_cfg or {}).get('bump_front'):
        m['bumpf'] = 1
    if (gen_cfg or {}).get('stop_strict'):
        m['stop19'] = 1
    return m


def one_decision_recovery(rt, gen, child, goal_meta, ocfg, skip=('continue', 'goal')):
    """After a candidate whose default continuation fails: at the next decision state, try each candidate once,
    then the default policy; recovered iff one succeeds (stops at the first success)."""
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    if child.terminal:
        return dict(recovered=False, terminal=child.outcome, tried=[])
    queries, _ = gen.propose(child.legal, child.execution)
    tried = []
    for q in queries:
        if q.kind == 'continue' or q.label in skip:      # the default continuation: known to fail
            continue
        ch = rt.run(child, q, horizon=min(horizon_for(q, ocfg), MAX_EPISODE_STEPS-child.index))
        end = default_rollout(rt, ch, goal_meta) if not ch.terminal else ch
        tried.append((q.label, end.outcome))
        if end.outcome == 'success':
            break
    return dict(recovered=any(o == 'success' for _, o in tried), terminal=None, tried=tried,
                offered=len(queries))


def run_one(result_json, out_dir, n_states=4, gen_kw=None, goal_meta=None, M=16, steps=48, recovery=0):
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
    gcfg = gen_kw if gen_kw is not None else r['generator']
    gen = QueryGenerator(dyn, GeneratorConfig(**gcfg))
    if goal_meta is None:
        goal_meta = default_meta_of(gcfg)
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
            labels, children = [], []
            for q in queries:
                t = time.perf_counter()
                ch = rt.run(st, q, horizon=min(horizon_for(q, ocfg), MAX_EPISODE_STEPS-st.index))
                end = default_rollout(rt, ch, goal_meta) if not ch.terminal else ch
                labels.append(dict(label=q.label, kind=q.kind, outcome=end.outcome, steps=int(end.index),
                                   query_outcome=ch.outcome, seconds=round(time.perf_counter()-t, 2)))
                children.append(ch)
            if recovery:
                bad = [k for k, c in enumerate(labels) if c['outcome'] != 'success']
                rng = np.random.default_rng(int(st.index)+7919*i)
                first = [k for k in bad if queries[k].kind == 'continue' or queries[k].label == 'goal']
                rest = [k for k in bad if k not in first]
                order = first+[rest[j] for j in rng.permutation(len(rest))]
                for k in order[:recovery]:
                    t = time.perf_counter()
                    labels[k]['recovery'] = one_decision_recovery(rt, gen, children[k], goal_meta, ocfg)
                    labels[k]['recovery']['seconds'] = round(time.perf_counter()-t, 2)
            rows.append(dict(decision=i, critical=i in crit, index=int(st.index), geodesic=float(st.geodesic),
                             tracks=len(st.legal.tracks), stuck=bool(diag.get('stuck')), chosen=chosen,
                             candidates=labels, features=X.tolist(), generator_ms=round(gen_ms, 1),
                             features_ms=round(feat_ms, 1)))
        rec = dict(key=spec.key, planner=r['planner'], K=len(queries) if rows else None, goal_meta=goal_meta,
                   generator=gcfg, recovery=recovery,
                   solution_steps=r['replay']['steps'], n_decisions=len(dec), critical=crit,
                   feature_names=FEATURE_NAMES, states=rows)
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
        return rec
    finally:
        rt.close()


def fit_logreg(X, y, l2=1.0, iters=30):
    """L2-regularised logistic regression (Newton); returns a scoring function."""
    mu, sd = X.mean(0), X.std(0)+1e-6
    Z = np.hstack([(X-mu)/sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    R = l2*np.diag(np.r_[np.ones(Z.shape[1]-1), 0.])
    for _ in range(iters):
        q = 1./(1.+np.exp(-np.clip(Z@w, -30, 30)))
        g = Z.T@(q-y)+R@w
        H = (Z*(q*(1-q))[:, None]).T@Z+R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return lambda Xn: np.hstack([(Xn-mu)/sd, np.ones((len(Xn), 1))])@w


def state_features(X):
    """Per-candidate features plus their deviation from the state's mean (within-state ranking cues)."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, X-X.mean(0, keepdims=True)])


def within_auc(scores, ok):
    """P(a good candidate outranks a bad one of the same state); ties 1/2."""
    s = np.asarray(scores, dtype=float); y = np.asarray(ok, dtype=bool)
    if y.all() or not y.any():
        return None
    pos, neg = s[y], s[~y]
    return float((pos[:, None] > neg[None, :]).mean()+0.5*(pos[:, None] == neg[None, :]).mean())


def summarize(out_dir):
    from collections import Counter
    from eventwm_graph.querygen.l3_interface import auc
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    recs = [r for r in recs if r.get('states')]
    if not recs:
        return {}
    names = recs[0]['feature_names']
    ix = {n: k for k, n in enumerate(names)}
    S = []                      # flat list of states
    for e, rec in enumerate(recs):
        for st in rec['states']:
            ok = np.asarray([c['outcome'] == 'success' for c in st['candidates']], dtype=bool)
            labs = [c['label'] for c in st['candidates']]
            kinds = [c['kind'] for c in st['candidates']]
            dflt = kinds.index('continue') if 'continue' in kinds else (labs.index('goal') if 'goal' in labs else None)
            S.append(dict(ep=e, ok=ok, X=np.asarray(st['features'], dtype=float), dflt=dflt,
                          chosen=labs.index(st['chosen']) if st['chosen'] in labs else
                          (dflt if st['chosen'] == 'continue' else None),
                          critical=st['critical'], tracks=st['tracks'], cands=st['candidates'],
                          gen_ms=st['generator_ms'], feat_ms=st['features_ms']))
    n = len(S)
    good = np.asarray([s['ok'].sum() for s in S]); K = np.asarray([len(s['ok']) for s in S])
    mixed = [s for s in S if 0 < s['ok'].sum() < len(s['ok'])]

    def top1(score_fn):
        """Share of states whose top-scored candidate succeeds (ties: first in generator order)."""
        return float(np.mean([s['ok'][int(np.argmax(score_fn(s)))] for s in S]))
    # legal single-feature scorers
    scorers = dict(progress=lambda s: s['X'][:, ix['progress_m']],
                   neg_p_overlap=lambda s: -s['X'][:, ix['p_overlap']],
                   dmin_p05=lambda s: s['X'][:, ix['dmin_p05']],
                   progress_minus_risk=lambda s: s['X'][:, ix['progress_m']]-3.*s['X'][:, ix['p_overlap']])
    # leave-one-episode-out logistic regression on the legal features (+ within-state deviations)
    E = sorted({s['ep'] for s in S})
    loeo = {}
    for e in E:
        tr = [s for s in S if s['ep'] != e]
        if not tr:
            continue
        Xt = np.vstack([state_features(s['X']) for s in tr]); yt = np.concatenate([s['ok'] for s in tr]).astype(float)
        if yt.min() == yt.max():
            continue
        f = fit_logreg(Xt, yt)
        for k, s in enumerate(S):
            if s['ep'] == e:
                loeo[k] = f(state_features(s['X']))
    sel = dict(any_good_upper_bound=float(np.mean(good > 0)),
               random=float(np.mean(good/K)),
               default=float(np.mean([s['ok'][s['dflt']] for s in S if s['dflt'] is not None])),
               oracle_choice=float(np.mean([s['ok'][s['chosen']] for s in S if s['chosen'] is not None])),
               **{f'legal_{k}': top1(f) for k, f in scorers.items()})
    if loeo:
        sel['legal_logreg_loeo'] = float(np.mean([S[k]['ok'][int(np.argmax(v))] for k, v in loeo.items()]))
    wauc = {k: [within_auc(f(s), s['ok']) for s in mixed] for k, f in scorers.items()}
    if loeo:
        wauc['logreg_loeo'] = [within_auc(loeo[k], S[k]['ok']) for k in loeo if 0 < S[k]['ok'].sum() < len(S[k]['ok'])]
    wauc = {k: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None)
            for k, v in wauc.items()}
    # recovery after a bad candidate
    rec_rows = [(s, c) for s in S for c in s['cands'] if c.get('recovery') is not None]
    bad_out = Counter(c['outcome'] for s in S for c in s['cands'] if c['outcome'] != 'success')
    recov = dict(tested=len(rec_rows),
                 recovered=float(np.mean([c['recovery']['recovered'] for _, c in rec_rows])) if rec_rows else None,
                 recovered_critical=float(np.mean([c['recovery']['recovered'] for s, c in rec_rows if s['critical']]))
                 if any(s['critical'] for s, _ in rec_rows) else None,
                 recovered_noncritical=float(np.mean([c['recovery']['recovered'] for s, c in rec_rows if not s['critical']]))
                 if any(not s['critical'] for s, _ in rec_rows) else None,
                 terminal_during_candidate=int(sum(bool(c['recovery'].get('terminal')) for _, c in rec_rows)),
                 by_failure={o: float(np.mean([c['recovery']['recovered'] for _, c in rec_rows if c['outcome'] == o]))
                             for o in sorted({c['outcome'] for _, c in rec_rows})})
    crit = [s for s in S if s['critical']]
    summ = dict(episodes=len(recs), states=n, K_mean=float(K.mean()),
                good_candidates_mean=float(good.mean()),
                good_candidates_hist={str(k): int((good == k).sum()) for k in range(int(K.max())+1)},
                states_with_0_good=float(np.mean(good == 0)), states_with_1_good=float(np.mean(good == 1)),
                states_with_ge2_good=float(np.mean(good >= 2)), mixed_states=len(mixed),
                critical_states=len(crit),
                critical_good_mean=float(np.mean([s['ok'].sum() for s in crit])) if crit else None,
                critical_default_ok=float(np.mean([s['ok'][s['dflt']] for s in crit if s['dflt'] is not None]))
                if crit else None,
                tracked_states=int(sum(s['tracks'] > 0 for s in S)),
                top1_success=sel, within_state_auc=wauc,
                bad_candidate_outcomes=dict(bad_out), recovery=recov,
                auc_progress=auc(np.concatenate([s['X'][:, ix['progress_m']] for s in S]),
                                 np.concatenate([s['ok'] for s in S])),
                generator_ms_p50=float(np.median([s['gen_ms'] for s in S])),
                features_ms_p50=float(np.median([s['feat_ms'] for s in S])),
                candidate_rollout_s_p50=float(np.median([c['seconds'] for s in S for c in s['cands']])))
    (Path(out_dir)/'summary.json').write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize', 'watch'])
    ap.add_argument('results', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--states', type=int, default=4)
    ap.add_argument('--gen', default='')
    ap.add_argument('--default-meta', default='auto')
    ap.add_argument('--recovery', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'summarize':
        print(json.dumps(summarize(a.out), indent=1))
        return
    if a.mode == 'watch':
        # results = Oracle suite directories: process their PASS results (this shard) as they appear, until every
        # suite has written its summary.json and nothing is left
        import zlib
        skip = set()
        while True:
            complete = all((Path(d)/'summary.json').exists() for d in a.results)
            files = sorted(f for d in a.results for f in Path(d).glob('*.json')
                           if f.name not in ('run.json', 'summary.json'))
            todo = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard and f.name not in skip
                    and not (Path(a.out)/f.name).exists()]
            for f in todo:
                try:
                    verdict = json.loads(f.read_text()).get('verdict')
                except Exception:
                    continue                      # being written: next scan
                if verdict != 'PASS':
                    skip.add(f.name)
                    continue
                try:
                    rec = run_one(str(f), a.out, a.states, json.loads(a.gen) if a.gen else None,
                                  None if a.default_meta == 'auto' else json.loads(a.default_meta), recovery=a.recovery)
                except Exception as e:            # one episode must not end the study
                    import traceback
                    traceback.print_exc()
                    print(json.dumps(dict(file=f.name, error=repr(e))), flush=True)
                    skip.add(f.name)
                    continue
                if rec:
                    print(json.dumps(dict(key=rec['key'], states=len(rec['states']), good=[
                        sum(c['outcome'] == 'success' for c in st['candidates']) for st in rec['states']])), flush=True)
            if complete and all(f.name in skip or (Path(a.out)/f.name).exists() for f in files
                                if zlib.crc32(f.name.encode()) % a.shards == a.shard):
                break
            time.sleep(120)
        return
    for f in a.results:
        name = Path(f).name
        if (Path(a.out)/name).exists():
            continue
        rec = run_one(f, a.out, a.states, json.loads(a.gen) if a.gen else None,
                      None if a.default_meta == 'auto' else json.loads(a.default_meta), recovery=a.recovery)
        if rec:
            print(json.dumps(dict(key=rec['key'], states=len(rec['states']), good=[sum(c['outcome'] == 'success'
                                  for c in s['candidates']) for s in rec['states']])), flush=True)


if __name__ == '__main__':
    main()
