"""Offline fitting / evaluation of the learned listwise ranker (``riskrank.py``, selector option ``L3_MW_RISK``).

TRAIN only for fitting, features, thresholds and margins: 5-fold cross-validation grouped by EPISODE (fold =
crc32(episode key) % 5) over the extracted TRAIN states (``riskrank extract``); the selector's deviation threshold is
tuned inside every outer training split by an inner 4-fold grouped cross-validation (out-of-fold ranker scores), so the
outer test fold never informs its own threshold.  The frozen model is fitted on every TRAIN state, its threshold tuned on
the outer out-of-fold scores.  Dev (``evaluate``): once, with the frozen model.

Metrics (per fold and pooled):
* within-state AUC of the risk score (-ranker score) for a real contact (or STOP away from the goal) within 32 / 64
  steps, averaged over the mixed states (both outcomes present), next to the rule's own -oc (cont 16 / 48);
* decisions vs the exact outcome of every candidate (``shortprobe._key_real``) at 32 and 64 steps: deviation rate,
  chosen candidate exact-collides, critical states (the default collides, some candidate does not) fixed, breaks (the
  default is safe, the chosen collides), chosen = exact best, mean exact value (contact 0, running 2 + half the legal
  route gain share, goal 3) -- for the default, the current rule (crit_rule.py: CV people, cont 16 / 48, no discovery,
  EPS 0), the ranker's argmax and the ranker selector.

LightGBM 4.x core API (PYTHONPATH with lightgbm; RISKFIT_THREADS threads, default 2).  The frozen model
``configs/l3_riskrank_m6b`` = preset ``m6b`` (150 rounds):

    python scripts/l3_run.py eventwm_graph.l3.riskfit cv --feat output/agents/m6_ranker/feat/orca_train_blind \\
        --out output/agents/m6_ranker/cv_final --params m6b --model output/agents/m6_ranker/model
    python scripts/l3_run.py eventwm_graph.l3.riskfit evaluate --model configs/l3_riskrank_m6b \\
        --feat output/agents/m6_ranker/feat/orca_dev --out output/agents/m6_ranker/dev

Results 2026-10-30 (ORCA native; reports output/agents/m6_ranker/cv_final/cv.txt, dev/dev.txt).  TRAIN CV pooled (785
states, 153 episodes): within-state AUC (-score) 0.518 at 32 steps (rule -oc16 0.574) / 0.491 at 64 (-oc48 0.557);
selector at 64 steps: value 2.184 vs rule16 2.143 / rule48 2.170 / default 2.127, exact best 32.0% vs 25.0 / 22.5 /
16.8%, chosen collides 100 vs 99 / 88 / 100.  Dev (283 states, one pass): value 2.185 vs 2.114 / 2.096 / 2.092, exact
best 29.7% vs 23.0 / 24.0 / 12.7%, collides 29 vs 34 / 35 / 31.  The gain is progress, not contact foresight (no
within-state contact discrimination beyond the rule's own oc), and much of it is a query prior: the ranker deviates to
the thru goal variants ('_th') in ~63% of TRAIN states and "always st_goal_th" scores TRAIN value 2.180 (collides 94) --
a fixed st_goal_th policy LOST SR online earlier (docs/l3/REPORT.md: 43.1% vs floor 49.2%).  The TRAIN states lie on the
floor's trajectories (running query = the goal query); after the ranker's own deviations the states are off that
distribution.  Validate in closed-loop SR before use.
"""
import argparse
import glob
import json
import math
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

from .riskrank import relevance, HAZARD_PARAMS

FOLDS = 5
PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3], learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=40, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              max_bin=63, verbose=-1, seed=0, num_threads=int(os.environ.get('RISKFIT_THREADS', '2')),
              lambdarank_truncation_level=10)
ROUNDS = 150
# parameter presets (``--params <name>``); m6b = the frozen model configs/l3_riskrank_m6b: a safety-weighted label gain
# (contact 0, running 6..10 by progress, goal 12) and small trees
PRESETS = dict(default={}, m6b=dict(label_gain=[0, 6, 7, 8, 9, 10, 12], num_leaves=7, min_data_in_leaf=60))


def fold_of(key, folds=FOLDS):
    return zlib.crc32(str(key).encode()) % folds


def load(feat_dir):
    """Extracted states of one or more feature directories (comma-separated, e.g. TRAIN orca_train_blind +
    orca_train2_blind); a decision (episode key + decision index) present twice is kept once (the first directory)."""
    recs, seen = [], set()
    for d in str(feat_dir).split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.pkl'))):
            r = pickle.load(open(f, 'rb'))
            k = (r['key'], int(r['decision']), r.get('config'))
            if k in seen:
                continue
            seen.add(k)
            recs.append(r)
    return recs


def vocab_of(recs):
    labs = sorted({l for r in recs for l in r['labels']})
    return {l: i for i, l in enumerate(labs)}


def matrix(recs, vocab, names=None):
    """Stacked features with the label column filled from ``vocab`` (unknown labels -1)."""
    names = names or list(recs[0]['names'])
    Xs = []
    for r in recs:
        X = np.asarray(r['X'], np.float64).copy()
        nm = list(r['names'])
        if nm != names:
            X = X[:, [nm.index(n) for n in names]]
        X[:, names.index('c_label')] = [vocab.get(l, -1) for l in r['labels']]
        Xs.append(X)
    return Xs, names


def exact_value(key, ctg0, w):
    """Exact value of an outcome key (the outcome rule's scale): contact / STOP away 0 (+ later = slightly better),
    running 2 + 0.5 x the legal route gain share, goal 3 (+ sooner = slightly better)."""
    c, v = key
    if c == 0.:
        return 0.5*float(v)/w
    if c == 3.:
        return 3.+0.5*(1.+float(v)/w)
    ctg = -float(v)
    g = (ctg0-ctg)/max(ctg0, 1e-6) if math.isfinite(ctg0) else 0.
    return 2.+0.5*float(np.clip(g, 0., 0.999))


def fit(Xs, rels, params=None, rounds=ROUNDS, names=None):
    """One LightGBM lambdarank booster; ``params['n_seeds']`` > 1: a list of boosters (seeds 0..n-1, scores averaged)."""
    import lightgbm as lgb
    p = dict(PARAMS)
    p.update(params or {})
    n_seeds = int(p.pop('n_seeds', 1))
    X = np.concatenate(Xs)
    y = np.concatenate([np.asarray(r, np.float64) for r in rels])
    ds = lgb.Dataset(X, label=y, group=[len(x) for x in Xs], feature_name=names or 'auto',
                     categorical_feature=['c_label'] if names and 'c_label' in names else 'auto', free_raw_data=False)
    if n_seeds <= 1:
        return lgb.train(p, ds, num_boost_round=int(rounds))
    return [lgb.train(dict(p, seed=int(p.get('seed', 0))+k), ds, num_boost_round=int(rounds)) for k in range(n_seeds)]


def predict(model, Xs):
    X = np.concatenate(Xs)
    ms = model if isinstance(model, list) else [model]
    s = np.mean([m.predict(X, num_threads=PARAMS['num_threads']) for m in ms], 0)
    out, i = [], 0
    for x in Xs:
        out.append(np.asarray(s[i:i+len(x)], np.float64)); i += len(x)
    return out


def choose(s, dflt, thr):
    best = int(np.argmax(s))
    return best if s[best]-s[dflt] > thr else int(dflt)


def decision_metrics(recs, choices, w):
    n = len(recs)
    dev = coll = crit = fix = brk = best = 0
    val = 0.
    for r, ch in zip(recs, choices):
        K = [tuple(k) for k in r['keys'][w]]
        cl = np.asarray([k[0] == 0. for k in K])
        df = int(r['default'])
        dev += int(ch != df)
        coll += int(cl[ch])
        if cl[df] and not cl.all():
            crit += 1
            fix += int(not cl[ch])
        if not cl[df] and cl[ch]:
            brk += 1
        best += int(K[ch] == max(K))
        val += exact_value(K[ch], r['ctg0'], w)
    return dict(states=n, deviation=dev/max(n, 1), chosen_collides=coll, critical=crit, fixed=fix, broken=brk,
                exact_best=best/max(n, 1), value=val/max(n, 1))


def within_auc(recs, risks, w):
    """Mean within-state AUC (mixed states) of ``risks`` (higher = riskier) for a real contact within ``w``."""
    aucs = []
    for r, x in zip(recs, risks):
        cl = np.asarray([tuple(k)[0] == 0. for k in r['keys'][w]])
        if cl.all() or not cl.any():
            continue
        a, b = np.asarray(x)[cl], np.asarray(x)[~cl]
        gt = (a[:, None] > b[None, :]).mean()
        eq = (a[:, None] == b[None, :]).mean()
        aucs.append(gt+0.5*eq)
    return (float(np.mean(aucs)) if aucs else float('nan')), len(aucs)


def tune_threshold(recs, scores, w=64, grid=None):
    """Deviation threshold maximising the mean exact value at ``w`` (ties: the larger threshold)."""
    margins = np.asarray([s.max()-s[int(r['default'])] for r, s in zip(recs, scores)])
    if grid is None:
        grid = np.unique(np.concatenate([[0.], np.quantile(margins, np.linspace(0, 0.99, 100))]))
    best = None
    for t in grid:
        ch = [choose(s, int(r['default']), t) for r, s in zip(recs, scores)]
        v = decision_metrics(recs, ch, w)['value']
        if best is None or v > best[1]+1e-12 or (abs(v-best[1]) <= 1e-12 and t > best[0]):
            best = (float(t), v)
    return best


def summary(recs, scores, thr):
    out = {}
    for w in (32, 64):
        rk = within_auc(recs, [-s for s in scores], w)
        oc16 = within_auc(recs, [-np.asarray(r['rule'][16]['oc']) for r in recs], w)
        oc48 = within_auc(recs, [-np.asarray(r['rule'][48]['oc']) for r in recs], w)
        out['auc%d' % w] = dict(ranker=rk[0], rule_oc16=oc16[0], rule_oc48=oc48[0], mixed_states=rk[1])
        sel = [choose(s, int(r['default']), thr) for r, s in zip(recs, scores)]
        amax = [int(np.argmax(s)) for s in scores]
        out['dec%d' % w] = dict(
            default=decision_metrics(recs, [int(r['default']) for r in recs], w),
            rule16=decision_metrics(recs, [int(r['choice'][16]) for r in recs], w),
            rule48=decision_metrics(recs, [int(r['choice'][48]) for r in recs], w),
            ranker_argmax=decision_metrics(recs, amax, w),
            ranker_selector=decision_metrics(recs, sel, w),
            oracle=decision_metrics(recs, [int(np.argmax([exact_value(tuple(k), r['ctg0'], w) for k in r['keys'][w]]))
                                           for r in recs], w))
    out['thr'] = thr
    return out


def rels_of(recs, w=64, **kw):
    return [relevance([tuple(k) for k in r['keys'][w]], **kw) for r in recs]


def run_cv(recs, vocab, params=None, rounds=ROUNDS, inner=True, label_w=64, tune_w=64, verbose=True):
    Xs, names = matrix(recs, vocab)
    rels = rels_of(recs, label_w)
    folds = np.asarray([fold_of(r['key']) for r in recs])
    oof = [None]*len(recs)
    per_fold = []
    for f in range(FOLDS):
        tr = np.flatnonzero(folds != f)
        te = np.flatnonzero(folds == f)
        thr = 0.
        if inner:
            # inner grouped CV on the training split only: out-of-fold scores -> threshold
            sub = [recs[i] for i in tr]
            ifold = np.asarray([zlib.crc32(('inner:'+str(recs[i]['key'])).encode()) % 4 for i in tr])
            ioof = [None]*len(tr)
            for g in range(4):
                a = np.flatnonzero(ifold != g); b = np.flatnonzero(ifold == g)
                m = fit([Xs[tr[i]] for i in a], [rels[tr[i]] for i in a], params, rounds, names)
                for i, s in zip(b, predict(m, [Xs[tr[i]] for i in b])):
                    ioof[i] = s
            thr = tune_threshold(sub, ioof, tune_w)[0]
        m = fit([Xs[i] for i in tr], [rels[i] for i in tr], params, rounds, names)
        sc = predict(m, [Xs[i] for i in te])
        for i, s in zip(te, sc):
            oof[i] = s
        fs = summary([recs[i] for i in te], sc, thr)
        fs['fold'] = f
        fs['n_train'] = int(len(tr)); fs['n_test'] = int(len(te))
        per_fold.append(fs)
        if verbose:
            print('fold %d: train %d test %d thr %.3f | AUC32 %.3f (oc16 %.3f) AUC64 %.3f (oc16 %.3f) | value64 sel %.4f rule16 %.4f def %.4f'
                  % (f, len(tr), len(te), thr, fs['auc32']['ranker'], fs['auc32']['rule_oc16'], fs['auc64']['ranker'],
                     fs['auc64']['rule_oc16'], fs['dec64']['ranker_selector']['value'], fs['dec64']['rule16']['value'],
                     fs['dec64']['default']['value']), flush=True)
    return oof, per_fold, names


def fmt_dec(d):
    return ('dev %3.0f%% | collides %3d | crit %3d fixed %3d | broken %3d | exact best %4.1f%% | value %.4f'
            % (100*d['deviation'], d['chosen_collides'], d['critical'], d['fixed'], d['broken'], 100*d['exact_best'], d['value']))


def report(s, title):
    lines = ['== %s (%d states; threshold %.4f)' % (title, s['dec64']['default']['states'], s['thr'])]
    for w in (32, 64):
        a = s['auc%d' % w]
        lines.append('  within-state AUC, real contact within %d steps (%d mixed states): ranker %.3f | rule -oc16 %.3f | rule -oc48 %.3f'
                     % (w, a['mixed_states'], a['ranker'], a['rule_oc16'], a['rule_oc48']))
    for w in (32, 64):
        lines.append('  decisions vs exact outcome within %d steps:' % w)
        for k in ('default', 'rule16', 'rule48', 'ranker_argmax', 'ranker_selector', 'oracle'):
            lines.append('    %-16s %s' % (k, fmt_dec(s['dec%d' % w][k])))
    return '\n'.join(lines)


def save_model(model, names, vocab, thr, out, info):
    """``model.txt`` (one booster) or ``model_<k>.txt`` (a seed ensemble; scores averaged) + ``meta.json``."""
    Path(out).mkdir(parents=True, exist_ok=True)
    ms = model if isinstance(model, list) else [model]
    files = ['model.txt'] if len(ms) == 1 else ['model_%d.txt' % k for k in range(len(ms))]
    for m, f in zip(ms, files):
        m.save_model(str(Path(out, f)))
        Path(out, f[:-4]+'.json').write_text(json.dumps(m.dump_model()))     # numpy fallback (riskrank.NumpyTrees)
    meta = dict(names=names, labels_vocab=vocab, thr=float(thr), models=files,
                hazard_params=json.loads(Path(HAZARD_PARAMS).read_text()), **info)
    Path(out, 'meta.json').write_text(json.dumps(meta, indent=1))


def load_model(path):
    import lightgbm as lgb
    meta = json.loads(Path(path, 'meta.json').read_text())
    ms = [lgb.Booster(model_file=str(Path(path, f))) for f in meta.get('models', ['model.txt'])]
    return (ms[0] if len(ms) == 1 else ms), meta


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cv')
    c.add_argument('--feat', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--model', default=None, help='also fit and save the frozen model (all TRAIN states)')
    c.add_argument('--rounds', type=int, default=ROUNDS)
    c.add_argument('--params', default='{}')
    c.add_argument('--label-w', type=int, default=64)
    c.add_argument('--tune-w', type=int, default=64)
    c.add_argument('--no-inner', action='store_true')
    e = sub.add_parser('evaluate')
    e.add_argument('--model', required=True)
    e.add_argument('--feat', required=True)
    e.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'cv':
        t0 = time.perf_counter()
        recs = load(a.feat)
        vocab = vocab_of(recs)
        params = (dict(PRESETS[a.params]) if a.params in PRESETS else
                  json.loads(Path(a.params).read_text()) if a.params.endswith('.json') else json.loads(a.params))
        oof, per_fold, names = run_cv(recs, vocab, params, a.rounds, not a.no_inner, a.label_w, a.tune_w)
        thr_all = tune_threshold(recs, oof, a.tune_w)[0]
        pooled = summary(recs, oof, thr_all)
        # pooled selector = each fold's own (inner-tuned) threshold applied to its test states
        folds = np.asarray([fold_of(r['key']) for r in recs])
        sel = [choose(s, int(r['default']), per_fold[int(folds[i])]['thr']) for i, (r, s) in enumerate(zip(recs, oof))]
        for w in (32, 64):
            pooled['dec%d' % w]['ranker_selector'] = decision_metrics(recs, sel, w)
        pooled['thr'] = float('nan')
        Path(a.out).mkdir(parents=True, exist_ok=True)
        text = [report(f, 'TRAIN CV fold %d (test episodes crc32(key)%%5 == %d; train %d states)' % (f['fold'], f['fold'], f['n_train']))
                for f in per_fold]
        text.append(report(pooled, 'TRAIN CV pooled out-of-fold (selector: each fold its inner-CV threshold)'))
        text.append('final threshold (tuned on the pooled out-of-fold scores): %.4f' % thr_all)
        Path(a.out, 'cv.txt').write_text('\n'.join(text)+'\n')
        Path(a.out, 'cv.json').write_text(json.dumps(dict(per_fold=per_fold, pooled=pooled, thr_final=thr_all,
                                                          params=dict(PARAMS, **params), rounds=a.rounds,
                                                          label_w=a.label_w, tune_w=a.tune_w, n=len(recs)), indent=1))
        with open(Path(a.out, 'oof.pkl'), 'wb') as fh:
            pickle.dump(dict(files=[r['file'] for r in recs], scores=oof), fh)
        print('\n'.join(text))
        if a.model:
            Xs, names = matrix(recs, vocab)
            m = fit(Xs, rels_of(recs, a.label_w), params, a.rounds, names)
            settings = recs[0].get('settings', {})
            save_model(m, names, vocab, thr_all, a.model, dict(
                settings=settings, rounds=a.rounds, params=dict(PARAMS, **params), label_w=a.label_w, tune_w=a.tune_w,
                train=dict(feat=a.feat, states=len(recs), episodes=len({r['key'] for r in recs})),
                note='LightGBM lambdarank on TRAIN exact outcomes (labels only); LEGAL features (riskrank.features)'))
            gains = np.mean([b.feature_importance('gain') for b in (m if isinstance(m, list) else [m])], 0)
            imp = sorted(zip(gains, names), reverse=True)[:30]
            Path(a.model, 'importance.txt').write_text('\n'.join('%-22s %.1f' % (n, g) for g, n in imp)+'\n')
            print('model -> %s (thr %.4f)' % (a.model, thr_all))
        print('%.0f s' % (time.perf_counter()-t0))
    elif a.cmd == 'evaluate':
        m, meta = load_model(a.model)
        recs = load(a.feat)
        Xs, names = matrix(recs, meta['labels_vocab'], meta['names'])
        sc = predict(m, Xs)
        s = summary(recs, sc, float(meta['thr']))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        text = report(s, 'DEV (frozen model %s, one pass)' % a.model)
        Path(a.out, 'dev.txt').write_text(text+'\n')
        Path(a.out, 'dev.json').write_text(json.dumps(s, indent=1))
        print(text)


if __name__ == '__main__':
    main()
