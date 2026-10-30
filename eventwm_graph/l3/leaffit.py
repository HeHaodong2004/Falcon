"""Learned long-horizon value on candidate leaves: fitting and candidate-ranking evaluation (analysis).

Data: ``leafdata.py`` rows (one per labelled candidate: legal / privileged features at the decision and at the exact
leaf after the candidate's interval; the exact outcome of candidate + default policy to the episode end).

Models (LightGBM, binary objective P(success); scene-grouped cross-validation):
  Q_state       decision-state legal features + candidate family        (a critic WITHOUT imagination)
  V_leaf        legal features of the exact leaf                         (perfect 0.77 s imagination + learned value)
  V_leaf_ctx    leaf + decision-state legal features
  Q_state_priv  Q_state + privileged features at the decision            (privileged critic, no imagination)
  V_leaf_priv   V_leaf_ctx + privileged features at the leaf             (privileged critic on the exact leaf)
Heuristics (no learning): leaf progress, leaf progress with a closest-approach penalty.

Ranking metrics per held-out state: critical top-1 (states whose default fails while some candidate succeeds: does
the top-scored candidate succeed?), within-state pairwise AUC (success vs failure pairs), and a deviation-policy proxy
over all states (take the top candidate when it beats the default by a margin).

    PYTHONPATH=<lightgbm dir> python scripts/l3_run.py eventwm_graph.l3.leaffit --data output/leafdata/R10 \
        [--folds 5 --fractions 0.25,0.5,1 --save output/leafvalue/v1]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(data_dirs):
    from .leafvalue import FAMILY_KEYS, family
    parts = []
    for d in str(data_dirs).split(','):
        for f in sorted(Path(d).glob('*.npz')):
            z = np.load(f, allow_pickle=False)
            if not len(z['y']):
                continue
            ep = str(z['episode'])
            scene = ep.split('__')[0].rsplit('_', 1)[0]
            n = len(z['y'])
            fam = np.asarray([[float(family(l) == k) for k in FAMILY_KEYS] for l in z['label']], np.float32)
            extra = {}
            if 'Xl32' in z.files and len(z['Xl32']) == n:
                extra = dict(Xl32=z['Xl32'], Pl32=z['Pl32'], t32=z['t32'].astype(int))
            parts.append(dict(Xs=z['Xs'], Xl=z['Xl'], Ps=z['Ps'], Pl=z['Pl'], fam=fam, y=z['y'].astype(int), **extra,
                              steps=z['steps'].astype(int), dflt=z['is_default'].astype(bool),
                              group=np.asarray(['%s#%d' % (ep, k) for k in z['decision']]),
                              scene=np.asarray([scene]*n), config=np.asarray([str(z['config'])]*n),
                              label=z['label'].astype(str)))
    keys = set(parts[0])
    for p in parts:
        keys &= set(p)
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return out


def utility(y, steps):
    """Exact outcome order of the rollout reference: success (sooner better) > alive > timeout > failure (later)."""
    u = np.zeros(len(y))
    u[y == 4] = 3.+0.5*(1.-steps[y == 4]/500.)
    u[y == 3] = 2.
    u[y == 2] = 1.
    f = (y == 0) | (y == 1)
    u[f] = 0.5*steps[f]/500.
    return u


FEATURE_SETS = {
    'Q_state': ('Xs', 'fam'),
    'V_leaf': ('Xl',),
    'V_leaf_ctx': ('Xl', 'Xs'),
    'Q_state_priv': ('Xs', 'fam', 'Ps'),
    'V_leaf_priv': ('Xl', 'Xs', 'Pl'),
    'V_leaf32': ('Xl32',),
    'V_leaf32_ctx': ('Xl32', 'Xs'),
    'V_leaf32_priv': ('Xl32', 'Xs', 'Pl32'),
}


def matrix(D, keys, idx=None):
    X = np.concatenate([D[k] for k in keys], 1)
    return X if idx is None else X[idx]


def fit(X, y, rounds=300, seed=0):
    import lightgbm as lgb
    params = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=int(__import__('os').environ.get('L3_LGB_THREADS', '2')))
    return lgb.train(params, lgb.Dataset(X, y.astype(int)), rounds)


def heuristic_scores(D, idx, name):
    from .leafvalue import LEGAL_NAMES
    Xl = D['Xl'][idx]
    col = {n: i for i, n in enumerate(LEGAL_NAMES)}
    prog = Xl[:, col['progress']]
    if name == 'progress':
        return prog-10.*Xl[:, col['terminal_collision']]
    if name == 'progress32':
        # the exact short-horizon reference's rule on the 32-step leaf: collision first, then legal route progress
        X2 = D['Xl32'][idx]
        return X2[:, col['progress']]-100.*X2[:, col['terminal_collision']]+100.*X2[:, col['terminal_success']]
    if name == 'progress_safe':
        pen = np.clip(0.6-Xl[:, col['min_dca']], 0., None)
        return prog-10.*Xl[:, col['terminal_collision']]-8.*pen
    raise ValueError(name)


def rank_metrics(D, idx, score, margins=(0., 0.05, 0.1, 0.2)):
    """Within-state ranking metrics of ``score`` (aligned with ``idx``)."""
    g = D['group'][idx]
    succ = D['y'][idx] == 4
    dflt = D['dflt'][idx]
    u = utility(D['y'][idx], D['steps'][idx])
    order = np.argsort(g, kind='stable')
    bounds = np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1], True])
    crit_hit, crit_rand, crit_n = 0., 0., 0
    pairs, conc = 0, 0.
    states = 0
    pol = {m: [0., 0., 0] for m in margins}          # chosen success, default success, deviations
    regret = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = order[a:b]
        if len(j) < 2 or not dflt[j].any():
            continue
        states += 1
        s, ok, d = score[j], succ[j], int(np.flatnonzero(dflt[j])[0])
        top = int(np.argmax(s))
        regret.append(float(u[j].max()-u[j][top]))
        if not ok[d] and ok.any():
            crit_n += 1
            crit_hit += float(ok[top])
            alt = np.delete(np.arange(len(j)), d)
            crit_rand += float(ok[alt].mean())
        pos, neg = s[ok], s[~ok]
        if len(pos) and len(neg):
            cmp = (pos[:, None] > neg[None]).astype(float)+0.5*(pos[:, None] == neg[None])
            conc += cmp.sum(); pairs += cmp.size
        for m in margins:
            k = top if s[top] > s[d]+m else d
            pol[m][0] += float(ok[k]); pol[m][1] += float(ok[d]); pol[m][2] += int(k != d)
    return dict(states=states, critical=crit_n, crit_top1=crit_hit/max(crit_n, 1), crit_random=crit_rand/max(crit_n, 1),
                pair_auc=conc/max(pairs, 1), regret=float(np.mean(regret)) if regret else math.nan,
                policy={str(m): dict(success=v[0]/max(states, 1), default=v[1]/max(states, 1),
                                     deviation=v[2]/max(states, 1)) for m, v in pol.items()})


def target(D, name):
    """'success': eventual success of candidate + default (the labels); 'nocoll32': no collision within 32 steps of
    the decision (a short-horizon exact target)."""
    if name == 'success':
        return (D['y'] == 4).astype(int)
    if name == 'nocoll32':
        return (D['t32'] != 0).astype(int)
    raise ValueError(name)


def crossval(D, folds=5, fractions=(1.,), seed=0, sets=None, rounds=300, tname='success'):
    rng = np.random.default_rng(seed)
    scenes = np.unique(D['scene'])
    rng.shuffle(scenes)
    fold_of = {s: i % folds for i, s in enumerate(scenes)}
    f = np.asarray([fold_of[s] for s in D['scene']])
    succ = target(D, tname)
    res = {}
    for frac in fractions:
        for name in (sets or FEATURE_SETS):
            if name in FEATURE_SETS and any(k not in D for k in FEATURE_SETS[name]):
                continue
            per = []
            for k in range(folds):
                te = np.flatnonzero(f == k)
                tr_scenes = [s for s in scenes if fold_of[s] != k]
                tr_scenes = tr_scenes[:max(1, int(round(frac*len(tr_scenes))))]
                tr = np.flatnonzero(np.isin(D['scene'], tr_scenes))
                if name in ('progress', 'progress_safe', 'progress32'):
                    if name == 'progress32' and 'Xl32' not in D:
                        break
                    sc = heuristic_scores(D, te, name)
                else:
                    keys = FEATURE_SETS[name]
                    b = fit(matrix(D, keys, tr), succ[tr], rounds=rounds, seed=seed+k)
                    sc = b.predict(matrix(D, keys, te))
                m = rank_metrics(D, te, sc)
                m['train_rows'] = int(len(tr))
                per.append(m)
            if not per:
                continue
            agg = _aggregate(per)
            res['%s@%.2f' % (name, frac)] = agg
            print('%-14s frac %.2f  crit top-1 %.3f (random %.3f, n=%d)  pair AUC %.3f  regret %.3f  policy@0.05 %.3f vs '
                  'default %.3f (dev %.2f)' % (name, frac, agg['crit_top1'], agg['crit_random'], agg['critical'],
                                                agg['pair_auc'], agg['regret'], agg['policy']['0.05']['success'],
                                                agg['policy']['0.05']['default'], agg['policy']['0.05']['deviation']),
                  flush=True)
    return res


def _aggregate(per):
    """Pool fold metrics weighted by their state counts (critical metrics by critical counts)."""
    n = np.asarray([p['states'] for p in per], float)
    c = np.asarray([p['critical'] for p in per], float)
    out = dict(states=int(n.sum()), critical=int(c.sum()),
               crit_top1=float((c*[p['crit_top1'] for p in per]).sum()/max(c.sum(), 1)),
               crit_random=float((c*[p['crit_random'] for p in per]).sum()/max(c.sum(), 1)),
               pair_auc=float(np.mean([p['pair_auc'] for p in per])),
               regret=float((n*[p['regret'] for p in per]).sum()/max(n.sum(), 1)), policy={})
    for m in per[0]['policy']:
        out['policy'][m] = {k: float((n*[p['policy'][m][k] for p in per]).sum()/max(n.sum(), 1))
                            for k in ('success', 'default', 'deviation')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='output/leafdata/R10')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fractions', default='1')
    ap.add_argument('--sets', default='progress,progress_safe,progress32,' + ','.join(FEATURE_SETS))
    ap.add_argument('--saveonly', action='store_true', help='skip the cross-validation')
    ap.add_argument('--imag', default=None, help='leafimag.py output: compare exact and imagined 32-step leaves')
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--save', default=None, help='fit V_leaf on ALL rows and save the model here (selector)')
    ap.add_argument('--save-set', default='V_leaf')
    ap.add_argument('--target', default='success', help='success | nocoll32 (training target; ranking metrics use '
                                                         'the eventual outcome)')
    a = ap.parse_args()
    D = load(a.data)
    succ = D['y'] == 4
    print('rows %d states %d scenes %d success %.3f collision %.3f timeout %.3f' % (
        len(D['y']), len(np.unique(D['group'])), len(np.unique(D['scene'])), succ.mean(), (D['y'] == 0).mean(),
        (D['y'] == 2).mean()), flush=True)
    if a.imag:
        imagined_comparison(D, a.imag, a.folds, a.seed, a.rounds)
        return
    res = {} if a.saveonly else crossval(D, a.folds, tuple(float(x) for x in a.fractions.split(',')), a.seed, a.sets.split(','), a.rounds,
                   a.target)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    if a.save:
        b = fit(matrix(D, FEATURE_SETS[a.save_set]), target(D, a.target), rounds=a.rounds, seed=a.seed)
        Path(a.save).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save, 'model.txt')))
        Path(a.save, 'model.json').write_text(json.dumps(b.dump_model()))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(set=a.save_set, keys=FEATURE_SETS[a.save_set],
                                                             rows=int(len(succ)), data=a.data)))



# ------------------------------------------------------------------------------------------------ imagined leaves
def load_imag(imag_dir, D):
    """Align ``leafimag.py`` rows with the exact-leaf rows of ``D`` by (episode, decision, label); returns the index
    into D of every aligned row and the imagined arrays (Xi [n, W, F], wi, ci, si)."""
    key = {(g.split('#')[0], int(g.split('#')[1]), l): i for i, (g, l) in enumerate(zip(D['group'], D['label']))}
    idx, Xi, wi, ci, si = [], [], [], [], []
    for f in sorted(Path(imag_dir).glob('*.npz')):
        z = np.load(f, allow_pickle=False)
        ep = str(z['episode'])
        for r in range(len(z['y'])):
            k = (ep, int(z['decision'][r]), str(z['label'][r]))
            if k in key:
                idx.append(key[k]); Xi.append(z['Xi'][r]); wi.append(z['wi'][r]); ci.append(z['ci'][r]); si.append(z['si'][r])
    return np.asarray(idx), np.asarray(Xi), np.asarray(wi), np.asarray(ci), np.asarray(si)


def imagined_comparison(D, imag_dir, folds=5, seed=0, rounds=300):
    """Same states, same labels, same learner: exact vs imagined 32-step leaves."""
    idx, Xi, wi, ci, si = load_imag(imag_dir, D)
    if not len(idx):
        print('no aligned imagined rows')
        return
    # restrict to complete states (every labelled candidate of the state imagined)
    groups = D['group'][idx]
    full = {g for g in np.unique(groups) if (groups == g).sum() == (D['group'] == g).sum()}
    keep = np.asarray([g in full for g in groups])
    idx, Xi, wi, ci, si = idx[keep], Xi[keep], wi[keep], ci[keep], si[keep]
    print('aligned rows %d in %d complete states' % (len(idx), len(full)))
    rng = np.random.default_rng(seed)
    scenes = np.unique(D['scene'])
    rng.shuffle(scenes)
    fold_of = {s: i % folds for i, s in enumerate(scenes)}
    f_all = np.asarray([fold_of[s] for s in D['scene']])
    succ = target(D, 'success')
    sel = np.zeros(len(D['y']), bool); sel[idx] = True
    pos = {int(j): r for r, j in enumerate(idx)}
    W = wi/np.maximum(wi.sum(1, keepdims=True), 1e-9)
    from .leafvalue import LEGAL_NAMES
    col = {n: i for i, n in enumerate(LEGAL_NAMES)}
    scores = {k: np.zeros(len(idx)) for k in ('exact_progress32', 'exact_V32', 'imag_rule', 'imag_V32exact', 'imag_V32imag')}
    for k in range(folds):
        te_rows = np.flatnonzero((f_all[idx] == k))
        tr = np.flatnonzero((f_all != k))
        if not len(te_rows):
            continue
        te = idx[te_rows]
        # exact leaf: heuristic and learned value (trained on all exact rows of the other folds)
        X2 = D['Xl32'][te]
        scores['exact_progress32'][te_rows] = X2[:, col['progress']]-100.*X2[:, col['terminal_collision']]+100.*X2[:, col['terminal_success']]
        bx = fit(D['Xl32'][tr], succ[tr], rounds=rounds, seed=seed+k)
        scores['exact_V32'][te_rows] = bx.predict(D['Xl32'][te])
        # imagined leaves: the legal 1.5 s rule (contact -> 0, STOP at goal -> top, else progress), the exact-trained
        # value applied per world, and a value trained on imagined leaves of the other folds
        Xw = Xi[te_rows]                                        # [n, W, F]
        n, nw, F = Xw.shape
        prog = Xw[:, :, col['progress']]
        c = ci[te_rows] > 0.5
        stop_goal = si[te_rows] == 1
        stop_bad = si[te_rows] == 2
        ww = W[te_rows]
        rule = np.where(c | stop_bad, -100., np.where(stop_goal, 100., prog))
        scores['imag_rule'][te_rows] = (ww*rule).sum(1)
        vx = bx.predict(Xw.reshape(-1, F)).reshape(n, nw)
        scores['imag_V32exact'][te_rows] = (ww*np.where(c | stop_bad, 0., np.where(stop_goal, 1., vx))).sum(1)
        tr_i = np.flatnonzero(f_all[idx] != k)
        Xtr = Xi[tr_i].reshape(-1, F)
        ytr = np.repeat(succ[idx[tr_i]], nw)
        wtr = W[tr_i].reshape(-1)
        ok = wtr > 0
        bi = fit(Xtr[ok], ytr[ok], rounds=rounds, seed=seed+k)
        vi = bi.predict(Xw.reshape(-1, F)).reshape(n, nw)
        scores['imag_V32imag'][te_rows] = (ww*np.where(c | stop_bad, 0., np.where(stop_goal, 1., vi))).sum(1)
    for name, sc in scores.items():
        m = rank_metrics(D, idx, sc)
        print('%-18s crit top-1 %.3f (random %.3f, n=%d)  pair AUC %.3f  regret %.3f  policy@0.05 %.3f vs default %.3f '
              '(dev %.2f)' % (name, m['crit_top1'], m['crit_random'], m['critical'], m['pair_auc'], m['regret'],
                              m['policy']['0.05']['success'], m['policy']['0.05']['default'], m['policy']['0.05']['deviation']))


if __name__ == '__main__':
    main()
