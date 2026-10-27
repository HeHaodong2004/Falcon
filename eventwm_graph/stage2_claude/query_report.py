"""Metrics, scene bootstrap and report for the query-ranking benchmark (query_bench.py outputs).

Definitions (fixed before any result was inspected):
  comparison sets  : '1.0' (primary: each Query truncated at its saved node within 0.05 s of t0 + 1 s), '0.5', '2.0',
                     'full' (complete executed outcomes; durations differ). A Root is eligible in a set with >= 2
                     distinct outcomes (Queries ending at the same node are one outcome).
  actual gain G    : initially UNKNOWN cells (original candidate domain) observed FREE or BLOCKED by the outcome.
  predicted gain   : model = sum of FREE + BLOCKED probabilities over the same domain ('expected');
                     model:hard = cells predicted FREE or BLOCKED at the frozen operating point.
  baselines        : random (exact expectation), trajectory distance / duration / heading change, legal-map frustum,
                     legal visibility and known-visible counts (query_bench heuristic) — no learning, no privileged data.
  pairs            : decisive if |dG| >= max(3, 0.1 max(G_i, G_j)); near ties (0 < |dG| below that) and exact ties are
                     reported separately; a predicted tie scores 0.5. Principal = Root-macro accuracy over decisive pairs.
  regret           : G_best - G_selected (selected = argmax predicted; ties in the prediction -> mean over the tied set);
                     normalized by (G_best - G_min) only for Roots with range >= 3 cells; hit = exact best,
                     near hit = within max(3, 0.1 G_best).
  bootstrap        : 1,000 scene resamples (seed 0) of Root-level values; paired differences resample the same scenes.
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .query_bench import DEST

SETS = ('1.0', '0.5', '2.0', 'full')
MODELS = ('v7', 'v11', 'v12', 'v15', 'v16b', 'v17a')
FULL_DEV_IOU = dict(v7=0.4563, v11=0.4688, v12=0.4716, v15=0.4784, v16b=0.4809, v17a=None)
TAU_ABS, TAU_REL = 3, .1
B = 1000


def train_val_calibration(name):
    """Binned (20 revelation x 10 BLOCKED-given-revelation) recalibration fitted on the model's 400-Root TRAIN-validation
    selection histogram (select.py cache; 128 px packets) -> (P(FREE | bin), P(BLOCKED | bin)) arrays of shape (20, 10).
    Sparse bins fall back to the revelation-bin marginal. Development data is not used."""
    import hashlib, pickle
    from eventwm_graph.stage2_repair.diagnose import sha
    from .select import RB as SRB, BB as SBB
    sel = DEST.parent/'selection'/(name+'.json')
    if not sel.exists():
        return None
    ck = json.loads(sel.read_text())['checkpoint']
    key = hashlib.sha256(('%s:%d:%s:%s' % (sha(ck), 400, SRB*1000+SBB, '')).encode()).hexdigest()[:16]
    path = DEST.parent/'selection'/('hist_%s.pkl' % key)
    if not path.exists():
        return None
    recs = pickle.loads(path.read_bytes())
    H = np.zeros((3, SRB, SBB))
    for _, root in recs:
        for h, _m in root:
            H += h
    H = H.reshape(3, 20, SRB//20, 10, SBB//10).sum((2, 4))           # -> (3, 20, 10)
    n = H.sum(0)
    marg = H.sum(2); mn = marg.sum(0)
    pf_m = marg[1]/np.maximum(mn, 1); pb_m = marg[2]/np.maximum(mn, 1)
    pf = np.where(n >= 200, H[1]/np.maximum(n, 1), pf_m[:, None]); pb = np.where(n >= 200, H[2]/np.maximum(n, 1), pb_m[:, None])
    return pf, pb


def load(name):
    return json.loads(gzip.decompress((DEST/'predictions'/(name+'.json.gz')).read_bytes()))


def rates(cm):
    cm = np.asarray(cm, np.float64)
    tp = cm[1, 1]+cm[2, 2]
    return tp/max(cm[1:].sum()+cm[:, 1:].sum()-tp, 1e-9)


def groups(records, set_name):
    """root -> list of outcome records in the comparison set (distinct outcomes)."""
    g = defaultdict(list)
    for r in records:
        if any(s[0] == set_name for s in r['sets']):
            g[r['root_id']].append(r)
    return {k: v for k, v in g.items() if len(v) >= 2}


def scores(r, method, heur):
    if method == 'distance':
        return r['distance']
    if method == 'duration':
        return r['duration']
    if method == 'heading_change':
        return r['rotation']
    if method in ('frustum', 'legal_visible', 'known_visible'):
        return heur[(r['root_id'], r['end_node'])][method]
    raise KeyError(method)


GAIN_KIND = ['total']


def gain(r):
    k = GAIN_KIND[0]
    return r['actual_free'] if k == 'free' else r['actual_blocked'] if k == 'blocked' else r['actual_free']+r['actual_blocked']


def nested_pairs(root_id, end_nodes):
    """Pairs (end_a, end_b) where one outcome's executed path is a prefix of the other's (full set only)."""
    from .query_bench import load_cache
    traj = {o['end_node']: o['trajectory'] for o in load_cache(root_id)['outcomes']}
    out = set()
    for a in end_nodes:
        for b in end_nodes:
            ta, tb = traj[a], traj[b]
            if a != b and len(ta) < len(tb) and np.array_equal(tb[:len(ta)], ta):
                out.add(frozenset((a, b)))
    return out


def root_stats(outs, pred, nested=()):
    """Pairwise / regret statistics of one Root (pred: predicted gains aligned with outs; None = random expectation).
    Pairs in `nested` (prefix-related outcomes) are counted separately and excluded from the pairwise scores."""
    G = np.array([gain(o) for o in outs], np.float64); n = len(G)
    res = dict(n=n, decisive=0, correct=0., near=0, near_correct=0., exact_ties=0, nested=0, nested_correct=0.,
               bins=defaultdict(lambda: [0, 0.]))
    for i in range(n):
        for j in range(i+1, n):
            d = G[i]-G[j]
            if frozenset((outs[i]['end_node'], outs[j]['end_node'])) in nested:
                if d != 0:
                    res['nested'] += 1
                    res['nested_correct'] += .5 if pred is None or pred[i] == pred[j] else float(np.sign(pred[i]-pred[j]) == np.sign(d))
                continue
            if d == 0:
                res['exact_ties'] += 1; continue
            tol = max(TAU_ABS, TAU_REL*max(G[i], G[j]))
            if pred is None:
                c = .5
            else:
                dp = pred[i]-pred[j]
                c = .5 if dp == 0 else float(np.sign(dp) == np.sign(d))
            if abs(d) < tol:
                res['near'] += 1; res['near_correct'] += c; continue
            res['decisive'] += 1; res['correct'] += c
            rel = abs(d)/max(G[i], G[j])
            key = '0.1-0.25' if rel < .25 else '0.25-0.5' if rel < .5 else '>=0.5'
            res['bins'][key][0] += 1; res['bins'][key][1] += c
    best = G.max(); rng = best-G.min()
    if pred is None:
        sel = G.mean(); hit = float((G == best).mean()); near_hit = float((best-G <= max(TAU_ABS, TAU_REL*best)).mean())
    else:
        p = np.asarray(pred, np.float64); top = p == p.max()
        sel = G[top].mean(); hit = float((G[top] == best).mean())
        near_hit = float((best-G[top] <= max(TAU_ABS, TAU_REL*best)).mean())
    res.update(best=best, regret=best-sel, range=rng, norm_regret=(best-sel)/rng if rng >= TAU_ABS else None,
               hit=hit, near_hit=near_hit)
    return res


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(values_by_scene, b=B, seed=0):
    """values_by_scene: scene -> list of Root values (None skipped). Mean over Roots of resampled scenes."""
    scenes = sorted(values_by_scene)
    sums = np.array([sum(v for v in values_by_scene[s] if v is not None) for s in scenes], np.float64)
    cnts = np.array([sum(v is not None for v in values_by_scene[s]) for s in scenes], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenes), (b, len(scenes)))
    with np.errstate(invalid='ignore'):
        stats = sums[idx].sum(1)/cnts[idx].sum(1)
    point = sums.sum()/max(cnts.sum(), 1)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return dict(mean=float(point), ci95=[float(lo), float(hi)], roots=int(cnts.sum()), scenes=int((cnts > 0).sum()))


def method_set_metrics(groups_, preds, scene_of, nested=None):
    """preds: root -> list aligned with groups_[root] (None = random expectation)."""
    per_root = {}
    for root, outs in groups_.items():
        per_root[root] = root_stats(outs, preds[root], (nested or {}).get(root, ()))
    by_scene = lambda f: _by_scene(per_root, scene_of, f)
    acc = by_scene(lambda s: s['correct']/s['decisive'] if s['decisive'] else None)
    out = dict(
        roots=len(per_root), scenes=len({scene_of[r] for r in per_root}),
        outcomes=int(sum(s['n'] for s in per_root.values())),
        decisive_pairs=int(sum(s['decisive'] for s in per_root.values())),
        near_tie_pairs=int(sum(s['near'] for s in per_root.values())),
        exact_tie_pairs=int(sum(s['exact_ties'] for s in per_root.values())),
        pairwise_root_macro=boot(acc),
        pairwise_pooled=float(sum(s['correct'] for s in per_root.values())/max(sum(s['decisive'] for s in per_root.values()), 1)),
        pairwise_scene_macro=float(np.mean([np.mean([v for v in vs if v is not None]) for vs in acc.values() if any(v is not None for v in vs)])),
        near_tie_accuracy=float(sum(s['near_correct'] for s in per_root.values())/max(sum(s['near'] for s in per_root.values()), 1)),
        by_relative_difference={k: dict(pairs=int(sum(s['bins'][k][0] for s in per_root.values())),
                                        accuracy=float(sum(s['bins'][k][1] for s in per_root.values())/max(sum(s['bins'][k][0] for s in per_root.values()), 1)))
                                for k in ('0.1-0.25', '0.25-0.5', '>=0.5')},
        regret_mean=boot(by_scene(lambda s: s['regret'])),
        regret_normalized_mean=boot(by_scene(lambda s: s['norm_regret'])),
        regret_median=float(np.median([s['regret'] for s in per_root.values()])),
        regret_p90=float(np.percentile([s['regret'] for s in per_root.values()], 90)),
        top1_hit=boot(by_scene(lambda s: s['hit'])),
        top1_near_hit=boot(by_scene(lambda s: s['near_hit'])),
        nested_pairs_excluded=int(sum(s['nested'] for s in per_root.values())),
        nested_pair_accuracy=float(sum(s['nested_correct'] for s in per_root.values())/max(sum(s['nested'] for s in per_root.values()), 1)),
        zero_range_roots=int(sum(s['range'] == 0 for s in per_root.values())),
        small_range_roots=int(sum(0 < s['range'] < TAU_ABS for s in per_root.values())))
    return out, per_root


def _by_scene(per_root, scene_of, f):
    d = defaultdict(list)
    for root, s in per_root.items():
        d[scene_of[root]].append(f(s))
    return d


def gain_metrics(records_in_set, scene_of):
    by_scene = defaultdict(list); rows = []
    for r in records_in_set:
        G = gain(r); P = r['exp_free']+r['exp_blocked']; H = r['hard_free']+r['hard_blocked']
        rows.append((G, P, H, r['actual_free'], r['exp_free'], r['actual_blocked'], r['exp_blocked']))
    a = np.array(rows, np.float64)
    G, P, H = a[:, 0], a[:, 1], a[:, 2]
    per_root_err = defaultdict(list)
    for r, (g, p) in zip(records_in_set, zip(G, P)):
        per_root_err[r['root_id']].append(abs(p-g))
    for root, errs in per_root_err.items():
        by_scene[scene_of[root]].append(float(np.mean(errs)))
    within = []
    roots = defaultdict(list)
    for r in records_in_set:
        roots[r['root_id']].append(r)
    for root, outs in roots.items():
        s = spearman([o['exp_free']+o['exp_blocked'] for o in outs], [gain(o) for o in outs])
        if s is not None:
            within.append(s)
    ious = np.array([rates(r['cm']) for r in records_in_set if gain(r) > 0 or (r['hard_free']+r['hard_blocked']) > 0])
    cm = sum(np.asarray(r['cm'], np.float64) for r in records_in_set); cmu = sum(np.asarray(r['cm_never_observed'], np.float64) for r in records_in_set)
    return dict(per_outcome_iou=dict(outcomes=int(len(ious)), mean=float(ious.mean()), median=float(np.median(ious)),
                                     p25=float(np.percentile(ious, 25)), p75=float(np.percentile(ious, 75)),
                                     share_below_0_3=float((ious < .3).mean()), share_above_0_6=float((ious > .6).mean())),
                region_iou=dict(pooled_all=float(rates(cm)), never_observed_cells=float(rates(cmu)), observed_cells=float(rates(cm-cmu)),
                                targets_in_never_observed=float(cmu[1:].sum()/max(cm[1:].sum(), 1))),
                outcomes=len(rows), actual_gain_mean=float(G.mean()), expected_gain_mean=float(P.mean()),
                mae_expected=float(np.abs(P-G).mean()), mae_expected_root_macro=boot(by_scene),
                bias_expected=float((P-G).mean()), relative_bias_expected=float((P-G).sum()/max(G.sum(), 1)),
                mae_free=float(np.abs(a[:, 4]-a[:, 3]).mean()), mae_blocked=float(np.abs(a[:, 6]-a[:, 5]).mean()),
                bias_free=float((a[:, 4]-a[:, 3]).mean()), bias_blocked=float((a[:, 6]-a[:, 5]).mean()),
                mae_hard=float(np.abs(H-G).mean()), bias_hard=float((H-G).mean()),
                pearson=float(np.corrcoef(P, G)[0, 1]), spearman=spearman(P, G),
                within_root_spearman_mean=float(np.mean(within)) if within else None, within_root_spearman_roots=len(within))


def false_free(records_in_set):
    cm = np.zeros((3, 3)); cmu = np.zeros((3, 3))
    for r in records_in_set:
        cm += np.asarray(r['cm']); cmu += np.asarray(r['cm_never_observed'])
    def block(c):
        pf = c[:, 1].sum()
        return dict(predicted_free=int(pf), free_true=int(c[1, 1]), free_but_blocked=int(c[2, 1]), free_but_unknown=int(c[0, 1]),
                    free_precision=float(c[1, 1]/max(pf, 1)), free_recall=float(c[1, 1]/max(c[1].sum(), 1)),
                    rate_free_but_blocked=float(c[2, 1]/max(pf, 1)), rate_free_but_unknown=float(c[0, 1]/max(pf, 1)),
                    blocked_precision=float(c[2, 2]/max(c[:, 2].sum(), 1)), blocked_recall=float(c[2, 2]/max(c[2].sum(), 1)),
                    label_aware_iou_pooled=float(rates(c)))
    return dict(all_cells=block(cm), never_observed_cells=block(cmu), observed_cells=block(cm-cmu))


def calibration(cal):
    out = {}
    nb = cal['bins']; centers = (np.arange(nb)+.5)/nb
    for k in ('revelation', 'free', 'blocked'):
        h = np.asarray(cal[k], np.float64)            # [bin, truth]
        n = h.sum(1); pos = h[:, 1]
        rate = np.where(n > 0, pos/np.maximum(n, 1), np.nan)
        ece = float(np.nansum(n/n.sum()*np.abs(rate-centers)))
        # binned AUC (ties within a bin count 0.5)
        neg = h[:, 0]; cum_neg = np.cumsum(neg)-neg
        auc = float((pos*(cum_neg+.5*neg)).sum()/max(pos.sum()*neg.sum(), 1))
        rel = centers >= .05                               # bins with predicted probability >= 0.05
        ece_rel = float(np.nansum((n*np.abs(rate-centers))[rel])/max(n[rel].sum(), 1))
        mean_pred_rel = float((n*centers)[rel].sum()/max(n[rel].sum(), 1)); obs_rel = float(pos[rel].sum()/max(n[rel].sum(), 1))
        out[k] = dict(ece=ece, ece_p_ge_005=ece_rel, mean_predicted_p_ge_005=mean_pred_rel, observed_p_ge_005=obs_rel,
                      auc_binned=auc, base_rate=float(pos.sum()/n.sum()),
                      reliability=[dict(bin=f'{i/nb:.2f}-{(i+1)/nb:.2f}', cells=int(n[i]), observed=None if np.isnan(rate[i]) else round(float(rate[i]), 4))
                                   for i in range(nb) if n[i] > 0])
    cells = float(np.asarray(cal['revelation']).sum())
    out['brier_revelation'] = cal['brier_sums'][0]/cells
    out['brier_multiclass'] = cal['brier_sums'][1]/cells
    return out


def regimes(records_in_set, groups_, preds_by_method, scene_of):
    """Map accuracy per outcome regime and decision accuracy per pair regime."""
    def bucket(r):
        d = r['duration']; x = r['distance']; h = r['rotation']
        return {'duration': '<=0.5' if d <= .5 else '0.5-1' if d <= 1 else '1-2' if d <= 2 else '>2',
                'distance': 'turn-only' if x < .05 else '0.05-0.5' if x < .5 else '0.5-1.5' if x < 1.5 else '>1.5',
                'heading_change': '<0.3' if h < .3 else '0.3-1' if h < 1 else '1-2' if h < 2 else '>2',
                'history': 'short' if r['history_seconds'] <= .632 else 'long',
                'unseen_direction': 'yes' if r['unseen_yaw_max'] > math.pi/4 else 'no'}
    acc = defaultdict(lambda: dict(cm=np.zeros((3, 3)), n=0, abs_err=0., G=0.))
    for r in records_in_set:
        for k, v in bucket(r).items():
            a = acc[(k, v)]; a['cm'] += np.asarray(r['cm']); a['n'] += 1
            a['abs_err'] += abs(r['exp_free']+r['exp_blocked']-gain(r)); a['G'] += gain(r)
    out = dict(map={})
    for (k, v), a in sorted(acc.items()):
        out['map'].setdefault(k, {})[v] = dict(outcomes=a['n'], label_aware_iou_pooled=float(rates(a['cm'])),
                                                gain_mae=a['abs_err']/max(a['n'], 1), actual_gain_mean=a['G']/max(a['n'], 1))
    # decision accuracy by the pair's regime (both outcomes of a pair in the same bucket)
    dec = {}
    for method, preds in preds_by_method.items():
        tally = defaultdict(lambda: [0, 0.])
        for root, outs in groups_.items():
            p = preds[root]
            for i in range(len(outs)):
                for j in range(i+1, len(outs)):
                    gi, gj = gain(outs[i]), gain(outs[j]); d = gi-gj
                    if d == 0 or abs(d) < max(TAU_ABS, TAU_REL*max(gi, gj)):
                        continue
                    c = .5 if p is None or p[i] == p[j] else float(np.sign(p[i]-p[j]) == np.sign(d))
                    bi, bj = bucket(outs[i]), bucket(outs[j])
                    for k in bi:
                        if bi[k] == bj[k]:
                            t = tally[(k, bi[k])]; t[0] += 1; t[1] += c
                    if abs(outs[i]['distance']-outs[j]['distance']) <= .2 and abs(outs[i]['rotation']-outs[j]['rotation']) <= .3:
                        t = tally[('kinematically_matched', 'dDistance<=0.2m,dHeading<=0.3rad')]; t[0] += 1; t[1] += c
        dec[method] = {f'{k}={v}': dict(pairs=n, accuracy=c/max(n, 1)) for (k, v), (n, c) in sorted(tally.items()) if n >= 30}
    out['decision_pairwise_by_regime'] = dec
    return out


def main(models=MODELS, out='report'):
    preds = {m: load(m) for m in models if (DEST/'predictions'/(m+'.json.gz')).exists()}
    heur_path = DEST/'predictions'/'heuristics.json.gz'
    heur = {(r['root_id'], r['end_node']): r for r in json.loads(gzip.decompress(heur_path.read_bytes()))['records']} if heur_path.exists() else {}
    base = next(iter(preds.values()))['records']
    recal = {}
    for m, p in preds.items():
        if all('hist_rb' in r for r in p['records'][:50]):
            c = train_val_calibration(m)
            if c is not None:
                recal[m] = c
    scene_of = {r['root_id']: r['scene_id'] for r in base}
    audit = json.loads((DEST/'audit.json').read_text())['summary']
    report = dict(definitions=__doc__, audit=audit, models={m: dict(full_dev_iou=FULL_DEV_IOU.get(m), checkpoint=p['checkpoint'],
                  sensor_directory=p['sensor_directory'], roots=p['roots']) for m, p in preds.items()}, sets={})
    per_root_out = {}
    for s in SETS:
        g_ref = groups(base, s)
        keyset = {(root, o['end_node']) for root, outs in g_ref.items() for o in outs}
        methods = {}
        for m, p in preds.items():
            g = groups(p['records'], s)
            assert {(root, o['end_node']) for root, outs in g.items() for o in outs} == keyset, 'identical evaluation groups'
            g = {root: sorted(outs, key=lambda o: o['end_node']) for root, outs in g.items()}
            methods[m] = (g, {root: [o['exp_free']+o['exp_blocked'] for o in outs] for root, outs in g.items()})
            methods[m+':hard'] = (g, {root: [o['hard_free']+o['hard_blocked'] for o in outs] for root, outs in g.items()})
            if m in recal:
                pf, pb = recal[m]
                methods[m+':recal'] = (g, {root: [float((np.asarray(o['hist_rb']).reshape(20, 10)*(pf+pb)).sum()) for o in outs] for root, outs in g.items()})
        g0 = {root: sorted(outs, key=lambda o: o['end_node']) for root, outs in g_ref.items()}
        methods['random'] = (g0, {root: None for root in g0})
        for h in ('distance', 'duration', 'heading_change')+(('frustum', 'legal_visible', 'known_visible') if heur else ()):
            methods[h] = (g0, {root: [scores(o, h, heur) for o in outs] for root, outs in g0.items()})
        res = dict(eligible=dict(roots=len(g0), scenes=len({scene_of[r] for r in g0}), outcomes=sum(len(v) for v in g0.values())), ranking={}, gain={},
                   false_free={}, regimes={}, paired_vs={})
        root_vals = {}
        nested = {root: nested_pairs(root, [o['end_node'] for o in outs]) for root, outs in g0.items()} if s == 'full' else None
        for name, (g, pr) in methods.items():
            res['ranking'][name], prs = method_set_metrics(g, pr, scene_of, nested)
            root_vals[name] = {r: (st['correct']/st['decisive'] if st['decisive'] else None, st['regret'], st['hit']) for r, st in prs.items()}
            per_root_out.setdefault(s, {})[name] = {r: dict(n=st['n'], decisive=st['decisive'], correct=st['correct'], regret=st['regret'],
                                                            hit=st['hit'], best=st['best'], range=st['range']) for r, st in prs.items()}
        for m, p in preds.items():
            recs = [o for outs in methods[m][0].values() for o in outs]
            res['gain'][m] = gain_metrics(recs, scene_of)
            res['false_free'][m] = false_free(recs)
            if m in recal:
                pf, pb = recal[m]
                rc = [dict(o, exp_free=float((np.asarray(o['hist_rb']).reshape(20, 10)*pf).sum()),
                           exp_blocked=float((np.asarray(o['hist_rb']).reshape(20, 10)*pb).sum())) for o in recs]
                res['gain'][m+':recal'] = gain_metrics(rc, scene_of)
        # paired scene bootstrap of differences against the reference methods
        for a in [m for m in preds]:
            for b_ in ('random', 'distance', 'duration', 'legal_visible', 'frustum', 'known_visible', 'v7', 'v11', 'v12', 'v15', 'v16b'):
                if b_ not in root_vals or b_ == a:
                    continue
                diffs = {}
                for k, label in ((0, 'pairwise'), (1, 'regret'), (2, 'top1_hit')):
                    d = defaultdict(list)
                    for r in g0:
                        va, vb = root_vals[a][r][k], root_vals[b_][r][k]
                        d[scene_of[r]].append(None if va is None or vb is None else va-vb)
                    diffs[label] = boot(d)
                res['paired_vs'][f'{a}-{b_}'] = diffs
        if s == '1.0':
            # ranking by FREE gain and by BLOCKED gain separately (models scored with the matching class sums)
            res['ranking_by_class'] = {}
            for kind, key in (('free', 'exp_free'), ('blocked', 'exp_blocked')):
                GAIN_KIND[0] = kind
                sub = {}
                for m in preds:
                    g = methods[m][0]
                    sub[m], _ = method_set_metrics(g, {root: [o[key] for o in outs] for root, outs in g.items()}, scene_of)
                for h in ('random', 'distance', 'legal_visible', 'frustum'):
                    if h in methods:
                        sub[h], _ = method_set_metrics(*methods[h], scene_of)
                res['ranking_by_class'][kind] = {m: dict(pairwise_root_macro=v['pairwise_root_macro'], top1_hit=v['top1_hit'],
                                                         regret_mean=v['regret_mean'], decisive_pairs=v['decisive_pairs']) for m, v in sub.items()}
            GAIN_KIND[0] = 'total'
        if s == '1.0' or s == 'full':
            focus = {k: v for k, v in methods.items() if k in ('v15', 'v16b', 'v17a', 'distance', 'duration', 'legal_visible', 'random')}
            ref_m = 'v16b' if 'v16b' in preds else next(iter(preds))
            res['regimes'] = regimes([o for outs in methods[ref_m][0].values() for o in outs], g0,
                                     {k: v[1] for k, v in focus.items()}, scene_of)
        report['sets'][s] = res
    report['calibration'] = {m: calibration(p['calibration']) for m, p in preds.items()}
    # per-scene results of the primary set (Root-macro within the scene)
    prim = per_root_out.get('1.0', {})
    scenes = defaultdict(dict)
    for name, roots in prim.items():
        for r, st in roots.items():
            d = scenes[scene_of[r]].setdefault(name, dict(roots=0, acc=[], regret=[]))
            d['roots'] += 1; d['regret'].append(st['regret'])
            if st['decisive']:
                d['acc'].append(st['correct']/st['decisive'])
    report['per_scene_primary'] = {sc: {n: dict(roots=d['roots'], pairwise=float(np.mean(d['acc'])) if d['acc'] else None,
                                                regret=float(np.mean(d['regret']))) for n, d in v.items()} for sc, v in sorted(scenes.items())}
    (DEST/(out+'.json')).write_text(json.dumps(report, indent=1, default=float))
    (DEST/(out+'_per_root.json')).write_text(json.dumps(per_root_out, default=float))
    print('wrote', DEST/(out+'.json'))
    return report


def fmt_ci(d, pct=False, digits=3):
    f = (lambda v: '%.1f%%' % (100*v)) if pct else (lambda v: ('%.'+str(digits)+'f') % v)
    return '%s [%s, %s]' % (f(d['mean']), f(d['ci95'][0]), f(d['ci95'][1]))


def tables(report, order=('v17a', 'v17a:recal', 'v16b', 'v16b:recal', 'v15', 'v15:recal', 'v12', 'v11', 'v7', 'legal_visible', 'frustum', 'known_visible', 'distance', 'duration',
                          'heading_change', 'random')):
    """Markdown tables of a report (narrative conclusions are written separately)."""
    lines = []
    for s, res in report['sets'].items():
        e = res['eligible']; rk = res['ranking']
        any_m = next(iter(rk.values()))
        lines.append(f"\n### Comparison set `{s}` — {e['roots']} Roots, {e['scenes']} scenes, {e['outcomes']} distinct outcomes; "
                     f"{any_m['decisive_pairs']} decisive pairs, {any_m['near_tie_pairs']} near ties, {any_m['exact_tie_pairs']} exact ties"
                     + (f"; {any_m['nested_pairs_excluded']} prefix-nested pairs excluded" if any_m.get('nested_pairs_excluded') else '') + "\n")
        lines.append('| method | pairwise acc. (Root-macro, 95 % CI) | pooled | top-1 hit | near hit | regret (cells) | normalized regret | median / P90 regret |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        names = [n for n in order if n in rk]+[n for n in rk if n not in order]
        for n in names:
            m = rk[n]
            lines.append(f"| {n} | {fmt_ci(m['pairwise_root_macro'], pct=True)} | {100*m['pairwise_pooled']:.1f}% | {fmt_ci(m['top1_hit'], pct=True)} | "
                         f"{100*m['top1_near_hit']['mean']:.1f}% | {fmt_ci(m['regret_mean'], digits=1)} | {fmt_ci(m['regret_normalized_mean'])} | "
                         f"{m['regret_median']:.0f} / {m['regret_p90']:.0f} |")
        lines.append('\n| method | acc. dG 10-25 % | acc. dG 25-50 % | acc. dG >= 50 % | near-tie acc. (dG < max(3, 10 %)) |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for n in names:
            b = rk[n]['by_relative_difference']
            lines.append(f"| {n} | {100*b['0.1-0.25']['accuracy']:.1f}% ({b['0.1-0.25']['pairs']}) | {100*b['0.25-0.5']['accuracy']:.1f}% ({b['0.25-0.5']['pairs']}) | "
                         f"{100*b['>=0.5']['accuracy']:.1f}% ({b['>=0.5']['pairs']}) | {100*rk[n]['near_tie_accuracy']:.1f}% |")
        if res['gain']:
            lines.append('\n| model | actual gain (mean) | expected gain (mean) | MAE | Root-macro MAE (CI) | bias | FREE MAE / bias | BLOCKED MAE / bias | hard-decision MAE / bias | Pearson / Spearman | within-Root Spearman |')
            lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
            for n in list(dict.fromkeys(x for m in order for x in (m, m+':recal') if x in res['gain'])):
                g = res['gain'][n]
                lines.append(f"| {n} | {g['actual_gain_mean']:.1f} | {g['expected_gain_mean']:.1f} | {g['mae_expected']:.1f} | {fmt_ci(g['mae_expected_root_macro'], digits=1)} | "
                             f"{g['bias_expected']:+.1f} | {g['mae_free']:.1f} / {g['bias_free']:+.1f} | {g['mae_blocked']:.1f} / {g['bias_blocked']:+.1f} | "
                             f"{g['mae_hard']:.1f} / {g['bias_hard']:+.1f} | {g['pearson']:.3f} / {g['spearman']:.3f} | {g['within_root_spearman_mean']:.3f} |")
            lines.append('\n| model | per-outcome IoU mean / median [IQR] | outcomes < 0.3 / > 0.6 | pooled IoU: all / previously observed cells / never-observed cells | targets in never-observed cells |')
            lines.append('| --- | ---: | ---: | ---: | ---: |')
            for n in [m for m in order if m in res['gain'] and ':' not in m]:
                g = res['gain'][n]; pi = g['per_outcome_iou']; ri = g['region_iou']
                lines.append(f"| {n} | {pi['mean']:.3f} / {pi['median']:.3f} [{pi['p25']:.3f}, {pi['p75']:.3f}] | {100*pi['share_below_0_3']:.0f}% / {100*pi['share_above_0_6']:.0f}% | "
                             f"{ri['pooled_all']:.3f} / {ri['observed_cells']:.3f} / {ri['never_observed_cells']:.3f} | {100*ri['targets_in_never_observed']:.0f}% |")
            lines.append('\n| model | FREE precision | FREE recall | predicted FREE | -> observed BLOCKED (rate) | -> still UNKNOWN (rate) | never-observed cells: FREE precision / ->UNKNOWN rate |')
            lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
            for n in [m for m in order if m in res['false_free'] and ':' not in m]:
                f, u = res['false_free'][n]['all_cells'], res['false_free'][n]['never_observed_cells']
                lines.append(f"| {n} | {f['free_precision']:.3f} | {f['free_recall']:.3f} | {f['predicted_free']} | {f['free_but_blocked']} ({100*f['rate_free_but_blocked']:.1f}%) | "
                             f"{f['free_but_unknown']} ({100*f['rate_free_but_unknown']:.1f}%) | {u['free_precision']:.3f} / {100*u['rate_free_but_unknown']:.1f}% |")
        rbc = res.get('ranking_by_class')
        if rbc:
            lines.append('\n| method | FREE-gain ranking: pairwise / top-1 hit (decisive pairs) | BLOCKED-gain ranking: pairwise / top-1 hit (decisive pairs) |')
            lines.append('| --- | ---: | ---: |')
            for n in [m for m in order if m in rbc['free']]:
                f, b = rbc['free'][n], rbc['blocked'][n]
                lines.append(f"| {n} | {fmt_ci(f['pairwise_root_macro'], pct=True)} / {100*f['top1_hit']['mean']:.1f}% ({f['decisive_pairs']}) | "
                             f"{fmt_ci(b['pairwise_root_macro'], pct=True)} / {100*b['top1_hit']['mean']:.1f}% ({b['decisive_pairs']}) |")
        pv = res.get('paired_vs', {})
        if pv:
            lines.append('\n| paired difference (scene bootstrap) | pairwise acc. | regret (cells) | top-1 hit |')
            lines.append('| --- | ---: | ---: | ---: |')
            for k, d in pv.items():
                if k.split('-')[0] in ('v17a', 'v16b', 'v15'):
                    lines.append(f"| {k} | {fmt_ci(d['pairwise'], pct=True)} | {fmt_ci(d['regret'], digits=1)} | {fmt_ci(d['top1_hit'], pct=True)} |")
        rg = res.get('regimes')
        if rg:
            lines.append('\n| regime (outcome) | outcomes | label-aware IoU (pooled) | gain MAE | mean actual gain |')
            lines.append('| --- | ---: | ---: | ---: | ---: |')
            for k, d in rg['map'].items():
                for v, a in d.items():
                    lines.append(f"| {k}={v} | {a['outcomes']} | {a['label_aware_iou_pooled']:.3f} | {a['gain_mae']:.1f} | {a['actual_gain_mean']:.1f} |")
            dec = rg['decision_pairwise_by_regime']
            keys = sorted({k for d in dec.values() for k in d})
            meth = [m for m in ('v17a', 'v16b', 'v15', 'legal_visible', 'distance', 'duration') if m in dec]
            lines.append('\n| pair regime (both outcomes) | pairs | ' + ' | '.join(meth) + ' |')
            lines.append('| --- | ---: | ' + ' | '.join('---:' for _ in meth) + ' |')
            for k in keys:
                n = next(dec[m][k]['pairs'] for m in meth if k in dec[m])
                lines.append(f'| {k} | {n} | ' + ' | '.join(f"{100*dec[m][k]['accuracy']:.1f}%" if k in dec[m] else '—' for m in meth) + ' |')
    cal = report.get('calibration', {})
    if cal:
        lines.append('\n### Calibration (all candidate cells of all benchmark outcomes)\n')
        lines.append('| model | Brier (revelation) | Brier (3-class) | revelation ECE / ECE(p>=.05) / AUC | FREE ECE(p>=.05) / AUC | BLOCKED ECE(p>=.05) / AUC | cells p>=.05: mean predicted vs observed revealed |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for n in [m for m in order if m in cal]:
            c = cal[n]
            lines.append(f"| {n} | {c['brier_revelation']:.4f} | {c['brier_multiclass']:.4f} | {c['revelation']['ece']:.4f} / {c['revelation']['ece_p_ge_005']:.3f} / {c['revelation']['auc_binned']:.3f} | "
                         f"{c['free']['ece_p_ge_005']:.3f} / {c['free']['auc_binned']:.3f} | {c['blocked']['ece_p_ge_005']:.3f} / {c['blocked']['auc_binned']:.3f} | "
                         f"{c['revelation']['mean_predicted_p_ge_005']:.3f} vs {c['revelation']['observed_p_ge_005']:.3f} |")
        ref = 'v16b' if 'v16b' in cal else next(iter(cal))
        lines.append(f'\nReliability of the revelation probability ({ref}; predicted bin -> observed revealed fraction):\n')
        lines.append('| bin | cells | observed |'); lines.append('| --- | ---: | ---: |')
        for b in cal[ref]['revelation']['reliability']:
            lines.append(f"| {b['bin']} | {b['cells']} | {b['observed']} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--models', nargs='*', default=list(MODELS))
    pa.add_argument('--out', default='report')
    pa.add_argument('--tables-md', help='also write the Markdown tables to this path')
    a = pa.parse_args()
    rep = main(tuple(a.models), a.out)
    if a.tables_md:
        Path(a.tables_md).write_text('# Query-ranking benchmark — generated tables (`query_report.py`)\n\n'
                                     'Primary set `1.0`; 95 % CIs from 1,000 scene bootstrap resamples. See QUERY_BENCH_REPORT.md for '
                                     'definitions and conclusions.\n' + tables(json.loads(json.dumps(rep, default=float))) + '\n')
        print('wrote', a.tables_md)
