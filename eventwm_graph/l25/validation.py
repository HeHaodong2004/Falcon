"""Offline scalar risk diagnostics; no dependency from the online compiler."""
from collections import defaultdict
import numpy as np


def risk_metrics(rows, bootstrap=1000, seed=912):
    if not rows:
        return dict(status='NO_SUPPORTED_LABELS', n=0)
    p = np.array([r['prediction'] for r in rows])
    y = np.array([r['label'] for r in rows], float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError('probabilities and binary matching labels required')
    by_root, by_scene = defaultdict(list), defaultdict(list)
    for r, score in zip(rows, (p-y)**2):
        by_root[(r['scene_id'], r['root_id'])].append(float(score))
    for (scene, root), scores in by_root.items():
        by_scene[scene].append(np.mean(scores))
    scene_scores = np.array([np.mean(v) for k, v in sorted(by_scene.items())])
    rng = np.random.default_rng(seed)
    draws = scene_scores[rng.integers(0, len(scene_scores), (bootstrap, len(scene_scores)))].mean(1)
    bins = []
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        bins.append(dict(lower=float(lo), upper=float(hi), n=int(mask.sum()),
                         predicted=float(p[mask].mean()) if mask.any() else None,
                         frequency=float(y[mask].mean()) if mask.any() else None))
    positive, negative = p[y == 1], p[y == 0]
    auc = float(((positive[:, None] > negative).sum()+.5*(positive[:, None] == negative).sum())/(len(positive)*len(negative))) if len(positive) and len(negative) else None
    rank_pairs, rank_score = 0, 0.
    groups = defaultdict(list)
    for row in rows:
        groups[row['root_id']].append(row)
    for group in groups.values():
        for a in group:
            for b in group:
                if a['label'] == 1 and b['label'] == 0:
                    rank_pairs += 1
                    rank_score += (a['prediction'] > b['prediction'])+.5*(a['prediction'] == b['prediction'])
    return dict(status='DESCRIPTIVE_DEVELOPMENT_ONLY', n=len(rows), roots=len(by_root), scenes=len(by_scene),
                positives=int(y.sum()), observed_rate=float(y.mean()), prediction_mean=float(p.mean()),
                brier_row_mean=float(np.mean((p-y)**2)), brier_scene_root_balanced=float(scene_scores.mean()),
                brier_95_scene_bootstrap=np.quantile(draws, [.025, .975]).tolist(),
                reliability=bins, ece=float(sum(b['n']*abs(b['predicted']-b['frequency']) for b in bins if b['n'])/len(rows)),
                auroc=auc, within_root_ranking_concordance=rank_score/rank_pairs if rank_pairs else None,
                within_root_opposite_label_pairs=rank_pairs,
                interval_scope='resample scenes, average candidates within root then roots within scene; fixed checkpoint/bank seed; no training variance')
