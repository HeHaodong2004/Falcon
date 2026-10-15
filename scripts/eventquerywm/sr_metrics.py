"""Probability metrics. Collision-positive discrimination; safe-positive BCE."""
import numpy as np


def ranking_metrics(target, probability):
    """Tie-aware average precision and trapezoidal ROC-AUC, collision positive."""
    order = np.argsort(-probability, kind='stable')
    y, p = target[order], probability[order]
    positives, negatives = int(y.sum()), int(len(y)-y.sum())
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p)-1]
    tp = np.r_[0., np.cumsum(y)[ends]]
    fp = np.r_[0., (ends+1)-np.cumsum(y)[ends]]
    ap = float(np.sum(np.diff(tp/positives)*tp[1:]/(tp[1:]+fp[1:]))) if positives else None
    auc = float(np.trapz(tp/positives, fp/negatives)) if positives and negatives else None
    return ap, auc


def metrics(sr_labels, p_safe, logits=None, bins=15):
    sr, p = np.asarray(sr_labels, dtype=np.int64), np.asarray(p_safe, dtype=np.float64)
    if not len(sr) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid evaluation predictions')
    collision, pc = 1-sr, 1-p
    predicted = (pc >= .5).astype(np.int64)
    tp, fn = int(np.sum((collision == 1) & (predicted == 1))), int(np.sum((collision == 1) & (predicted == 0)))
    tn, fp = int(np.sum((collision == 0) & (predicted == 0))), int(np.sum((collision == 0) & (predicted == 1)))
    ap, roc = ranking_metrics(collision, pc)
    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        bce = float(np.mean(np.logaddexp(0, z)-sr*z))
    else:
        clipped = np.clip(p, 1e-12, 1-1e-12)
        bce = float(np.mean(-sr*np.log(clipped)-(1-sr)*np.log1p(-clipped)))
    bucket = np.minimum((pc*bins).astype(int), bins-1)
    ece = sum(float(np.mean(bucket == k))*abs(float(pc[bucket == k].mean())-float(collision[bucket == k].mean()))
              for k in range(bins) if np.any(bucket == k))
    return dict(count=len(sr), bce=bce, accuracy=(tp+tn)/len(sr),
                collision_pr_auc_average_precision=ap, collision_roc_auc=roc,
                collision_recall=tp/(tp+fn) if tp+fn else None,
                brier_score=float(np.mean((p-sr)**2)), collision_ece=ece, ece_equal_width_bins=bins,
                confusion_matrix=[[tn,fp],[fn,tp]],
                confusion_matrix_order='rows=true, columns=predicted; [safe, collision]; collision threshold >=0.5',
                positive_class='collision (SR=0) for PR-AUC, ROC-AUC, recall, confusion, ECE; safe (SR=1) for model/BCE; Brier equivalent',
                p_safe_distribution=dict(min=float(p.min()), max=float(p.max()), mean=float(p.mean()),
                    quantiles={str(q):float(np.quantile(p,q)) for q in (0., .1, .25, .5, .75, .9, 1.)},
                    mean_collision_samples=float(p[sr == 0].mean()) if np.any(sr == 0) else None,
                    mean_safe_samples=float(p[sr == 1].mean()) if np.any(sr == 1) else None))


def majority_baseline(sr_labels, train_safe_prior):
    sr = np.asarray(sr_labels)
    return dict(probability_source='constant safe prior estimated from optimization TRAIN only',
                train_safe_prior=train_safe_prior,
                empirical_prior=metrics(sr,np.full(len(sr),train_safe_prior)),
                hard_majority_accuracy=float(np.mean(sr == int(train_safe_prior >= .5))),
                hard_majority_collision_recall=0. if train_safe_prior >= .5 else 1.)
