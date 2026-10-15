"""Root-aware physical and probabilistic metrics; no scalar-SR shortcuts."""
import math
import numpy as np
import torch


def trajectory_energy(positions, truth, valid, weights):
    """Proper energy score for each observed joint path, normalized by dimensions.

    positions B,S,T,N,2; truth B,T,N,2; valid B,T,N. A root is the
    sampling unit, regardless of its agent count or number of candidate queries.
    Missing/post-contact targets contribute neither zeros nor negative evidence.
    """
    mask = valid.to(positions.dtype)[..., None]
    count = mask.sum((1, 2, 3)) * 2
    error = ((positions - truth[:, None]) * mask[:, None]).flatten(2)
    distances = torch.sqrt(error.square().sum(-1) / count[:, None].clamp_min(1) + 1e-10)
    difference = ((positions[:, :, None] - positions[:, None, :]) * mask[:, None, None]).flatten(3)
    pair = torch.sqrt(difference.square().sum(-1) / count[:, None, None].clamp_min(1) + 1e-10)
    scores = (weights * distances).sum(1) - .5 * (weights[:, :, None] * weights[:, None, :] * pair).sum((1, 2))
    return scores, count > 0


def trajectory_nll(positions, truth, valid, weights, sigma):
    """Joint scene Gaussian-mixture NLL per supported coordinate.

    Gaussian observation error is conditional on one coherent trajectory mode;
    modes are shared across all agents and times. This score is distinct from
    discrete scenario diversity or calibrated physical uncertainty.
    """
    mask = valid.to(positions.dtype)[..., None]
    count = mask.sum((1, 2, 3)) * 2
    sigma = torch.as_tensor(sigma, device=positions.device, dtype=positions.dtype).clamp_min(.01)
    while sigma.ndim < positions.ndim:
        sigma = sigma.unsqueeze(-1)
    logp = -.5 * ((positions - truth[:, None]) / sigma).square() - sigma.log() - .5 * math.log(2 * math.pi)
    joint = (logp * mask[:, None]).sum((2, 3, 4))
    nll = -torch.logsumexp(weights.clamp_min(1e-8).log() + joint, 1) / count.clamp_min(1)
    return nll, count > 0


def mean_valid(value, valid):
    mask = valid.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def interval_event_nll(cumulative, event_time, event_valid, exposure, times, planned_duration):
    """Exact discrete first-contact likelihood, right censor otherwise.

    A collision falls into (previous clock, current clock]. Right-censored
    exposure contributes survival only through the last fully exposed interval.
    Partial final intervals require explicit extra knots, not invented safety.
    """
    zero = torch.zeros_like(cumulative[..., :1])
    previous = torch.cat((zero, cumulative[..., :-1]), -1)
    event = event_valid[..., None] & (event_time[..., None] <= times) & (event_time[..., None] > times - (times[1]-times[0] if len(times)>1 else times[0]))
    event &= times <= planned_duration[..., None] + 1e-5
    event_mass = (cumulative - previous).clamp_min(1e-7)
    event_loss = -(event_mass.log() * event).sum(-1)
    supported = (times <= exposure[..., None] + 1e-5) & (times <= planned_duration[..., None] + 1e-5)
    index = (supported.long().sum(-1) - 1).clamp_min(0)
    final_cdf = cumulative.gather(-1, index[..., None]).squeeze(-1)
    survival_loss = -(1-final_cdf).clamp_min(1e-7).log()
    has_event = event.any(-1)
    valid = has_event | supported.any(-1)
    return torch.where(has_event, event_loss, survival_loss), valid


def binary_summary(target, probability):
    y = np.asarray(target, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], np.clip(p[keep], 1e-7, 1-1e-7)
    if not len(y):
        return {'count': 0}
    order = np.argsort(-p, kind='stable')
    ys = y[order]
    # Threshold-tied AP: no ordering benefit within equal-score groups.
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y)-1]
    tp = np.cumsum(ys)[ends]
    ap = float(np.sum(np.diff(np.r_[0., tp]) * tp / (ends+1)) / y.sum()) if y.sum() else None
    bins = np.minimum((p*10).astype(int), 9)
    calibration = []
    ece = 0.
    for i in range(10):
        m = bins == i
        if m.any():
            error = abs(p[m].mean()-y[m].mean())
            ece += error*m.mean()
            calibration.append({'bin': i, 'n': int(m.sum()), 'predicted': float(p[m].mean()), 'observed': float(y[m].mean())})
    return {'count': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'brier': float(np.mean((p-y)**2)), 'nll': float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'average_precision': ap, 'ece10': float(ece), 'reliability': calibration,
            'positive_nll': float(-np.log(p[y==1]).mean()) if y.sum() else None}


def paired_root_interval(a, b=None, seed=20261014, repetitions=500):
    """Percentile bootstrap over roots; paired arrays must share root ordering."""
    a = np.asarray(a, float)
    values = a if b is None else a-np.asarray(b, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'roots': 0, 'mean': None, 'low': None, 'high': None}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    return {'roots': int(len(values)), 'mean': float(values.mean()),
            'low': float(np.quantile(boot, .025)), 'high': float(np.quantile(boot, .975)),
            'repetitions': repetitions, 'unit': 'root', 'paired_difference': b is not None}


def candidate_concordance(labels, risks):
    """One root's matched action ranking, ties counted as half, collision=1."""
    y, p = np.asarray(labels), np.asarray(risks)
    pos, neg = p[y==1], p[y==0]
    if not len(pos) or not len(neg):
        return None
    delta = pos[:, None]-neg[None, :]
    return float(((delta>0)+.5*(delta==0)).mean())
