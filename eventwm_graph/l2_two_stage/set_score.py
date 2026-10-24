"""Proper kernel score for complete joint finite multisets of future paths.

This is an OFFLINE score, never a model input or an observation simulator.
Every present episode and unattributed person contributes one complete path.
No identity matching, nearest-neighbor rescue, or supported-person subset is
used. Coincident distinct people contribute repeatedly to the finite measure.

For future-only paths x, normalize the vector by sqrt(number of native times).
The inner kernel is k(x,z)=exp(-||x-z||^2/(2*sigma_path^2)). A world multiset A
has Hilbert embedding a=sum_{x in A} phi(x), without division by person count.
The outer kernel is K(A,B)=exp(-||a-b||^2/(2*sigma_set^2)). Gaussian kernels on
Hilbert space are positive definite. Thus

    S(P,Y)=E[K(W,W')]-2*E[K(W,Y)]+K(Y,Y)

is proper: its excess expected score against Q is ||mu_P-mu_Q||^2 >= 0.
No strict-propriety claim is made here. The default path bandwidth3m follows
the inherited population-kernel scale; outer bandwidth2 is in embedding units.
Both are fixed score configuration, not parameters to select on model results.

``iid_u`` estimates the first expectation with off-diagonal independent-world
pairs. It is unbiased for the generating law when worlds are iid/equal-weight;
an individual estimate may be negative and MUST NOT be clipped. ``finite``
instead evaluates the exact kernel score of a declared weighted finite mixture,
including diagonal terms. These are distinct estimands and are labeled below.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


class KernelScoreResourceError(RuntimeError):
    """Scoring refused a declared allocation; no paths were silently omitted."""


@dataclass(frozen=True)
class JointSetKernelScore:
    score: Optional[float]
    status: str
    estimator: str
    worlds: int
    forecast_counts: Tuple[int, ...]
    truth_count: int
    future_times: Tuple[float, ...]
    path_bandwidth: float
    set_bandwidth: float
    world_pair_mean: Optional[float] = None
    world_truth_mean: Optional[float] = None
    reason: Optional[str] = None
    version: str = 'joint_future_multiset_kernel_score_1'


def _presence(values, shape):
    values = np.asarray(values)
    if values.shape != shape:
        raise ValueError('presence must be constant per world/person, shape [W,N]')
    if values.dtype.kind != 'b':
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
            raise ValueError('invalid presence values')
        if not ((values == 0) | (values == 1)).all():
            raise ValueError('presence must contain only booleans or 0/1')
    return values.astype(bool, copy=False)


def _set_gram(paths, membership, path_bandwidth, set_bandwidth):
    """All world-set kernels via one path Gram and membership contractions.

    membership[path,world] is1 for each path instance, including duplicates.
    There are no Python loops over world pairs or trajectory associations.
    """
    if len(paths):
        norm = np.einsum('ij,ij->i', paths, paths)
        distance2 = norm[:, None] + norm[None, :] - 2 * (paths @ paths.T)
        np.maximum(distance2, 0., out=distance2)
        inner = np.exp(-distance2 / (2 * path_bandwidth ** 2))
        embedded = membership.T @ inner @ membership
    else:
        embedded = np.zeros((membership.shape[1], membership.shape[1]))
    diagonal = np.diag(embedded)
    distance2 = diagonal[:, None] + diagonal[None, :] - 2 * embedded
    np.maximum(distance2, 0., out=distance2)
    return np.exp(-distance2 / (2 * set_bandwidth ** 2))


def joint_future_set_kernel_score(
    positions,
    present,
    truth,
    *,
    times,
    horizon=None,
    target_valid=None,
    weights=None,
    estimator='iid_u',
    path_bandwidth=3.,
    set_bandwidth=2.,
    time_tolerance=1e-8,
    max_path_instances=4096,
):
    """Score whole coherent population futures at identical native timestamps.

    positions: [world,time,padded_person,xy]; present: [world,padded_person].
    truth: [time,physical_person,xy], with every physical person's whole path.
    times: strictly increasing offsets in seconds from the prediction anchor.
    t=0 is always excluded. Pass an actual native endpoint as ``horizon``;
    nominal around-.5/1/2/3 labels belong in the caller's result metadata.

    target_valid is a truth-availability mask [time] or [time,person]. It is
    NOT sensor visibility. Any requested missing target censors the complete
    population score; we never delete that person or stitch a partial path.
    Missing future endpoint support returns ``truth_censored``. No positive
    future points returns ``no_future_support``. Nonfinite present forecasts
    or uncensored truth are errors, not successful scores.

    Arbitrary normalized weights require estimator='finite'. The iid U-score
    assumes independent draws from the complete generating law and requires
    equal weights. A max-path allocation refusal is explicit; the caller must
    retain it as a failed score, never exclude the root to claim improvement.
    """
    x = np.asarray(positions)
    y = np.asarray(truth)
    ts = np.asarray(times, dtype=np.float64)
    if x.ndim != 4 or x.shape[-1] != 2 or y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError('forecast/truth must be [W,T,N,2] and [T,P,2]')
    if x.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf':
        raise ValueError('trajectory coordinates must be real numeric arrays')
    worlds, length, people, _ = x.shape
    if not worlds or y.shape[0] != length:
        raise ValueError('empty world bank or mismatched native time dimensions')
    if (ts.shape != (length,) or not length or not np.isfinite(ts).all() or
            (ts < 0).any() or (np.diff(ts) <= 0).any()):
        raise ValueError('times must be strictly increasing nonnegative offsets')
    if not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError('invalid timestamp tolerance')
    if (not math.isfinite(path_bandwidth) or path_bandwidth <= 0 or
            not math.isfinite(set_bandwidth) or set_bandwidth <= 0):
        raise ValueError('kernel bandwidths must be positive and finite')
    if estimator not in ('iid_u', 'finite'):
        raise ValueError('unknown kernel-score estimator')
    if estimator == 'iid_u' and worlds < 2:
        raise ValueError('iid off-diagonal score requires at least2 worlds')
    presence = _presence(present, (worlds, people))
    counts = tuple(int(v) for v in presence.sum(-1))
    if weights is None:
        probability = np.full(worlds, 1. / worlds)
    else:
        probability = np.asarray(weights, dtype=np.float64)
        if (probability.shape != (worlds,) or not np.isfinite(probability).all() or
                (probability < 0).any() or abs(float(probability.sum()) - 1) > 1e-10):
            raise ValueError('world weights must be finite, nonnegative and sum to1')
    if estimator == 'iid_u' and not np.allclose(probability, 1. / worlds, rtol=0, atol=1e-12):
        raise ValueError('nonuniform weights need exact finite-mixture estimator')
    future = ts > 0.
    reason = None
    if horizon is not None:
        endpoint = float(horizon)
        if not math.isfinite(endpoint) or endpoint <= 0:
            raise ValueError('horizon must be a positive native endpoint')
        future &= ts <= endpoint + time_tolerance
        if ts[-1] < endpoint - time_tolerance:
            reason = 'native truth does not reach the requested endpoint'
        elif not np.any(np.abs(ts - endpoint) <= time_tolerance):
            raise ValueError('horizon is not a supplied native endpoint')
    selected = np.flatnonzero(future)
    common = dict(estimator=estimator, worlds=worlds, forecast_counts=counts,
        truth_count=y.shape[1], future_times=tuple(float(v) for v in ts[selected]),
        path_bandwidth=float(path_bandwidth), set_bandwidth=float(set_bandwidth))
    if reason is not None:
        return JointSetKernelScore(None, 'truth_censored', reason=reason, **common)
    if not len(selected):
        return JointSetKernelScore(None, 'no_future_support',
            reason='current point alone is not a future trajectory', **common)
    if target_valid is not None:
        valid = np.asarray(target_valid)
        if valid.shape not in ((length,), (length, y.shape[1])):
            raise ValueError('truth validity mask has wrong shape')
        valid = _presence(valid, valid.shape)
        if not valid[selected].all():
            return JointSetKernelScore(None, 'truth_censored',
                reason='a required time/person truth is unavailable', **common)
    total = sum(counts) + y.shape[1]
    if (isinstance(max_path_instances, bool) or
            not isinstance(max_path_instances, int) or max_path_instances < 0):
        raise ValueError('invalid path-instance allocation limit')
    if total > max_path_instances:
        raise KernelScoreResourceError('complete score requires %d path instances; limit=%d' %
            (total, max_path_instances))
    predicted_paths = x[:, selected].transpose(0, 2, 1, 3).reshape(
        worlds * people, len(selected) * 2)[presence.ravel()]
    true_paths = y[selected].transpose(1, 0, 2).reshape(
        y.shape[1], len(selected) * 2)
    if not np.isfinite(predicted_paths).all() or not np.isfinite(true_paths).all():
        raise ValueError('nonfinite present trajectory or uncensored population truth')
    # Translation is immaterial to either Gaussian kernel. Subtract one common
    # reference before the dot-product Gram to avoid large-origin cancellation.
    paths = np.concatenate((predicted_paths, true_paths), axis=0).astype(np.float64)
    if total:
        paths = (paths - paths[:1]) / math.sqrt(len(selected))
    owner = np.repeat(np.arange(worlds), people)[presence.ravel()]
    owner = np.concatenate((owner, np.full(len(true_paths), worlds)))
    membership = np.zeros((total, worlds + 1))
    membership[np.arange(total), owner] = 1.
    gram = _set_gram(paths, membership, path_bandwidth, set_bandwidth)
    predicted_gram, cross = gram[:worlds, :worlds], gram[:worlds, worlds]
    if estimator == 'iid_u':
        pair_mean = float((predicted_gram.sum() - np.trace(predicted_gram)) /
                          (worlds * (worlds - 1)))
    else:
        pair_mean = float(probability @ predicted_gram @ probability)
    cross_mean = float(probability @ cross)
    score = pair_mean - 2 * cross_mean + float(gram[worlds, worlds])
    if not math.isfinite(score):
        raise FloatingPointError('nonfinite complete-population kernel score')
    return JointSetKernelScore(float(score), 'supported',
        world_pair_mean=pair_mean, world_truth_mean=cross_mean, **common)
