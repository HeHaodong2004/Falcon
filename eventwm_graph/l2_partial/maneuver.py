"""Bounded native-clock maneuver prior and exact-position disintegration.

At the *start* of each canonical native interval, a tracked person's velocity
is retained with probability ``exp(-rate * interval_dt)`` or changed to
``retention * old_velocity + N(0, reset_std**2 I)``. ``retention=0`` preserves
the original independent reset experiment; ``retention=1`` is a mean-preserving
velocity jump that does not attenuate the learned OU target. Position is
unchanged. Both alternatives then use the existing integrated OU kernel,
including its persistent intention and process uncertainty.

This is a discrete native-clock model. It is NOT a semigroup if a caller draws
new reset alternatives at arbitrary propagation boundaries. Call
``transition_components`` once at a native interval's start, retain its
components across partial continuations, and use ``advance_components`` for
the remainder. Forecast sampling must call ``sample_resets`` at exactly those
same interval starts. ``interval_dt`` controls the reset probability even when
``dt`` advances only part of that interval.

Conditioning is exact within each Gaussian component, with exact observed
positions. Collapsing the conditional mixture to one Gaussian preserves its
first two moments but loses higher moments and multimodality; this is an
explicit assumed-density approximation, separate from persistent intention
modes. Gaussian reset speeds are not hard speed-bounded. Neither population
counts nor tracker identifiers enter these helpers.
"""
from typing import Tuple
import math

import numpy as np

from .probability import transition_matrices


NATIVE_INTERVAL_SECONDS = .048
DEFAULT_MANEUVER_RATE_HZ = 2.
DEFAULT_RESET_STD_MPS = 2.


def reset_probability(rate: float = DEFAULT_MANEUVER_RATE_HZ,
                      interval_dt: float = NATIVE_INTERVAL_SECONDS) -> float:
    """One reset opportunity per full native interval, including partial calls."""
    if not math.isfinite(rate) or rate < 0:
        raise ValueError('maneuver rate must be finite and nonnegative')
    if not math.isfinite(interval_dt) or interval_dt <= 0:
        raise ValueError('native interval duration must be finite and positive')
    return -math.expm1(-rate * interval_dt)


def advance_components(mean: np.ndarray, covariance: np.ndarray, dt: float,
                       intent: np.ndarray, diffusion: np.ndarray,
                       relaxation: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
    """Advance existing components; never draw or reweight a maneuver.

    ``mean`` has shape ``(..., K, 4)``, covariance ``(..., K, 4, 4)``,
    intent ``(..., 2)`` and diffusion ``(...)``. The component axis is K.
    """
    if not math.isfinite(dt) or dt < 0:
        raise ValueError('component duration must be finite and nonnegative')
    if not math.isfinite(relaxation) or relaxation < 0:
        raise ValueError('relaxation must be finite and nonnegative')
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    target = np.asarray(intent, dtype=float)
    q = np.asarray(diffusion, dtype=float)
    if m.shape[-1:] != (4,) or c.shape != m.shape + (4,):
        raise ValueError('invalid Gaussian component shapes')
    if target.shape != m.shape[:-2] + (2,):
        raise ValueError('persistent intent shape mismatch')
    if q.shape not in ((), m.shape[:-2]) or np.any(q < 0):
        raise ValueError('invalid diffusion shape or value')
    if not all(np.isfinite(a).all() for a in (m, c, target, q)):
        raise ValueError('nonfinite Gaussian transition parameter')
    F, Q, a, e = transition_matrices(dt, relaxation)
    result = m @ F.T
    result[..., :2] += (dt - a) * target[..., None, :]
    result[..., 2:] += (1 - e) * target[..., None, :]
    cov = F @ c @ F.T + q[..., None, None, None] * Q
    if relaxation > 0:
        # Match probability.advance and the shared sampled dynamics exactly.
        cov[..., :2, :2] += np.eye(2) * (.0025 * dt)
    return result, (cov + cov.swapaxes(-1, -2)) * .5


def transition_components(h, dt: float,
                          rate: float = DEFAULT_MANEUVER_RATE_HZ,
                          reset_std: float = DEFAULT_RESET_STD_MPS,
                          interval_dt: float = NATIVE_INTERVAL_SECONDS,
                          retention: float = 0.
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create continue/reset components and advance part of one native interval.

    Returns means ``(N, 2, 4)``, covariances ``(N, 2, 4, 4)``, and shared
    component probabilities ``(2,)``. Each person's reset indicator is an
    independent draw conditional on this joint hypothesis. The caller may keep
    a bounded association beam while integrating this local mixture exactly.
    """
    p = reset_probability(rate, interval_dt)
    if not math.isfinite(reset_std) or reset_std < 0:
        raise ValueError('reset standard deviation must be finite and nonnegative')
    if not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError('maneuver retention must lie in [0,1]')
    if not math.isfinite(dt) or dt < 0 or dt > interval_dt + 1e-9:
        raise ValueError('transition_components advances at most one native interval')
    original = np.asarray(h.mean, dtype=float)
    covariance = np.asarray(h.covariance, dtype=float)
    if original.ndim != 2 or original.shape[1] != 4 or covariance.shape != original.shape + (4,):
        raise ValueError('invalid hypothesis Gaussian shapes')
    n = len(original)
    means = np.repeat(original[:, None, :], 2, axis=1)
    covs = np.repeat(covariance[:, None, :, :], 2, axis=1)
    means[:, 1, 2:] *= retention
    # Linear velocity retention plus independent jump noise. In particular,
    # retention=1 preserves both the old mean and position/velocity covariance.
    covs[:, 1, :2, 2:] *= retention
    covs[:, 1, 2:, :2] *= retention
    covs[:, 1, 2:, 2:] = retention**2 * covariance[:, 2:, 2:] + np.eye(2) * reset_std**2
    intent = original[:, 2:] if h.intent is None else np.asarray(h.intent, dtype=float)
    if intent.shape != (n, 2):
        raise ValueError('expand persistent intention hypotheses before maneuver transition')
    diffusion = np.full(n, .16) if h.diffusion is None else np.asarray(h.diffusion, dtype=float)
    means, covs = advance_components(means, covs, dt, intent, diffusion, float(h.relaxation))
    return means, covs, np.array([1. - p, p])


def moment_match(mean: np.ndarray, covariance: np.ndarray,
                 probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preserve local mixture means and covariances, including between-mode variance."""
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    w = np.asarray(probs, dtype=float)
    if m.shape[-1:] != (4,) or c.shape != m.shape + (4,):
        raise ValueError('invalid mixture shapes')
    try:
        w = np.broadcast_to(w, m.shape[:-1])
    except ValueError as error:
        raise ValueError('mixture probability shape mismatch') from error
    if (not np.isfinite(w).all() or np.any(w < 0) or
            not np.allclose(w.sum(-1), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('mixture probabilities must be normalized')
    result = np.sum(w[..., None] * m, axis=-2)
    delta = m - result[..., None, :]
    cov = np.sum(w[..., None, None] *
                 (c + delta[..., :, None] * delta[..., None, :]), axis=-3)
    return result, (cov + cov.swapaxes(-1, -2)) * .5


def _condition_component(mean, covariance, position):
    """Linear Gaussian disintegration with an explicit support dimension."""
    P = (covariance[:2, :2] + covariance[:2, :2].T) * .5
    aa, bb, dd = float(P[0, 0]), float(P[0, 1]), float(P[1, 1])
    tolerance = 1e-12 * max(1., abs(aa), abs(bb), abs(dd))
    determinant = aa * dd - bb * bb
    d = np.asarray(position, dtype=float) - mean[:2]
    if aa > tolerance and dd > tolerance and determinant > tolerance * (aa + dd):
        # Common case: avoid a LAPACK dispatch for every candidate association.
        inverse = np.array([[dd, -bb], [-bb, aa]]) / determinant
        gain = covariance[:, :2] @ inverse
        conditional_mean = mean + gain @ d
        conditional_cov = covariance - gain @ covariance[:2, :]
        conditional_mean[:2] = position
        conditional_cov[:2, :] = 0.
        conditional_cov[:, :2] = 0.
        log_density = -math.log(2 * math.pi) - .5 * (math.log(determinant) + d @ inverse @ d)
        return (conditional_mean, (conditional_cov + conditional_cov.T) * .5), float(log_density), 2
    eig, U = np.linalg.eigh(P)
    if eig.min() < -tolerance:
        raise ValueError('negative predictive position covariance')
    positive = eig > tolerance
    rank = int(positive.sum())
    coordinates = U.T @ d
    if np.linalg.norm(coordinates[~positive]) > 1e-8:
        return None, -np.inf, rank
    if rank:
        basis = U[:, positive]
        inverse = (basis / eig[positive]) @ basis.T
        gain = covariance[:, :2] @ inverse
        conditional_mean = mean + gain @ d
        conditional_cov = covariance - gain @ covariance[:2, :]
        log_density = -.5 * (rank * math.log(2 * math.pi) +
                             np.log(eig[positive]).sum() +
                             np.sum(coordinates[positive]**2 / eig[positive]))
    else:
        conditional_mean = mean.copy()
        conditional_cov = covariance.copy()
        log_density = 0.
    conditional_mean[:2] = position
    conditional_cov[:2, :] = 0.
    conditional_cov[:, :2] = 0.
    conditional_cov = (conditional_cov + conditional_cov.T) * .5
    return (conditional_mean, conditional_cov), float(log_density), rank


def support_ranks(mean: np.ndarray, covariance: np.ndarray,
                  probs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Position reference dimension for every track/detection pair.

    Inputs match ``condition_components_many``. Returns int8 ``[N,D]`` with
    values 0 (matching atom), 1 (compatible affine line), 2 (plane density),
    or 3 (no compatible positive-probability component). Zero-weight mixture
    components do not establish an observation reference measure.

    The caller must compare these dimensions across ALL assignments and
    positive joint hypotheses before adding likelihoods. A 2-D density has
    zero Radon-Nikodym derivative on an atomic or line-supported component of
    the common reference measure. At intersections of distinct affine lines,
    the conditioning point is a null event for both line laws; this predicate
    declares compatible supports but does not resolve that nonunique null-set
    conditional. Ordinary full-rank native transitions and exact position atoms
    avoid that exceptional case.
    """
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    w = np.asarray(probs, dtype=float)
    z = np.asarray(positions, dtype=float)
    if m.ndim != 3 or m.shape[2] != 4 or c.shape != m.shape + (4,) or z.ndim != 2 or z.shape[1] != 2:
        raise ValueError('invalid support-rank dimensions')
    n, k, _ = m.shape
    try:
        w = np.broadcast_to(w, (n, k))
    except ValueError as error:
        raise ValueError('invalid support-rank mixture weights') from error
    if (not all(np.isfinite(a).all() for a in (m, c, w, z)) or np.any(w < 0)
            or not np.allclose(w.sum(axis=1), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid support-rank parameters')
    ranks = np.full((n, len(z)), 3, dtype=np.int8)
    if not n or not len(z):
        return ranks
    aa, dd = c[..., 0, 0], c[..., 1, 1]
    bb = (c[..., 0, 1] + c[..., 1, 0]) * .5
    tolerance = 1e-12 * np.maximum(1., np.maximum(np.abs(aa), np.maximum(np.abs(bb), np.abs(dd))))
    determinant = aa * dd - bb * bb
    full = (aa > tolerance) & (dd > tolerance) & (determinant > tolerance * (aa + dd))
    positive = w > 0
    # Every point is supported by a nondegenerate Gaussian. This is the common
    # path and needs neither eigendecomposition nor a per-detection loop.
    ranks[np.any(full & positive, axis=1)] = 2
    for i in np.flatnonzero(np.any(~full & positive, axis=1)):
        for component in np.flatnonzero(~full[i] & positive[i]):
            for j, point in enumerate(z):
                conditional, _, rank = _condition_component(m[i, component], c[i, component], point)
                if conditional is not None:
                    ranks[i, j] = min(ranks[i, j], rank)
    return ranks


def condition_components(mean: np.ndarray, covariance: np.ndarray,
                         probs: np.ndarray, z: np.ndarray,
                         return_weights: bool = False):
    """Condition all compatible components, then match posterior velocity moments.

    Inputs have shape ``(K,4)``, ``(K,4,4)``, ``(K,)`` and ``(2,)``.
    Returns ``((mean,covariance), log_evidence)`` or ``(None,-inf)``.
    ``return_weights=True`` appends the normalized posterior component weights.

    Exact position atoms and continuous densities must not be added as if they
    shared a reference measure. A matching atom dominates continuous components
    at its atom; otherwise densities are evaluated on compatible support. For
    the normal nondegenerate native kernel, evidence is a 2-D position density.
    """
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    w = np.asarray(probs, dtype=float)
    z = np.asarray(z, dtype=float)
    if m.ndim != 2 or m.shape[1] != 4 or c.shape != (len(m), 4, 4) or w.shape != (len(m),) or z.shape != (2,):
        raise ValueError('invalid conditioning shapes')
    if (not all(np.isfinite(a).all() for a in (m, c, w, z)) or np.any(w < 0)
            or not np.isclose(w.sum(), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid conditioning parameters')
    values = []
    for i in np.flatnonzero(w > 0):
        conditional, logp, rank = _condition_component(m[i], c[i], z)
        if conditional is not None:
            values.append((int(i), conditional, math.log(w[i]) + logp, rank))
    posterior = np.zeros(len(w))
    if not values:
        result = (None, -np.inf)
    else:
        minimum_rank = min(value[3] for value in values)
        values = [value for value in values if value[3] == minimum_rank]
        logs = np.asarray([value[2] for value in values])
        maximum = float(logs.max())
        evidence_scaled = np.exp(logs - maximum)
        weights = evidence_scaled / evidence_scaled.sum()
        for value, probability in zip(values, weights):
            posterior[value[0]] = probability
        # The posterior weights have just been normalized; avoid repeating the
        # public moment_match validation inside every association candidate.
        components = np.stack([value[1][0] for value in values])
        result_mean = np.sum(weights[:, None] * components, axis=0)
        delta = components - result_mean
        result_cov = np.sum(weights[:, None, None] *
                            (np.stack([value[1][1] for value in values]) +
                             delta[:, :, None] * delta[:, None, :]), axis=0)
        result_cov = (result_cov + result_cov.T) * .5
        result_mean[:2] = z
        result_cov[:2, :] = 0.
        result_cov[:, :2] = 0.
        result = ((result_mean, result_cov), maximum + math.log(float(evidence_scaled.sum())))
    return (*result, posterior) if return_weights else result


def condition_components_many(mean: np.ndarray, covariance: np.ndarray,
                              probs: np.ndarray, positions: np.ndarray,
                              return_weights: bool = False):
    """All track/detection associations in one batched disintegration.

    Inputs have shape ``(N,K,4)``, ``(N,K,4,4)``, ``(K,)`` or ``(N,K)``, and
    ``(D,2)``. Returns ``((mean[N,D,4], covariance[N,D,4,4]), log_evidence[N,D])``.
    Optional posterior weights have shape ``(N,D,K)``. Unsupported pairs have
    ``-inf`` evidence and zero placeholder moments, never positive probability.

    Build this table once per source hypothesis; assigning one slot does not
    change another slot's conditional Gaussian under the represented model.
    Degenerate components use the scalar support-aware oracle above.
    """
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    w = np.asarray(probs, dtype=float)
    z = np.asarray(positions, dtype=float)
    if m.ndim != 3 or m.shape[2] != 4 or c.shape != m.shape + (4,) or z.ndim != 2 or z.shape[1] != 2:
        raise ValueError('invalid batched conditioning shapes')
    n, k, _ = m.shape
    try:
        w = np.broadcast_to(w, (n, k))
    except ValueError as error:
        raise ValueError('invalid batched mixture weights') from error
    if (not all(np.isfinite(a).all() for a in (m, c, w, z)) or np.any(w < 0)
            or not np.allclose(w.sum(1), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid batched conditioning parameters')
    result_mean = np.zeros((n, len(z), 4))
    result_cov = np.zeros((n, len(z), 4, 4))
    log_evidence = np.full((n, len(z)), -np.inf)
    posterior = np.zeros((n, len(z), k))
    if not n or not len(z):
        result = ((result_mean, result_cov), log_evidence)
        return (*result, posterior) if return_weights else result
    aa, dd = c[..., 0, 0], c[..., 1, 1]
    bb = (c[..., 0, 1] + c[..., 1, 0]) * .5
    determinant = aa * dd - bb**2
    tolerance = 1e-12 * np.maximum(1., np.maximum(np.abs(aa), np.maximum(np.abs(bb), np.abs(dd))))
    if not np.all((aa > tolerance) & (dd > tolerance) & (determinant > tolerance * (aa + dd))):
        for i in range(n):
            for j in range(len(z)):
                conditioned, evidence, weights = condition_components(m[i], c[i], w[i], z[j], return_weights=True)
                if conditioned is not None:
                    result_mean[i, j], result_cov[i, j] = conditioned
                log_evidence[i, j] = evidence
                posterior[i, j] = weights
        result = ((result_mean, result_cov), log_evidence)
        return (*result, posterior) if return_weights else result
    inverse = np.stack([dd, -bb, -bb, aa], axis=-1).reshape(n, k, 2, 2) / determinant[..., None, None]
    gain = c[..., :, :2] @ inverse
    conditional_cov = c - gain @ c[..., :2, :]
    delta = z[None, :, None, :] - m[:, None, :, :2]
    log_density = (-math.log(2 * math.pi) - .5 * np.log(determinant)[:, None, :] -
                   .5 * np.einsum('ndki,nkij,ndkj->ndk', delta, inverse, delta))
    log_weights = np.full(w.shape, -np.inf)
    np.log(w, out=log_weights, where=w > 0)
    log_joint = log_density + log_weights[:, None, :]
    maximum = log_joint.max(axis=-1, keepdims=True)
    posterior = np.exp(log_joint - maximum)
    scaled_evidence = posterior.sum(axis=-1, keepdims=True)
    posterior /= scaled_evidence
    log_evidence = maximum[..., 0] + np.log(scaled_evidence[..., 0])
    conditional_mean = m[:, None, :, :] + np.einsum('nkij,ndkj->ndki', gain, delta)
    conditional_mean[..., :2] = z[None, :, None, :]
    result_mean = np.einsum('ndk,ndki->ndi', posterior, conditional_mean)
    differences = conditional_mean - result_mean[:, :, None, :]
    result_cov = (np.einsum('ndk,nkij->ndij', posterior, conditional_cov) +
                  np.einsum('ndk,ndki,ndkj->ndij', posterior, differences, differences))
    result_cov = (result_cov + result_cov.swapaxes(-1, -2)) * .5
    result_mean[..., :2] = z[None, :, :]
    result_cov[..., :2, :] = 0.
    result_cov[..., :, :2] = 0.
    result = ((result_mean, result_cov), log_evidence)
    return (*result, posterior) if return_weights else result


def sample_resets(states: np.ndarray, rng: np.random.Generator,
                  rate: float = DEFAULT_MANEUVER_RATE_HZ,
                  reset_std: float = DEFAULT_RESET_STD_MPS,
                  interval_dt: float = NATIVE_INTERVAL_SECONDS,
                  retention: float = 0.
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample only interval-start resets; caller then applies its usual OU step.

    The reset mask is independent per world/person, conditional on current joint
    state. This does not create people, move positions, or reset intentions.
    Persist the resulting states if prediction stops inside this native interval.
    """
    p = reset_probability(rate, interval_dt)
    if not math.isfinite(reset_std) or reset_std < 0:
        raise ValueError('reset standard deviation must be finite and nonnegative')
    if not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError('maneuver retention must lie in [0,1]')
    states = np.asarray(states, dtype=float)
    if states.shape[-1:] != (4,) or not np.isfinite(states).all():
        raise ValueError('invalid sampled states')
    mask = rng.random(states.shape[:-1]) < p
    result = states.copy()
    resets = retention * states[..., 2:] + rng.normal(0., reset_std, size=states.shape[:-1] + (2,))
    result[..., 2:] = np.where(mask[..., None], resets, result[..., 2:])
    return result, mask
