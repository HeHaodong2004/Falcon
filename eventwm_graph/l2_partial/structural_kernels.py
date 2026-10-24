"""Probability-preserving kernels for the structural optimization campaign.

These operators retain the repair model's finite Sobol/Gauss-Hermite measures,
pointwise observation law, native endpoints, and nearest-person tie rule. They
do not fit parameters. ``cdf_diagnostic=False`` omits only an offline diagnostic;
the CDF never supplies certificate posterior weights. Compilation is explicit
initialization overhead and is not a claimed real-time operation.
"""
from collections import OrderedDict
import math
import threading

import numpy as np
from numba import njit

from .hidden import _gaussians, _position_quadrature
from .maneuver import condition_components_many, moment_match


_NORMAL_CACHE = OrderedDict()
_NORMAL_CACHE_BYTES = 8 * 1024 * 1024
_NORMAL_CACHE_ENTRIES = 16
_NORMAL_CACHE_LOCK = threading.RLock()


def clear_kernel_caches():
    """Clear retained Sobol buffers; JIT code and fixed GH rules are separate."""
    with _NORMAL_CACHE_LOCK:
        _NORMAL_CACHE.clear()


def kernel_cache_info():
    with _NORMAL_CACHE_LOCK:
        return {'sobol_entries': len(_NORMAL_CACHE),
                'sobol_bytes': sum(a.nbytes for a in _NORMAL_CACHE.values()),
                'sobol_max_entries': _NORMAL_CACHE_ENTRIES,
                'sobol_max_bytes': _NORMAL_CACHE_BYTES}


def _sobol_normals(n, points):
    # Independent model instances share this immutable rule cache. Generation
    # is also serialized so concurrent first calls cannot multiply its memory
    # or race move/iteration/eviction. This is a cooperative cache lock, not a
    # hard real-time scheduling guarantee.
    with _NORMAL_CACHE_LOCK:
        return _sobol_normals_locked(n, points)


def _sobol_normals_locked(n, points):
    from scipy.special import ndtri
    from scipy.stats import qmc
    key = (int(n), int(points))
    if key in _NORMAL_CACHE:
        _NORMAL_CACHE.move_to_end(key)
        return _NORMAL_CACHE[key]
    normal = ndtri(np.clip(qmc.Sobol(n * 2, scramble=True, seed=73).random_base2(
        int(math.log2(points))), 1e-10, 1 - 1e-10)).reshape(points, n, 2)
    normal.setflags(write=False)
    if normal.nbytes <= _NORMAL_CACHE_BYTES:
        _NORMAL_CACHE[key] = normal
        while (len(_NORMAL_CACHE) > _NORMAL_CACHE_ENTRIES or
               sum(a.nbytes for a in _NORMAL_CACHE.values()) > _NORMAL_CACHE_BYTES):
            _NORMAL_CACHE.popitem(last=False)
    return normal


def _condition_positions_many(mean, covariance, positions):
    """Same conditional as probability.condition_position over S x N points.

    Inverses and velocity covariances depend on the person, not the retained
    Sobol stratum, and are computed once. Exact Diracs preserve the reference
    scalar support threshold. Output arrays own their memory.
    """
    S, N, _ = positions.shape
    cm = np.broadcast_to(mean, (S, N, 4)).copy()
    cv = np.broadcast_to(covariance, (S, N, 4, 4)).copy()
    P = covariance[:, :2, :2]
    singular = np.max(np.abs(P), axis=(1, 2)) < 1e-12
    if np.any(singular):
        distance = np.linalg.norm(positions[:, singular] - mean[None, singular, :2], axis=-1)
        if np.any(distance > 1e-8):
            raise ValueError('reference Dirac condition has unsupported Sobol point')
    keep = ~singular
    if np.any(keep):
        p = P[keep]
        aa = p[:, 0, 0] + 1e-12; bb = p[:, 0, 1]
        cc = p[:, 1, 0]; dd = p[:, 1, 1] + 1e-12
        determinant = aa * dd - bb * cc
        if np.any(determinant <= 0):
            raise ValueError('reference conditional has nonpositive determinant')
        inv = np.stack([dd, -bb, -cc, aa], axis=-1).reshape(-1, 2, 2) / determinant[:, None, None]
        gain = covariance[keep, :, :2] @ inv
        delta = positions[:, keep] - mean[None, keep, :2]
        cm[:, keep] = mean[None, keep] + np.einsum('nij,snj->sni', gain, delta)
        cm[:, keep, :2] = positions[:, keep]
        c = covariance[keep] - gain @ covariance[keep, :2, :]
        c[:, :2, :] = 0.; c[:, :, :2] = 0.
        c = (c + c.swapaxes(-1, -2)) * .5
        cv[:, keep] = c
    return cm, cv


def _certificate_components(mean, covariance, weight, boxes, worlds, points, cdf_diagnostic):
    """Unnormalized component tuples and likelihood under the old finite rule."""
    N = len(mean)
    if N == 0:
        return [(float(weight), mean, covariance)], float(weight), float(weight), 0
    positions = mean[:, :2]; P = covariance[:, :2, :2]
    if np.max(np.abs(P)) < 1e-12:
        valid = all(not np.any(np.all((positions >= lo) & (positions <= hi), axis=1))
                    for lo, hi in boxes)
        mass = float(weight) if valid else 0.
        return ([(mass, mean, covariance)] if valid else []), mass, mass, 0
    normal = _sobol_normals(N, points)
    eig, U = np.linalg.eigh(P)
    sample = positions + np.einsum('nij,pnj->pni', U, normal * np.sqrt(np.maximum(eig, 0))[None])
    valid = np.ones(points, bool)
    for lo, hi in boxes:
        valid &= ~np.any(np.all((sample >= lo) & (sample <= hi), axis=-1), axis=1)
    mass = float(weight * valid.mean())
    analytic = None
    if cdf_diagnostic:
        if len(boxes) == 1:
            from scipy.stats import multivariate_normal
            lo, hi = boxes[0]; empty = 1.
            for m, p in zip(positions, P):
                dist = multivariate_normal(mean=m, cov=p + np.eye(2) * 1e-14, allow_singular=True)
                inside = dist.cdf(hi) - dist.cdf([lo[0], hi[1]]) - dist.cdf([hi[0], lo[1]]) + dist.cdf(lo)
                empty *= max(0., min(1., 1 - inside))
            analytic = float(weight * empty)
        else:
            analytic = mass  # Reference behavior; NOT a union-rectangle CDF.
    ids = np.flatnonzero(valid)
    if not len(ids):
        return [], mass, analytic, points
    count = min(worlds, len(ids))
    chosen = ids[np.minimum(len(ids) - 1, ((np.arange(count) + .5) * len(ids) / count).astype(int))]
    cm, cv = _condition_positions_many(mean, covariance, sample[chosen])
    return [(mass / count, cm[i], cv[i]) for i in range(count)], mass, analytic, points


def condition_all_empty_fast(hypotheses, boxes, unresolved_mass, *, worlds=16,
                             points=256, cdf_diagnostic=False):
    """Same tracked+hidden truncated Sobol posterior as repair_filter.

    Count alternatives and all retained strata are preserved, including private
    hidden states. The contamination likelihood remains 0.5 per rectangle.
    With the default diagnostic setting, CDF/error fields are explicitly None.
    The input must already contain deduplicated legal certificate rectangles,
    as in the existing observation API.
    """
    if int(points) != points or points <= 0 or int(points) & (int(points) - 1):
        raise ValueError('Sobol points must be a positive power of two')
    if int(worlds) != worlds or worlds <= 0:
        raise ValueError('certificate strata must be positive')
    worlds = int(worlds); points = int(points)
    boxes = [(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)) for lo, hi in boxes]
    for lo, hi in boxes:
        if (lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all()
                or np.any(hi < lo)):
            raise ValueError('invalid certified rectangle')
    candidates = []; empirical = 0.; analytic = 0.; quadrature_points = 0
    for h in hypotheses:
        counts = enumerate(h.hidden_pmf) if h.hidden_exchangeable else [(int(round(h.expected_hidden)), 1.)]
        for k, p in counts:
            if p <= 0 or h.weight <= 0:
                continue
            hm = np.repeat(h.hidden_mean, k, axis=0) if h.hidden_exchangeable else h.hidden_mean
            hc = np.repeat(h.hidden_covariance, k, axis=0) if h.hidden_exchangeable else h.hidden_covariance
            mean = np.concatenate([h.mean, hm]); covariance = np.concatenate([h.covariance, hc])
            components, mass, cdf, _ = _certificate_components(
                mean, covariance, h.weight * p, boxes, worlds, points, cdf_diagnostic)
            empirical += mass
            # The old wrapper receives only unsupported_empty when this
            # count alternative has zero quadrature mass; retain that behavior.
            if cdf_diagnostic and mass > 0:
                analytic += cdf
            # Match the reference diagnostic's count: a successful count
            # alternative reports points even when its positions are Diracs.
            if mass > 0:
                quadrature_points += points
            for weight, cm, cv in components:
                if mass <= 0 or weight <= 0:
                    continue
                pmf = np.zeros_like(h.hidden_pmf); pmf[k] = 1.
                candidates.append(h.fork(weight=weight, mean=cm[:h.n], covariance=cv[:h.n],
                    hidden_mean=cm[h.n:], hidden_covariance=cv[h.n:], hidden_pmf=pmf,
                    hidden_exchangeable=False, maneuver_mean=None, maneuver_covariance=None,
                    maneuver_probability=None, hidden_maneuver_mean=None,
                    hidden_maneuver_covariance=None, hidden_maneuver_probability=None))
    residual = unresolved_mass * .5 ** len(boxes); denominator = empirical + residual
    if denominator <= 0:
        return [h.fork(weight=0.) for h in hypotheses], 1., {'unsupported_empty': True,
            'empty_cdf_diagnostic_status': 'NOT_COMPUTED' if not cdf_diagnostic else 'COMPUTED'}
    return [h.fork(weight=h.weight / denominator) for h in candidates], residual / denominator, {
        'empty_evidence_quadrature': empirical,
        'empty_evidence_cdf': analytic if cdf_diagnostic else None,
        'empty_evidence_absolute_error': abs(empirical - analytic) if cdf_diagnostic else None,
        'empty_quadrature_points': quadrature_points,
        'empty_includes_hidden_population': True,
        'empty_cdf_diagnostic_status': ('SINGLE_RECTANGLE_REFERENCE' if len(boxes) == 1 else
                                       'REFERENCE_EMPIRICAL_FALLBACK') if cdf_diagnostic else 'NOT_COMPUTED'}


def missed_hidden_moments_fast(mean, covariance, robot_pose, regime, law, order=15):
    """Same GH points and analytic conditionals, accumulated in position space.

    A linear conditional maps the two-dimensional centered GH covariance to
    four state dimensions. This avoids constructing a Q x 4 x 4 outer product
    for every person. Equality is numerical (floating-point roundoff), not a
    new quadrature or posterior approximation.
    """
    mean, covariance = _gaussians(mean, covariance)
    if not len(mean):
        return np.empty(0), mean.copy(), covariance.copy()
    nodes, base_weights = _position_quadrature(order)
    Cpp = covariance[:, :2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(Cpp)
    if np.any(eigenvalues < -1e-8):
        raise ValueError('indefinite hidden position covariance')
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    delta = np.einsum('mij,qj->mqi', factor, nodes)
    positions = mean[:, None, :2] + delta
    pd = np.asarray(law.detection(positions, robot_pose, int(regime)), dtype=float)
    if (pd.shape != positions.shape[:-1] or not np.isfinite(pd).all()
            or np.any((pd < 0) | (pd > 1))):
        raise ValueError('invalid hidden detection probability')
    weights = base_weights[None] * (1. - pd)
    mass = weights.sum(axis=1)
    weights /= np.maximum(mass, np.finfo(float).tiny)[:, None]
    K = covariance[:, :, :2] @ np.linalg.pinv(Cpp)
    conditional_covariance = covariance - K @ covariance[:, :2, :]
    shift = np.einsum('mq,mqk->mk', weights, delta)
    total_weight = weights.sum(axis=1)
    postmean = mean * total_weight[:, None] + np.einsum('mij,mj->mi', K, shift)
    # Center using the actual finite normalized weight sum. This also handles
    # impossible misses (zero mass), whose placeholder covariance is C|p.
    center = shift / np.maximum(total_weight, np.finfo(float).tiny)[:, None]
    centered = delta - center[:, None]
    position_covariance = np.einsum('mq,mqi,mqj->mij', weights, centered, centered)
    postcov = conditional_covariance + K @ position_covariance @ K.swapaxes(-1, -2)
    tiny_offset = (mean + np.einsum('mij,mj->mi', K, center)) - postmean
    postcov += total_weight[:, None, None] * tiny_offset[:, :, None] * tiny_offset[:, None, :]
    return mass, postmean, (postcov + postcov.swapaxes(-1, -2)) * .5


class ObservationWorkspace:
    """One-observation packed emissions; never retained across belief updates.

    ``tracked_misses[i]`` matches law.missed_moments(hypotheses[i], pose).
    ``hidden_emissions[i]`` matches hidden.hypothesis_emissions(...).
    Shared covariance/mean blocks are deduplicated by contents only within
    this invocation. Different optical regimes and GH orders remain separate.
    Positive disintegration still uses the existing rank-aware implementation.
    """
    def __init__(self, hypotheses, z, robot_pose, law):
        from .observation import EndpointObservationLaw
        z = np.asarray(z, dtype=float).reshape(-1, 2)
        self.tracked_misses = [None] * len(hypotheses)
        self.hidden_emissions = [None] * len(hypotheses)
        specs = []; groups = {}; unique = {}
        standard_tracked = getattr(type(law), 'missed_moments', None) is EndpointObservationLaw.missed_moments
        for i, h in enumerate(hypotheses):
            for hidden in (False, True):
                if not hidden and not standard_tracked:
                    self.tracked_misses[i] = law.missed_moments(h, robot_pose)
                    continue
                prefix = 'hidden_' if hidden else ''
                mean = h.hidden_mean if hidden else h.mean
                covariance = h.hidden_covariance if hidden else h.covariance
                mean, covariance = _gaussians(mean, covariance)
                mm = getattr(h, prefix + 'maneuver_mean')
                mixture = mm is not None
                if mixture:
                    mm = np.asarray(mm, dtype=float)
                    cc = np.asarray(getattr(h, prefix + 'maneuver_covariance'), dtype=float)
                    if mm.ndim != 3 or mm.shape[0] != len(mean) or mm.shape[2] != 4 or cc.shape != mm.shape + (4,):
                        raise ValueError('invalid maneuver component dimensions')
                    M, K, _ = mm.shape
                    pp = np.broadcast_to(np.asarray(getattr(h, prefix + 'maneuver_probability'), dtype=float), (M, K))
                    if (not np.isfinite(pp).all() or np.any(pp < 0)
                            or not np.allclose(pp.sum(1), 1., atol=1e-10, rtol=1e-10)):
                        raise ValueError('invalid maneuver probabilities')
                else:
                    M = len(mean); K = 1
                    mm = mean[:, None]; cc = covariance[:, None]; pp = np.ones((M, 1))
                flatm = mm.reshape(-1, 4); flatc = cc.reshape(-1, 4, 4)
                order = 15 if hidden or (len(flatm) and np.max(np.linalg.eigvalsh(flatc[:, :2, :2])) > .25) else 9
                group = (int(h.visibility_mode), order)
                key = (group, flatm.shape, flatm.tobytes(), flatc.tobytes())
                if key not in unique:
                    unique[key] = len(specs)
                    specs.append((flatm, flatc))
                    groups.setdefault(group, []).append(unique[key])
                spec = (i, hidden, unique[key], mm, cc, pp, mixture)
                if not hasattr(self, '_specs'):
                    self._specs = []
                self._specs.append(spec)
        results = [None] * len(specs)
        for (regime, order), indices in groups.items():
            means = np.concatenate([specs[k][0] for k in indices])
            covs = np.concatenate([specs[k][1] for k in indices])
            mass, cm, cv = missed_hidden_moments_fast(means, covs, robot_pose, regime, law, order)
            cursor = 0
            for k in indices:
                end = cursor + len(specs[k][0])
                results[k] = (mass[cursor:end], cm[cursor:end], cv[cursor:end])
                cursor = end
        positive_cache = {}; detection_cache = {}
        for i, hidden, index, mm, cc, pp, mixture in getattr(self, '_specs', []):
            h = hypotheses[i]; M, K, _ = mm.shape
            mass, cm, cv = results[index]
            if mixture:
                mass = mass.reshape(M, K); total = np.sum(pp * mass, axis=1)
                postmean = np.zeros((M, 4)); postcov = np.zeros((M, 4, 4))
                good = total > 0
                if np.any(good):
                    weights = pp[good] * mass[good] / total[good, None]
                    postmean[good], postcov[good] = moment_match(
                        cm.reshape(M, K, 4)[good], cv.reshape(M, K, 4, 4)[good], weights)
            else:
                total, postmean, postcov = mass, cm, cv
            if not hidden:
                self.tracked_misses[i] = (total, postmean, postcov)
                continue
            key = (index, pp.tobytes())
            if key not in positive_cache:
                positive_cache[key] = condition_components_many(mm, cc, pp, z)
            conditioned, density = positive_cache[key]
            regime = int(h.visibility_mode)
            if regime not in detection_cache:
                pd = np.asarray(law.detection(z, robot_pose, regime), dtype=float)
                if pd.shape != (len(z),) or not np.isfinite(pd).all() or np.any((pd < 0) | (pd > 1)):
                    raise ValueError('invalid hidden detection probability')
                logpd = np.full_like(pd, -np.inf)
                np.log(pd, out=logpd, where=pd > 0)
                detection_cache[regime] = logpd
            self.hidden_emissions[i] = (total, postmean, postcov, conditioned,
                                       density + detection_cache[regime][None])
        # Workspace retains only outputs, not duplicated input byte keys or
        # temporary concatenations. Its lifetime is bounded by one observation.
        if hasattr(self, '_specs'):
            del self._specs


@njit(cache=True, fastmath=False)
def _event_features_kernel(positions, velocities, robots, hidden, age, elapsed,
                           forward, sine, cosine, speed_maximum, lateral_maximum):
    Q, T, _ = robots.shape; W, _, N, _ = positions.shape
    out = np.empty((Q, W, T, 12), np.float32)
    for q in range(Q):
        for w in range(W):
            for t in range(T):
                nearest = -1; minimum = 100.; count = 0
                for n in range(N):
                    x = positions[w, t, n, 0]; z = positions[w, t, n, 1]
                    if np.isfinite(x):
                        count += 1
                    dx = x - robots[q, t, 0]; dz = z - robots[q, t, 2]
                    d = math.sqrt(dx * dx + dz * dz)
                    if not np.isfinite(d):
                        d = 100.
                    if nearest < 0 or d < minimum:
                        minimum = d; nearest = n
                speed = 0.; lateral = 10.
                if nearest >= 0:
                    vx = velocities[w, t, nearest, 0]; vz = velocities[w, t, nearest, 1]
                    speed = math.sqrt(vx * vx + vz * vz)
                    if np.isnan(speed):
                        speed = 0.
                    elif np.isinf(speed):
                        speed = speed_maximum
                    dx = positions[w, t, nearest, 0] - robots[q, t, 0]
                    dz = positions[w, t, nearest, 1] - robots[q, t, 2]
                    lateral = abs(dx * sine[q, t] + dz * cosine[q, t])
                    if np.isnan(lateral):
                        lateral = 10.
                    elif np.isinf(lateral):
                        lateral = lateral_maximum
                out[q, w, t, 0] = 1.
                for k in range(4):
                    radius = (.4, .8, 1.5, 3.)[k]
                    out[q, w, t, k + 1] = math.exp(-minimum * minimum / (2 * radius * radius))
                out[q, w, t, 5] = hidden[w] / 4
                out[q, w, t, 6] = count / 4
                out[q, w, t, 7] = forward[q, t]
                out[q, w, t, 8] = speed / 3
                out[q, w, t, 9] = math.exp(-lateral * lateral / .5)
                out[q, w, t, 10] = age[q, w, t]
                out[q, w, t, 11] = elapsed[q, w, t]
    return out


def event_features_many_fast(positions, velocities, robots, hidden, age, elapsed):
    """Fused Q/W/native-time nearest-human features; no Q x W x T x N buffer.

    First-index distance ties, nonfinite padding, all endpoints and all twelve
    features match queries.event_features_many. Numba is single-threaded and
    does not use fastmath. Public wrapper preparation belongs in full latency.
    """
    positions = np.asarray(positions); velocities = np.asarray(velocities)
    robots = np.asarray(robots); hidden = np.asarray(hidden)
    if (positions.ndim != 4 or positions.shape[-1] != 2 or velocities.shape != positions.shape
            or robots.ndim != 3 or robots.shape[-1] != 4 or robots.shape[1] != positions.shape[1]
            or hidden.shape != (positions.shape[0],)):
        raise ValueError('event feature dimensions')
    Q, T, _ = robots.shape; W = positions.shape[0]; shape = (Q, W, T)
    forward = np.concatenate([np.zeros((Q, 1), bool),
        np.linalg.norm(np.diff(robots[:, :, [0, 2]], axis=1), axis=-1) > .05], axis=1)
    elapsed = np.asarray(elapsed)
    if elapsed.ndim == 2:
        elapsed = elapsed[:, None, :]
    age = np.broadcast_to(np.minimum(np.asarray(age), 5) / 5, shape)
    elapsed = np.broadcast_to(elapsed / 3, shape)
    sine = np.sin(robots[:, :, 3]); cosine = np.cos(robots[:, :, 3])
    speed_dtype = velocities.dtype if np.issubdtype(velocities.dtype, np.floating) else np.dtype(float)
    lateral_dtype = np.result_type(positions.dtype, robots.dtype, sine.dtype)
    return _event_features_kernel(positions, velocities, robots, hidden, age, elapsed,
                                 forward, sine, cosine, float(np.finfo(speed_dtype).max),
                                 float(np.finfo(lateral_dtype).max))


def warmup_kernels():
    """Compile the normal float64 event kernel outside online timing."""
    p = np.zeros((1, 2, 1, 2)); r = np.zeros((1, 2, 4))
    event_features_many_fast(p, p, r, np.zeros(1), 0., np.arange(2) * .048)
