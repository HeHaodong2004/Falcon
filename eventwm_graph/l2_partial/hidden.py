"""Persistent broad hidden-person law, independent of the robot's query.

The initial Gaussian is anchored once to an arrived legal robot pose. It is a
spatial probability density, NOT a traversability/visibility certificate and
NOT a known position in unobserved geometry. The continuous transition substep
is zero-target integrated OU. The same substep is available for analytic
filtering and shared-world sampling. ``repair_model`` additionally applies the
shared native-clock maneuver/jump law; the pure OU helpers here do not insert
that discrete event and must not be presented as its complete transition.

An exchangeable hidden population has one iid Gaussian density and a count
pmf. A shared world instead has a fixed number of private hidden states; their
identities are internal latent bookkeeping, never simulator identities. Those
states must remain separate when conditioning a successor. Averaging them
back into one iid density loses observation-induced dependence.

Exact first detections use p_D(z) * phi_position(z) and Gaussian conditional
velocity. A miss uses position-only Gauss-Hermite integration and analytic
conditional velocity moments, followed by Gaussian moment matching. That
integral/closure is approximate; neither
this module nor its subset dynamic program makes it exact. The subset program
is exact only conditional on its supplied emission likelihoods.
"""
from functools import lru_cache

import numpy as np

from .maneuver import advance_components, condition_components_many, moment_match
from .probability import transition_matrices


HIDDEN_DIFFUSION = .25
HIDDEN_RELAXATION = .7
HIDDEN_POSITION_VARIANCE = 100.
HIDDEN_VELOCITY_VARIANCE = 4.


def _gaussians(mean, covariance):
    m = np.asarray(mean, dtype=float)
    c = np.asarray(covariance, dtype=float)
    if m.ndim != 2 or m.shape[1:] != (4,) or c.shape != (len(m), 4, 4):
        raise ValueError('hidden Gaussians require [M,4] and [M,4,4]')
    if not np.isfinite(m).all() or not np.isfinite(c).all():
        raise ValueError('nonfinite hidden Gaussian')
    return m, c


def prepare_prior(robot_pose):
    """Create the prior ONCE; never recenter after a robot action or a miss."""
    pose = np.asarray(robot_pose, dtype=float)
    if pose.shape != (4,) or not np.isfinite(pose).all():
        raise ValueError('legal robot pose must be [x,y,z,yaw]')
    mean = np.array([[pose[0], pose[2], 0., 0.]])
    covariance = np.diag([HIDDEN_POSITION_VARIANCE, HIDDEN_POSITION_VARIANCE,
                          HIDDEN_VELOCITY_VARIANCE, HIDDEN_VELOCITY_VARIANCE])[None]
    return mean, covariance


def propagate_hidden_gaussian(mean, covariance, dt):
    """Semigroup OU propagation of each exchangeable/private Gaussian."""
    mean, covariance = _gaussians(mean, covariance)
    m, c = advance_components(mean[:, None], covariance[:, None], dt,
                              np.zeros((len(mean), 2)),
                              np.full(len(mean), HIDDEN_DIFFUSION),
                              HIDDEN_RELAXATION)
    return m[:, 0], c[:, 0]


@lru_cache(maxsize=8)
def _position_quadrature(order):
    if int(order) != order or order < 3:
        raise ValueError('hidden integration order must be at least three')
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    a, b = np.meshgrid(nodes, nodes, indexing='ij')
    points = np.sqrt(2.) * np.stack([a.ravel(), b.ravel()], axis=-1)
    weights = (np.outer(weights, weights) / np.pi).ravel()
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


def missed_hidden_moments(mean, covariance, robot_pose, regime, law, order=15):
    """Integrate the SAME pointwise p_D law over position; moment-match miss.

    Default 15x15 Gauss-Hermite is bounded and stronger than the tracked
    nine-point rule for this broad prior. It is not exact. ``order`` is exposed
    for convergence diagnostics; high-order references belong offline. Full
    state moments use analytic X|position, so no four-dimensional quadrature
    or extra velocity approximation is introduced. Degenerate position support
    uses the Gaussian pseudoinverse conditional and is preserved.
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
    conditional_mean = mean[:, None, :] + np.einsum('mij,mqj->mqi', K, delta)
    conditional_covariance = covariance - K @ covariance[:, :2, :]
    posterior_mean = np.einsum('mq,mqk->mk', weights, conditional_mean)
    centered = conditional_mean - posterior_mean[:, None, :]
    posterior_covariance = conditional_covariance + np.einsum(
        'mq,mqi,mqj->mij', weights, centered, centered)
    posterior_covariance = (posterior_covariance + posterior_covariance.swapaxes(-1, -2)) * .5
    return mass, posterior_mean, posterior_covariance


def first_detection_moments(mean, covariance, z, robot_pose, regime, law):
    """Exact-coordinate conditional Gaussians and unnormalized log densities.

    Returns ``((mean[M,D,4],cov[M,D,4,4]),log_joint[M,D])``. Unsupported pairs
    have -inf density and placeholder zero moments. A caller must not retain
    those moments as a supported posterior. No birth/clutter density is mixed
    into the hidden-person likelihood.
    """
    mean, covariance = _gaussians(mean, covariance)
    z = np.asarray(z, dtype=float).reshape(-1, 2)
    moments, log_density = condition_components_many(
        mean[:, None], covariance[:, None], np.ones((len(mean), 1)), z)
    pd = np.asarray(law.detection(z, robot_pose, int(regime)), dtype=float)
    if pd.shape != (len(z),) or np.any((pd < 0) | (pd > 1)) or not np.isfinite(pd).all():
        raise ValueError('invalid hidden detection probability')
    log_pd = np.full_like(pd, -np.inf)
    np.log(pd, out=log_pd, where=pd > 0)
    return moments, log_density + log_pd[None]


def hypothesis_emissions(h, z, robot_pose, law):
    """Hidden miss and exact detections under the retained predictive measure.

    Returns ``(miss[M], missmean[M,4], misscov[M,4,4],
    (condmean[M,D,4], condcov[M,D,4,4]), logjoint[M,D])``. Interior native-clock
    maneuver alternatives are integrated individually before any posterior
    moment matching. Thus a no-observation continuation does not silently
    replace rare jump components with their summary Gaussian. Quadrature and
    posterior moment matching remain declared approximations.
    """
    z = np.asarray(z, dtype=float).reshape(-1, 2)
    mean, covariance = _gaussians(h.hidden_mean, h.hidden_covariance)
    mixture_mean = getattr(h, 'hidden_maneuver_mean', None)
    if mixture_mean is None:
        mass, postmean, postcov = missed_hidden_moments(
            mean, covariance, robot_pose, h.visibility_mode, law)
        conditioned, logjoint = first_detection_moments(
            mean, covariance, z, robot_pose, h.visibility_mode, law)
        return mass, postmean, postcov, conditioned, logjoint
    mm = np.asarray(mixture_mean, dtype=float)
    cc = np.asarray(h.hidden_maneuver_covariance, dtype=float)
    pp = np.asarray(h.hidden_maneuver_probability, dtype=float)
    if mm.ndim != 3 or mm.shape[0] != len(mean) or mm.shape[2] != 4 or cc.shape != mm.shape + (4,):
        raise ValueError('invalid hidden maneuver component dimensions')
    M, K, _ = mm.shape
    pp = np.broadcast_to(pp, (M, K))
    if (np.any(pp < 0) or not np.isfinite(pp).all()
            or not np.allclose(pp.sum(axis=1), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid hidden maneuver probabilities')
    mass, cm, cv = missed_hidden_moments(
        mm.reshape(-1, 4), cc.reshape(-1, 4, 4), robot_pose, h.visibility_mode, law)
    mass = mass.reshape(M, K)
    total = np.sum(pp * mass, axis=1)
    postmean = np.zeros_like(mean); postcov = np.zeros_like(covariance)
    good = total > 0
    if np.any(good):
        weights = pp[good] * mass[good] / total[good, None]
        postmean[good], postcov[good] = moment_match(
            cm.reshape(M, K, 4)[good], cv.reshape(M, K, 4, 4)[good], weights)
    conditioned, logdensity = condition_components_many(mm, cc, pp, z)
    pd = np.asarray(law.detection(z, robot_pose, int(h.visibility_mode)), dtype=float)
    if pd.shape != (len(z),) or not np.isfinite(pd).all() or np.any((pd < 0) | (pd > 1)):
        raise ValueError('invalid hidden detection probability')
    logpd = np.full_like(pd, -np.inf)
    np.log(pd, out=logpd, where=pd > 0)
    return total, postmean, postcov, conditioned, logdensity + logpd[None]


def sample_hidden_states(mean, covariance, count, rng, exchangeable=True):
    """Sample private hidden states once per common exogenous world.

    Exchangeable input must have one Gaussian; private input has exactly
    ``count`` Gaussians. No robot pose is accepted, preventing query-dependent
    recentring. Samples are hypotheses, never legal observed coordinates.
    """
    mean, covariance = _gaussians(mean, covariance)
    if int(count) != count or count < 0:
        raise ValueError('hidden sample count must be a nonnegative integer')
    count = int(count)
    if exchangeable:
        if len(mean) != 1:
            raise ValueError('iid hidden prior must contain one Gaussian')
        mean = np.broadcast_to(mean, (count, 4))
        covariance = np.broadcast_to(covariance, (count, 4, 4))
    elif len(mean) != count:
        raise ValueError('private hidden state count mismatch')
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues < -1e-8):
        raise ValueError('indefinite hidden covariance')
    noise = rng.normal(size=(count, 4)) * np.sqrt(np.maximum(eigenvalues, 0))
    return mean + np.einsum('nij,nj->ni', eigenvectors, noise)


def advance_hidden_samples(states, dt, rng):
    """Sample the same OU kernel as propagate_hidden_gaussian, in one batch."""
    states = np.asarray(states, dtype=float)
    if states.shape[-1:] != (4,) or not np.isfinite(states).all():
        raise ValueError('invalid private hidden states')
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('invalid hidden time interval')
    if dt == 0:
        return states.copy()
    F, Q, _, _ = transition_matrices(float(dt), HIDDEN_RELAXATION)
    # Q is positive definite for dt>0. Eigenfactorization also handles tiny
    # positive intervals without adding undeclared numerical process noise.
    eigenvalues, eigenvectors = np.linalg.eigh(Q * HIDDEN_DIFFUSION)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None]
    result = states @ F.T + rng.normal(size=states.shape) @ factor.T
    result[..., :2] += rng.normal(size=states[..., :2].shape) * np.sqrt(.0025 * dt)
    return result


def hidden_subset_likelihoods(assign, miss):
    """Likelihood that private hidden slots explain each detection subset.

    All unassigned hidden slots are missed. Entry ``mask`` sums every
    injective assignment from that subset of detections to the private slots;
    unselected detections may be explained by tracked people elsewhere. The
    reference measure is the unordered finite-set measure (1/D!). Complexity
    is O(M D 2**D); callers must enforce their observation-size budget.
    """
    assign = np.asarray(assign, dtype=float)
    miss = np.asarray(miss, dtype=float)
    if assign.ndim != 2 or miss.shape != (len(assign),):
        raise ValueError('invalid private hidden likelihood dimensions')
    if (not np.isfinite(assign).all() or not np.isfinite(miss).all()
            or np.any(assign < 0) or np.any((miss < 0) | (miss > 1))):
        raise ValueError('invalid private hidden likelihood')
    M, D = assign.shape
    masks = np.arange(1 << D)
    dp = np.zeros(1 << D)
    dp[0] = 1.
    for i in range(M):
        nxt = dp * miss[i]
        for j in range(D):
            selected = masks[(masks & (1 << j)) != 0]
            nxt[selected] += dp[selected ^ (1 << j)] * assign[i, j]
        dp = nxt
    return dp


def exchangeable_count_update(pmf, miss, first_likelihood):
    """Count posterior when this iid hidden population explains all b points.

    Likelihood at original count H is
    ``falling(H,b) * miss**(H-b) * product(first_likelihood)``.
    ``first_likelihood[j]`` is p_D(z_j)phi_position(z_j), not a normalized birth
    proposal. Returns absolute evidence and remaining-count pmf. Zero evidence
    returns a zero pmf and must be handled as unsupported, never normalized.
    """
    pmf = np.asarray(pmf, dtype=float)
    first_likelihood = np.asarray(first_likelihood, dtype=float)
    if (pmf.ndim != 1 or not len(pmf) or np.any(pmf < 0)
            or not np.isfinite(pmf).all() or abs(pmf.sum() - 1) > 1e-8):
        raise ValueError('invalid hidden count prior')
    if (not np.isfinite(miss) or not 0 <= miss <= 1 or first_likelihood.ndim != 1
            or not np.isfinite(first_likelihood).all() or np.any(first_likelihood < 0)):
        raise ValueError('invalid exchangeable emission likelihood')
    b = len(first_likelihood)
    count = np.arange(len(pmf))
    falling = np.ones(len(pmf))
    for j in range(b):
        falling *= np.maximum(count - j, 0)
    terms = pmf * falling * miss ** np.maximum(count - b, 0)
    terms *= float(np.prod(first_likelihood))
    evidence = float(terms.sum())
    posterior = np.zeros_like(pmf)
    if evidence > 0 and b < len(pmf):
        posterior[:len(pmf) - b] = terms[b:] / evidence
    return evidence, posterior
