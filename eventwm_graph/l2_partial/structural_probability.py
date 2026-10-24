"""Local conservative probability accounting; no invented residual point law.

For a declared input measure P >= sum_i w_i P_i, the remainder has mass u.
If L_i is the evidence of P_i and the remainder's evidence is at most M,
then P(.|o) >= sum_i w_i L_i/(sum_j w_j L_j + u*M) P_i(.|o).

This statement requires the actual component conditionals and a common
dominating observation measure. Moment matching and numerical quadrature are
NOT automatically covered by this proof. Callers must report those errors
separately. Unknown density bounds give a vacuous, explicitly unresolved
result rather than a fabricated posterior or event probability.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


GUARANTEE_SCOPE = (
    "local lower-submeasure accounting for the declared incoming component "
    "measures, their supplied exact conditionals/evidences, and a certified "
    "residual evidence upper bound; excludes unbounded Gaussian projection, "
    "quadrature, model error, and earlier unaccounted approximation"
)


@dataclass(frozen=True)
class CountSupport:
    """Possible total counts, or an unbounded integer support above minimum."""

    values: Optional[Tuple[int, ...]] = None
    minimum: int = 0

    def __post_init__(self):
        if int(self.minimum) != self.minimum or self.minimum < 0:
            raise ValueError("count minimum must be a nonnegative integer")
        object.__setattr__(self, "minimum", int(self.minimum))
        if self.values is not None:
            values = tuple(self.values)
            if any(int(v) != v or v < 0 for v in values):
                raise ValueError("population support must contain nonnegative integers")
            values = tuple(sorted(set(int(v) for v in values if v >= self.minimum)))
            object.__setattr__(self, "values", values)
            if values:
                object.__setattr__(self, "minimum", min(values))

    @property
    def maximum(self):
        return None if self.values is None else max(self.values, default=-1)

    @property
    def impossible(self):
        return self.values == ()

    def after_detections(self, count):
        if int(count) != count or count < 0:
            raise ValueError("observed count must be a nonnegative integer")
        return CountSupport(self.values, max(self.minimum, int(count)))

    def union(self, other):
        if self.impossible:
            return other
        if other.impossible:
            return self
        if self.values is None or other.values is None:
            return CountSupport(None, min(self.minimum, other.minimum))
        return CountSupport(self.values + other.values)


@dataclass(frozen=True)
class ObservationMeasure:
    """A common reference, not merely the numerical units of a density.

    Set dimensions are per observed point: zero for an atom, one for a line,
    two for planar Lebesgue measure. For a mixed/singular set, reference_id
    must identify the shared atoms/lines; a rank alone does not identify them.
    """

    kind: str = "event"
    dimensions: Tuple[int, ...] = ()
    reference_id: Optional[str] = "event-probability"

    def __post_init__(self):
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if self.kind not in ("event", "set"):
            raise ValueError("unknown observation measure")
        if self.kind == "event" and self.dimensions:
            raise ValueError("event measure has no position dimensions")
        if any(d not in (0, 1, 2) for d in self.dimensions):
            raise ValueError("position support dimensions must be 0, 1 or 2")

    @property
    def reference_identified(self):
        return bool(self.reference_id)


@dataclass(frozen=True)
class EvidenceBound:
    log_upper: float
    measure: ObservationMeasure
    status: str
    provenance: str

    def __post_init__(self):
        if math.isnan(self.log_upper):
            raise ValueError("NaN evidence bound")
        if not self.status or not self.provenance:
            raise ValueError("evidence bound requires status and provenance")

    @classmethod
    def unknown(cls, measure, reason="no certified residual evidence bound"):
        return cls(math.inf, measure, "UNBOUNDED_EVIDENCE", reason)

    @classmethod
    def event(cls, measure=None, upper=1.0, provenance="event probability is at most one"):
        measure = measure or ObservationMeasure()
        if measure.kind != "event":
            raise ValueError("an event probability bound is not a continuous density bound")
        if not math.isfinite(upper) or not 0 <= upper <= 1:
            raise ValueError("event upper probability must be in [0,1]")
        return cls(math.log(upper) if upper else -math.inf, measure,
                   "EVENT_PROBABILITY_BOUND", provenance)


@dataclass(frozen=True)
class SubmeasureUpdate:
    retained_weights: np.ndarray
    unresolved_mass: float
    log_evidence_lower: float
    log_evidence_upper: float
    status: str
    posterior_valid: bool
    residual_count_support: CountSupport
    bound_status: str
    guarantee_scope: str = GUARANTEE_SCOPE
    approximation_notes: Tuple[str, ...] = ()

    def metadata(self):
        return {
            "status": self.status, "posterior_valid": self.posterior_valid,
            "bound_status": self.bound_status, "guarantee_scope": self.guarantee_scope,
            "approximation_notes": list(self.approximation_notes),
            "residual_count_support": self.residual_count_support.values,
            "residual_count_minimum": self.residual_count_support.minimum,
            "point_residual_law": False,
            "positive_evidence_certified": math.isfinite(self.log_evidence_lower),
        }


def _weights(weights, unresolved_mass):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be a finite nonnegative vector")
    u = float(unresolved_mass)
    if not math.isfinite(u) or not 0 <= u <= 1:
        raise ValueError("unresolved mass must be in [0,1]")
    if abs(float(w.sum()) + u - 1.) > 1e-10:
        raise ValueError("retained plus unresolved mass must equal one")
    return w, u


def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return -math.inf
    return float(np.logaddexp.reduce(values))


def _coerce_support(value):
    if isinstance(value, CountSupport):
        return value
    return CountSupport(None if value is None else tuple(value))


def bayes_submeasure(
    prior_weights,
    component_log_likelihoods,
    unresolved_mass,
    residual_bound,
    *,
    component_measure,
    component_count_supports=None,
    residual_count_support=None,
    observed_count=None,
    approximation_notes=(),
):
    """Return a lower posterior submeasure and an explicit unspecified remainder.

    The caller supplies component posterior distributions separately. This
    function never changes a density, supplies a fake residual likelihood,
    or certifies an approximate component conditional as exact.
    """
    w, u = _weights(prior_weights, unresolved_mass)
    log_lik = np.asarray(component_log_likelihoods, dtype=float)
    if log_lik.shape != w.shape or np.isnan(log_lik).any() or np.isposinf(log_lik).any():
        raise ValueError("component log likelihoods must be finite or negative infinity")
    if not isinstance(component_measure, ObservationMeasure):
        raise TypeError("declare the common component observation measure")
    probability_measure = component_measure.kind == "event" or (
        component_measure.kind == "set" and all(d == 0 for d in component_measure.dimensions)
    )
    if probability_measure and np.any(log_lik > 1e-12):
        raise ValueError("event/atomic-set likelihood cannot exceed probability one")
    if residual_bound is None:
        residual_bound = EvidenceBound.unknown(component_measure)
    if residual_bound.measure != component_measure:
        raise ValueError("component and residual bounds use different reference measures")
    if observed_count is None:
        observed_count = len(component_measure.dimensions) if component_measure.kind == "set" else 0
    if int(observed_count) != observed_count or observed_count < 0:
        raise ValueError("invalid observed count")
    observed_count = int(observed_count)
    if component_measure.kind == "set" and observed_count != len(component_measure.dimensions):
        raise ValueError("set size and observed count disagree")
    if component_count_supports is None:
        supports = [CountSupport() for _ in w]
    else:
        if len(component_count_supports) != len(w):
            raise ValueError("one count support is required per retained component")
        supports = [_coerce_support(s) for s in component_count_supports]
    active = (w > 0) & np.isfinite(log_lik)
    for is_active, support in zip(active, supports):
        if is_active and support.after_detections(observed_count).impossible:
            raise ValueError("positive evidence contradicts retained population support")
    old_residual = _coerce_support(residual_count_support).after_detections(observed_count)
    log_m = float(residual_bound.log_upper)
    bound_status = residual_bound.status
    if old_residual.impossible:
        log_m = -math.inf
        bound_status = "IMPOSSIBLE_COUNT"
    # An unidentified singular reference cannot supply a finite global bound.
    if u and not component_measure.reference_identified and log_m != -math.inf:
        log_m = math.inf
        bound_status = "UNIDENTIFIED_REFERENCE"
    log_w = np.full_like(w, -np.inf)
    np.log(w, out=log_w, where=w > 0)
    terms = log_w + log_lik
    log_a = _logsumexp(terms)
    log_b = math.log(u) + log_m if u else -math.inf
    log_den = float(np.logaddexp(log_a, log_b))
    posterior_valid = log_den != -math.inf
    if log_den == -math.inf:
        result = np.zeros_like(w)
        status = "UNSUPPORTED_OBSERVATION"
    elif log_den == math.inf:
        result = np.zeros_like(w)
        status = "VACUOUS_RESIDUAL_BOUND"
    else:
        result = np.exp(terms - log_den)
        if log_b == -math.inf:
            # The residual has exactly zero posterior evidence. Ordinary
            # floating-point normalization must not fabricate a tiny unknown
            # channel that becomes vacuous at the next unbounded-density update.
            result /= result.sum()
            status = "EXACT_FINITE_BAYES"
        else:
            # Round down rather than silently normalize lower masses upward.
            total = float(result.sum())
            if total > 1:
                result /= np.nextafter(total, math.inf)
            result = np.nextafter(result, np.zeros_like(result))
            status = "LOCAL_SUBMEASURE_BOUND"
    remainder = 0. if posterior_valid and log_b == -math.inf else float(max(0., 1. - result.sum()))
    # A loose normalizer transfers part of retained components into the
    # unspecified remainder. Its support is not only the old residual's.
    count_support = CountSupport(())
    if u and log_m != -math.inf:
        count_support = count_support.union(old_residual)
    for is_active, support in zip(active, supports):
        if is_active:
            count_support = count_support.union(support.after_detections(observed_count))
    if not posterior_valid:
        # A recovery state must keep arrived count constraints, while explicitly
        # refusing to call itself a posterior of the contradicted old model.
        count_support = CountSupport(None, observed_count)
    result.setflags(write=False)
    return SubmeasureUpdate(
        result, remainder, log_a, log_den, status, posterior_valid,
        count_support, bound_status, approximation_notes=tuple(approximation_notes),
    )


def event_probability_bounds(weights, event_probabilities, unresolved_mass):
    w, u = _weights(weights, unresolved_mass)
    p = np.asarray(event_probabilities, dtype=float)
    if p.shape != w.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("one valid event probability is required per component")
    lower = float(w @ p)
    return lower, min(1., lower + u)


def partition_probability_bounds(weights, component_probabilities, unresolved_mass):
    """Return coupled branch bounds, retaining the residual allocation simplex."""
    w, u = _weights(weights, unresolved_mass)
    p = np.asarray(component_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(w) or not np.isfinite(p).all():
        raise ValueError("partition probabilities must be a finite component-by-branch matrix")
    if np.any(p < 0) or np.any(p > 1) or np.any(np.abs(p.sum(1) - 1.) > 1e-10):
        raise ValueError("each component must define a normalized partition")
    lower = w @ p
    return {
        "lower": lower, "upper": np.minimum(1., lower + u),
        "residual_simplex_mass": u,
        "coupling": "one nonnegative residual allocation across branches summing to residual_simplex_mass",
        "point_branch_probabilities": False,
    }


def prune_submeasure(weights, kept_indices, unresolved_mass):
    """Preserve omitted mass exactly; caller must union its physical support."""
    w, u = _weights(weights, unresolved_mass)
    ids = np.asarray(kept_indices)
    if ids.ndim != 1 or any(int(i) != i for i in ids):
        raise ValueError("kept indices must be an integer vector")
    ids = ids.astype(int)
    if len(set(ids.tolist())) != len(ids) or np.any((ids < 0) | (ids >= len(w))):
        raise ValueError("duplicate or invalid kept index")
    keep = w[ids].copy()
    removed = np.ones(len(w), dtype=bool)
    removed[ids] = False
    return keep, min(1., float(u + w[removed].sum()))


def gaussian_noise_log_supremum(position_noise_covariance, *, elapsed_time):
    """Supremum for convolution with caller-certified independent Gaussian noise.

    A positive elapsed time and positive-definite supplied noise covariance are
    required. A posterior moment covariance is NOT such a kernel certificate.
    At zero time, singular noise or an unknown lower noise floor, use infinity.
    """
    elapsed_time = float(elapsed_time)
    if not math.isfinite(elapsed_time) or elapsed_time < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    cov = np.asarray(position_noise_covariance, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("position noise covariance must be finite symmetric 2x2")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-12:
        raise ValueError("position noise covariance is not positive semidefinite")
    if elapsed_time == 0 or eigenvalues.min() <= 0:
        return math.inf
    return float(-math.log(2. * math.pi) - .5 * np.log(eigenvalues).sum())


def finite_set_evidence_bound(
    count_support, per_detection_log_suprema, measure, *,
    joint_domination_certified=False,
    provenance="caller-certified joint assignment density domination on a common reference",
):
    """Conservative injective-assignment bound, with explicit support statuses.

    For multiple points, products of MARGINAL density suprema are insufficient:
    a perfectly correlated joint law can be singular despite smooth marginals.
    The caller must certify that every joint assignment density is bounded by
    the supplied product. Independent additive Gaussian noise across people is
    one sufficient condition, even when their pre-transition states correlate.
    """
    if measure.kind != "set":
        raise ValueError("finite-set evidence needs a set observation measure")
    support = _coerce_support(count_support)
    D = len(measure.dimensions)
    upper = np.asarray(per_detection_log_suprema, dtype=float)
    if upper.shape != (D,) or np.isnan(upper).any():
        raise ValueError("one non-NaN log supremum is required per detection")
    if support.maximum is not None and support.maximum < D:
        return EvidenceBound(-math.inf, measure, "IMPOSSIBLE_COUNT", provenance)
    if not D:
        return EvidenceBound(0., measure, "EMPTY_SET_EVENT_BOUND", "empty-set observation is an event")
    if not measure.reference_identified:
        return EvidenceBound.unknown(measure, "singular/mixed reference support has not been identified")
    if all(d == 0 for d in measure.dimensions):
        return EvidenceBound(0., measure, "ATOMIC_SET_PROBABILITY_BOUND", "unit finite-set atom has probability at most one")
    if D > 1 and not joint_domination_certified:
        return EvidenceBound.unknown(measure, "marginal density bounds do not certify joint assignment domination")
    if support.maximum is None:
        return EvidenceBound.unknown(measure, "no certified finite population upper support")
    if np.isposinf(upper).any():
        return EvidenceBound.unknown(measure, "at least one point density has no finite upper bound")
    N = support.maximum
    log_assignments = sum(math.log(N - j) for j in range(D))
    value = -math.inf if np.isneginf(upper).any() else log_assignments + float(upper.sum())
    return EvidenceBound(value, measure, "INJECTIVE_SET_DENSITY_BOUND", provenance)


def canonical_impulse_velocity_variance(
    variance, intervals, *, interval=.048, jump_rate=2., jump_std=2.,
    relaxation=.7, diffusion=.25,
):
    """Exact second moment of the declared mean-preserving canonical impulse law.

    Resets occur at interval START with Bernoulli probability
    1-exp(-jump_rate*interval); this is not a claim of a full Poisson process.
    Partial API intervals must reuse the canonical latent and are not extra
    trials. This helper audits the law; it does not select its parameters.
    """
    if int(intervals) != intervals or intervals < 0:
        raise ValueError("interval count must be a nonnegative integer")
    parameters = (interval, jump_rate, jump_std, relaxation, diffusion)
    if not all(math.isfinite(x) for x in parameters) or interval <= 0 or min(parameters[1:]) < 0:
        raise ValueError("invalid impulse/OU parameters")
    v = np.asarray(variance, dtype=float)
    if not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("variance must be finite and nonnegative")
    n = int(intervals)
    p = -math.expm1(-jump_rate * interval)
    decay = math.exp(-2. * relaxation * interval)
    if relaxation == 0:
        return v + n * (p * jump_std**2 + diffusion * interval)
    one_minus_decay = -math.expm1(-2. * relaxation * interval)
    noise = decay * p * jump_std**2 + diffusion * one_minus_decay / (2. * relaxation)
    total_decay = math.exp(-2. * relaxation * interval * n)
    geometric = -math.expm1(-2. * relaxation * interval * n) / one_minus_decay
    return total_decay * v + noise * geometric
