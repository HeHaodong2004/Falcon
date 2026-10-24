"""Prepared exact Bernoulli/Poisson count conditioning, without a count cutoff.

Preparation is parameter dependent and belongs inside each model call. Repeated
rejection blocks reuse its immutable suffix recursion. A finite Poisson prefix
is only one branch of a mixture: its strictly positive remaining tail is sampled
separately and uses the original unbounded conditional-Poisson implementation.
No global CDF is rounded to one and no possible count is discarded.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson

from .count import suffix_tails
from .types import immutable_array

VERSION = 'prepared_count_suffix_shared_grid_explicit_unbounded_tail_2'


def _reference_truncated(rate, lower, size, rng):
    """Original count.py inverse/recurrence, including explicit numeric failure."""
    survival = poisson.sf(lower-1, rate)
    if not survival > 0:
        raise RuntimeError('truncated Poisson tail underflow')
    uniform = (rng.random(size)+.5*np.finfo(float).eps).clip(
        np.finfo(float).tiny, 1-np.finfo(float).eps)
    sampled = poisson.isf(uniform*survival, rate)
    rare = rate < float(lower) or survival < 1e-8
    fallback = np.ones(size, bool) if rare else (~np.isfinite(sampled)) | (sampled < lower)
    if fallback.any():
        u = uniform[fallback]
        mass = float(np.exp(poisson.logpmf(lower, rate)-poisson.logsf(lower-1, rate)))
        cumulative = mass
        value = int(lower)
        answer = np.full(len(u), value, int)
        pending = u > cumulative
        while pending.any():
            value += 1
            mass *= rate/value
            cumulative += mass
            answer[pending] = value
            pending = u > cumulative
            if value-lower > 100000 or (mass == 0 and pending.any()):
                raise RuntimeError('conditional Poisson recurrence exhausted numeric support')
        sampled[fallback] = answer
    if not np.isfinite(sampled).all():
        raise RuntimeError('truncated Poisson inverse unsupported')
    return sampled.astype(np.int64)


@dataclass(frozen=True)
class _PoissonPrefix:
    lower: int
    upper: int
    conditional_cdf: np.ndarray
    log_pmf: np.ndarray
    tail_probability: float
    reference_only: bool


def _prepare_prefix(rate, lower, shared=None):
    """Finite-branch conditional CDF plus a positive, unbounded tail branch."""
    empty = immutable_array(np.empty(0, np.float64))
    log_survival = float(poisson.logsf(lower-1, rate) if shared is None else shared[0][lower])
    if not np.isfinite(log_survival):
        # This state may be unreachable after certain Bernoulli sources. Do not
        # turn its numerical failure into either fabricated or truncated counts.
        return _PoissonPrefix(lower, lower-1, empty, empty, math.nan, True)
    upper_limit = max(lower, int(math.ceil(rate+8*math.sqrt(rate+1))))
    if upper_limit-lower > 65536:
        return _PoissonPrefix(lower, lower-1, empty, empty, math.nan, True)
    values = np.arange(lower, upper_limit+1, dtype=np.int64)
    log_tails = (poisson.logsf(values, rate) if shared is None else shared[0][values+1])-log_survival
    candidates = np.flatnonzero(log_tails <= math.log(1e-6))
    if not len(candidates):
        return _PoissonPrefix(lower, lower-1, empty, empty, math.nan, True)
    at = int(candidates[0])
    upper = int(values[at])
    tail = float(np.exp(log_tails[at]))
    if not np.isfinite(tail) or not 0 < tail < 1:
        # In particular, an underflowed positive tail is NOT silently set to 0.
        return _PoissonPrefix(lower, lower-1, empty, empty, math.nan, True)
    log_pmf = np.asarray(poisson.logpmf(values[:at+1], rate) if shared is None else shared[1][values[:at+1]], np.float64)
    finite_probability = np.exp(log_pmf-logsumexp(log_pmf))
    cdf = np.cumsum(finite_probability)
    cdf /= cdf[-1]
    # This is conditional on the finite branch. The global tail stays positive.
    cdf[-1] = 1.
    return _PoissonPrefix(lower, upper, immutable_array(cdf),
                          immutable_array(log_pmf), tail, False)


@dataclass(frozen=True, init=False)
class PreparedCountSampler:
    """Immutable q(count allocation | total >= minimum), or a fixed total.

    ``sample`` returns Bernoulli source indicators and Poisson hidden counts.
    It changes the random-number coupling relative to count.py, not the law.
    ``log_normalizer`` is the same tail/PMF suffix value as that implementation.
    All neural-parameter-dependent preparation must remain inside timed inference.
    """
    rate: float
    minimum: int
    total: object
    log_normalizer: float
    suffix: np.ndarray
    probabilities: np.ndarray
    prefixes: tuple

    def __init__(self, logits, age, rate, minimum, certain=None, total=None):
        logits = np.asarray(logits, np.float64)
        age = np.asarray(age, np.float64)
        rate = float(rate)
        if logits.ndim != 1 or age.shape != logits.shape or not np.isfinite(logits).all():
            raise ValueError('finite one-dimensional logits and matching ages required')
        if not np.isfinite(age).all() or np.any(age < 0) or not np.isfinite(rate) or rate < 0:
            raise ValueError('finite nonnegative age and Poisson rate required')
        if int(minimum) != minimum or minimum < 0:
            raise ValueError('nonnegative integer minimum required')
        minimum = int(minimum)
        forced = age < 1e-7
        if certain is not None:
            certain = np.asarray(certain, bool)
            if certain.shape != age.shape:
                raise ValueError('certain source mask mismatch')
            forced |= certain
        yes = np.where(forced, 0., -np.logaddexp(0., -logits))
        no = np.where(forced, -math.inf, -np.logaddexp(0., logits))
        if total is None:
            table, yes, no = suffix_tails(logits, age, rate, minimum, forced)
            maximum = minimum
            # Every prefix uses the same integer Poisson values. Calculate the
            # exact SciPy functions in two vector calls instead of redispatching
            # SF/PMF machinery for each possible lower count.
            upper = max(minimum, int(math.ceil(rate+8*math.sqrt(rate+1))))
            shared = None
            if minimum and upper <= 65536:
                shared = (poisson.logsf(np.arange(-1, upper+1), rate),
                          poisson.logpmf(np.arange(upper+1), rate))
            prefixes = tuple(_prepare_prefix(rate, lower, shared) for lower in range(1, minimum+1))
        else:
            if int(total) != total or total < minimum:
                raise ValueError('fixed integer total must be at least the minimum')
            total = maximum = int(total)
            table = np.empty((len(logits)+1, total+1), np.float64)
            table[-1] = poisson.logpmf(np.arange(total+1), rate)
            for i in range(len(logits)-1, -1, -1):
                table[i, 0] = no[i]+table[i+1, 0]
                table[i, 1:] = np.logaddexp(no[i]+table[i+1, 1:], yes[i]+table[i+1, :-1])
            prefixes = ()
        normalizer = float(table[0, maximum])
        if not np.isfinite(normalizer):
            raise RuntimeError('population normalizer underflow; no fabricated worlds')
        probability = np.zeros((len(logits), maximum+1), np.float64)
        needed = np.arange(maximum+1)
        for i in range(len(logits)):
            supported = np.isfinite(table[i])
            with np.errstate(invalid='ignore'):
                logp = yes[i]+table[i+1, np.maximum(needed-1, 0)]-table[i]
            probability[i, supported] = np.exp(logp[supported]).clip(0, 1)
            if total is not None:
                probability[i, 0] = 0.
        for key, value in (('rate', rate), ('minimum', minimum), ('total', total),
                           ('log_normalizer', normalizer), ('suffix', immutable_array(table)),
                           ('probabilities', immutable_array(probability)), ('prefixes', prefixes)):
            object.__setattr__(self, key, value)

    def sample(self, worlds, rng):
        if int(worlds) != worlds or worlds < 0 or not isinstance(rng, np.random.Generator):
            raise ValueError('nonnegative integer world count and numpy Generator required')
        worlds = int(worlds)
        needed = np.full(worlds, self.minimum if self.total is None else self.total, np.int64)
        present = np.zeros((worlds, len(self.probabilities)), bool)
        for i in range(len(self.probabilities)):
            present[:, i] = rng.random(worlds) < self.probabilities[i, needed]
            needed = np.maximum(needed-present[:, i], 0)
        if self.total is not None:
            return present, needed
        counts = np.zeros(worlds, np.int64)
        for lower in np.unique(needed):
            selected = np.flatnonzero(needed == lower)
            if lower == 0:
                counts[selected] = rng.poisson(self.rate, len(selected))
                continue
            prefix = self.prefixes[int(lower)-1]
            if prefix.reference_only:
                counts[selected] = _reference_truncated(self.rate, int(lower), len(selected), rng)
                continue
            tail = rng.random(len(selected)) < prefix.tail_probability
            finite = selected[~tail]
            counts[finite] = prefix.lower+np.searchsorted(prefix.conditional_cdf,
                rng.random(len(finite)), side='right')
            if tail.any():
                counts[selected[tail]] = _reference_truncated(
                    self.rate, prefix.upper+1, int(tail.sum()), rng)
        if np.any(present.sum(-1)+counts < self.minimum):
            raise RuntimeError('minimum-population sampler invariant failed')
        return present, counts
