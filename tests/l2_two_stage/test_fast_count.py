"""Prepared count law checks against analytic probabilities and original sampler."""
import itertools
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import poisson

from eventwm_graph.l2_two_stage.count import sample_conditioned, sample_total_count
from eventwm_graph.l2_two_stage.fast_count import PreparedCountSampler,_prepare_prefix


def test_suffix_and_bernoulli_draws_match_reference():
    for rate in (0., .001, .3, 4., 100.):
        for minimum in (0, 1, 3, 5):
            logits = np.array([-.9, .2, 1.5, -2.])
            age = np.array([.2, 0., 2., .8])
            if rate == 0. and minimum > len(logits):
                continue
            sampler = PreparedCountSampler(logits, age, rate, minimum)
            expected, _, z = sample_conditioned(logits, age, rate, minimum, 128,
                                                np.random.default_rng(731))
            actual, counts = sampler.sample(128, np.random.default_rng(731))
            np.testing.assert_array_equal(actual, expected)
            assert sampler.log_normalizer == z
            assert np.all(actual.sum(1)+counts >= minimum)


def test_fixed_total_is_identical_to_reference_including_rng_coupling():
    for total in (1, 3, 8):
        logits = np.array([-.9, .2, 1.5, -2.])
        age = np.array([.2, 0., 2., .8])
        sampler = PreparedCountSampler(logits, age, .7, 1, total=total)
        expected, counts, z = sample_total_count(logits, age, .7, total, 200,
                                                np.random.default_rng(218))
        actual, hidden = sampler.sample(200, np.random.default_rng(218))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(hidden, counts)
        assert sampler.log_normalizer == z


def test_prefix_mixture_recovers_analytic_pmf_and_preserves_positive_tail():
    for rate in (1e-12, .03, .7, 5., 100.):
        sampler = PreparedCountSampler([], [], rate, 6)
        for prefix in sampler.prefixes:
            assert not prefix.reference_only
            assert 0 < prefix.tail_probability < 1
            assert prefix.conditional_cdf[-1] == 1.
            probability = np.diff(np.r_[0., prefix.conditional_cdf])*(1-prefix.tail_probability)
            expected = np.exp(poisson.logpmf(np.arange(prefix.lower, prefix.upper+1), rate)
                              -poisson.logsf(prefix.lower-1, rate))
            np.testing.assert_allclose(probability, expected, rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(prefix.tail_probability,
                np.exp(poisson.logsf(prefix.upper, rate)-poisson.logsf(prefix.lower-1, rate)),
                rtol=0, atol=0)
            with pytest.raises(ValueError):
                prefix.conditional_cdf.setflags(write=True)


def test_shared_grid_prefixes_match_independent_scalar_preparation_bitwise():
    for rate in (0.,1e-200,1e-12,.03,.7,5.,100.,1000.):
        sampler=PreparedCountSampler(np.zeros(7),np.ones(7),rate,6)
        for prefix in sampler.prefixes:
            scalar=_prepare_prefix(rate,prefix.lower)
            assert (prefix.lower,prefix.upper,prefix.reference_only)==(
                scalar.lower,scalar.upper,scalar.reference_only)
            assert prefix.tail_probability==scalar.tail_probability or (
                np.isnan(prefix.tail_probability) and np.isnan(scalar.tail_probability))
            np.testing.assert_array_equal(prefix.log_pmf,scalar.log_pmf)
            np.testing.assert_array_equal(prefix.conditional_cdf,scalar.conditional_cdf)


def test_explicit_tail_branch_returns_counts_above_finite_prefix():
    class Zeros(np.random.Generator):
        def random(self, size=None, dtype=np.float64, out=None):
            return np.zeros(size, dtype=dtype)
    sampler = PreparedCountSampler([], [], .2, 3)
    prefix = sampler.prefixes[2]
    calls = []
    def tail(rate, lower, size, rng):
        calls.append((rate, lower, size))
        return np.full(size, lower+37, np.int64)
    with patch('eventwm_graph.l2_two_stage.fast_count._reference_truncated', tail):
        present, counts = sampler.sample(7, Zeros(np.random.PCG64(9)))
    assert present.shape == (7, 0)
    assert calls == [(.2, prefix.upper+1, 7)]
    assert np.all(counts > prefix.upper)


def test_joint_bernoulli_poisson_law_matches_analytic_held_count_probabilities():
    logits = np.array([-.8, .3])
    rate = .7
    minimum = 2
    sampler = PreparedCountSampler(logits, [1., 1.], rate, minimum)
    present, counts = sampler.sample(180000, np.random.default_rng(1234))
    p = expit(logits)
    for bits in itertools.product((0, 1), repeat=2):
        source_mass = np.prod(np.where(bits, p, 1-p))
        matching = np.all(present == bits, axis=1)
        for hidden in range(5):
            expected = source_mass*poisson.pmf(hidden, rate)/np.exp(sampler.log_normalizer)
            if sum(bits)+hidden < minimum:
                expected = 0.
            actual = np.mean(matching & (counts == hidden))
            tolerance = 6*np.sqrt(expected*(1-expected)/len(counts))+2/len(counts)
            assert abs(actual-expected) <= tolerance


def test_numeric_underflow_uses_explicit_reference_and_invalid_inputs_fail():
    # Positive tail below scipy's representable SF is never turned into zero.
    sampler = PreparedCountSampler([], [], 1e-200, 1)
    assert sampler.prefixes[0].reference_only
    _, counts = sampler.sample(16, np.random.default_rng(91))
    np.testing.assert_array_equal(counts, np.ones(16, np.int64))
    with pytest.raises(ValueError):
        sampler.suffix.setflags(write=True)
    with pytest.raises(ValueError):
        sampler.probabilities.setflags(write=True)
    for args in (([0.], [-1.], 1., 0), ([0.], [], 1., 0),
                 ([0.], [1.], -1., 0), ([0.], [1.], 1., 1.5)):
        with pytest.raises(ValueError):
            PreparedCountSampler(*args)
    with pytest.raises(RuntimeError):
        PreparedCountSampler([], [], 0., 1)
