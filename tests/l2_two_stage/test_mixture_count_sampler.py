"""Mixture posterior/count laws and exact one-component RNG delegation."""
import itertools

import numpy as np
import pytest
from scipy.special import expit, logsumexp
from scipy.stats import poisson

from eventwm_graph.l2_two_stage.fast_count import PreparedCountSampler
from eventwm_graph.l2_two_stage.mixture_count_sampler import PreparedMixtureCountSampler


def _constraint(logits, age, rate, minimum, certain=None, total=None):
    probability = expit(logits)
    forced = np.asarray(age) < 1e-7
    if certain is not None:
        forced |= certain
    probability = np.where(forced, 1., probability)
    result = 0.
    for bits in itertools.product((0, 1), repeat=len(logits)):
        weight = np.prod(np.where(bits, probability, 1-probability))
        if total is None:
            result += weight*poisson.sf(minimum-sum(bits)-1, rate)
        else:
            result += weight*poisson.pmf(total-sum(bits), rate)
    return result


@pytest.mark.parametrize('total', [None, 3, 7])
@pytest.mark.parametrize('worlds', [0, 1, 97])
def test_single_component_is_exact_old_sampler_and_rng(total, worlds):
    logits = np.array([-.9, .2, 1.5, -2.])
    age = np.array([.2, 0., 2., .8])
    certain = np.array([False, False, False, True])
    old = PreparedCountSampler(logits, age, .7, 2, certain=certain, total=total)
    new = PreparedMixtureCountSampler(logits, age, [.7], [731.], 2,
                                      certain=certain, total=total)
    r0 = np.random.default_rng(231)
    r1 = np.random.default_rng(231)
    a, b = old.sample(worlds, r0)
    c, d = new.sample(worlds, r1)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert old.log_normalizer == new.log_normalizer
    assert r0.bit_generator.state == r1.bit_generator.state
    np.testing.assert_array_equal(r0.random(31), r1.random(31))


@pytest.mark.parametrize('total', [None, 2, 4, 8])
def test_component_posterior_and_normalizer_match_enumeration(total):
    logits = np.array([-.8, .3, 1.1])
    age = np.array([1., 0., .9])
    rates = np.array([.03, .7, 9.])
    count_logits = np.array([.6, -1.1, .2])
    new = PreparedMixtureCountSampler(logits, age, rates, count_logits, 2,
                                      total=total)
    prior = np.exp(count_logits-logsumexp(count_logits))
    constraints = np.array([_constraint(logits, age, rate, 2, total=total)
                            for rate in rates])
    z = np.dot(prior, constraints)
    np.testing.assert_allclose(np.exp(new.log_component_normalizers), constraints,
                               rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(np.exp(new.component_log_probabilities), prior*constraints/z,
                               rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(np.exp(new.log_normalizer), z, rtol=3e-14)


@pytest.mark.parametrize('total', [None, 3])
def test_joint_allocations_match_analytic_mixture_with_floor(total):
    logits = np.array([-.8, .3])
    rates = np.array([.15, 3.2])
    count_logits = np.array([.2, -.4])
    minimum = 2
    sampler = PreparedMixtureCountSampler(logits, [1., 1.], rates,
                                          count_logits, minimum, total=total)
    present, hidden = sampler.sample(100000, np.random.default_rng(19312))
    assert np.all(present.sum(1)+hidden >= minimum)
    if total is not None:
        assert np.all(present.sum(1)+hidden == total)
    probability = expit(logits)
    prior = np.exp(count_logits-logsumexp(count_logits))
    for bits in itertools.product((0, 1), repeat=2):
        mass = np.prod(np.where(bits, probability, 1-probability))
        matching = np.all(present == bits, axis=1)
        for count in range(9):
            compatible = sum(bits)+count >= minimum
            if total is not None:
                compatible &= sum(bits)+count == total
            expected = (mass*np.dot(prior, poisson.pmf(count, rates))
                        /np.exp(sampler.log_normalizer) if compatible else 0.)
            empirical = np.mean(matching & (hidden == count))
            tolerance = 6*np.sqrt(expected*(1-expected)/len(hidden))+2/len(hidden)
            assert abs(empirical-expected) <= tolerance


def test_zero_weight_padding_preserves_indices_and_single_active_rng():
    sampler = PreparedMixtureCountSampler([.2], [1.], [.1, .7, 3.],
                                          [-np.inf, .3, -np.inf], 1)
    old = PreparedCountSampler([.2], [1.], .7, 1)
    np.testing.assert_array_equal(sampler.active_components, [1])
    np.testing.assert_array_equal(sampler.component_log_probabilities,
                                  [-np.inf, 0., -np.inf])
    rng = np.random.default_rng(251)
    old_rng = np.random.default_rng(251)
    a, b = sampler.sample(51, rng)
    c, d = old.sample(51, old_rng)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert rng.bit_generator.state == old_rng.bit_generator.state


def test_finite_log_component_is_not_pruned_when_ordinary_probability_underflows():
    class RareComponent(np.random.Generator):
        def gumbel(self, loc=0., scale=1., size=None):
            result = np.zeros(size)
            result[:, 1] = 1000.
            return result
    sampler = PreparedMixtureCountSampler([], [], [.01, 900.], [0., -900.], 0)
    assert np.isfinite(sampler.component_log_probabilities).all()
    assert np.exp(sampler.component_log_probabilities[1]) == 0.
    present, hidden = sampler.sample(8, RareComponent(np.random.PCG64(7)))
    assert present.shape == (8, 0)
    assert np.all(hidden > 500)


def test_preparation_is_immutable_and_allocation_has_no_count_ceiling():
    rates = np.array([.3, 500.])
    weights = np.array([-.1, .5])
    sampler = PreparedMixtureCountSampler([], [], rates, weights, 0)
    rates[:] = 1.
    weights[:] = 0.
    np.testing.assert_array_equal(sampler.rates, [.3, 500.])
    for field in ('rates', 'log_count_probabilities', 'log_component_normalizers',
                  'component_log_probabilities', 'active_components'):
        with pytest.raises(ValueError):
            getattr(sampler, field).setflags(write=True)
    _, hidden = sampler.sample(200, np.random.default_rng(21))
    assert hidden.max() > 500


@pytest.mark.parametrize('rates,weights', [([], []), ([.1], [0., 1.]),
    ([0., 1.], [0., 1.]), ([-1.], [0.]), ([np.inf], [0.]),
    ([.1], [np.nan]), ([.1], [np.inf]), ([.1], [-np.inf]),
    ([1+0j], [0.]), ([1.], [0+0j]), ([True], [0.])])
def test_invalid_family_parameters_fail_closed(rates, weights):
    with pytest.raises(ValueError):
        PreparedMixtureCountSampler([], [], rates, weights, 0)


def test_active_numerical_failure_is_not_removed_from_mixture():
    with pytest.raises(RuntimeError):
        PreparedMixtureCountSampler([], [], [1e-200, 1.], [0., 0.], 7)
    with pytest.raises(RuntimeError):
        PreparedMixtureCountSampler([], [], [1., 2.], [-1e308, 1e308], 0)


@pytest.mark.parametrize('worlds', [-1, 1.5, True])
def test_invalid_sampling_arguments_reject(worlds):
    sampler = PreparedMixtureCountSampler([], [], [1., 2.], [0., 0.], 0)
    with pytest.raises(ValueError):
        sampler.sample(worlds, np.random.default_rng(7))
