"""Oracles for the bounded maneuver law, independent of filter integration."""
import math
import unittest

import numpy as np

from eventwm_graph.l2_partial.maneuver import (
    advance_components, condition_components, condition_components_many, moment_match, reset_probability,
    sample_resets, transition_components, support_ranks,
)
from eventwm_graph.l2_partial.probability import advance, transition_matrices
from eventwm_graph.l2_partial.state import Hypothesis


def hypothesis():
    return Hypothesis(
        1., np.array([[0., 0., 1., 0.]]), np.diag([.01, .01, .2, .2])[None],
        np.zeros(1), np.zeros(1, int), np.array([1.]),
        intent=np.array([[1., 0.]]), diffusion=np.array([.16]),
        mode=np.zeros(1, int), relaxation=.7,
    )


class ManeuverLawTests(unittest.TestCase):
    def test_support_dimensions_atoms_lines_planes_and_impossible_points(self):
        means = np.zeros((3, 3, 4))
        covariances = np.tile(np.eye(4), (3, 3, 1, 1))
        covariances[:, 0, :2, :2] = 0.
        covariances[:, 1, 1, 1] = 0.
        probs = np.array([[.2, .3, .5], [0., .6, .4], [1., 0., 0.]])
        ranks = support_ranks(means, covariances, probs, [[0., 0.], [1., 0.], [0., 1.]])
        np.testing.assert_array_equal(ranks, [[0, 1, 2], [1, 1, 2], [0, 3, 3]])

    def test_full_rank_support_ignores_zero_weight_atoms(self):
        means = np.zeros((2, 2, 4)); covariance = np.tile(np.eye(4), (2, 2, 1, 1))
        covariance[:, 1, :2, :2] = 0.
        ranks = support_ranks(means, covariance, [1., 0.], [[0., 0.], [1e6, -1e6]])
        np.testing.assert_array_equal(ranks, np.full((2, 2), 2))

    def test_mean_preserving_jump_retains_ou_mean_and_cross_covariance(self):
        h = hypothesis()
        c = h.covariance.copy(); c[0, 0, 2] = c[0, 2, 0] = .03
        h = h.fork(covariance=c)
        m, c, p = transition_components(h, .048, retention=1.)
        mean, covariance = moment_match(m, c, p)
        original = advance(h, .048)
        F, _, _, _ = transition_matrices(.048, h.relaxation)
        expected_extra = p[1] * 4 * (F[:, 2:] @ F[:, 2:].T)
        np.testing.assert_allclose(mean, original.mean, atol=1e-12)
        np.testing.assert_allclose(covariance[0] - original.covariance[0], expected_extra, atol=1e-12)
        m0, c0, _ = transition_components(h, 0., retention=.3)
        np.testing.assert_allclose(m0[:, 1, 2:], .3 * h.mean[:, 2:])
        np.testing.assert_allclose(c0[:, 1, :2, 2:], .3 * h.covariance[:, :2, 2:])
        np.testing.assert_allclose(c0[:, 1, 2:, 2:], .09 * h.covariance[:, 2:, 2:] + 4*np.eye(2))

    def test_jump_retention_avoids_learned_target_attenuation(self):
        jump = hypothesis(); old = hypothesis()
        for _ in range(40):
            m, c, p = transition_components(jump, .048, retention=1.)
            mean, covariance = moment_match(m, c, p)
            jump = jump.fork(mean=mean, covariance=covariance)
            m, c, p = transition_components(old, .048, retention=0.)
            mean, covariance = moment_match(m, c, p)
            old = old.fork(mean=mean, covariance=covariance)
        self.assertAlmostEqual(jump.mean[0, 2], 1., places=12)
        self.assertLess(old.mean[0, 2], .3)

    def test_sampled_mean_preserving_jump_matches_analytic_moments(self):
        h = hypothesis(); rng = np.random.default_rng(184)
        m, c, p = transition_components(h, 0., retention=1.)
        mean, covariance = moment_match(m, c, p)
        states = rng.multivariate_normal(h.mean[0], h.covariance[0], size=12000)
        sampled, _ = sample_resets(states, rng, retention=1.)
        np.testing.assert_allclose(sampled.mean(0), mean[0], atol=.035)
        np.testing.assert_allclose(np.cov(sampled, rowvar=False, bias=True), covariance[0], atol=.06)

    def test_continue_component_is_existing_transition(self):
        h = hypothesis()
        m, c, p = transition_components(h, .048)
        expected = advance(h, .048)
        np.testing.assert_allclose(m[:, 0], expected.mean, atol=1e-12)
        np.testing.assert_allclose(c[:, 0], expected.covariance, atol=1e-12)
        self.assertAlmostEqual(p[1], 1 - math.exp(-2 * .048))

    def test_partial_interval_preserves_reset_draw_and_composes(self):
        h = hypothesis()
        direct_m, direct_c, direct_p = transition_components(h, .048)
        m, c, p = transition_components(h, .019)
        m, c = advance_components(m, c, .029, h.intent, h.diffusion, h.relaxation)
        np.testing.assert_allclose(m, direct_m, atol=1e-12)
        np.testing.assert_allclose(c, direct_c, atol=1e-12)
        np.testing.assert_array_equal(p, direct_p)
        with self.assertRaises(ValueError):
            transition_components(h, .096)

    def test_known_posterior_mixture_and_exact_position(self):
        means = np.array([[0., 0., 1., 0.], [0., 0., -1., 0.]])
        covariance = np.repeat(np.eye(4)[None], 2, axis=0)
        covariance[:, 0, 2] = covariance[:, 2, 0] = .5
        (m, c), logp, weights = condition_components(
            means, covariance, np.array([.25, .75]), np.array([2., 0.]),
            return_weights=True)
        # Both position marginals agree: weights stay unchanged. Conditional
        # velocities are 2 and 0 with variance .75, giving total variance 1.5.
        np.testing.assert_allclose(weights, [.25, .75], atol=1e-12)
        np.testing.assert_allclose(m, [2., 0., .5, 0.], atol=1e-12)
        self.assertAlmostEqual(c[2, 2], 1.5)
        self.assertAlmostEqual(logp, -math.log(2 * math.pi) - 2)
        np.testing.assert_array_equal(c[:2, :], np.zeros((2, 4)))
        np.testing.assert_array_equal(c[:, :2], np.zeros((4, 2)))

    def test_atom_is_not_added_to_continuous_density(self):
        means = np.zeros((2, 4))
        means[1, 2] = 5.
        covariance = np.stack([np.diag([0., 0., 1., 1.]), np.eye(4)])
        (m, _), logp, weights = condition_components(
            means, covariance, np.array([.3, .7]), [0., 0.], return_weights=True)
        np.testing.assert_array_equal(weights, [1., 0.])
        self.assertEqual(m[2], 0.)
        self.assertAlmostEqual(logp, math.log(.3))

    def test_unsupported_exact_position_stays_unsupported(self):
        means = np.zeros((2, 4))
        covariance = np.zeros((2, 4, 4))
        result, logp, weights = condition_components(
            means, covariance, np.array([.5, .5]), [1., 0.], return_weights=True)
        self.assertIsNone(result)
        self.assertEqual(logp, -np.inf)
        np.testing.assert_array_equal(weights, [0., 0.])

    def test_moment_matching_keeps_between_component_variance(self):
        means = np.array([[0., 0., -2., 0.], [0., 0., 2., 0.]])
        m, c = moment_match(means, np.zeros((2, 4, 4)), [.5, .5])
        np.testing.assert_array_equal(m, np.zeros(4))
        self.assertEqual(c[2, 2], 4.)

    def test_sampled_kernel_matches_transition_moments(self):
        h = hypothesis()
        m, c, probs = transition_components(h, .048)
        mean, covariance = moment_match(m, c, probs)
        rng = np.random.default_rng(183)
        count = 12000
        states = rng.multivariate_normal(h.mean[0], h.covariance[0], size=count)
        states, flag = sample_resets(states, rng)
        F, Q, a, e = transition_matrices(.048, h.relaxation)
        samples = states @ F.T
        samples[:, :2] += (.048 - a) * h.intent[0]
        samples[:, 2:] += (1 - e) * h.intent[0]
        samples += rng.multivariate_normal(np.zeros(4), Q * h.diffusion[0], size=count)
        samples[:, :2] += rng.normal(size=(count, 2)) * np.sqrt(.0025 * .048)
        self.assertLess(abs(flag.mean() - probs[1]), .012)
        np.testing.assert_allclose(samples.mean(0), mean[0], atol=.035)
        np.testing.assert_allclose(np.cov(samples.T, bias=True), covariance[0], atol=.05)

    def test_resets_preserve_positions_and_disabled_rate(self):
        original = np.arange(40, dtype=float).reshape(10, 4)
        changed, _ = sample_resets(original, np.random.default_rng(4), rate=100.)
        np.testing.assert_array_equal(changed[:, :2], original[:, :2])
        unchanged, flags = sample_resets(original, np.random.default_rng(4), rate=0.)
        np.testing.assert_array_equal(unchanged, original)
        self.assertFalse(flags.any())
        self.assertEqual(reset_probability(0.), 0.)

    def test_batched_conditioning_matches_scalar_oracle(self):
        h = hypothesis()
        m, c, p = transition_components(h, .048)
        m = np.repeat(m, 6, axis=0)
        c = np.repeat(c, 6, axis=0)
        m[:, :, 0] += np.arange(6)[:, None]
        z = np.array([[.04, .02], [1.2, -.2], [5.8, 1.], [3., -.1]])
        (means, covariances), evidence, weights = condition_components_many(m, c, p, z, return_weights=True)
        for i in range(len(m)):
            for j in range(len(z)):
                (sm, sc), se, sw = condition_components(m[i], c[i], p, z[j], return_weights=True)
                np.testing.assert_allclose(means[i, j], sm, atol=1e-11)
                np.testing.assert_allclose(covariances[i, j], sc, atol=1e-11)
                np.testing.assert_allclose(evidence[i, j], se, atol=1e-11)
                np.testing.assert_allclose(weights[i, j], sw, atol=1e-11)

    def test_batched_degenerate_support(self):
        m = np.zeros((1, 2, 4))
        c = np.zeros((1, 2, 4, 4))
        (means, _), evidence = condition_components_many(m, c, [.5, .5], [[0., 0.], [1., 0.]])
        np.testing.assert_array_equal(means[0, 0], np.zeros(4))
        self.assertEqual(evidence[0, 0], 0.)
        self.assertEqual(evidence[0, 1], -np.inf)


if __name__ == '__main__':
    unittest.main()
