"""Small mathematical oracles for local residual bounds and time composition.

No collection or confirmation data is loaded. These are explicit finite-model
or Gaussian-transition oracles, not empirical predictive acceptance tests.
"""
import math
import unittest

import numpy as np

from eventwm_graph.l2_partial.structural_probability import (
    CountSupport, EvidenceBound, ObservationMeasure,
    bayes_submeasure, canonical_impulse_velocity_variance,
    event_probability_bounds, finite_set_evidence_bound,
    gaussian_noise_log_supremum, partition_probability_bounds,
    prune_submeasure,
)


EVENT = ObservationMeasure()


class ResidualSubmeasureOracles(unittest.TestCase):
    def update(self, w, likelihood, u, bound=1., **kwargs):
        return bayes_submeasure(
            w, np.log(np.asarray(likelihood, float)), u,
            EvidenceBound.event(upper=bound), component_measure=EVENT, **kwargs,
        )

    def test_finite_removed_mixture_contained_over_multiple_updates(self):
        # Actual four-state prior is known only to this oracle. Two physical
        # states are retained; later residual also contains their unused weight.
        actual = np.array([.20, .30, .10, .40])
        retained = actual[:2].copy()
        u = float(actual[2:].sum())
        observations = (
            np.array([.30, .90, .10, .40]),
            np.array([.80, .15, .90, .20]),
            np.array([.10, .60, .45, .95]),
        )
        for likelihood in observations:
            actual = actual * likelihood
            actual /= actual.sum()
            result = self.update(retained, likelihood[:2], u)
            retained, u = result.retained_weights, result.unresolved_mass
            np.testing.assert_array_less(retained - 1e-14, actual[:2])
            self.assertAlmostEqual(float(retained.sum()) + u, 1.)
            for event in (
                np.array([0., 1., 1., 0.]),
                np.array([1., 0., 0., 1.]),
                np.array([.2, .7, .4, .9]),
            ):
                lower, upper = event_probability_bounds(retained, event[:2], u)
                truth = float(actual @ event)
                self.assertLessEqual(lower, truth + 1e-13)
                self.assertGreaterEqual(upper, truth - 1e-13)

    def test_exact_bayes_when_no_residual_even_if_bound_is_infinite(self):
        result = bayes_submeasure(
            [.3, .7], np.log([.2, .8]), 0.,
            EvidenceBound.unknown(EVENT), component_measure=EVENT,
        )
        np.testing.assert_allclose(result.retained_weights, [.06 / .62, .56 / .62], atol=1e-15)
        self.assertEqual(result.unresolved_mass, 0.)
        self.assertEqual(result.status, "EXACT_FINITE_BAYES")
        self.assertTrue(result.posterior_valid)

    def test_pruned_atom_is_not_assigned_zero_diffuse_density(self):
        measure = ObservationMeasure("set", (0,), "point-atom-at-x=2")
        bound = finite_set_evidence_bound(CountSupport((1,)), [0.], measure)
        # Both retained and removed people can explain precisely the same atom.
        result = bayes_submeasure(
            [.4], [0.], .6, bound, component_measure=measure,
            component_count_supports=[(1,)], residual_count_support=(1,),
        )
        self.assertAlmostEqual(result.retained_weights[0], .4)
        self.assertAlmostEqual(result.unresolved_mass, .6)
        self.assertEqual(result.residual_count_support.values, (1,))

    def test_continuous_density_above_one_uses_density_bound(self):
        measure = ObservationMeasure("set", (2,), "planar-lebesgue")
        log_m = gaussian_noise_log_supremum(np.eye(2) * .0001, elapsed_time=.048)
        self.assertGreater(math.exp(log_m), 1000.)
        bound = finite_set_evidence_bound((1,), [log_m], measure)
        result = bayes_submeasure(
            [.5], [0.], .5, bound, component_measure=measure,
            component_count_supports=[(1,)], residual_count_support=(1,),
        )
        actual_retained = .5 / (.5 + .5 * math.exp(log_m))
        self.assertLessEqual(result.retained_weights[0], actual_retained + 1e-15)
        self.assertLess(result.retained_weights[0], .001)
        with self.assertRaises(ValueError):
            EvidenceBound.event(measure=measure)

    def test_unknown_density_is_vacuous_and_preserves_count_constraints(self):
        measure = ObservationMeasure("set", (2, 2), "planar-lebesgue")
        result = bayes_submeasure(
            [.7], np.log([.2]), .3, EvidenceBound.unknown(measure),
            component_measure=measure, component_count_supports=[(2, 4)],
            residual_count_support=CountSupport(None), observed_count=2,
            approximation_notes=("Gaussian projection not certified",),
        )
        np.testing.assert_array_equal(result.retained_weights, [0.])
        self.assertEqual(result.unresolved_mass, 1.)
        self.assertEqual(result.status, "VACUOUS_RESIDUAL_BOUND")
        self.assertEqual(result.residual_count_support.minimum, 2)
        self.assertIsNone(result.residual_count_support.values)
        self.assertIn("excludes", result.guarantee_scope)
        self.assertEqual(result.approximation_notes, ("Gaussian projection not certified",))
        self.assertFalse(result.metadata()["point_residual_law"])

    def test_loose_normalizer_residual_contains_retained_count_support(self):
        result = self.update(
            [.7], [.2], .3,
            component_count_supports=[(4,)], residual_count_support=(6,),
        )
        self.assertEqual(result.residual_count_support.values, (4, 6))

    def test_impossible_count_removes_only_incompatible_residual(self):
        measure = ObservationMeasure("set", (2, 2), "planar-lebesgue")
        result = bayes_submeasure(
            [.7], np.log([.2]), .3, EvidenceBound.unknown(measure),
            component_measure=measure, component_count_supports=[(2,)],
            residual_count_support=(0, 1),
        )
        self.assertAlmostEqual(result.retained_weights[0], 1.)
        self.assertEqual(result.unresolved_mass, 0.)
        self.assertEqual(result.residual_count_support.values, (2,))

    def test_impossible_observation_is_not_called_valid_posterior(self):
        result = bayes_submeasure(
            [.5], [-math.inf], .5, EvidenceBound.event(upper=0.),
            component_measure=EVENT,
        )
        self.assertFalse(result.posterior_valid)
        self.assertEqual(result.status, "UNSUPPORTED_OBSERVATION")
        self.assertEqual(result.unresolved_mass, 1.)
        self.assertEqual(result.log_evidence_upper, -math.inf)
        self.assertFalse(result.metadata()["positive_evidence_certified"])

    def test_log_scale_invariance_large_positive_and_negative_offsets(self):
        measure = ObservationMeasure("set", (2,), "planar-lebesgue")
        base = None
        for offset in (-1000., 0., 1000.):
            result = bayes_submeasure(
                [.2, .3], np.log([.4, .8]) + offset, .5,
                EvidenceBound(math.log(2.) + offset, measure, "TEST_BOUND", "analytic oracle"),
                component_measure=measure,
            )
            if base is None:
                base = result.retained_weights
            np.testing.assert_allclose(result.retained_weights, base, rtol=2e-13)
            self.assertTrue(np.isfinite(result.retained_weights).all())

    def test_common_reference_is_mandatory(self):
        point = ObservationMeasure("set", (0,), "atom-A")
        other = ObservationMeasure("set", (0,), "atom-B")
        with self.assertRaises(ValueError):
            bayes_submeasure([.5], [0.], .5, EvidenceBound.unknown(other), component_measure=point)
        unknown = ObservationMeasure("set", (1,), None)
        result = bayes_submeasure(
            [.5], [0.], .5,
            EvidenceBound(0., unknown, "UNCERTIFIED_TEST", "unknown line"),
            component_measure=unknown,
        )
        self.assertEqual(result.status, "VACUOUS_RESIDUAL_BOUND")

    def test_probability_measure_rejects_density_above_one(self):
        with self.assertRaises(ValueError):
            self.update([.5], [2.], .5)
        with self.assertRaises(ValueError):
            bayes_submeasure([.5], [math.inf], .5, EvidenceBound.event(), component_measure=EVENT)

    def test_positive_evidence_rejects_impossible_retained_count(self):
        measure = ObservationMeasure("set", (2, 2), "planar-lebesgue")
        with self.assertRaises(ValueError):
            bayes_submeasure(
                [.5], [0.], .5, EvidenceBound.unknown(measure),
                component_measure=measure, component_count_supports=[(1,)],
            )

    def test_partition_reconstruction_and_shared_residual_simplex(self):
        weights = np.array([.3, .7])
        emissions = np.array([[.2, .8], [.9, .1]])
        reconstructed = np.zeros(2)
        for j in range(2):
            posterior = self.update(weights, emissions[:, j], 0.)
            reconstructed += math.exp(posterior.log_evidence_lower) * posterior.retained_weights
        np.testing.assert_allclose(reconstructed, weights, atol=1e-15)
        bounded = partition_probability_bounds(weights * .8, emissions, .2)
        self.assertAlmostEqual(float(bounded["lower"].sum()) + bounded["residual_simplex_mass"], 1.)
        for residual_allocation in ([.2, 0.], [0., .2], [.08, .12]):
            truth = bounded["lower"] + residual_allocation
            self.assertAlmostEqual(float(truth.sum()), 1.)
            self.assertTrue(np.all(truth <= bounded["upper"] + 1e-15))
        self.assertFalse(bounded["point_branch_probabilities"])

    def test_pruning_mass_and_immutable_result(self):
        keep, u = prune_submeasure([.1, .2, .3], [2, 0], .4)
        np.testing.assert_allclose(keep, [.3, .1])
        self.assertAlmostEqual(u, .6)
        result = self.update(keep, [.8, .2], u)
        self.assertFalse(result.retained_weights.flags.writeable)
        with self.assertRaises(ValueError):
            result.retained_weights[0] = 1.
        with self.assertRaises(ValueError):
            prune_submeasure([.6], [0, 0], .4)
        with self.assertRaises(ValueError):
            self.update([.7], [.5], .4)


class DensityBoundOracles(unittest.TestCase):
    def test_gaussian_supremum_and_zero_time_guard(self):
        cov = np.array([[.3, .1], [.1, .2]])
        expected = -math.log(2. * math.pi) - .5 * math.log(np.linalg.det(cov))
        self.assertAlmostEqual(gaussian_noise_log_supremum(cov, elapsed_time=.1), expected)
        self.assertEqual(gaussian_noise_log_supremum(cov, elapsed_time=0.), math.inf)
        self.assertEqual(gaussian_noise_log_supremum(np.diag([.1, 0.]), elapsed_time=.1), math.inf)
        with self.assertRaises(ValueError):
            gaussian_noise_log_supremum(np.diag([.1, -.2]), elapsed_time=.1)

    def test_exhaustive_injective_assignments_below_set_bound(self):
        # All six assignments of two observations to three persons.
        from itertools import permutations
        assignment = np.array([[.4, 2.], [.7, .5], [.2, 1.5]])
        miss = np.array([.8, .3, .9])
        exact = 0.
        for a in permutations(range(3), 2):
            unused = set(range(3)) - set(a)
            exact += assignment[a[0], 0] * assignment[a[1], 1] * math.prod(miss[i] for i in unused)
        measure = ObservationMeasure("set", (2, 2), "planar-lebesgue")
        bound = finite_set_evidence_bound(
            (1, 3), np.log(assignment.max(0)), measure, joint_domination_certified=True,
        )
        self.assertLessEqual(exact, math.exp(bound.log_upper))
        self.assertAlmostEqual(math.exp(bound.log_upper), 6 * .7 * 2.)

    def test_atomic_mixed_unknown_count_and_unknown_support(self):
        atomic = ObservationMeasure("set", (0, 0), "two-distinct-atoms")
        self.assertEqual(finite_set_evidence_bound(None, [0., 0.], atomic).log_upper, 0.)
        mixed = ObservationMeasure("set", (0, 1, 2), "declared-atom-line-and-plane")
        bounded = finite_set_evidence_bound(
            (3, 4), [0., math.log(2.), math.log(5.)], mixed, joint_domination_certified=True,
        )
        self.assertAlmostEqual(math.exp(bounded.log_upper), 4 * 3 * 2 * 2 * 5)
        self.assertEqual(finite_set_evidence_bound(None, [0., 0., 0.], mixed).log_upper, math.inf)
        unidentified = ObservationMeasure("set", (0, 1, 2), None)
        self.assertEqual(finite_set_evidence_bound((4,), [0., 0., 0.], unidentified).log_upper, math.inf)
        impossible = finite_set_evidence_bound((0, 1), [0., 0.], atomic)
        self.assertEqual(impossible.log_upper, -math.inf)

    def test_smooth_marginals_do_not_bound_correlated_joint_density(self):
        # X1=X2 with a smooth common Gaussian has bounded marginal densities but
        # a singular two-person joint law. A marginal bound is insufficient.
        measure = ObservationMeasure("set", (2, 2), "planar-lebesgue")
        bound = finite_set_evidence_bound((2,), [0., 0.], measure)
        self.assertEqual(bound.log_upper, math.inf)
        self.assertIn("joint", bound.provenance)


class CanonicalLawOracles(unittest.TestCase):
    def test_closed_form_matches_independent_step_recurrence(self):
        for kappa in (0., .7, 1e-8):
            for n in (0, 1, 62):
                variance = .3
                dt = .048
                p = 1 - math.exp(-2. * dt)
                decay = math.exp(-2. * kappa * dt)
                q = .25 * dt if kappa == 0 else .25 * (-math.expm1(-2. * kappa * dt)) / (2. * kappa)
                for _ in range(n):
                    variance = decay * (variance + p * 4.) + q
                closed = canonical_impulse_velocity_variance(.3, n, relaxation=kappa)
                self.assertAlmostEqual(float(closed), variance, places=11)

    def test_existing_analytic_kernel_matches_variance_law(self):
        from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
        from eventwm_graph.l2_partial.state import Hypothesis
        h = Hypothesis(
            1., np.zeros((1, 4)), np.zeros((1, 4, 4)), np.zeros(1),
            np.zeros(1, int), np.array([1.]), intent=np.zeros((1, 2)),
            diffusion=np.array([.25]), mode=np.zeros(1, int), relaxation=.7,
        )
        model = RepairedGraphBeliefWorldModel(maneuver_retention=1.)
        advanced = model.advance_hypothesis(h, 62 * .048, 0.)
        expected = canonical_impulse_velocity_variance(0., 62)
        np.testing.assert_allclose(np.diag(advanced.covariance[0])[2:], expected, rtol=1e-11, atol=1e-11)

    def test_analytic_three_second_segmentation_keeps_partial_latents(self):
        from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
        from eventwm_graph.l2_partial.state import Hypothesis
        model = RepairedGraphBeliefWorldModel(maneuver_retention=1.)
        h = Hypothesis(
            1., np.array([[1., 2., .7, -.2], [3., 0., -.3, .4]]),
            np.tile(np.diag([.1, .2, .3, .4]), (2, 1, 1)), np.zeros(2),
            np.arange(2), np.array([1.]), intent=np.array([[.5, .1], [-.1, .2]]),
            diffusion=np.array([.25, .4]), mode=np.zeros(2, int), relaxation=.7,
        )
        for start in (0., .013, 1.23456789):
            direct = model.advance_hypothesis(h, 3., start)
            for segments in ((1.5, 1.5), (1., 1., 1.)):
                recursive = h
                now = start
                for dt in segments:
                    recursive = model.advance_hypothesis(recursive, dt, now)
                    now += dt
                np.testing.assert_allclose(recursive.mean, direct.mean, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(recursive.covariance, direct.covariance, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(recursive.maneuver_probability, direct.maneuver_probability, atol=1e-13)
                self.assertEqual(recursive.maneuver_until, direct.maneuver_until)
                np.testing.assert_array_equal(recursive.mode, direct.mode)
                np.testing.assert_array_equal(recursive.hidden_pmf, direct.hidden_pmf)


if __name__ == "__main__":
    unittest.main()
