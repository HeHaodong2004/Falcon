"""Adversarial integrated probability tests; all fixtures are synthetic."""
import unittest

import numpy as np

from eventwm_graph.l2_partial.observation import EndpointObservationLaw
from eventwm_graph.l2_partial.state import BeliefState, Hypothesis, ObservedGraph
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState, WorldCarrier


class ConstantDetection(EndpointObservationLaw):
    version = "constant-detection-probability-oracle"

    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)

    def detection(self, positions, robot_pose, regime):
        positions = np.asarray(positions)
        values = np.asarray(self.probabilities)[np.asarray(regime, int)]
        if values.ndim:
            values = values.reshape(values.shape + (1,) * (positions.ndim - 1 - values.ndim))
        return np.broadcast_to(values, positions.shape[:-1]).copy()


def graph():
    return ObservedGraph("structural-probability-oracle", 0, .2, 0., {})


def person(weight=1., x=2., diffusion=0., regime=0, slots=1, covariance=0.):
    means = np.zeros((slots, 4))
    means[:, 0] = x + np.arange(slots) * 2.
    return Hypothesis(
        weight, means, np.tile(np.eye(4) * covariance, (slots, 1, 1)),
        np.zeros(slots), np.arange(slots), np.array([1.]),
        intent=np.zeros((slots, 2)), diffusion=np.full(slots, diffusion),
        mode=np.zeros(slots, int), relaxation=0., height=np.zeros(slots),
        visibility_mode=regime, hidden_mean=np.zeros((1, 4)),
        hidden_covariance=np.eye(4)[None], hidden_exchangeable=True,
    )


def observation(t, xs=()):
    return {
        "timestamp": t, "robot_pose": [0., 0., 0., 0.], "static_updates": [],
        "tracks": [
            {"position": [float(x), 0., 0.], "last_seen_time": t,
             "visible_now": True, "valid": True} for x in xs
        ],
    }


def model(probabilities=(.2, .8), **kwargs):
    return StructuralGraphBeliefWorldModel(
        kind="memory_cv", maneuver_rate=0., population_prior=[0., 1.],
        observation_law=ConstantDetection(probabilities), **kwargs,
    )


def belief(wm, hypotheses, unresolved=0., evidence=None, timestamp=0.):
    return StructuralBeliefState(
        graph(), hypotheses, timestamp, np.zeros(4), unresolved_mass=unresolved,
        model_version=wm.model_version, evidence={} if evidence is None else evidence,
    ).validate()


class DegenerateDetectionProbabilityTests(unittest.TestCase):
    def test_certain_positive_detections_do_not_disappear_via_zero_misses(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                wm = model((1., 1.), optimized=optimized)
                prior = belief(wm, [person(slots=2)])
                post = wm.observe(prior, observation(0., (2., 4.)))
                self.assertFalse(post.evidence["unsupported_observation"])
                self.assertAlmostEqual(post.unresolved_mass, 0., places=12)
                self.assertEqual(len(post.hypotheses), 1)
                np.testing.assert_array_equal(post.hypotheses[0].mean[:, :2], [[2., 0.], [4., 0.]])
                np.testing.assert_array_equal(post.hypotheses[0].covariance[:, :2, :], 0.)

    def test_impossible_positive_detection_has_zero_evidence(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                wm = model((0., 0.), optimized=optimized)
                post = wm.observe(belief(wm, [person()]), observation(0., (2.,)))
                self.assertTrue(post.evidence["unsupported_observation"])
                self.assertEqual(post.unresolved_mass, 1.)
                self.assertEqual(sum(h.weight for h in post.hypotheses), 0.)

    def test_impossible_empty_set_has_zero_evidence(self):
        wm = model((1., 1.))
        post = wm.observe(belief(wm, [person()]), observation(0.))
        self.assertTrue(post.evidence["unsupported_observation"])
        self.assertEqual(post.unresolved_mass, 1.)


class CoarseEmptyRefinementTests(unittest.TestCase):
    def carrier_belief(self, wm, name, endpoint_count):
        p = np.array([.2, .8])
        no = (1 - p) ** endpoint_count
        event_likelihood = no if name == "survived_no_detection" else 1 - no
        weights = event_likelihood / event_likelihood.sum()
        times = np.arange(endpoint_count + 1) * .048
        worlds = (person(.5, regime=0), person(.5, regime=1))
        positions = np.broadcast_to(np.array([2., 0.]), (2, len(times), 1, 2)).copy()
        velocities = np.zeros_like(positions)
        conditional_empty = () if name == "survived_no_detection" else tuple(
            ((1 - (1 - p) ** (endpoint_count - 1)) * (1 - p) / (1 - no)).tolist()
        )
        event = (name, float(times[-1]), tuple(np.zeros(4)), conditional_empty)
        carrier = WorldCarrier(
            times, positions, velocities, worlds, weights, 0., 73, "oracle-origin",
            len(times) - 1, (event,),
        )
        hs = [h.fork(weight=float(w)) for h, w in zip(worlds, weights)]
        return belief(wm, hs, timestamp=float(times[-1])).fork(carrier=carrier)

    def test_empty_at_no_detection_branch_endpoint_adds_no_evidence(self):
        wm = model()
        prior = self.carrier_belief(wm, "survived_no_detection", 1)
        expected = prior.carrier.weights.copy()
        post = wm.observe(prior, observation(prior.timestamp))
        np.testing.assert_allclose(post.carrier.weights, expected, atol=1e-13)

    def test_last_empty_after_any_detection_uses_conditional_factor(self):
        wm = model()
        prior = self.carrier_belief(wm, "survived_detection", 2)
        # Event D_at_first AND N_at_last has p*(1-p)=.16 in both regimes.
        post = wm.observe(prior, observation(prior.timestamp))
        np.testing.assert_allclose(post.carrier.weights, [.5, .5], atol=1e-13)

    def test_one_endpoint_any_then_empty_is_an_explicit_contradiction(self):
        from eventwm_graph.l2_partial.structural_budget import UnsupportedConditioning
        wm = model()
        prior = self.carrier_belief(wm, "survived_detection", 1)
        with self.assertRaises(UnsupportedConditioning):
            wm.observe(prior, observation(prior.timestamp))


class ResidualEnvelopeIntegrationTests(unittest.TestCase):
    def test_coarse_observation_cannot_reveal_retained_support_membership(self):
        wm = model((.5, .5))
        prior = belief(
            wm, [person(.5)], unresolved=.5,
            evidence={"residual_count_support": [1]},
        )
        future = wm.predict_shared(prior, .048, num_worlds=2, seed=73)
        # A physical event with likelihood .5 in both represented and removed
        # worlds leaves the actual residual mass .5. The caller does not know
        # the removed event likelihood, so the returned envelope must contain it.
        branch_weights = future.weights * .5
        successor = wm.successor_at(
            future, 1, branch_weights, np.zeros(4),
            refinement=(np.ones(2), np.ones(2)),
        )
        self.assertGreaterEqual(successor.unresolved_mass, .5 - 1e-12)
        self.assertLessEqual(sum(h.weight for h in successor.hypotheses), .5 + 1e-12)

    def test_existing_exact_population_support_survives_an_empty_observation(self):
        wm = model((.1, .1))
        prior = belief(
            wm, [person(.7, slots=6)], unresolved=.3,
            evidence={"residual_count_support": [6], "maximum_simultaneous_detections": 6},
        )
        post = wm.observe(prior, observation(.048))
        support = post.evidence["residual_count_support"]
        self.assertTrue(support)
        self.assertTrue(all(n == 6 for n in support), support)
        self.assertEqual(post.evidence["maximum_simultaneous_detections"], 6)

    def test_noise_floor_bound_cannot_be_inferred_from_model_kind_alone(self):
        # This exact linear-Gaussian fixture has no projection or quadrature
        # error. A removed q=.1 component has narrower transition density than
        # the old assumed memory_cv q=.16 noise floor.
        wm = model((1., 1.))
        retained = person(.5, diffusion=1.)
        removed = person(.5, diffusion=.1)
        full = belief(wm, [retained, removed])
        partial = belief(
            wm, [retained], unresolved=.5,
            evidence={"residual_count_support": [1]},
        )
        row = observation(.144, (2.,))
        exact = wm.observe(full, row)
        lower = wm.observe(partial, row)
        actual_retained = sum(h.weight for h in exact.hypotheses if h.diffusion[0] == 1.)
        claimed_lower = sum(h.weight for h in lower.hypotheses if h.diffusion[0] == 1.)
        self.assertAlmostEqual(actual_retained, 1. / 11., places=8)
        self.assertLessEqual(claimed_lower, actual_retained + 1e-12)


class CarrierRandomStreamTests(unittest.TestCase):
    def test_fixed_private_hidden_state_does_not_consume_a_new_initial_draw(self):
        # A state-independent coarse event conveys no information. The direct
        # and extended carrier must keep the same common-world innovations;
        # private hidden Dirac states are already sampled physical states.
        for hidden_count in (0, 1):
            with self.subTest(hidden_count=hidden_count):
                wm = model((.5, .5))
                h = person().fork(
                    hidden_pmf=np.array([0., 1.]) if hidden_count else np.array([1.]),
                    hidden_mean=np.array([[4., 0., 0., 0.]]),
                    hidden_covariance=np.zeros((1, 4, 4)),
                    hidden_exchangeable=True,
                )
                prior = belief(wm, [h])
                direct = wm.predict_shared(prior, .192, num_worlds=4, seed=73)
                first = wm.predict_shared(prior, .096, num_worlds=4, seed=73)
                child = wm.successor_at(
                    first, len(first.times) - 1, first.weights * .5, np.zeros(4))
                continued = wm.predict_shared(child, .096, num_worlds=4, seed=73)
                np.testing.assert_array_equal(direct.positions[:, :3], first.positions)
                np.testing.assert_allclose(
                    direct.positions[:, -1], continued.positions[:, -1], atol=1e-12, rtol=0)
                np.testing.assert_allclose(
                    direct.velocities[:, -1], continued.velocities[:, -1], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
