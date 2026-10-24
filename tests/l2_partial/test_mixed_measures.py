"""Mixed-measure association and incompatible-model regression oracles.

Exact position atoms have probability mass; a 2-D Gaussian contributes a
density. They cannot be added across assignments as if they had the same
reference measure. These tests require no dataset or trained checkpoint.
"""
import unittest

import numpy as np

from eventwm_graph.l2_partial.state import BeliefState
from test_observation_repair import (
    _belief, _graph, _hypothesis, _observation, _query, _wm,
    ConstantLaw, HalfSpaceLaw,
)


class MixedMeasureAssociationTests(unittest.TestCase):
    def test_tracked_atom_excludes_diffuse_hidden_birth(self):
        h = _hypothesis(position=(2., 0.), hidden=1).fork(
            hidden_mean=np.array([[2., 0., 0., 0.]]),
            hidden_covariance=np.eye(4)[None])
        b = BeliefState(_graph(), [h], 0., np.zeros(4)).validate()
        posterior = _wm().observe(b, _observation(0., True, position=(2., 0.)))
        self.assertEqual(posterior.evidence['birth_probability'], 0.)
        self.assertEqual(posterior.unresolved_mass, 0.)
        self.assertTrue(all(h.n == 1 and h.expected_hidden == 1 for h in posterior.hypotheses))

    def test_private_hidden_atom_excludes_diffuse_tracked_assignment(self):
        h = _hypothesis(position=(2., 0.), hidden=1).fork(
            covariance=np.eye(4)[None], hidden_mean=np.array([[2., 0., 0., 0.]]),
            hidden_covariance=np.zeros((1, 4, 4)), hidden_exchangeable=False)
        b = BeliefState(_graph(), [h], 0., np.zeros(4)).validate()
        posterior = _wm().observe(b, _observation(0., True, position=(2., 0.)))
        self.assertAlmostEqual(posterior.evidence['birth_probability'], 1.)
        self.assertEqual(posterior.unresolved_mass, 0.)
        self.assertTrue(all(h.n == 2 and h.expected_hidden == 0 for h in posterior.hypotheses))

    def test_atomic_reference_is_shared_across_joint_hypotheses(self):
        atom = _hypothesis(.001, position=(2., 0.)).fork(slot_ids=np.array([10]))
        diffuse = _hypothesis(.999, position=(2., 0.)).fork(
            covariance=np.eye(4)[None], slot_ids=np.array([20]))
        b = BeliefState(_graph(), [atom, diffuse], 0., np.zeros(4)).validate()
        posterior = _wm().observe(b, _observation(0., True, position=(2., 0.)))
        self.assertEqual(posterior.unresolved_mass, 0.)
        self.assertTrue(all(h.slot_ids[0] == 10 for h in posterior.hypotheses if h.weight > 0))

    def test_zero_weight_atom_does_not_suppress_supported_density(self):
        atom = _hypothesis(0., position=(2., 0.)).fork(slot_ids=np.array([10]))
        diffuse = _hypothesis(1., position=(2., 0.)).fork(
            covariance=np.eye(4)[None], slot_ids=np.array([20]))
        b = BeliefState(_graph(), [atom, diffuse], 0., np.zeros(4)).validate()
        posterior = _wm().observe(b, _observation(0., True, position=(2., 0.)))
        self.assertEqual(posterior.unresolved_mass, 0.)
        self.assertTrue(all(h.slot_ids[0] == 20 for h in posterior.hypotheses if h.weight > 0))

    def test_zero_hidden_population_does_not_create_an_atomic_emitter(self):
        h = _hypothesis(position=(2., 0.), hidden=0).fork(
            covariance=np.eye(4)[None], hidden_mean=np.array([[2., 0., 0., 0.]]),
            hidden_covariance=np.zeros((1, 4, 4)), hidden_exchangeable=True)
        b = BeliefState(_graph(), [h], 0., np.zeros(4)).validate()
        posterior = _wm().observe(b, _observation(0., True, position=(2., 0.)))
        self.assertEqual(posterior.unresolved_mass, 0.)
        self.assertTrue(all(h.n == 1 and h.expected_hidden == 0 for h in posterior.hypotheses))


class ModelVersionGuardTests(unittest.TestCase):
    def test_different_observation_laws_have_different_versions(self):
        a, b = _wm(ConstantLaw()), _wm(HalfSpaceLaw())
        self.assertNotEqual(a.model_version, b.model_version)

    def test_foreign_observation_law_future_is_rejected(self):
        a, b, belief = _wm(ConstantLaw()), _wm(HalfSpaceLaw()), _belief()
        future = a.predict_shared(belief, .048)
        with self.assertRaises(ValueError):
            b.evaluate_queries(belief, [_query()], future)

    def test_foreign_transition_future_is_rejected(self):
        a, b, belief = _wm(maneuver_rate=0.), _wm(maneuver_rate=2.), _belief()
        future = a.predict_shared(belief, .048)
        with self.assertRaises(ValueError):
            b.evaluate_queries(belief, [_query()], future)

    def test_foreign_versioned_belief_requires_explicit_migration(self):
        a, b = _wm(maneuver_rate=0.), _wm(maneuver_rate=2.)
        bound = a.propagate(_belief(), .048)
        with self.assertRaises(ValueError):
            b.propagate(bound, .096)


if __name__ == '__main__':
    unittest.main()
