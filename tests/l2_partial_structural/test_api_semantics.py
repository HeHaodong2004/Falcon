"""Failure-oriented public API contracts for the structural L2 revision.

These are synthetic mathematical/API fixtures, with no trained checkpoints,
dataset access, simulator truth, or calibration choices. They deliberately
import the new structural class: absence of that implementation is not a skip
or a passing test. The coordinator schedules execution after the class exists.

Execution likelihoods are component-indexed and bound to the exact prior by a
support fingerprint. Their stable evidence ID represents one physical event.
Sensor identity is independent of graph updates and execution evidence.
"""

import copy
import unittest

import numpy as np

from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.state import BeliefState, Hypothesis, ObservedGraph


def _model(**kwargs):
    return StructuralGraphBeliefWorldModel(
        kind="memory_cv", maneuver_rate=0., population_prior=[1.], **kwargs)


def _graph():
    return ObservedGraph(
        "structural-api-fixture", 1, .2, 0.,
        {(i, j): (1, 0.) for i in range(-15, 16) for j in range(-5, 6)})


def _belief(model, timestamp=0., unresolved=0.):
    hypotheses = []
    for i, (weight, x) in enumerate(((.35, -2.), (.65, 2.))):
        covariance = np.diag([.4, .4, .3, .3])
        covariance[0, 2] = covariance[2, 0] = .08
        hypotheses.append(Hypothesis(
            weight=weight * (1 - unresolved),
            mean=np.array([[x, 0., .5, -.1]]),
            covariance=covariance[None],
            last_seen=np.array([timestamp]), slot_ids=np.array([i]),
            hidden_pmf=np.array([1.]), intent=np.array([[.5, -.1]]),
            diffusion=np.array([.1]), mode=np.array([0]), relaxation=0.,
            height=np.array([0.]), visibility_mode=1,
            birth_time=np.array([timestamp]), birth_position=np.array([[x, 0.]])))
    return BeliefState(
        _graph(), hypotheses, timestamp, np.zeros(4),
        unresolved_mass=unresolved, model_version=model.model_version).validate()


def _observation(timestamp, positions=(), **extra):
    row = {
        "timestamp": timestamp, "robot_pose": [0., 0., 0., 0.],
        "static_updates": [],
        "tracks": [{"position": [x, .9, z], "last_seen_time": timestamp,
                    "visible_now": True, "valid": True, "track_id": 900 + i}
                   for i, (x, z) in enumerate(positions)],
    }
    row.update(extra)
    return row


def _trace(prior, likelihood=(.2, .8), evidence_id="trace-1", **extra):
    trace = {
        "evidence_id": evidence_id, "support_fingerprint": prior.fingerprint(),
        "world_likelihood": list(likelihood), "unresolved_likelihood": .4,
    }
    trace.update(extra)
    return trace


def _update(cell):
    return {"cell": list(cell), "state": 2, "height": 0.,
            "source": "observed_static_wall_inflated_025m"}


class PhysicalAssertions(unittest.TestCase):
    """Compare posterior content, without requiring private ledger names."""

    def assertPhysicalEqual(self, actual, expected, atol=2e-11):
        actual.validate()
        expected.validate()
        self.assertAlmostEqual(actual.timestamp, expected.timestamp, places=12)
        self.assertEqual(actual.terminal, expected.terminal)
        self.assertAlmostEqual(actual.unresolved_mass, expected.unresolved_mass, places=11)
        np.testing.assert_allclose(actual.robot_pose, expected.robot_pose, atol=atol, rtol=0)
        key = lambda h: (tuple(h.slot_ids), h.visibility_mode)
        left, right = sorted(actual.hypotheses, key=key), sorted(expected.hypotheses, key=key)
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(key(a), key(b))
            self.assertAlmostEqual(a.weight, b.weight, places=11)
            for field in ("mean", "covariance", "last_seen", "slot_ids", "hidden_pmf",
                          "intent", "diffusion", "mode", "height"):
                av, bv = getattr(a, field), getattr(b, field)
                if av is None or bv is None:
                    self.assertIs(av, bv)
                else:
                    np.testing.assert_allclose(av, bv, atol=atol, rtol=0, err_msg=field)


class ExecutionEvidenceTests(PhysicalAssertions):
    def test_survival_applies_when_observation_is_invalid(self):
        model = _model()
        prior = _belief(model)
        posterior = model.observe(
            prior, _observation(.144, observation_valid=False), _trace(prior))
        expected = model.propagate(prior, .144)
        weights = np.array([h.weight for h in expected.hypotheses]) * [.2, .8]
        weights /= weights.sum()
        expected = expected.fork(hypotheses=[
            h.fork(weight=float(w)) for h, w in zip(expected.hypotheses, weights)])
        self.assertPhysicalEqual(posterior, expected)
        self.assertFalse(posterior.evidence.get("negative_evidence_available", False))

    def test_survival_applies_without_a_human_sensor(self):
        model = _model()
        prior = _belief(model)
        invalid = model.observe(prior, _observation(.144, observation_valid=False), _trace(prior))
        unavailable = model.observe(
            prior, _observation(.144, human_sensor_available=False), _trace(prior))
        self.assertPhysicalEqual(unavailable, invalid)
        self.assertFalse(unavailable.evidence.get("negative_evidence_available", False))

    def test_termination_is_independent_of_sensor_availability(self):
        for sensor in ({"observation_valid": False}, {"human_sensor_available": False}):
            with self.subTest(sensor=sensor):
                model = _model()
                prior = _belief(model)
                posterior = model.observe(
                    prior, _observation(.117, **sensor),
                    _trace(prior, episode_terminated=True))
                posterior.validate()
                self.assertTrue(posterior.terminal)
                self.assertEqual(posterior.timestamp, .117)
                absorbed = model.propagate(posterior, .261)
                self.assertTrue(absorbed.terminal)
                self.assertAlmostEqual(sum(h.weight for h in absorbed.hypotheses)
                                       + absorbed.unresolved_mass, 1.)

    def test_survival_replay_is_once_before_support_validation(self):
        model = _model()
        prior = _belief(model)
        row = _observation(.144, observation_valid=False)
        trace = _trace(prior)
        once = model.observe(prior, row, trace)
        self.assertNotEqual(once.fingerprint(), prior.fingerprint())
        twice = model.observe(once, copy.deepcopy(row), copy.deepcopy(trace))
        self.assertPhysicalEqual(twice, once)

    def test_same_trace_later_accompanying_sensor_is_not_applied_twice(self):
        model = _model()
        prior = _belief(model)
        trace = _trace(prior)
        once = model.observe(prior, _observation(.144, observation_valid=False), trace)
        sensor = _observation(.144)
        expected = model.observe(once, sensor)
        actual = model.observe(once, sensor, trace)
        self.assertPhysicalEqual(actual, expected)

    def test_conflicting_trace_replay_is_rejected(self):
        model = _model()
        prior = _belief(model)
        trace = _trace(prior)
        row = _observation(0., observation_valid=False)
        once = model.observe(prior, row, trace)
        changed = {**trace, "world_likelihood": [.8, .2]}
        saved = once.to_dict()
        with self.assertRaises(ValueError):
            model.observe(once, row, changed)
        self.assertEqual(once.to_dict(), saved)

    def test_likelihood_is_bound_to_its_prior_support(self):
        model = _model()
        prior = _belief(model)
        other = prior.fork(hypotheses=[h.fork(mean=h.mean + [1., 0., 0., 0.])
                                     for h in prior.hypotheses])
        with self.assertRaises(ValueError):
            model.observe(other, _observation(0., observation_valid=False), _trace(prior))

    def test_invalid_component_and_residual_likelihoods_rejected(self):
        cases = [
            {"world_likelihood": [-.1, .8]}, {"world_likelihood": [1.1, .8]},
            {"world_likelihood": [np.nan, .8]}, {"world_likelihood": [np.inf, .8]},
            {"world_likelihood": [.8]}, {"world_likelihood": [[.2], [.8]]},
            {"unresolved_likelihood": -.1}, {"unresolved_likelihood": 1.1},
            {"unresolved_likelihood": np.nan}, {"unresolved_likelihood": np.inf},
        ]
        for changed in cases:
            with self.subTest(changed=changed):
                model = _model()
                prior = _belief(model, unresolved=.1)
                saved = prior.to_dict()
                with self.assertRaises(ValueError):
                    model.observe(prior, _observation(0., observation_valid=False),
                                  _trace(prior, **changed))
                self.assertEqual(prior.to_dict(), saved)


class IndependentEvidenceIdentityTests(PhysicalAssertions):
    def test_new_graph_delta_does_not_repeat_the_same_sensor_evidence(self):
        model = _model()
        prior = _belief(model)
        row = _observation(0.)
        once = model.observe(prior, row)
        changed = {**row, "static_updates": [_update((40, 40))]}
        twice = model.observe(once, changed)
        self.assertPhysicalEqual(twice, once)
        self.assertEqual(twice.graph.get((40, 40))[0], 2)
        self.assertEqual(once.graph.get((40, 40))[0], 0)

    def test_trace_only_extension_does_not_repeat_sensor_evidence(self):
        model = _model()
        prior = _belief(model)
        row = _observation(0.)
        observed = model.observe(prior, row)
        trace = _trace(observed)
        weights = np.array([h.weight for h in observed.hypotheses]) * [.2, .8]
        weights /= weights.sum()
        expected = observed.fork(hypotheses=[
            h.fork(weight=float(w)) for h, w in zip(observed.hypotheses, weights)])
        actual = model.observe(observed, row, trace)
        self.assertPhysicalEqual(actual, expected)

    def test_graph_only_arrival_does_not_consume_later_human_evidence(self):
        model = _model()
        prior = _belief(model)
        updates = [_update((40, 40))]
        graph_only = _observation(0., human_sensor_available=False, static_updates=updates)
        intermediate = model.observe(prior, graph_only)
        human = _observation(0., static_updates=updates)
        actual = model.observe(intermediate, human)
        expected = model.observe(prior, human)
        self.assertPhysicalEqual(actual, expected)
        self.assertEqual(actual.graph.materialize(), expected.graph.materialize())

    def test_graph_delta_and_trace_can_both_arrive_after_cached_sensor(self):
        model = _model()
        prior = _belief(model)
        sensor = _observation(0.)
        observed = model.observe(prior, sensor)
        trace = _trace(observed)
        trace_only = model.observe(observed, sensor, trace)
        graph_and_trace = model.observe(
            observed, {**sensor, "static_updates": [_update((40, 40))]}, trace)
        self.assertPhysicalEqual(graph_and_trace, trace_only)
        self.assertEqual(graph_and_trace.graph.get((40, 40))[0], 2)

    def test_privileged_extras_and_tracker_ids_do_not_change_evidence(self):
        model = _model()
        prior = _belief(model)
        sensor = _observation(.048, [(1.5, .1)])
        a = model.observe(prior, sensor)
        changed = copy.deepcopy(sensor)
        changed["tracks"][0]["track_id"] = -999
        changed["tracks"][0]["estimated_velocity"] = [999., 999., 999.]
        changed["privileged_future"] = [[100., 200.]]
        changed["full_scene_geometry"] = "must not be used"
        b = model.observe(prior, changed)
        self.assertPhysicalEqual(a, b)


class SuccessorDomainTests(PhysicalAssertions):
    def test_joint_branch_weights_reject_invalid_domains_without_mutation(self):
        model = _model()
        belief = _belief(model)
        future = model.predict_shared(belief, .144, num_worlds=4, seed=19)
        good = future.weights.copy()
        cases = [
            np.r_[-.01, good[1:]], np.r_[np.nan, good[1:]],
            np.r_[np.inf, good[1:]], good[:-1], good[:, None],
            good * 1.1, np.r_[good[0] + .01, good[1:] * .5],
        ]
        original_positions = future.positions.copy()
        source = belief.to_dict()
        for weights in cases:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    model.successor_at(future, 1, weights, np.zeros(4))
                self.assertEqual(belief.to_dict(), source)
                np.testing.assert_array_equal(future.positions, original_positions)

    def test_invalid_successor_endpoint_is_rejected(self):
        model = _model()
        future = model.predict_shared(_belief(model), .144, num_worlds=4, seed=19)
        for index in (-1, len(future.times), 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.successor_at(future, index, future.weights, np.zeros(4))

    def test_foreign_model_future_is_rejected_at_public_successor(self):
        first, other = _model(), _model(max_hypotheses=8)
        self.assertNotEqual(first.model_version, other.model_version)
        future = first.predict_shared(_belief(first), .144, num_worlds=4, seed=19)
        with self.assertRaises(ValueError):
            other.successor_at(future, 1, future.weights, np.zeros(4))

    def test_zero_mass_branch_is_rejected_or_explicitly_unsupported(self):
        model = _model()
        future = model.predict_shared(_belief(model), .144, num_worlds=4, seed=19)
        try:
            successor = model.successor_at(future, 1, np.zeros(4), np.zeros(4))
        except ValueError:
            return  # An impossible conditional may be explicitly refused.
        successor.validate()
        self.assertEqual(successor.unresolved_mass, 1.)
        self.assertEqual(sum(h.weight for h in successor.hypotheses), 0.)

    def test_valid_joint_branch_has_valid_mass_and_exact_endpoint_clock(self):
        model = _model()
        prior = _belief(model, timestamp=.0130000001)
        future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
        weights = future.weights * np.array([.2, .8, .4, .9])
        successor = model.successor_at(future, 2, weights, np.zeros(4))
        successor.validate()
        self.assertEqual(successor.timestamp, future.times[2])
        self.assertAlmostEqual(sum(h.weight for h in successor.hypotheses)
                               + successor.unresolved_mass, 1.)
        for h in successor.hypotheses:
            self.assertAlmostEqual(h.n + h.expected_hidden, 1.)


class ExactClockTests(PhysicalAssertions):
    def test_finite_horizon_and_explicit_clock_are_required(self):
        model = _model()
        prior = _belief(model)
        for horizon in (-1., np.nan, np.inf, -np.inf):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                model.predict_shared(prior, horizon, num_worlds=4)
        for times in ([], [0., np.nan], [0., np.inf], [-.01, 0., .144],
                      [.048, .144], [0., .145]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                model.predict_shared(prior, .144, num_worlds=4, times=times)

    def test_nonfinite_propagation_and_observation_time_rejected(self):
        for timestamp in (np.nan, np.inf, -np.inf):
            with self.subTest(timestamp=timestamp):
                model = _model()
                prior = _belief(model)
                saved = prior.to_dict()
                with self.assertRaises(ValueError):
                    model.propagate(prior, timestamp)
                for sensor in ({}, {"observation_valid": False}, {"human_sensor_available": False}):
                    with self.subTest(sensor=sensor), self.assertRaises(ValueError):
                        model.observe(prior, _observation(timestamp, **sensor))
                self.assertEqual(prior.to_dict(), saved)

    def test_exact_observation_preserves_absolute_timestamp_and_coordinates(self):
        model = _model()
        prior = _belief(model, timestamp=.0130000001)
        timestamp = .1170000003
        position = (1.2500000007, -.1250000004)
        posterior = model.observe(prior, _observation(timestamp, [position]))
        posterior.validate()
        self.assertEqual(posterior.timestamp, timestamp)
        self.assertGreater(sum(h.weight for h in posterior.hypotheses), 0.)
        for h in posterior.hypotheses:
            if h.weight <= 0:
                continue
            seen = np.flatnonzero(h.last_seen == timestamp)
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(h.mean[seen[0], :2], position)
            np.testing.assert_array_equal(h.covariance[seen[0], :2], np.zeros((2, 4)))
            self.assertEqual(h.height[seen[0]], .9)

    def test_detection_with_a_different_clock_is_rejected(self):
        model = _model()
        prior = _belief(model)
        row = _observation(.117, [(1.25, 0.)])
        row["tracks"][0]["last_seen_time"] += .001
        with self.assertRaises(ValueError):
            model.observe(prior, row)

    def test_future_samples_keep_absolute_non_native_origin(self):
        model = _model()
        prior = _belief(model, timestamp=.0130000001)
        relative = np.array([0., .048, .096, .1170000003])
        future = model.predict_shared(prior, relative[-1], num_worlds=4, times=relative)
        np.testing.assert_array_equal(future.times, prior.timestamp + relative)
        self.assertEqual(future.times[-1], prior.timestamp + relative[-1])
        self.assertEqual(prior.timestamp, .0130000001)

    def test_zero_duration_does_not_change_a_prepared_belief(self):
        model = _model()
        prior = _belief(model, timestamp=.0130000001)
        # Prepare once so this assertion cannot mistake initialization of an
        # explicitly absent latent for information gained by a zero-time call.
        prior = model.propagate(prior, prior.timestamp)
        after = model.propagate(prior, prior.timestamp)
        self.assertPhysicalEqual(after, prior)
        future = model.predict_shared(prior, 0., num_worlds=4)
        np.testing.assert_array_equal(future.times, [prior.timestamp])

    def test_old_observation_is_rejected_even_without_a_sensor(self):
        model = _model()
        prior = _belief(model, timestamp=.144)
        for sensor in ({}, {"observation_valid": False}, {"human_sensor_available": False}):
            with self.subTest(sensor=sensor), self.assertRaises(ValueError):
                model.observe(prior, _observation(.096, **sensor))


if __name__ == "__main__":
    unittest.main()
