"""Independent adversarial tests of finite-carrier API and persistence.

No learned parameters or data are loaded. Tests exercise public operations and
serialized public state rather than accepting an immutable dataclass label as
proof that the represented probability law is immutable or validated.
"""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.l2_partial.structural_budget import (
    ACTIVE_DEADLINE, InferenceBudgetExceeded, UnsupportedConditioning,
)
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from test_api_semantics import _model, _belief, _observation, _trace, _update, PhysicalAssertions


def _query(actions=2):
    poses = [[i * 25 / 120, 0., 0., 0.] for i in range(actions + 1)]
    return {
        "query_id": "synthetic-carrier-query", "executor_version": EXECUTOR_VERSION,
        "compiled": {"planned_poses": poses, "actions": ["forward"] * actions,
                     "graph_revision": 1},
        "waypoints": [poses[-1][:3]], "final_yaw": 0.,
    }


def _finite_child(model=None):
    model = model or _model()
    prior = _belief(model)
    future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
    child = model.successor_at(
        future, 1, future.weights * np.array([.2, .8, .4, .9]), np.zeros(4))
    if child.carrier is None:
        raise AssertionError("fixture must exercise an actual finite conditional carrier")
    return model, prior, future, child


def _coarse_branch(kind):
    model = _model()
    prior = _belief(model)
    future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
    result = model.evaluate_queries(
        prior, [_query()], future, budget_ms=10000., refined_queries=1).predictions[0]
    branch = next(b for b in result.observation_branches if b.observation["kind"] == kind)
    return model, prior, future, result, branch.successor


class CarrierSerializationTests(PhysicalAssertions):
    def test_json_roundtrip_preserves_nonzero_modes_and_float32_latents(self):
        for use_float32 in (False, True):
            with self.subTest(float32_latents=use_float32):
                model = _model()
                prior = _belief(model)
                hypotheses = []
                for i, h in enumerate(prior.hypotheses):
                    changes = {"mode": np.array([i + 1], dtype=np.int64)}
                    if use_float32:
                        changes.update(intent=h.intent.astype(np.float32),
                                       diffusion=h.diffusion.astype(np.float32))
                    hypotheses.append(h.fork(**changes))
                prior = StructuralBeliefState.wrap(prior.fork(hypotheses=hypotheses))
                future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
                child = model.successor_at(future, 1, future.weights * .5, np.zeros(4))
                for label, original in (("continuous", prior), ("carrier", child)):
                    with self.subTest(state=label):
                        restored = StructuralBeliefState.from_dict(
                            json.loads(json.dumps(original.to_dict(), allow_nan=False)))
                        self.assertPhysicalEqual(restored, original)
                        self.assertEqual(restored.fingerprint(), original.fingerprint())

    def test_restoring_state_does_not_modify_the_input_dictionary(self):
        _, _, _, child = _finite_child()
        value = child.to_dict()
        before = copy.deepcopy(value)
        StructuralBeliefState.from_dict(value)
        self.assertEqual(value, before)

    def test_export_dictionary_does_not_alias_live_evidence(self):
        _, _, _, child = _finite_child()
        before = copy.deepcopy(child.evidence)
        exported = child.to_dict()
        exported["evidence"]["successor_measure"] = "accidentally changed export"
        self.assertEqual(child.evidence, before)

    def test_coarse_ledger_json_roundtrip_preserves_fingerprint(self):
        _, _, _, _, child = _coarse_branch("survived_detection")
        restored = StructuralBeliefState.from_dict(json.loads(json.dumps(child.to_dict())))
        self.assertPhysicalEqual(restored, child)
        self.assertEqual(restored.fingerprint(), child.fingerprint())

    def test_changed_future_path_changes_fingerprint_or_is_rejected(self):
        model, _, _, child = _finite_child()
        changed = copy.deepcopy(child.to_dict())
        endpoint = child.carrier.index + 1
        changed["carrier"]["positions"][0][endpoint][0][0] += 2.
        try:
            restored = StructuralBeliefState.from_dict(changed)
        except ValueError:
            return  # An integrity-bound serialized tape may reject edits.
        original_end = model.propagate(child, child.carrier.times[endpoint])
        changed_end = model.propagate(restored, child.carrier.times[endpoint])
        self.assertNotEqual(original_end.hypotheses[0].mean[0, 0],
                            changed_end.hypotheses[0].mean[0, 0])
        self.assertNotEqual(restored.fingerprint(), child.fingerprint())

    def test_carrier_domains_are_validated_on_restore(self):
        _, _, _, child = _finite_child()
        cases = {
            "negative weight": lambda c: c["weights"].__setitem__(0, -.1),
            "weight total": lambda c: c["weights"].__setitem__(0, .99),
            "weight shape": lambda c: c.__setitem__("weights", [[w] for w in c["weights"]]),
            "nonfinite time": lambda c: c["times"].__setitem__(1, float("nan")),
            "unordered time": lambda c: c["times"].__setitem__(1, -.048),
            "endpoint index": lambda c: c.__setitem__("index", 1000),
            "noninteger endpoint": lambda c: c.__setitem__("index", 1.5),
            "world count": lambda c: c["worlds"].pop(),
            "path shape": lambda c: c["positions"].pop(),
        }
        for name, corrupt in cases.items():
            with self.subTest(case=name):
                value = copy.deepcopy(child.to_dict())
                corrupt(value["carrier"])
                with self.assertRaises(ValueError):
                    StructuralBeliefState.from_dict(value)

    def test_carrier_and_public_endpoint_cannot_describe_different_states(self):
        _, _, _, child = _finite_child()
        value = copy.deepcopy(child.to_dict())
        value["timestamp"] += .012
        with self.assertRaises(ValueError):
            StructuralBeliefState.from_dict(value)

    def test_restored_latent_buffers_remain_read_only(self):
        _, _, _, child = _finite_child()
        restored = StructuralBeliefState.from_dict(json.loads(json.dumps(child.to_dict())))
        with self.assertRaises(ValueError):
            restored.carrier.worlds[0].intent[0, 0] += 10.


class CarrierContinuationTests(PhysicalAssertions):
    def test_past_endpoint_ledger_is_compacted_after_continuation(self):
        model, _, _, _, child = _coarse_branch("survived_no_detection")
        self.assertTrue(child.carrier.observation_ledger)
        continued = model.propagate(child, .144)
        self.assertTrue(all(event[1] >= continued.timestamp - 1e-8
                            for event in continued.carrier.observation_ledger))
        np.testing.assert_array_equal(continued.carrier.weights, child.carrier.weights)

    def test_foreign_future_cannot_use_the_unbound_belief_escape_hatch(self):
        first, other = _model(), _model(max_hypotheses=8)
        prior = _belief(first).fork(model_version="unbound")
        future = first.predict_shared(prior, .144, num_worlds=4, seed=19)
        with self.assertRaises(ValueError):
            other.evaluate_queries(prior, [_query()], future, budget_ms=10000.)

    def test_unobserved_successor_rejects_mutated_continuous_source(self):
        model = _model()
        prior = _belief(model)
        future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
        prior.hypotheses[0].mean[0, 0] += 10.
        with self.assertRaises(ValueError):
            model.successor_at(future, 1, future.weights, np.zeros(4))

    def test_no_observation_uses_existing_joint_path_at_stored_endpoint(self):
        model, _, _, child = _finite_child()
        endpoint = child.carrier.index + 1
        continued = model.propagate(child, child.carrier.times[endpoint])
        np.testing.assert_array_equal(
            np.stack([h.mean[:, :2] for h in continued.hypotheses]),
            child.carrier.positions[:, endpoint, :1])
        np.testing.assert_array_equal(continued.carrier.weights, child.carrier.weights)
        self.assertEqual(continued.carrier.origin_fingerprint, child.carrier.origin_fingerprint)

    def test_extending_a_carrier_keeps_its_existing_future_prefix(self):
        model, _, _, child = _finite_child()
        extended = model.predict_shared(child, .240, num_worlds=4, seed=7)
        old = child.carrier
        for i, t in enumerate(extended.times):
            matching = np.flatnonzero(np.isclose(old.times, t, rtol=0, atol=1e-12))
            if len(matching):
                np.testing.assert_array_equal(extended.positions[:, i], old.positions[:, matching[0]])
                np.testing.assert_array_equal(extended.velocities[:, i], old.velocities[:, matching[0]])
        np.testing.assert_array_equal(extended.weights, old.weights)

    def test_off_grid_carrier_interpolation_is_explicitly_unsupported(self):
        model, _, _, child = _finite_child()
        with self.assertRaises(UnsupportedConditioning) as caught:
            model.propagate(child, child.timestamp + .012)
        self.assertEqual(caught.exception.status, "UNSUPPORTED_CONTINUOUS_REFINEMENT")

    def test_exact_detection_on_coarse_carrier_is_refused_without_mutation(self):
        model, _, _, child = _finite_child()
        before = copy.deepcopy(child.to_dict())
        with self.assertRaises(UnsupportedConditioning):
            model.observe(child, _observation(child.timestamp, [(1.25, 0.)]))
        self.assertEqual(child.to_dict(), before)

    def test_same_endpoint_no_detection_refinement_does_not_repeat_miss(self):
        model, _, _, _, child = _coarse_branch("survived_no_detection")
        row = _observation(child.timestamp, robot_pose=child.robot_pose.tolist())
        updated = model.observe(child, row)
        self.assertPhysicalEqual(updated, child)
        # The evidence ledger may gain an identity, but its physical law must not.
        np.testing.assert_array_equal(updated.carrier.weights, child.carrier.weights)

    def test_same_endpoint_empty_after_any_detection_uses_conditional_event(self):
        model, _, _, result, child = _coarse_branch("survived_detection")
        future = model.predict_shared(_belief(model), .144, num_worlds=4, seed=19)
        index = result.diagnostics["endpoint_index"]
        indices = np.arange(index + 1)
        poses = np.asarray(_query()["compiled"]["planned_poses"])
        no_all, no_before, no_last = model.observation_law.no_detection(
            future, indices, poses, return_parts=True)
        ratio = (1 - no_before) * no_last / (1 - no_all)
        expected = child.carrier.weights * ratio
        expected /= expected.sum()
        updated = model.observe(
            child, _observation(child.timestamp, robot_pose=child.robot_pose.tolist()))
        np.testing.assert_allclose(updated.carrier.weights, expected, atol=2e-14, rtol=0)


class AtomicOnlineResultTests(unittest.TestCase):
    def test_initialized_and_actually_observed_graph_storage_is_immutable(self):
        # This exercises base initialize's replacement of the history graph
        # with the legal root snapshot, not just wrap(raw BeliefState).
        model = StructuralGraphBeliefWorldModel(
            kind="memory_cv", maneuver_rate=0., population_prior=[0., 1.])
        snapshot = {
            "revision": 1, "timestamp": 0., "resolution": .2, "height": 0.,
            "cells": [{"cell": [i, j], "state": 1, "height": 0.}
                      for i in range(-15, 16) for j in range(-5, 6)],
        }
        initial = model.initialize([_observation(0., [(2., 0.)])], snapshot)
        observed = model.observe(
            initial, _observation(.048, [(2.01, 0.)]),
            {"evidence_id": "actual-graph-arrival", "start_timestamp": 0.,
             "end_timestamp": .048, "episode_terminated": False})
        with_delta = model.observe(
            observed, _observation(.048, [(2.01, 0.)],
                                   static_updates=[_update((40, 40))]))
        self.assertEqual(initial.graph.get((40, 40))[0], 0)
        self.assertEqual(observed.graph.get((40, 40))[0], 0)
        self.assertEqual(with_delta.graph.get((40, 40))[0], 2)
        for label, belief in (("initialized", initial), ("arrived", observed),
                              ("delta", with_delta)):
            with self.subTest(stage=label):
                graph = belief.graph
                while graph is not None:
                    with self.assertRaises(TypeError):
                        graph.cells[(41, 41)] = (1, 0.)
                    graph = graph.parent
        # Mutating the caller's snapshot cannot change the model's owned map.
        snapshot["cells"][0]["state"] = 2
        self.assertEqual(initial.graph.get((-15, -5))[0], 1)

    def test_invalid_whole_observation_cannot_certify_new_graph_cells(self):
        model = _model()
        prior = _belief(model)
        row = _observation(.048, observation_valid=False, static_updates=[_update((40, 40))])
        updated = model.observe(prior, row, _trace(prior, episode_terminated=True))
        self.assertTrue(updated.terminal)
        self.assertEqual(updated.graph.get((40, 40))[0], 0)

    def test_current_endpoint_evidence_capacity_is_bounded_and_explicit(self):
        model = _model()
        belief = _belief(model)
        for i in range(32):
            belief = model.observe(belief, _observation(0., human_observation_id="frame-" + str(i)))
        before = copy.deepcopy(belief.to_dict())
        with self.assertRaises(UnsupportedConditioning) as caught:
            model.observe(belief, _observation(0., human_observation_id="frame-over-capacity"))
        self.assertEqual(caught.exception.status, "UNSUPPORTED_EVIDENCE_CAPACITY")
        self.assertEqual(belief.to_dict(), before)

    def test_timeout_without_returned_belief_keeps_actual_observation_pending(self):
        model = _model()
        prior = _belief(model)
        before = copy.deepcopy(prior.to_dict())

        def expire_after_observe(*args, **kwargs):
            deadline = ACTIVE_DEADLINE.get()
            raise InferenceBudgetExceeded(
                "forecast:injected", 10., deadline.budget_ms, deadline.pending_observation)

        with patch.object(model, "predict_shared", side_effect=expire_after_observe):
            with self.assertRaises(InferenceBudgetExceeded) as caught:
                model.online_call(prior, _observation(.048), [], horizon=.144, budget_ms=10.)
        self.assertEqual(prior.to_dict(), before)
        self.assertTrue(caught.exception.pending_observation)


class NonExecutionRiskTests(unittest.TestCase):
    """Root source-audit finding: residual mass cannot create an event at n=0.

    Written before the coordinator's fix; the first API-example failure was a
    separate executed graph witness. These no-execution cases become measured
    regression evidence only when this suite is subsequently run.
    """

    def assertNoNewNativeEvent(self, prediction):
        self.assertEqual(tuple(prediction.collision_interval), (0., 0.))
        self.assertEqual(tuple(prediction.psc_interval), (0., 0.))
        self.assertEqual(prediction.collision_probability, 0.)
        self.assertEqual(prediction.psc_probability, 0.)
        self.assertEqual(prediction.time_to_conflict["unresolved_first_event_mass"], 0.)
        self.assertEqual(len(prediction.time_to_conflict["times_s"]), 0)
        self.assertEqual(prediction.evaluated_interval[0], prediction.evaluated_interval[1])

    def test_absorbed_terminal_has_no_new_event_and_returns_same_belief(self):
        model = _model()
        prior = _belief(model, unresolved=.2).fork(terminal=True)
        future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
        prediction = model.evaluate_queries(
            prior, [_query()], future, budget_ms=10000., refined_queries=1).predictions[0]
        self.assertNoNewNativeEvent(prediction)
        self.assertEqual(prediction.termination_probabilities["absorbed"], 1.)
        self.assertEqual(len(prediction.observation_branches), 1)
        branch = prediction.observation_branches[0]
        self.assertEqual(branch.probability, 1.)
        self.assertEqual(branch.observation["kind"], "terminal_absorbed")
        self.assertIs(branch.successor, prior)

    def test_zero_horizon_has_no_new_event_despite_unresolved_population(self):
        model = _model()
        prior = _belief(model, unresolved=.2)
        future = model.predict_shared(prior, 0., num_worlds=4, seed=19)
        prediction = model.evaluate_queries(
            prior, [_query()], future, budget_ms=10000., refined_queries=1).predictions[0]
        self.assertNoNewNativeEvent(prediction)
        self.assertFalse(prediction.query_complete)
        self.assertEqual(prediction.remaining_query_cursor["action_index"], 0)
        np.testing.assert_array_equal(prediction.duration_samples,
                                      np.zeros_like(prediction.duration_samples))

    def test_invalidated_route_is_known_nonexecution_not_epistemic_failure(self):
        model = _model()
        prior = _belief(model, unresolved=.2)
        prior = prior.fork(graph=prior.graph.update([_update((1, 0))]))
        future = model.predict_shared(prior, .144, num_worlds=4, seed=19)
        prediction = model.evaluate_queries(
            prior, [_query()], future, budget_ms=10000., refined_queries=1).predictions[0]
        self.assertNoNewNativeEvent(prediction)
        self.assertFalse(prediction.diagnostics["static_support_checked"])
        self.assertEqual(prediction.termination_probabilities["route_invalidated"], 1.)
        self.assertEqual(prediction.termination_probabilities.get("unresolved", 0.), 0.)
        self.assertFalse(prediction.query_complete)
        self.assertEqual(prediction.remaining_query_cursor["action_index"], 0)


if __name__ == "__main__":
    unittest.main()
