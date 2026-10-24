"""Synthetic public-API example; no dataset, fitting, collection, or policy.

The default uses a declared one-person synthetic prior and simple dynamics.
--frozen-checkpoints loads the four checksum-verified exploratory artifacts,
without training. Both modes use the same synthetic legal input fixture.
Run only after coordinating the research CPU allowance. This is a correctness
example with a generous fixture budget, not a latency measurement.
"""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.l2_partial.structural_budget import InferenceBudgetExceeded, UnsupportedConditioning
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState


ASSETS = {
    "dynamics": ("output/l2_partial_graph_wm_explore_20261021/learned_posdiff/checkpoint.pt",
                 "57c91156101320e2be6a02aee23d214a0405959e823b73be3786b0287bd25c05"),
    "population": ("output/l2_partial_graph_wm_explore_20261021/train.npy.population.json",
                   "b7df625ac62e0eae09656fa32532b72e28039af140fc7b2803e7866b8f1f539c"),
    "events": ("output/l2_partial_graph_wm_repair_20261021/native_events_v5_scene.json",
               "feec91f6918963f618c8a44694fb210bcd92549c9a92e52fc394f2c0b6aa736b"),
    "optics": ("output/l2_partial_graph_wm_repair_20261021/observation_fit.json",
               "ace29a5160b4a8510ac96d3c7bfb3cbad0936fc809c2dd35bf302aaa874e8793"),
}


def build_model(frozen_checkpoints=False, optimized=True):
    kwargs = {"optimized": optimized, "residual_policy": "bound",
              "maneuver_rate": 2., "reset_std": 2., "maneuver_retention": 1.}
    provenance = {}
    if frozen_checkpoints:
        from eventwm_graph.l2_partial.neural import LearnedDynamics
        from eventwm_graph.l2_partial.queries import NativeEventModel
        import torch
        torch.set_num_threads(1)
        paths = {}
        for name, (relative, expected) in ASSETS.items():
            path = REPO / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("frozen artifact checksum mismatch: " + name)
            paths[name] = path
            provenance[name] = {"path": relative, "sha256": actual}
        kwargs.update(
            dynamics=LearnedDynamics.load(str(paths["dynamics"])),
            population_prior=json.loads(paths["population"].read_text()),
            event_model=NativeEventModel(json.loads(paths["events"].read_text())["coefficients"]),
            observation_checkpoint=str(paths["optics"]))
    else:
        kwargs["population_prior"] = [0., 1.]  # The synthetic fixture declares one person.
    return StructuralGraphBeliefWorldModel(**kwargs), provenance


def observation(timestamp, position, robot_pose=(0., 0., 0., 0.)):
    return {
        "timestamp": float(timestamp), "robot_pose": list(robot_pose),
        "observation_valid": True, "human_sensor_available": True,
        "static_updates": [],
        "tracks": [{"position": [float(position), 0., 0.], "last_seen_time": float(timestamp),
                    "visible_now": True, "valid": True}],
    }


def legal_fixture():
    history = [observation(0., 2.)]
    graph = {"revision": 1, "timestamp": 0., "resolution": .2, "height": 0.,
             "cells": [{"cell": [i, j], "state": 1, "height": 0.}
                       for i in range(-15, 16) for j in range(-10, 11)]}
    return history, graph


def queries(graph_version, steps=6):
    result = []
    for turn in (False, True):
        poses = [[0. if turn else i * 25 / 120, 0., 0., i * 10 / 120 if turn else 0.]
                 for i in range(steps + 1)]
        result.append({
            "query_id": "turn" if turn else "forward", "executor_version": EXECUTOR_VERSION,
            "compiled": {"planned_poses": poses, "actions": ["left" if turn else "forward"] * steps,
                         "graph_revision": graph_version},
            "waypoints": [poses[-1][:3]], "final_yaw": poses[-1][3],
        })
    return result


def prediction_summary(prediction):
    return {
        "query_id": prediction.query_id,
        "collision_upper_estimate": prediction.collision_probability,
        "collision_interval": prediction.collision_interval,
        "psc_upper_estimate": prediction.psc_probability,
        "psc_interval": prediction.psc_interval,
        "evaluated_interval": prediction.evaluated_interval,
        "query_complete": prediction.query_complete,
        "remaining_query_cursor": prediction.remaining_query_cursor,
        "unresolved_mass": prediction.unresolved_uncertainty,
        "branches": [{"kind": b.observation["kind"], "represented_mass": b.probability,
                      "probability_interval": b.observation.get("probability_interval"),
                      "continuous_refinement_supported": b.successor.evidence.get("continuous_bridge_supported"),
                      "epistemic_remainder": b.observation["kind"] == "unresolved_physical_outcome"}
                     for b in prediction.observation_branches],
    }


def run(frozen_checkpoints=False, optimized=True):
    model, assets = build_model(frozen_checkpoints, optimized)
    history, graph = legal_fixture()
    initial = model.initialize(history, graph)
    initial.validate()
    assert initial.graph.get((100, 100))[0] == 0  # Outside observed support remains UNKNOWN.

    # Arrived legal input, not a sampled future promoted to real evidence.
    arrived = observation(.048, 2.01)
    trace_metadata = {"evidence_id": "synthetic-arrival-1", "start_timestamp": 0.,
                      "end_timestamp": .048, "episode_terminated": False}
    actual = model.observe(initial, arrived, trace_metadata)
    actual.validate()

    try:
        actual.graph.cells[(100, 100)] = (1, 0.)
    except TypeError:
        graph_write_rejected = True
    else:
        raise AssertionError("structural graph storage must reject direct mutation")

    restored_actual = StructuralBeliefState.from_dict(
        json.loads(json.dumps(actual.to_dict(), allow_nan=False)))
    assert restored_actual.fingerprint() == actual.fingerprint()

    plans = queries(actual.graph.version)
    shared = model.predict_shared(actual, horizon=.144, num_worlds=16, seed=7)
    batch = model.evaluate_queries(actual, plans, shared, budget_ms=5000., refined_queries=2)
    assert batch.shared_future is shared  # Both reference routes use the same human worlds.
    first = batch.predictions[0]
    assert not first.query_complete

    # No observation/event evidence: preserve the continuous transition law.
    unobserved = model.successor_at(shared, 1, shared.weights,
                                   np.asarray(plans[0]["compiled"]["planned_poses"][1]))
    continued_unobserved = model.propagate(unobserved, float(shared.times[2]))
    assert continued_unobserved.carrier is None
    continued_unobserved.validate()

    # Selecting this hypothetical branch is an API demonstration, not a policy.
    branch = next(b for b in first.observation_branches
                  if b.observation["kind"] == "survived_no_detection")
    hypothetical = branch.successor
    hypothetical.validate()
    continuation_future = model.predict_shared(hypothetical, .048, num_worlds=16, seed=7)
    continuation = model.evaluate_queries(
        hypothetical, [plans[0]], continuation_future, budget_ms=5000., refined_queries=1)
    assert continuation.predictions[0].remaining_query_cursor["action_index"] > first.remaining_query_cursor["action_index"]

    encoded = json.dumps(hypothetical.to_dict(), allow_nan=False)
    restored = StructuralBeliefState.from_dict(json.loads(encoded))
    assert restored.fingerprint() == hypothetical.fingerprint()

    before = hypothetical.fingerprint()
    try:
        model.observe(hypothetical, observation(hypothetical.timestamp, 2.03, hypothetical.robot_pose))
    except UnsupportedConditioning as exc:
        refinement_status = exc.status
    else:
        raise AssertionError("finite coarse carrier cannot silently become a continuous exact posterior")
    assert hypothetical.fingerprint() == before

    # The actual continuous belief is retained separately from speculation.
    actual_after_observation = model.observe(
        actual, observation(hypothetical.timestamp, 2.03, hypothetical.robot_pose))
    actual_after_observation.validate()

    pending = observation(actual.timestamp + .048, 2.02)
    before = actual.fingerprint()
    try:
        model.online_call(actual, pending, plans, horizon=.144, num_worlds=16, seed=7, budget_ms=0.)
    except InferenceBudgetExceeded as exc:
        timeout = {"status": exc.status, "pending_observation": exc.pending_observation,
                   "stage": exc.stage, "budget_ms": exc.budget_ms}
        assert exc.pending_observation
    else:
        raise AssertionError("zero budget must return an explicit atomic timeout")
    assert actual.fingerprint() == before

    return {
        "schema": "l2_structural_api_example_1", "data": "synthetic legal observations only",
        "training_performed": False, "frozen_pretrained_assets": assets,
        "optimized": optimized, "model_version": model.model_version,
        "executor_version": EXECUTOR_VERSION, "graph_write_rejected": graph_write_rejected,
        "actual_timestamp": actual.timestamp, "actual_updated_timestamp": actual_after_observation.timestamp,
        "shared_worlds": len(shared.worlds), "predictions": [prediction_summary(p) for p in batch.predictions],
        "continuation": prediction_summary(continuation.predictions[0]),
        "analytic_successor_continues": True, "continuous_json_roundtrip": True,
        "carrier_json_roundtrip": True,
        "unsupported_refinement_status": refinement_status, "atomic_timeout": timeout,
        "limitations": ["finite hypothetical carriers have no continuous exact-observation bridge",
                        "reported physical-branch masses are lower contributions with coupled residual intervals",
                        "fixture budgets are not online latency measurements",
                        "trace metadata carries no survival likelihood; no automatic native-trace adapter is implied"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-checkpoints", action="store_true")
    parser.add_argument("--reference-kernels", action="store_true")
    args = parser.parse_args()
    value = run(args.frozen_checkpoints, not args.reference_kernels)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
