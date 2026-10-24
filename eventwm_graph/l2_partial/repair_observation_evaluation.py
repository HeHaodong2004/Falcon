"""Development-only calibration of the actual generative coarse observation law.

The event is contact, or survival with no/any visible-center detections over the
entire planned interval, excluding the root. The inherited untracked-appearance
head is never used. Future worlds are regenerated from legal inputs and must
exactly match the frozen native-event cache before actual observations are read.

Residual contact follows the same declared hazard law. Residual survival has no
specified optical distribution, so its no/any-detection allocation remains an
interval. Proper categorical scores are bounded over that allocation, without
inventing a point prediction. This module has no fitting or holdout entry point.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import time
import unittest

import numpy as np
from scipy.special import expit
import torch

from .data import read
from .neural import LearnedDynamics
from .queries import event_features
from .repair_events import (_cohort_rows, _factory, _json, _output_path, _publish,
    _sha, _sources as _core_sources, event_clock_inputs, load_cache,
    model_configuration, planned_steps, RESIDUAL_HAZARD, STEP_SECONDS)
from .repair_evaluation import scene_ci
from .repair_risk import _infinite_aware_summary, CALIBRATION_BINS


SCHEMA = "partial_graph_generative_observation_evaluation_1"
CLASSES = ("contact", "survived_no_detections", "survived_any_detections")


def _sources():
    package = Path(__file__).parent
    return {**_core_sources(), "repair_observation_evaluation.py": _sha(__file__),
        "repair_evaluation.py": _sha(package / "repair_evaluation.py"),
        "repair_risk.py": _sha(package / "repair_risk.py")}


def _destination(path):
    base = Path(__file__).resolve().parents[2] / "output" / "l2_partial_graph_wm_repair_20261021"
    result = Path(path).resolve()
    if not base.is_dir() or base.resolve() != base or result == base or not result.is_relative_to(base):
        raise ValueError("observation artifacts must remain in the existing research repair output")
    return _output_path(result)


def native_class(event_record, observations):
    """Actual labels only; an absorbing contact overrides its detection history."""
    if len(observations) != event_record["observed_steps"]:
        raise ValueError("native observation prefix length mismatch")
    detection_count = sum(sum(bool(track["visible_now"] and track.get("valid", True))
                              for track in row["tracks"]) for row in observations)
    any_detection = detection_count > 0
    if event_record["first"][0] >= 0:
        if event_record["administrative_censoring"]:
            raise ValueError("contact cannot be administratively censored")
        target = 0
    elif event_record["administrative_censoring"]:
        target = None
    elif event_record["observed_steps"] == event_record["planned_steps"]:
        target = 2 if any_detection else 1
    else:
        raise ValueError("noncontact incomplete exposure is not marked censored")
    return {"target": target, "observed_any_detection": any_detection,
            "observed_detection_count": int(detection_count)}


def _same_configuration(actual, cached):
    # Runtime optical parameters are tuples; their saved JSON representation is
    # lists. Compare canonical serialized values without relaxing numeric checks.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(cached, sort_keys=True, allow_nan=False)


def prepare(snapshot, event_cache, checkpoint, population, cohort, output):
    torch.set_num_threads(1)
    source = _sources()
    output = _destination(output)
    metadata_path = _destination(str(output) + ".manifest.json")
    events, metadata = load_cache(event_cache, "dev")
    if metadata["source_sha256"] != _core_sources():
        raise ValueError("event-cache core sources differ from the current frozen model")
    for actual, expected, label in ((_sha(checkpoint), metadata["checkpoint_sha256"], "dynamics"),
            (_sha(population), metadata["population_sha256"], "population"),
            (_sha(Path(snapshot) / "manifest.json"), metadata["snapshot_manifest_sha256"], "snapshot")):
        if actual != expected:
            raise ValueError("event-cache provenance mismatch: " + label)
    if metadata["seed"] != 31 or metadata["worlds"] != 16 or metadata["horizon_seconds"] != 3.:
        raise ValueError("observation calibration requires the frozen3s/16world/seed31 workload")
    selected = _cohort_rows(snapshot, "dev", cohort)
    permitted = _cohort_rows(snapshot, "dev", None)
    selected_ids = {row["root_id"] for row, _ in selected}
    if (selected_ids != {row["root_id"] for row, _ in permitted} or len(selected) != 632
            or len({row["scene_id"] for row, _ in selected}) != 106):
        raise ValueError("observation calibration requires all frozen632 development roots /106 scenes")
    grouped = collections.defaultdict(dict)
    for record in events:
        if record["query_id"] in grouped[record["root_id"]]:
            raise ValueError("duplicate native-event cache query")
        grouped[record["root_id"]][record["query_id"]] = record
    if set(grouped) != selected_ids:
        raise ValueError("event-cache roots differ from the fixed development cohort")
    kwargs = metadata["model_kwargs"]
    optical_path = kwargs.get("observation_checkpoint")
    if not optical_path:
        raise ValueError("explicit fitted optical checkpoint required")
    optical_sha = _sha(optical_path)
    model = _factory(metadata["model_factory"])(dynamics=LearnedDynamics.load(checkpoint),
        population_prior=json.loads(Path(population).read_text()), **kwargs)
    if not getattr(model, "observation_repair", False) or not _same_configuration(model_configuration(model), metadata["model_configuration"]):
        raise ValueError("regenerated observation model differs from the event-cache model")
    result = []
    started = time.perf_counter()
    counts = collections.Counter()
    for index, (meta, folder) in enumerate(selected):
        # Read only legal inputs until every counterfactual observation law for
        # this root has been evaluated. No future observation or truth input.
        legal = read(folder / "legal_root.json.gz")
        queries = read(folder / "queries.json.gz")
        ids = [query["query_id"] for query in queries]
        if len(ids) != len(set(ids)) or set(ids) != set(grouped[meta["root_id"]]):
            raise ValueError("legal query inventory differs from native cache")
        belief = model.initialize(legal["history"], legal["partial_graph"])
        steps = [planned_steps(query) for query in queries]
        times = np.arange(max(steps) + 1, dtype=float) * STEP_SECONDS
        future = model.predict_shared(belief, float(times[-1]), 16, seed=31, times=times)
        hidden = np.array([hypothesis.expected_hidden for hypothesis in future.worlds])
        total = hidden + np.array([hypothesis.n for hypothesis in future.worlds])
        allowed = np.stack([total > 0, total > 0, hidden > 0], -1)
        age = np.mean([max(0., belief.timestamp - hypothesis.last_seen.mean()) if hypothesis.n else 0.
                       for hypothesis in future.worlds])
        predicted = []
        for query, planned in zip(queries, steps):
            event = grouped[meta["root_id"]][query["query_id"]]
            if planned != event["planned_steps"] or meta["scene_id"] != event["scene_id"]:
                raise ValueError("planned horizon or scene differs from native cache")
            if not np.array_equal(future.weights, event["weights"]) or future.unresolved_mass != event["unresolved"]:
                raise ValueError("regenerated world weights or unresolved mass differ from native cache")
            if not np.array_equal(allowed, event["allowed"]):
                raise ValueError("regenerated population risk support differs from native cache")
            poses = np.asarray(query["compiled"]["planned_poses"][:planned + 1], float)
            feature_age, feature_elapsed = event_clock_inputs(model, age, times[:planned + 1])
            features = event_features(future.positions[:, :planned + 1], future.velocities[:, :planned + 1],
                poses, hidden, feature_age, feature_elapsed)[:, 1:].astype(np.float32)
            if not np.array_equal(features, event["x"]):
                raise ValueError("regenerated full-planned float32 features differ from native cache")
            no_detection = model.observation_law.no_detection(future, np.arange(planned + 1), poses)
            if no_detection.shape != future.weights.shape or not np.isfinite(no_detection).all() or np.any((no_detection < 0) | (no_detection > 1)):
                raise ValueError("invalid world-conditioned no-detection probability")
            predicted.append({"schema": SCHEMA, "split": "dev", "root_id": meta["root_id"],
                "scene_id": meta["scene_id"], "query_id": query["query_id"],
                "no_detection": no_detection.copy(), "planned_steps": planned,
                "observed_steps": event["observed_steps"], "administrative_censoring": event["administrative_censoring"],
                "root_weight": event["root_weight"], "unresolved": event["unresolved"],
                "parity": "bitwise weights, allowed masks, unresolved mass and float32 native features"})
        # Actual legal observation labels are opened only after prediction.
        outcomes = read(folder / "outcomes.json.gz")
        nodes = read(folder / "future_observations.json.gz")
        if len(outcomes) != len(queries):
            raise ValueError("native outcomes/query cardinality mismatch")
        for prediction, outcome in zip(predicted, outcomes):
            event = grouped[meta["root_id"]][prediction["query_id"]]
            if min(len(outcome["node_ids"]), event["planned_steps"]) != event["observed_steps"]:
                raise ValueError("actual observation count differs from native cache")
            observations = [nodes[node_id]["observation"] for node_id in outcome["node_ids"][:event["observed_steps"]]]
            prediction.update(native_class(event, observations))
            result.append(prediction)
            counts["queries"] += 1
            counts["administratively_censored"] += prediction["target"] is None
            if prediction["target"] is not None:
                counts[CLASSES[prediction["target"]]] += 1
        if (index + 1) % 25 == 0:
            print(json.dumps({"roots": index + 1, "queries": len(result), "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    if _sources() != source or _sha(optical_path) != optical_sha:
        raise RuntimeError("observation evaluation sources/checkpoint changed during preparation")
    _publish(output, lambda stream: np.save(stream, np.array(result, dtype=object), allow_pickle=True))
    value = {"schema": SCHEMA, "split": "dev", "roots": len(selected), "scenes": 106,
        "source_sha256": source, "cache_sha256": _sha(output), "event_cache_sha256": _sha(event_cache),
        "event_cache_manifest": metadata, "cohort_sha256": _sha(cohort),
        "optical_checkpoint_sha256": optical_sha, "counts": dict(counts),
        "worlds": 16, "seed": 31, "threads": 1, "elapsed_s": time.perf_counter() - started,
        "targets": list(CLASSES), "parity": "exact per-query feature, world-weight and unresolved-mass parity verified",
        "information_boundary": "forecast/no-detection law from legal inputs only; actual future legal observation payload opened afterward; no privileged future read",
        "coarse_event": "any visible valid human center at any native post-action endpoint over full planned interval, root excluded; contact absorbing and takes precedence",
        "appearance_proxy": "native event head3 remains separate and is never used for this generative observation prediction"}
    _json(metadata_path, value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def probability_bounds(coefficients, event, observation):
    """Coupled categorical support: (C, N+a, D+r-a), 0<=a<=r."""
    x = np.asarray(event["x"], float)
    hazards = expit(np.clip(x @ np.asarray(coefficients)[0], -30., 30.))
    hazards *= np.asarray(event["allowed"])[:, None, 0]
    survival = np.prod(1 - hazards, axis=1)
    weights = np.asarray(event["weights"], float)
    no_detection = np.asarray(observation["no_detection"], float)
    if no_detection.shape != weights.shape or not np.isfinite(no_detection).all() or np.any((no_detection < 0) | (no_detection > 1)):
        raise ValueError("invalid cached no-detection law")
    residual_survival = float(event["unresolved"] * math.exp(event["planned_steps"] * math.log1p(-RESIDUAL_HAZARD)))
    contact = float(weights @ (1 - survival)) + event["unresolved"] - residual_survival
    absent = float(weights @ (survival * no_detection))
    detected = float(weights @ (survival * (1 - no_detection)))
    if event["planned_steps"] == 0:
        # Empty endpoint sets contain no detections under every optical law.
        absent += residual_survival
        residual_survival = 0.
    base = np.array([contact, absent, detected])
    if not np.isfinite(base).all() or np.any(base < -1e-12) or abs(base.sum() + residual_survival - 1) > 1e-7:
        raise ValueError("categorical observation mass mismatch")
    base = np.maximum(base, 0.)
    upper = base + np.array([0., residual_survival, residual_survival])
    return base, upper, residual_survival


def categorical_scores(lower, upper, residual_survival, target):
    """Rigorous log-score and coupled three-class Brier extrema, sum convention."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if target not in (0, 1, 2):
        raise ValueError("observed categorical outcome required")
    y = np.eye(3)[target]
    r = float(residual_survival)
    def brier(allocation):
        p = lower + np.array([0., allocation, r - allocation])
        return float(np.square(p - y).sum())
    optimal = float(np.clip((y[1] - lower[1] + lower[2] + r - y[2]) / 2, 0., r))
    nll_lower = -math.log(float(upper[target])) if upper[target] > 0 else None
    nll_upper = -math.log(float(lower[target])) if lower[target] > 0 else None
    return {"categorical_nll_lower": nll_lower, "categorical_nll_upper": nll_upper,
        "categorical_nll_lower_infinite": nll_lower is None,
        "categorical_nll_upper_infinite": nll_upper is None,
        "categorical_brier_lower": brier(optimal),
        "categorical_brier_upper": max(brier(0.), brier(r))}


def scored_query(coefficients, event, observation):
    lower, upper, residual = probability_bounds(coefficients, event, observation)
    target = observation["target"]
    row = {key: observation[key] for key in ("root_id", "scene_id", "query_id", "planned_steps", "observed_steps", "administrative_censoring", "observed_any_detection")}
    row.update(target=target, endpoint_eligible=target is not None,
        unresolved_optical_survival=residual, probability_lower=lower.tolist(), probability_upper=upper.tolist())
    if target is not None:
        row.update(categorical_scores(lower, upper, residual, target))
    for index, name in enumerate(CLASSES):
        outcome = int(target == index) if target is not None else None
        row[name] = outcome
        row["pred_lower_" + name] = float(lower[index])
        row["pred_upper_" + name] = float(upper[index])
        row["bias_lower_" + name] = float(lower[index] - outcome) if outcome is not None else None
        row["bias_upper_" + name] = float(upper[index] - outcome) if outcome is not None else None
        if outcome is not None:
            row["brier_lower_" + name] = float((np.clip(outcome, lower[index], upper[index]) - outcome) ** 2)
            row["brier_upper_" + name] = float(max((lower[index] - outcome) ** 2, (upper[index] - outcome) ** 2))
    return row


def _class_summary(rows, name):
    prefixes = {"predicted_lower": "pred_lower_", "predicted_upper": "pred_upper_",
        "bias_lower": "bias_lower_", "bias_upper": "bias_upper_",
        "brier_lower": "brier_lower_", "brier_upper": "brier_upper_"}
    summary = {key: scene_ci(rows, prefix + name) for key, prefix in prefixes.items()}
    summary["event_rate"] = scene_ci(rows, name)
    if summary["event_rate"]["scenes"] < 30:
        status = "INCONCLUSIVE_SCENE_SUPPORT"
    elif summary["bias_upper"]["hi"] < 0:
        status = "CREDIBLE_UNDERPREDICTION_EVEN_AT_UPPER_PROBABILITY_BOUND"
    elif summary["bias_lower"]["lo"] > 0:
        status = "CREDIBLE_OVERPREDICTION_EVEN_AT_LOWER_PROBABILITY_BOUND"
    else:
        status = "BOUNDED_BIAS_INTERVAL_INCLUDES_ZERO_OR_CANNOT_RESOLVE_SIGN"
    bins = []
    for index, (lo, hi) in enumerate(zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:])):
        final = index == len(CALIBRATION_BINS) - 2
        selected = [row for row in rows if row["pred_lower_" + name] >= lo
                    and (row["pred_lower_" + name] <= hi if final else row["pred_lower_" + name] < hi)]
        bins.append({"lower": lo, "upper": hi, "queries": len(selected),
            "observed": scene_ci(selected, name),
            "predicted_lower": scene_ci(selected, "pred_lower_" + name),
            "predicted_upper": scene_ci(selected, "pred_upper_" + name),
            "bias_lower": scene_ci(selected, "bias_lower_" + name),
            "bias_upper": scene_ci(selected, "bias_upper_" + name),
            "probability_interval_crosses_bin_upper_queries": sum(row["pred_upper_" + name] > hi for row in selected)})
    if sum(row["queries"] for row in bins) != len(rows):
        raise ValueError("probability-bound calibration bins dropped eligible queries")
    return {"summary": summary, "bias_status": status, "calibration_bins": bins,
        "bin_definition": "stratified by probability LOWER bound; every eligible query assigned once; predictions remain intervals and are not point-calibration claims"}


def report(observation_cache, event_cache, checkpoints, output):
    output = _destination(output)
    source = _sources()
    metadata = json.loads(Path(str(observation_cache) + ".manifest.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("split") != "dev":
        raise ValueError("only the fixed development observation cache is permitted")
    if _sha(observation_cache) != metadata["cache_sha256"] or _sha(event_cache) != metadata["event_cache_sha256"]:
        raise ValueError("observation/native feature-cache binding mismatch")
    observations = np.load(observation_cache, allow_pickle=True).tolist()
    events, event_metadata = load_cache(event_cache, "dev")
    lookup = {(row["root_id"], row["query_id"]): row for row in events}
    observation_keys = [(row["root_id"], row["query_id"]) for row in observations]
    if len(observation_keys) != len(set(observation_keys)) or set(observation_keys) != set(lookup):
        raise ValueError("observation/native cohort query mismatch")
    if len({row["root_id"] for row in observations}) != 632:
        raise ValueError("full632-root observation cohort required")
    checkpoints = json.loads(Path(checkpoints).read_text()) if isinstance(checkpoints, (str, Path)) else checkpoints
    if not checkpoints:
        raise ValueError("named frozen event checkpoints required")
    results, hashes = {}, {}
    for name, path in checkpoints.items():
        coefficients = np.asarray(json.loads(Path(path).read_text())["coefficients"], float)
        if coefficients.shape != (3, 12) or not np.isfinite(coefficients).all():
            raise ValueError("invalid native event coefficients")
        hashes[name] = _sha(path)
        scored = [scored_query(coefficients, lookup[(row["root_id"], row["query_id"])], row) for row in observations]
        eligible = [row for row in scored if row["endpoint_eligible"]]
        results[name] = {"eligible_queries": len(eligible), "excluded_administrative_queries": len(scored) - len(eligible),
            "categorical_nll_lower": _infinite_aware_summary(eligible, "categorical_nll_lower", "categorical_nll_lower_infinite"),
            "categorical_nll_upper": _infinite_aware_summary(eligible, "categorical_nll_upper", "categorical_nll_upper_infinite"),
            "categorical_brier_lower": scene_ci(eligible, "categorical_brier_lower"),
            "categorical_brier_upper": scene_ci(eligible, "categorical_brier_upper"),
            "unresolved_optical_survival": scene_ci(eligible, "unresolved_optical_survival"),
            "classes": {label: _class_summary(eligible, label) for label in CLASSES}, "query_rows": scored}
    if _sources() != source:
        raise RuntimeError("observation reporting sources changed during evaluation")
    value = {"schema": SCHEMA, "split": "dev", "source_sha256": source,
        "observation_cache_manifest": metadata, "event_cache_manifest_sha256": _sha(str(event_cache) + ".manifest.json"),
        "checkpoints_sha256": hashes, "models": results,
        "protocol": {"targets": list(CLASSES), "native_contact": "all contact queries retained; contact takes precedence over any earlier detections",
            "censoring": "exclude only explicitly administratively truncated queries from categorical endpoint scoring",
            "uncertainty": "same equal-scene root means and1000 scene bootstrap draws seed73; siblings are not independent",
            "scores": "categorical NLL and sum-of-three-class Brier bounded over the coupled residual-survival no/any-detection allocation",
            "residual_optics": "unresolved survival has no declared optical law; no point empty probability is fabricated; zero endpoints are logically no detections",
            "calibration_bins": "probability-lower-bound bins, retaining full prediction intervals and every eligible query",
            "limitations": "development diagnostics after train optical fitting/dev native calibration; no fresh confirmation and no model selection on old holdout",
            "appearance_proxy": "legacy untracked-appearance head is not used for this generative observation test"}}
    _json(output, value)
    return value


def selftest():
    from .observation import EndpointObservationLaw
    from types import SimpleNamespace

    class ObservationEvaluationTests(unittest.TestCase):
        def test_runtime_tuple_and_saved_list_configuration_parity(self):
            self.assertTrue(_same_configuration({"probability": (.2, .8)}, {"probability": [.2, .8]}))
            self.assertFalse(_same_configuration({"probability": (.2, .8)}, {"probability": [.21, .79]}))

        def test_two_endpoint_no_detection_excludes_root(self):
            class KnownLaw(EndpointObservationLaw):
                def detection(self, positions, robot_pose, regime):
                    return np.broadcast_to(np.array([.2, .4])[None, :, None], positions.shape[:-1])
            future = SimpleNamespace(worlds=[SimpleNamespace(visibility_mode=0)], positions=np.zeros((1, 3, 1, 2)))
            self.assertAlmostEqual(KnownLaw().no_detection(future, np.arange(3), np.zeros((3, 4)))[0], .8 * .6)

        def test_contact_overrides_detections_and_censor_is_not_negative(self):
            observation = {"tracks": [{"visible_now": True}]}
            event = {"observed_steps": 1, "planned_steps": 2, "first": np.array([0, -1, -1]), "administrative_censoring": False}
            self.assertEqual(native_class(event, [observation])["target"], 0)
            event.update(first=np.array([-1, -1, -1]), administrative_censoring=True)
            self.assertIsNone(native_class(event, [observation])["target"])
            event.update(planned_steps=1, administrative_censoring=False)
            self.assertEqual(native_class(event, [observation])["target"], 2)
            self.assertEqual(native_class(event, [{"tracks": []}])["target"], 1)

        def test_coupled_brier_bounds_match_all_residual_allocations(self):
            lower = np.array([.2, .1, .3]);r = .4;upper = lower + [0., r, r]
            for target in range(3):
                result = categorical_scores(lower, upper, r, target)
                values = [np.square(lower + [0., a, r-a] - np.eye(3)[target]).sum() for a in np.linspace(0, r, 1001)]
                self.assertLessEqual(result["categorical_brier_lower"], min(values) + 1e-12)
                self.assertAlmostEqual(result["categorical_brier_upper"], max(values))
                self.assertAlmostEqual(result["categorical_nll_lower"], -math.log(upper[target]))
                self.assertAlmostEqual(result["categorical_nll_upper"], -math.log(lower[target]))

        def test_common_world_contact_and_visibility_are_not_independent_marginals(self):
            x = np.zeros((2, 2, 12));x[..., 0] = 1.;x[1, :, 1] = 1.
            coefficients = np.zeros((3, 12));coefficients[0, 0] = math.log(.1/.9);coefficients[0, 1] = 2 * math.log(9.)
            event = {"x": x, "weights": np.array([.4, .4]), "allowed": np.ones((2, 3)), "unresolved": .2, "planned_steps": 2}
            lower, upper, residual = probability_bounds(coefficients, event, {"no_detection": np.array([1., 0.])})
            self.assertAlmostEqual(lower[1], .4 * .9**2)
            self.assertAlmostEqual(lower[2], .4 * .1**2)
            self.assertAlmostEqual(residual, .2 * .995**2)
            self.assertAlmostEqual(lower.sum() + residual, 1.)
            self.assertAlmostEqual(upper[0], lower[0])

        def test_unresolved_optics_can_make_nll_upper_unbounded(self):
            scores = categorical_scores([.2, 0., .3], [.2, .5, .8], .5, 1)
            self.assertTrue(scores["categorical_nll_upper_infinite"])
            self.assertIsNone(scores["categorical_nll_upper"])
            self.assertAlmostEqual(scores["categorical_nll_lower"], math.log(2.))

        def test_empty_endpoint_interval_is_logically_no_detections(self):
            event = {"x": np.empty((1, 0, 12)), "weights": np.array([.7]), "allowed": np.ones((1, 3)), "unresolved": .3, "planned_steps": 0}
            lower, upper, residual = probability_bounds(np.zeros((3, 12)), event, {"no_detection": np.ones(1)})
            np.testing.assert_allclose(lower, [0., 1., 0.])
            np.testing.assert_array_equal(lower, upper)
            self.assertEqual(residual, 0.)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ObservationEvaluationTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    for name in ("snapshot", "event_cache", "checkpoint", "population", "cohort", "output"):
        command.add_argument(name)
    command = commands.add_parser("report")
    for name in ("observation_cache", "event_cache", "checkpoints", "output"):
        command.add_argument(name)
    commands.add_parser("selftest")
    args = vars(parser.parse_args());command = args.pop("command")
    result = {"prepare": prepare, "report": report, "selftest": selftest}[command](**args)
    if result is not None and command == "report":
        print(json.dumps({"models": list(result["models"]), "output": args["output"]}), flush=True)


if __name__ == "__main__":
    main()
