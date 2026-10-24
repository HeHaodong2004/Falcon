"""Native event and fixed candidate-decision audit on one repaired future cache.

All compared coefficient sets receive identical cached legal-root futures and
full planned intervals. Administrative censoring contributes censored likelihood
but no endpoint binary or candidate-utility negative. This controls inference
features; historical coefficients were trained on different features, so the
comparison does NOT isolate stopping-time calibration alone.
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

from .queries import FEATURE_NAMES
from .repair_events import (EVENT_NAMES, RESIDUAL_HAZARD, batch, likelihood,
                            load_cache, probabilities, _json, _output_path, _sha, _sources)
from .repair_evaluation import scene_ci


SCHEMA = "partial_graph_native_risk_repair_report_1"
LOW_RISK_THRESHOLDS = (.05, .10)
CALIBRATION_BINS = (0., .05, .10, .20, .40, .60, .80, 1.)
UTILITY = {"progress_m": 1., "collision": -5., "psc": -1., "duration_s": -.1}


def _sources_for_report():
    return {**_sources(), "repair_risk.py": _sha(__file__),
            "repair_evaluation.py": _sha(Path(__file__).with_name("repair_evaluation.py"))}


def duration_progress(coefficients, record):
    """Native post-action contact endpoint or planned non-contact endpoint."""
    x = np.asarray(record["x"], float)
    collision = expit(np.clip(x @ np.asarray(coefficients)[0], -30., 30.))
    collision *= np.asarray(record["allowed"])[:, None, 0]
    before = np.c_[np.ones(len(x)), np.cumprod(1 - collision, axis=1)[:, :-1]]
    first = np.asarray(record["weights"]) @ (before * collision)
    n = int(record["planned_steps"])
    first += record["unresolved"] * (1 - RESIDUAL_HAZARD) ** np.arange(n) * RESIDUAL_HAZARD
    no_contact = max(0., 1 - float(first.sum()))
    elapsed = np.asarray(record["planned_elapsed"], float)
    progress = np.asarray(record["planned_cumulative_progress"], float)
    if elapsed.shape != (n + 1,) or progress.shape != (n + 1,):
        raise ValueError("decision reporting requires full legal planned duration/progress")
    return {"predicted_duration": float(first @ elapsed[1:] + no_contact * elapsed[-1]),
            "predicted_progress": float(first @ progress[1:] + no_contact * progress[-1]),
            "first_collision_mass": first.tolist(), "noncontact_mass": no_contact}


def _logscore(probability, outcome):
    mass = float(probability if outcome else 1 - probability)
    return float(-math.log(mass)) if mass > 0 else None


def scored_query(coefficients, record):
    p = probabilities(coefficients, record)
    if not np.isfinite(p).all() or np.any((p < -1e-8) | (p > 1 + 1e-8)):
        raise ValueError("invalid native event probability")
    p = np.clip(p, 0., 1.)
    y = np.asarray(record["first"]) >= 0
    eligible = bool(np.asarray(record["calibration_eligible"]).all())
    if not np.asarray(record["calibration_eligible"]).all() and np.asarray(record["calibration_eligible"]).any():
        raise ValueError("endpoint reporting requires one shared complete-followup cohort")
    c = torch.tensor(coefficients, dtype=torch.float32)
    with torch.no_grad():
        nll = float(likelihood(c, batch([record])))
    if math.isnan(nll):
        raise FloatingPointError("NaN first-event likelihood is an implementation failure")
    row = {"root_id": record["root_id"], "scene_id": record["scene_id"],
        "query_id": record["query_id"], "endpoint_eligible": eligible,
        "administrative_censoring": bool(record["administrative_censoring"]),
        "planned_steps": int(record["planned_steps"]), "observed_steps": int(record["observed_steps"]),
        "termination": record["termination"], "joint_censored_nll": nll if np.isfinite(nll) else None,
        "original_candidate_queries": int(record["original_candidate_queries"]),
        "joint_censored_nll_infinite": not np.isfinite(nll),
        "unresolved_mass": float(record["unresolved"])}
    for k, name in enumerate(EVENT_NAMES):
        row[name] = int(y[k])
        row["pred_" + name] = float(p[k])
        row["brier_" + name] = float((p[k] - y[k]) ** 2) if eligible else None
        row["bias_" + name] = float(p[k] - y[k]) if eligible else None
        row["binary_nll_" + name] = _logscore(p[k], y[k]) if eligible else None
        row["binary_nll_infinite_" + name] = eligible and row["binary_nll_" + name] is None
    motion = duration_progress(coefficients, record)
    row.update(motion)
    row["actual_duration"] = float(record["actual_observed_seconds"])
    row["actual_progress"] = float(record["actual_progress"])
    row["duration_absolute_error"] = abs(row["predicted_duration"] - row["actual_duration"]) if eligible else None
    row["progress_absolute_error"] = abs(row["predicted_progress"] - row["actual_progress"]) if eligible else None
    row["actual_utility"] = (row["actual_progress"] - 5 * y[0] - y[1] - .1 * row["actual_duration"]) if eligible else None
    row["predicted_utility"] = row["predicted_progress"] - 5 * p[0] - p[1] - .1 * row["predicted_duration"]
    n = int(record["planned_steps"])
    residual_collision = float(np.sum((1 - RESIDUAL_HAZARD) ** np.arange(n) * RESIDUAL_HAZARD))
    residual_psc = float(np.sum((1 - RESIDUAL_HAZARD) ** (2 * np.arange(n)) * RESIDUAL_HAZARD))
    collision_upper = min(1., p[0] + record["unresolved"] * (1 - residual_collision))
    psc_upper = min(1., p[1] + record["unresolved"] * (1 - residual_psc))
    row["conservative_utility"] = row["predicted_progress"] - 5 * collision_upper - psc_upper - .1 * row["predicted_duration"]
    row["depth0_utility"] = float(record["planned_cumulative_progress"][-1] - .1 * record["planned_elapsed"][-1])
    return row


def within_root_auc(rows, event):
    groups = collections.defaultdict(list)
    for row in rows:
        if row["endpoint_eligible"]:
            groups[row["root_id"]].append(row)
    scores = []
    for root, group in groups.items():
        positive = np.array([r["pred_" + event] for r in group if r[event]])
        negative = np.array([r["pred_" + event] for r in group if not r[event]])
        if not len(positive) or not len(negative):
            continue
        difference = positive[:, None] - negative[None]
        auc = np.mean((difference > 0) + .5 * (difference == 0))
        scores.append({"root_id": root, "scene_id": group[0]["scene_id"], "auc": float(auc),
                       "positive_queries": len(positive), "negative_queries": len(negative)})
    return {"summary": scene_ci(scores, "auc"), "root_rows": scores,
            "interpretation": "paired sibling discrimination; only roots with both native outcomes are informative"}


def _calibration(rows, event, lower, upper, include_upper=False):
    selected = [r for r in rows if r["endpoint_eligible"] and r["pred_" + event] >= lower
                and (r["pred_" + event] <= upper if include_upper else r["pred_" + event] < upper)]
    return {"lower": float(lower), "upper": float(upper), "include_upper": include_upper,
        "queries": len(selected), "predicted": scene_ci(selected, "pred_" + event),
        "observed": scene_ci(selected, event), "bias": scene_ci(selected, "bias_" + event),
        "brier": scene_ci(selected, "brier_" + event)}


def _infinite_aware_summary(rows, key, infinite_key):
    result = scene_ci(rows, key)
    bad = [r for r in rows if r[infinite_key]]
    result["infinite_queries"] = len(bad)
    result["infinite_roots"] = len({r["root_id"] for r in bad})
    if bad:
        result["finite_only_mean"] = result["mean"]
        result["finite_only_ci95"] = [result["lo"], result["hi"]]
        for field in ("mean", "lo", "hi", "one_sided_upper95"):
            result[field] = None
        result["status"] = "IMPOSSIBLE_OBSERVATIONS_UNDER_DECLARED_LAW"
    return result


def event_summary(rows, event):
    eligible = [r for r in rows if r["endpoint_eligible"]]
    bias = scene_ci(eligible, "bias_" + event)
    if bias["scenes"] < 30:
        status = "INCONCLUSIVE_SCENE_SUPPORT"
    elif bias["hi"] < 0:
        status = "CREDIBLE_UNDERPREDICTION"
    elif bias["lo"] > 0:
        status = "CREDIBLE_OVERPREDICTION"
    else:
        status = "BIAS_INTERVAL_INCLUDES_ZERO"
    low_risk = {}
    for threshold in LOW_RISK_THRESHOLDS:
        low = [r for r in eligible if r["pred_" + event] <= threshold]
        positives = [{**r, "low_predicted_risk": float(r["pred_" + event] <= threshold)}
                     for r in eligible if r[event]]
        low_risk[str(threshold)] = {"queries": len(low),
            "actual_event_rate_among_low_predictions": scene_ci(low, event),
            "fraction_of_positive_queries_given_low_risk": scene_ci(positives, "low_predicted_risk"),
            "positive_queries": len(positives)}
    bins = [_calibration(rows, event, lo, hi, i == len(CALIBRATION_BINS) - 2)
            for i, (lo, hi) in enumerate(zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]))]
    p = [r["pred_" + event] for r in eligible]
    boundaries = np.quantile(p, np.arange(11) / 10) if p else []
    deciles = [_calibration(rows, event, lo, hi, i == 9)
               for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    return {"event": event, "eligible_queries": len(eligible),
        "excluded_administrative_queries": len(rows) - len(eligible),
        "positive_queries": sum(r[event] for r in eligible),
        "brier": scene_ci(eligible, "brier_" + event),
        "bias": bias, "bias_status": status,
        "mean_predicted": scene_ci(eligible, "pred_" + event),
        "event_rate": scene_ci(eligible, event),
        "binary_log_score": _infinite_aware_summary(eligible, "binary_nll_" + event, "binary_nll_infinite_" + event),
        "within_root_auc": within_root_auc(rows, event), "low_predicted_risk": low_risk,
        "calibration_bins": bins, "predicted_risk_deciles": deciles,
        "bin_estimand": "equal-scene means of root conditional-bin means; decile boundaries are descriptive query quantiles, ties can make bins unequal"}


def decision_summary(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["root_id"]].append(row)
    decisions = []
    censored_roots = 0
    for root, group in groups.items():
        if any(r["original_candidate_queries"] != len(group) for r in group):
            raise ValueError("candidate set missing queries; cannot assess hindsight regret")
        if not all(r["endpoint_eligible"] for r in group):
            censored_roots += 1
            continue
        oracle = max(group, key=lambda r: (r["actual_utility"], r["query_id"]))
        for rule, key in (("native_expected_utility", "predicted_utility"),
                          ("residual_upper_risk", "conservative_utility"),
                          ("depth0_planned_progress", "depth0_utility")):
            chosen = max(group, key=lambda r: (r[key], r["query_id"]))
            decisions.append({"root_id": root, "scene_id": group[0]["scene_id"], "rule": rule,
                "candidate_queries": len(group), "chosen_query_id": chosen["query_id"],
                "oracle_query_id": oracle["query_id"], "actual_utility": chosen["actual_utility"],
                "oracle_utility": oracle["actual_utility"],
                "regret": oracle["actual_utility"] - chosen["actual_utility"],
                "selected_collision": chosen["collision"], "selected_psc": chosen["psc"],
                "selected_progress": chosen["actual_progress"], "selected_duration": chosen["actual_duration"]})
    summary = {}
    for rule in ("native_expected_utility", "residual_upper_risk", "depth0_planned_progress"):
        selected = [r for r in decisions if r["rule"] == rule]
        summary[rule] = {key: scene_ci(selected, key) for key in
            ("actual_utility", "oracle_utility", "regret", "selected_collision", "selected_psc",
             "selected_progress", "selected_duration")}
    eligible = [r for r in rows if r["endpoint_eligible"]]
    return {"utility_coefficients": UTILITY, "excluded_roots_with_administrative_censored_sibling": censored_roots,
        "complete_candidate_roots": len(groups) - censored_roots,
        "rules": summary, "root_rows": decisions,
        "duration_absolute_error": scene_ci(eligible, "duration_absolute_error"),
        "progress_absolute_error": scene_ci(eligible, "progress_absolute_error"),
        "scope": "observed curriculum candidate sets, planned<=3s endpoint consequences; progress is traveled reference-route arclength, not goal-distance reduction; hindsight oracle is diagnostic only, no closed-loop navigation claim"}


def report(cache, checkpoints, output, *, evaluation_kind="development"):
    torch.set_num_threads(1)
    if evaluation_kind != "development":
        raise ValueError("fresh confirmation requires a separately frozen cache protocol; legacy holdout is prohibited")
    output = _output_path(output)
    source = _sources_for_report()
    records, metadata = load_cache(cache, "dev")
    checkpoints = json.loads(Path(checkpoints).read_text()) if isinstance(checkpoints, (str, Path)) else checkpoints
    if not records or not checkpoints:
        raise ValueError("nonempty development cache and named coefficient sets required")
    for record in records:
        for required in ("planned_elapsed", "planned_cumulative_progress", "actual_progress",
                         "original_candidate_queries", "actual_observed_seconds"):
            if required not in record:
                raise ValueError("event cache predates decision diagnostics: " + required)
    results = {}
    hashes = {}
    started = time.perf_counter()
    for name, path in checkpoints.items():
        coefficients = np.asarray(json.loads(Path(path).read_text())["coefficients"], float)
        if coefficients.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(coefficients).all():
            raise ValueError("invalid named coefficient checkpoint")
        hashes[name] = _sha(path)
        scored = [scored_query(coefficients, record) for record in records]
        results[name] = {"events": {event: event_summary(scored, event) for event in EVENT_NAMES},
            "joint_censored_first_event_nll": _infinite_aware_summary(scored, "joint_censored_nll", "joint_censored_nll_infinite"),
            "decision": decision_summary(scored), "query_rows": scored}
        print(json.dumps({"checkpoint": name, "queries": len(scored),
                          "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    if _sources_for_report() != source:
        raise RuntimeError("risk-report sources changed during evaluation; result not published")
    result = {"schema": SCHEMA, "evaluation_kind": evaluation_kind,
        "source_sha256": source, "cache_manifest": metadata,
        "cache_sha256": _sha(cache), "coefficient_checkpoints": hashes,
        "queries": len(records), "roots": len({r["root_id"] for r in records}),
        "scenes": len({r["scene_id"] for r in records}), "threads": 1,
        "elapsed_s": time.perf_counter() - started,
        "protocol": {"probabilities": "full legal planned horizon; PSC/proxy appearance stop after absorbing contact and include terminal contact endpoint",
            "likelihood": "joint three-head first-event mixture likelihood over observed risk sets, with explicit right censoring and structural zeros",
            "censoring": "all administrative truncations excluded from endpoint metrics, including their known positives; retained in observed-prefix likelihood",
            "independence": "query means within each root, root means within each scene, scene bootstrap1000 seed73",
            "comparison": "identical repaired cached futures for all coefficients; isolates coefficient/calibration changes at evaluation, not every original stopping-time/training-feature difference",
            "calibration": "development fitted/checkpoint-selected results are descriptive and need fresh confirmation",
            "danger_thresholds": list(LOW_RISK_THRESHOLDS), "fixed_calibration_bins": list(CALIBRATION_BINS),
            "appearance": metadata["appearance_target"], "utility": UTILITY,
            "frozen_holdout": "no historical or protected holdout payload was read"},
        "models": results}
    _json(output, result)
    return result


def selftest():
    class RiskTests(unittest.TestCase):
        def test_post_action_duration_and_progress(self):
            x = np.zeros((1, 3, len(FEATURE_NAMES)))
            x[..., 0] = 1.
            c = np.zeros((3, len(FEATURE_NAMES)))
            c[0, 0] = math.log(.2 / .8)
            r = {"x": x, "weights": [1.], "allowed": np.ones((1, 3)), "unresolved": 0.,
                 "planned_steps": 3, "planned_elapsed": [0., 1., 2., 3.],
                 "planned_cumulative_progress": [0., 2., 4., 6.]}
            score = duration_progress(c, r)
            self.assertAlmostEqual(score["predicted_duration"], .2 + .16 * 2 + (.128 + .512) * 3)
            self.assertAlmostEqual(score["predicted_progress"], 2 * score["predicted_duration"])

        def test_auc_requires_mixed_native_outcomes_and_respects_ties(self):
            rows = [{"root_id": "a", "scene_id": "s", "endpoint_eligible": True,
                     "collision": y, "pred_collision": .5} for y in (0, 1)]
            rows.append({"root_id": "b", "scene_id": "t", "endpoint_eligible": True,
                         "collision": 1, "pred_collision": .9})
            result = within_root_auc(rows, "collision")
            self.assertEqual(result["summary"]["mean"], .5)
            self.assertEqual(result["summary"]["roots"], 1)

        def test_administrative_positive_is_excluded_from_binary_bin(self):
            rows = [{"root_id": "a", "scene_id": "s", "endpoint_eligible": False,
                     "collision": 1, "pred_collision": .5, "bias_collision": None, "brier_collision": None}]
            self.assertEqual(_calibration(rows, "collision", 0, 1, True)["queries"], 0)

        def test_zero_probability_positive_is_an_infinite_score(self):
            self.assertIsNone(_logscore(0., 1))
            self.assertEqual(_logscore(0., 0), 0.)

        def test_zero_action_candidate_has_zero_risk_duration_and_progress(self):
            r = {"x": np.empty((1, 0, len(FEATURE_NAMES))), "weights": [1.],
                 "allowed": np.ones((1, 3)), "unresolved": 0., "planned_steps": 0,
                 "planned_elapsed": [0.], "planned_cumulative_progress": [0.]}
            c = np.zeros((3, len(FEATURE_NAMES)))
            np.testing.assert_array_equal(probabilities(c, r), np.zeros(3))
            s = duration_progress(c, r)
            self.assertEqual(s["predicted_duration"], 0.)
            self.assertEqual(s["predicted_progress"], 0.)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RiskTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("report")
    for name in ("cache", "checkpoints", "output"):
        execute.add_argument(name)
    execute.add_argument("--evaluation-kind", default="development")
    commands.add_parser("selftest")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"report": report, "selftest": selftest}[command](**args)


if __name__ == "__main__":
    main()
