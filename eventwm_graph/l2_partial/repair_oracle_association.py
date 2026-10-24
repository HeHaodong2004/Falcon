"""Offline-only historical association oracle for the bounded repair study.

Physical identities are recovered from exact legal XYZ detections and the
ordered native Euclidean ranges in frozen privileged prefix labels. Ties or
missing labels are not guessed. Initial t=0 positions use ONLY each person's
initial waypoint, never their future route or a complete map. The oracle is
diagnostic supervision: it is not a deployable filter or an online model input.

The forced filters share the original sequential population/association prior.
Only the choice of identity is privileged. Their old Gaussian versus repaired
maneuver conditional innovation densities isolate process misspecification
from identity fragmentation. They do not evaluate the repaired set-emission law.
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
import torch

from .data import read, root_data
from .model import GraphBeliefWorldModel
from .repair_model import RepairedGraphBeliefWorldModel
from .state import BeliefState, Hypothesis, ObservedGraph
from .repair_events import _cohort_rows, _json, _output_path, _sha, _sources
from .repair_evaluation import gospa, population_score


SCHEMA = "partial_graph_offline_association_oracle_1"


def match_ranges(detection_positions, robot_position, native_ranges, tolerance=1e-5):
    """Unambiguous exact native range matches, with one-to-one conflict rejection."""
    positions = np.asarray(detection_positions, float).reshape(-1, 3)
    robot = np.asarray(robot_position, float)[:3]
    ranges = np.asarray(native_ranges, float)
    measured = np.linalg.norm(positions - robot, axis=-1)
    candidates = [np.flatnonzero(abs(ranges - d) <= tolerance).tolist() for d in measured]
    matched = [ids[0] if len(ids) == 1 else None for ids in candidates]
    counts = collections.Counter(i for i in matched if i is not None)
    statuses = []
    for j, identity in enumerate(matched):
        if len(candidates[j]) == 0:
            statuses.append("no_exact_range_match")
        elif len(candidates[j]) > 1:
            statuses.append("ambiguous_equal_ranges")
        elif counts[identity] > 1:
            matched[j] = None
            statuses.append("conflicting_one_to_one_match")
        else:
            statuses.append("exact_range_match")
    return matched, statuses, candidates


def identify_history(data, privileged):
    """Read only prefix labels and initial physical positions, not scene geometry."""
    labels = privileged["prefix_labels"]
    info = privileged["episode"]["info"]
    count = int(data["truth"].shape[1])
    initial = np.asarray([info[f"human_{i}_waypoint_0_position"] for i in range(count)], float).reshape(-1, 3)
    result = []
    counts = collections.Counter()
    for row in data["legal"]["history"]:
        timestamp = float(row["timestamp"])
        detections = sorted([r for r in row["tracks"] if r["visible_now"] and r.get("valid", True)],
                            key=lambda r: tuple(r["position"]))
        positions = np.asarray([r["position"] for r in detections], float).reshape(-1, 3)
        if abs(timestamp) <= 1e-8:
            distance = np.linalg.norm(positions[:, None] - initial[None], axis=-1)
            candidates = [np.flatnonzero(d <= 1e-5).tolist() for d in distance]
            identities = [c[0] if len(c) == 1 else None for c in candidates]
            used = collections.Counter(i for i in identities if i is not None)
            statuses = []
            for j, identity in enumerate(identities):
                status = "exact_initial_position" if identity is not None else "ambiguous_or_missing_initial_position"
                if identity is not None and used[identity] > 1:
                    identities[j] = None
                    status = "conflicting_one_to_one_match"
                statuses.append(status)
        else:
            matching = [label for label in labels if abs(float(label["timestamp"]) - timestamp) <= 1e-7]
            if len(matching) == 1:
                identities, statuses, candidates = match_ranges(positions, row["robot_pose"],
                    matching[0]["euclidean_center_distances_m"])
            else:
                identities = [None] * len(detections)
                statuses = ["missing_or_ambiguous_prefix_timestamp"] * len(detections)
                candidates = [[] for _ in detections]
        for status in statuses:
            counts[status] += 1
        result.append({"timestamp": timestamp, "detections": detections,
            "physical_ids": identities, "statuses": statuses, "candidate_ids": candidates})
    # Verify the index convention against the root future tape at the ONE time
    # shared by both stores. No matching to a later target is performed.
    for detection, identity in zip(result[-1]["detections"], result[-1]["physical_ids"]):
        if identity is not None:
            distance = np.linalg.norm(np.asarray(detection["position"])[[0, 2]] - data["truth"][0, identity])
            if distance > 1e-5:
                raise ValueError("prefix physical indices disagree with frozen future tape")
    return result, dict(counts)


def _initial(model, data, population):
    legal = data["legal"]
    g = legal["partial_graph"]
    graph = ObservedGraph("offline_oracle_legal_empty", 0, float(g["resolution"]), float(g["height"]), {})
    h = Hypothesis(1., np.empty((0, 4)), np.empty((0, 4, 4)), np.empty(0),
        np.empty(0, int), np.asarray(population).copy())
    model._initial_legal_pose = np.asarray(legal["history"][0]["robot_pose"], float)
    return BeliefState(graph, model.initial_hypotheses(h),
        float(legal["history"][0]["timestamp"]), np.asarray(legal["history"][0]["robot_pose"], float))


def forced_history(model, data, identity_rows, population):
    """Oracle clamping with the released sequential prior, without true count input."""
    belief = _initial(model, data, population)
    if len(belief.hypotheses) != 1:
        raise ValueError("oracle dynamics comparison requires one history-filter hypothesis")
    innovations = []
    trajectory = []
    unsupported = []
    for row, identities in zip(data["legal"]["history"], identity_rows):
        t = float(row["timestamp"])
        belief = model.propagate(belief, t)
        h = belief.hypotheses[0]
        used = set()
        for detection, physical, status in zip(identities["detections"], identities["physical_ids"], identities["statuses"]):
            if physical is None:
                unsupported.append({"timestamp": t, "status": status})
                continue
            z = np.asarray(detection["position"])[[0, 2]]
            available = h.n - len(used)
            counts = np.arange(len(h.hidden_pmf))
            denominator = available + counts
            inv = np.divide(1., denominator, out=np.zeros_like(denominator, float), where=denominator > 0)
            track_factor = float(h.hidden_pmf @ inv)
            transfer = h.hidden_pmf * counts * inv
            birth_factor = float(transfer.sum())
            matches = np.flatnonzero(h.slot_ids == physical)
            if len(matches):
                i = int(matches[0])
                if i in used:
                    raise ValueError("oracle identity reused within a legal detection set")
                conditioned, log_density = model.condition_track(h, i, z)
                if conditioned is None or not np.isfinite(log_density):
                    unsupported.append({"timestamp": t, "physical_id": physical,
                        "status": "known_identity_outside_dynamics_support"})
                    continue
                delta = z - h.mean[i, :2]
                covariance = h.covariance[i, :2, :2]
                chi2 = float(delta @ np.linalg.pinv(covariance) @ delta)
                distance = float(np.linalg.norm(z - np.asarray(row["robot_pose"])[[0, 2]]))
                broad_birth_density = -math.log(2 * math.pi * 100.) - distance ** 2 / 200.
                log_birth_odds = math.log(birth_factor) + broad_birth_density - math.log(track_factor) - log_density if birth_factor > 0 else None
                innovations.append({"timestamp": t, "physical_id": int(physical),
                    "age_seconds": float(t - h.last_seen[i]),
                    "innovation_distance_m": float(np.linalg.norm(delta)), "gaussian_moment_chi2": chi2,
                    "known_identity_log_density": float(log_density),
                    "birth_vs_correct_track_log_odds": log_birth_odds,
                    "false_birth_favored": bool(log_birth_odds is not None and log_birth_odds > 0),
                    "track_prior_factor": track_factor, "birth_prior_factor": birth_factor})
                mean = h.mean.copy()
                covariance = h.covariance.copy()
                mean[i], covariance[i] = conditioned
                seen = h.last_seen.copy()
                seen[i] = t
                pmf = h.hidden_pmf * inv
                pmf /= pmf.sum()
                height = h.height.copy() if h.height is not None else np.zeros(h.n)
                height[i] = detection["position"][1]
                h = h.fork(mean=mean, covariance=covariance, last_seen=seen,
                           hidden_pmf=pmf, height=height)
                used.add(i)
            else:
                if birth_factor <= 0:
                    unsupported.append({"timestamp": t, "physical_id": physical,
                        "status": "new_identity_outside_population_support"})
                    continue
                pmf = np.zeros_like(transfer)
                pmf[:-1] = transfer[1:] / birth_factor
                n = h.n
                height = h.height if h.height is not None else np.zeros(n)
                birth_time = h.birth_time if h.birth_time is not None else h.last_seen
                birth_position = h.birth_position if h.birth_position is not None else h.mean[:, :2]
                h = h.fork(mean=np.concatenate([h.mean, [[*z, 0., 0.]]]),
                    covariance=np.concatenate([h.covariance, np.diag([0., 0., 1., 1.])[None]]),
                    last_seen=np.r_[h.last_seen, t], slot_ids=np.r_[h.slot_ids, int(physical)],
                    hidden_pmf=pmf, height=np.r_[height, detection["position"][1]],
                    intent=np.concatenate([h.intent, np.zeros((1, 2))]), diffusion=np.r_[h.diffusion, .25],
                    mode=np.r_[h.mode, -1], birth_time=np.r_[birth_time, t],
                    birth_position=np.concatenate([birth_position, z[None]]))
                used.add(n)
        # Match the association-only filter's declared local moment compression.
        h = h.fork(maneuver_mean=None, maneuver_covariance=None, maneuver_probability=None)
        belief = belief.fork(hypotheses=[h], timestamp=t, robot_pose=np.asarray(row["robot_pose"]),
                            graph=belief.graph.update(row.get("static_updates", []))).validate()
        trajectory.append({"timestamp": t, "localized": h.n,
            "expected_hidden": h.expected_hidden, "expected_total": h.n + h.expected_hidden,
            "physical_slot_ids": h.slot_ids.tolist()})
    return {"innovations": innovations, "trajectory": trajectory, "unsupported": unsupported,
            "root_population": population_score(belief, data["truth"].shape[1]),
            "root_gospa": gospa(belief.hypotheses[0].mean[:, :2], data["truth"][0])}


def anonymous_history(model, data, population):
    belief = _initial(model, data, population)
    filter_model = model.history_filter()
    trajectory = []
    for row in data["legal"]["history"]:
        belief = filter_model.observe(belief, row)
        score = population_score(belief, data["truth"].shape[1])
        trajectory.append({"timestamp": float(row["timestamp"]),
            "conditional_expected_count": score["conditional_expected_count"],
            "conditional_localized_count": score["conditional_localized_count"],
            "conditional_hidden_count": score["conditional_hidden_count"],
            "unresolved_mass": belief.unresolved_mass,
            "true_count_probability_lower": score["true_count_probability_lower"]})
    return trajectory


def run(snapshot, population, cohort, output, *, maneuver_retention=1.):
    torch.set_num_threads(1)
    output = _output_path(output)
    selected = _cohort_rows(snapshot, "dev", cohort)
    if not 1 <= len(selected) <= 24:
        raise ValueError("oracle study is bounded to 1..24 development roots")
    source = {**_sources(), "repair_oracle_association.py": _sha(__file__),
              "repair_evaluation.py": _sha(Path(__file__).with_name("repair_evaluation.py"))}
    population_values = json.loads(Path(population).read_text())
    roots = []
    started = time.perf_counter()
    for meta, folder in selected:
        data = root_data(folder)
        privileged = read(folder / "privileged_initialization.json.gz")
        identity_rows, identity_counts = identify_history(data, privileged)
        old = GraphBeliefWorldModel(kind="memory_cv", population_prior=population_values)
        maneuver = RepairedGraphBeliefWorldModel(kind="memory_cv", observation_repair=False,
            maneuver_rate=2., reset_std=2., maneuver_retention=maneuver_retention, population_prior=population_values)
        full = RepairedGraphBeliefWorldModel(kind="memory_cv", observation_repair=True,
            maneuver_rate=2., reset_std=2., maneuver_retention=maneuver_retention, population_prior=population_values)
        root = {"root_id": meta["root_id"], "scene_id": meta["scene_id"],
            "true_population_offline_only": data["truth"].shape[1],
            "identity_mapping_counts": identity_counts,
            "identity_rows": [{k: value for k, value in row.items() if k != "detections"} for row in identity_rows],
            "oracle_gaussian": forced_history(old, data, identity_rows, population_values),
            "oracle_maneuver": forced_history(maneuver, data, identity_rows, population_values),
            "anonymous_gaussian": anonymous_history(old, data, population_values),
            "anonymous_maneuver": anonymous_history(maneuver, data, population_values),
            "anonymous_full": anonymous_history(full, data, population_values)}
        roots.append(root)
    summaries = {}
    for variant in ("oracle_gaussian", "oracle_maneuver"):
        observations = [r for root in roots for r in root[variant]["innovations"]]
        root_rates = [np.mean([r["false_birth_favored"] for r in root[variant]["innovations"]])
                      for root in roots if root[variant]["innovations"]]
        density = [r["known_identity_log_density"] for r in observations]
        summaries[variant] = {"matched_reobservations": len(observations),
            "roots_with_reobservations": len(root_rates),
            "root_mean_false_birth_favored_rate": float(np.mean(root_rates)) if root_rates else None,
            "pooled_known_identity_nll": float(-np.mean(density)) if density else None,
            "pooled_chi2_p95": float(np.percentile([r["gaussian_moment_chi2"] for r in observations], 95)) if observations else None,
            "unsupported_observations": sum(len(root[variant]["unsupported"]) for root in roots)}
    ending = {**_sources(), "repair_oracle_association.py": _sha(__file__),
              "repair_evaluation.py": _sha(Path(__file__).with_name("repair_evaluation.py"))}
    result = {"schema": SCHEMA, "split": "dev", "roots": len(roots),
        "source_sha256": source, "source_sha256_end": ending,
        "sources_changed_during_diagnostic": source != ending,
        "snapshot_manifest_sha256": _sha(Path(snapshot) / "manifest.json"),
        "population_prior_sha256": _sha(population), "cohort_sha256": _sha(cohort),
        "elapsed_s": time.perf_counter() - started, "threads": 1,
        "maneuver_rate": 2., "reset_std": 2., "maneuver_retention": maneuver_retention,
        "contract": {"identities": "exact XYZ legal positions + native prefix Euclidean ranges; t0 initial positions only; ties rejected",
            "forced_filter": "oracle identity-clamped original sequential prior and analytic conditional kinematics; true total count not supplied",
            "birth_odds": "original sequential hidden/track prior and sigma10m broad density; same formula in both process models",
            "privileged_boundary": "this module is offline diagnosis only; no true identities/count/map passed to deployed model",
            "limitations": "small selected diagnostic cohort; no independent statistical claim; mean dynamics likelihood is conditioned on oracle identities; new set-emission likelihood is not compared by birth odds"},
        "summary": summaries, "root_rows": roots}
    _json(output, result)
    print(json.dumps({"output": str(output), "roots": len(roots), "elapsed_s": result["elapsed_s"], "summary": summaries}), flush=True)
    return result


def selftest():
    class OracleTests(unittest.TestCase):
        def test_unique_ranges_recover_ordered_identity(self):
            ids, states, _ = match_ranges([[0, 0, 2], [3, 0, 0]], [0, 0, 0], [3, 2])
            self.assertEqual(ids, [1, 0])
            self.assertEqual(states, ["exact_range_match"] * 2)

        def test_range_ties_are_unsupported(self):
            ids, states, _ = match_ranges([[0, 0, 2]], [0, 0, 0], [2, 2])
            self.assertEqual(ids, [None])
            self.assertEqual(states, ["ambiguous_equal_ranges"])

        def test_reusing_one_physical_identity_is_rejected(self):
            ids, states, _ = match_ranges([[0, 0, 2], [2, 0, 0]], [0, 0, 0], [2])
            self.assertEqual(ids, [None, None])
            self.assertEqual(states, ["conflicting_one_to_one_match"] * 2)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OracleTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    for name in ("snapshot", "population", "cohort", "output"):
        execute.add_argument(name)
    execute.add_argument("--maneuver-retention", type=float, default=1.)
    commands.add_parser("selftest")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"run": run, "selftest": selftest}[command](**args)


if __name__ == "__main__":
    main()
