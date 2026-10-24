"""Bounded development-only joint-time trajectory scoring of frozen repair models.

This diagnostic does not fit or select parameters. It scores each root-visible
physical identity using its fixed root ancestry and an entire sampled path;
future positions never repair an identity assignment. The normalized path Energy
Score is proper for that conditional finite-world measure, not a complete
random-population score. Missing represented mass remains explicit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from eventwm_graph.l2_partial.data import root_data
from eventwm_graph.l2_partial.neural import LearnedDynamics
from eventwm_graph.l2_partial.queries import NativeEventModel
from eventwm_graph.l2_partial.repair_events import (
    _cohort_rows, _factory, _json, _output_path, _sha, model_configuration,
)
from eventwm_graph.l2_partial.repair_evaluation import (
    HORIZONS, _metric_sources, scene_ci, tag_root_ancestry, truth_at, visible_ids,
)


def path_energy(samples, truth, weights):
    """Exact Energy Score of a finite law on R^(times*2), scaled to metres."""
    samples = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 3 or truth.shape != samples.shape[1:]:
        raise ValueError("expected worlds x positive times x spatial coordinates")
    if samples.shape[1] == 0 or weights.shape != (samples.shape[0],):
        raise ValueError("invalid path/weight dimensions")
    if (not np.isfinite(samples).all() or not np.isfinite(truth).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., atol=1e-12)):
        raise ValueError("finite paths and a normalized nonnegative law required")
    x = samples.reshape(samples.shape[0], -1)
    y = truth.ravel()
    first = float(weights @ np.linalg.norm(x - y, axis=1))
    pair = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    return float((first - .5 * (weights @ pair @ weights)) / math.sqrt(len(truth)))


def fixed_ancestry_path_score(future, truths, target_ids, absolute_times):
    """Conditional per-target paths; no per-world or future target matching."""
    absolute_times = np.asarray(absolute_times, dtype=float)
    truths = np.asarray(truths, dtype=float)
    indices = [int(np.argmin(abs(future.times - t))) for t in absolute_times]
    if (not len(indices) or truths.shape[0] != len(indices)
            or any(abs(float(future.times[i]) - t) > 1e-6
                   for i, t in zip(indices, absolute_times))):
        raise ValueError("trajectory target timestamps must exactly match forecast")
    scores, masses, unsupported = [], [], []
    for identity in target_ids:
        paths, weights = [], []
        for w, hypothesis in enumerate(future.worlds):
            if future.weights[w] <= 0:
                continue
            if not hasattr(hypothesis, "diagnostic_ancestry"):
                raise ValueError("positive forecast mass lost fixed ancestry metadata")
            mapping = dict(hypothesis.diagnostic_ancestry)
            slots = [j for j, slot in enumerate(hypothesis.slot_ids)
                     if mapping.get(int(slot)) == identity]
            if len(slots) > 1:
                raise ValueError("fixed ancestry is not one-to-one")
            if len(slots) == 1:
                path = future.positions[w, indices, slots[0], :]
                if np.isfinite(path).all():
                    paths.append(path)
                    weights.append(float(future.weights[w]))
        mass = float(sum(weights))
        masses.append({"physical_identity_offline_only": int(identity), "represented_mass": mass})
        if mass <= 0:
            unsupported.append(int(identity))
            continue
        score = path_energy(np.asarray(paths), truths[:, identity, :], np.asarray(weights) / mass)
        scores.append(score)
    supported = bool(target_ids) and not unsupported
    return {
        "trajectory_energy_score": float(np.mean(scores)) if supported else None,
        "supported_targets_only_energy_score": float(np.mean(scores)) if scores else None,
        "target_count": len(target_ids), "unsupported_target_ids": unsupported,
        "all_targets_supported": supported, "target_represented_mass": masses,
        "represented_mass_mean": float(np.mean([r["represented_mass"] for r in masses])) if masses else 0.,
        "represented_mass_minimum": min((r["represented_mass"] for r in masses), default=0.),
        "unresolved_mass": float(future.unresolved_mass),
    }


class TimeBudgetExceeded(Exception):
    pass


def _alarm(_signum, _frame):
    raise TimeBudgetExceeded("bounded diagnostic wall-compute budget reached")


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not 0 < args.seconds <= 60:
        raise ValueError("this diagnostic permits at most60 seconds of numerical work")
    output = _output_path(args.output)
    if output.exists():
        raise FileExistsError("historical diagnostic outputs are immutable")
    variants = json.loads(Path(args.variants).read_text())
    if len(variants) != 6:
        raise ValueError("all six frozen repair variants are required")
    cohort = json.loads(Path(args.cohort).read_text())
    if not isinstance(cohort, list) or len(cohort) != 10 or len(set(cohort)) != 10:
        raise ValueError("only the declared ten-root development smoke cohort is allowed")
    roots = _cohort_rows(args.snapshot, "dev", cohort)
    if len(roots) != 10:
        raise ValueError("all declared development roots must be present")
    sources = {**_metric_sources(), "trajectory_script": _sha(__file__)}
    assets = {name: {"path": str(Path(getattr(args, name)).resolve()),
                     "sha256": _sha(getattr(args, name))}
              for name in ("checkpoint", "population", "event_checkpoint", "variants", "cohort")}
    for config in variants.values():
        optical = config.get("kwargs", {}).get("observation_checkpoint")
        if optical:
            assets["observation_checkpoint"] = {"path": str(Path(optical).resolve()), "sha256": _sha(optical)}
    records, completed = [], []
    status = "COMPLETE_BOUNDED_DESCRIPTIVE_DIAGNOSTIC"
    configuration = {}
    started = time.perf_counter()
    prior_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        learned = LearnedDynamics.load(args.checkpoint)
        population = json.loads(Path(args.population).read_text())
        event = NativeEventModel(json.loads(Path(args.event_checkpoint).read_text())["coefficients"])
        models = {name: _factory(config["factory"])(
            dynamics=learned if config.get("use_learned", True) else None,
            population_prior=population, event_model=event, **config.get("kwargs", {}))
            for name, config in variants.items()}
        configuration = {name: model_configuration(model) for name, model in models.items()}
        for meta, folder in roots:
            data = root_data(folder, True)
            legal = data["legal"]
            identities = visible_ids(data)
            # Prepare every variant before recording this root, preventing a
            # partial time-budget root from favoring earlier variants.
            root_records = []
            for name, model in models.items():
                base = model.initialize(legal["history"], legal["partial_graph"])
                belief = tag_root_ancestry(base, data["truth"][0])
                future = model.predict_shared(belief, 3., 16, seed=41, times=[0., *HORIZONS])
                available, truths = [], []
                for horizon in HORIZONS:
                    target = truth_at(data, belief.timestamp + horizon)
                    if target is None:
                        break
                    available.append(horizon)
                    truths.append(target)
                    score = fixed_ancestry_path_score(future, truths, identities,
                        belief.timestamp + np.asarray(available))
                    root_records.append({"root_id": meta["root_id"], "scene_id": meta["scene_id"],
                        "variant": name, "horizon": horizon, "scored_times_from_root": available.copy(),
                        "number_times": len(available), **score})
            records.extend(root_records)
            completed.append(meta["root_id"])
            print(json.dumps({"roots": len(completed), "elapsed_s": round(time.perf_counter() - started, 3)}), flush=True)
    except TimeBudgetExceeded:
        status = "INCOMPLETE_TIME_BUDGET_NO_ACCEPTANCE_CLAIM"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
    numerical_seconds = time.perf_counter() - started
    ending = {**_metric_sources(), "trajectory_script": _sha(__file__)}
    if ending != sources or any(_sha(v["path"]) != v["sha256"] for v in assets.values()):
        raise RuntimeError("frozen inference sources or checkpoints changed")
    summary = {}
    for name in variants:
        summary[name] = {}
        for horizon in HORIZONS:
            selected = [r for r in records if r["variant"] == name and r["horizon"] == horizon]
            metric = scene_ci(selected, "trajectory_energy_score")
            metric.update(eligible_roots=len(selected), unsupported_roots=sum(not r["all_targets_supported"] for r in selected),
                          represented_mass=scene_ci(selected, "represented_mass_mean"))
            summary[name][str(horizon)] = metric
    result = {"schema": "partial_graph_repair_joint_time_energy_1", "split": "dev",
        "status": status, "scientific_status": "DESCRIPTIVE_SMALL_SELECTED_DEVELOPMENT_DIAGNOSTIC",
        "requested_roots": 10, "completed_roots": len(completed), "completed_root_ids": completed,
        "scenes": len({r["scene_id"] for r in records}), "variants": variants,
        "resolved_model_configuration": configuration, "seeds": {"future": 41, "bootstrap": 73},
        "worlds": 16, "threads": 1, "seconds_limit": args.seconds, "numerical_seconds": numerical_seconds,
        "sources": sources, "sources_after": ending, "assets": assets,
        "snapshot_manifest_sha256": _sha(Path(args.snapshot) / "manifest.json"),
        "metric_contract": "Per-root-visible physical target, concatenate the same world path at every available0.5/1/2/3s prefix endpoint; exact finite weighted Energy Score divided by sqrt(numbertimes). Fixed ancestry from unsampled root; no future/world rematching. Unsupported targets make root score undefined, never zero. Per-target represented mass and unresolved mass remain explicit.",
        "limitations": "Conditional per-person joint-time score, not joint across humans or a full random-population score. Selected10 development roots, no fresh payload or parameter selection. Prefix-time support shrinks with horizon. Bootstrap intervals are descriptive and do not provide broad confirmation or separate seed uncertainty.",
        "summary": summary, "rows": records}
    _json(output, result)
    print(json.dumps({"output": str(output), "status": status, "completed_roots": len(completed),
                      "numerical_seconds": numerical_seconds}), flush=True)


def selftest():
    class Tests(unittest.TestCase):
        def test_deterministic_path_has_rms_spatial_error(self):
            score = path_energy([[[1., 0.], [2., 0.]]], [[0., 0.], [0., 0.]], [1.])
            self.assertAlmostEqual(score, math.sqrt(2.5), places=13)

        def test_same_marginals_different_temporal_coupling(self):
            coupled = np.array([[[-1., 0.], [-1., 0.]], [[1., 0.], [1., 0.]]])
            crossed = np.array([[[-1., 0.], [1., 0.]], [[1., 0.], [-1., 0.]]])
            truth = np.array([[1., 0.], [1., 0.]])
            for i in range(2):
                self.assertAlmostEqual(path_energy(coupled[:, i:i+1], truth[i:i+1], [.5, .5]),
                                       path_energy(crossed[:, i:i+1], truth[i:i+1], [.5, .5]))
            self.assertAlmostEqual(path_energy(coupled, truth, [.5, .5]), .5)
            self.assertGreater(path_energy(crossed, truth, [.5, .5]), .9)

        def test_missing_ancestry_is_not_rematched_or_dropped(self):
            world = SimpleNamespace(diagnostic_ancestry=((0, 0),), slot_ids=(0,))
            future = SimpleNamespace(worlds=[world], weights=np.array([.7]), times=np.array([.5, 1.]),
                positions=np.zeros((1, 2, 1, 2)), unresolved_mass=.3)
            result = fixed_ancestry_path_score(future, np.zeros((2, 2, 2)), [0, 1], [.5, 1.])
            self.assertIsNone(result["trajectory_energy_score"])
            self.assertEqual(result["unsupported_target_ids"], [1])
            self.assertEqual(result["target_represented_mass"][0]["represented_mass"], .7)
            self.assertEqual(result["target_represented_mass"][1]["represented_mass"], 0.)

        def test_empty_target_set_is_not_a_perfect_score(self):
            future = SimpleNamespace(worlds=[], weights=np.array([]), times=np.array([.5]),
                positions=np.empty((0, 1, 0, 2)), unresolved_mass=1.)
            self.assertIsNone(fixed_ancestry_path_score(future, np.empty((1, 0, 2)), [], [.5])["trajectory_energy_score"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    action = sub.add_parser("run")
    for name in ("snapshot", "checkpoint", "population", "event-checkpoint", "variants", "cohort", "output"):
        action.add_argument("--" + name, required=True)
    action.add_argument("--seconds", type=float, default=60.)
    args = parser.parse_args()
    selftest() if args.command == "selftest" else run(args)


if __name__ == "__main__":
    main()
