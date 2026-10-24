"""Narrow, evaluation-only adapter for the preselected 15 fresh repair roots.

The existing train/dev readers and fitters are deliberately unchanged. This
module never relabels fresh data as development data and has no fitting entry
point. ``freeze`` reads metadata and completed model checkpoints only. A later
``snapshot`` command opens exactly the previously selected sealed roots, after
validating the frozen design, and copies their checksum-verified payloads from a
read-only SQLite backup. ``evaluate`` accepts only that fresh snapshot.

All metric calculations reuse the development evaluator's pure scoring helpers.
The small per-root orchestration is explicit here to avoid weakening its split
guards. Fresh native caches have split='fresh', which the fitting loader rejects.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from eventwm_graph.stage1.storage import scene_split
from .data import read, root_data
from .neural import LearnedDynamics
from .queries import NativeEventModel, event_features
from .repair_events import (EVENT_NAMES, SCHEMA as EVENT_SCHEMA, STEP_SECONDS,
    _factory, _json, _output_path, _publish, _sha, event_clock_inputs,
    model_configuration, planned_steps, target_masks, validate_record)
from .repair_evaluation import (DEFAULT_VARIANTS, HORIZONS, CONDITIONING_HORIZONS, _metric_sources,
    _summaries, ancestry_energy, association_ambiguous, conditioning_pair,
    expected_set_scores, population_score, sampled_population_scores,
    tag_root_ancestry, truth_at, visible_ids)
from .repair_risk import (decision_summary, event_summary, scored_query,
                         _infinite_aware_summary)


DESIGN_SCHEMA = "partial_graph_repair_fresh_frozen_design_1"
SNAPSHOT_SCHEMA = "partial_graph_repair_fresh_snapshot_1"
REPORT_SCHEMA = "partial_graph_repair_fresh_evaluation_1"
EXPECTED_ROOTS = 15
EXPECTED_SCENES = 11
EXPECTED_CANDIDATE_DIGEST = "9aa28dd89114641ad9c20e52040408118f02106c2d5ff61f4aba38c1ebdbc5dc"
ROOT_SCHEMA = "observation_limited_query_conditioned_root_1"
PAYLOAD_FILES = frozenset(("legal_root.json.gz", "queries.json.gz", "outcomes.json.gz",
    "future_observations.json.gz", "privileged_future.npz",
    "privileged_initialization.json.gz", "privileged_step_labels.json.gz"))
CORE_CONTRACT_KEYS = ("map.py", "observer.py", "query.py", "metrics.py", "execution.py",
    "eventwm_graph/stage1/runtime.py", "scripts/collector/observation.py",
    "falcon/additional_action.py", "falcon/additional_metric.py")


def _sources():
    package = Path(__file__).parent
    return {**_metric_sources(), "repair_fresh.py": _sha(__file__),
        "repair_risk.py": _sha(package / "repair_risk.py"),
        "stage1/storage.py": _sha(package.parent / "stage1" / "storage.py")}


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path)}


def _read_reference(reference):
    path = Path(reference["path"])
    if _sha(path) != reference["sha256"]:
        raise ValueError("frozen file changed: " + str(path))
    return json.loads(path.read_text())


def _id_digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _research_output_path(path):
    """Validate containment before _output_path can create any parent folder."""
    base = Path(__file__).resolve().parents[2] / "output" / "l2_partial_graph_wm_repair_20261021"
    if not base.is_dir() or base.resolve() != base:
        raise ValueError("existing physical repair output directory required")
    resolved = Path(path).resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError("fresh artifacts must remain inside this research worktree's repair output")
    return _output_path(resolved)


def audit_candidates(candidates, old_manifest, *, expected_digest=EXPECTED_CANDIDATE_DIGEST):
    """Pure metadata-only legality, identity and scene-disjointness checks."""
    roots = candidates["roots"]
    ids = [r["root_id"] for r in roots]
    scenes = {r["scene_id"] for r in roots}
    if len(ids) != EXPECTED_ROOTS or len(set(ids)) != EXPECTED_ROOTS or len(scenes) != EXPECTED_SCENES:
        raise ValueError("fresh scope must be the exact preselected 15 roots / 11 scenes")
    if _id_digest(ids) != candidates["candidate_root_id_set_sha256"]:
        raise ValueError("candidate identity-set digest mismatch")
    if _id_digest(ids) != expected_digest:
        raise ValueError("candidate set differs from independently recorded preselection")
    old_roots = {r["root_id"] for r in old_manifest["rows"]}
    old_scenes = {r["scene_id"] for r in old_manifest["rows"]}
    if set(ids) & old_roots or scenes & old_scenes:
        raise ValueError("fresh roots/scenes overlap the previous research snapshot")
    old_versions = old_manifest["rows"][0]["versions"]
    for row in roots:
        if row["official_source_split"] != "train" or scene_split(row["scene_id"]) != "train":
            raise ValueError("protected source scene is not permitted")
        if row["schema"] != ROOT_SCHEMA or set(row["files"]) != PAYLOAD_FILES:
            raise ValueError("fresh root schema or payload inventory mismatch")
        if len(row["query_ids"]) != row["queries"] or len(set(row["query_ids"])) != row["queries"]:
            raise ValueError("invalid candidate query identity inventory")
        if Path(row["source"]).name != row["root_id"] or len(row["root_id"]) != 64:
            raise ValueError("invalid source root identity/path")
        for name in CORE_CONTRACT_KEYS:
            if row["versions"].get(name) != old_versions.get(name) or name not in old_versions:
                raise ValueError("fresh native contract mismatch: " + name)
    return {"roots": len(roots), "scenes": len(scenes),
        "queries": sum(r["queries"] for r in roots), "root_id_set_sha256": _id_digest(ids),
        "status": "METADATA_ONLY_DISJOINT_SCOPE_VERIFIED"}


def _validate_variants(variants, optical_path):
    """Explicitly permit the frozen unfitted-optics and association ablations."""
    variants = copy.deepcopy(variants)
    if not set(DEFAULT_VARIANTS) <= set(variants) or not 5 <= len(variants) <= 6:
        raise ValueError("fresh evaluation requires the five predeclared variants (at most one explicit diagnostic)")
    for name, variant in variants.items():
        if not variant["factory"].endswith(":RepairedGraphBeliefWorldModel"):
            continue
        kwargs = variant.setdefault("kwargs", {})
        if name == "association_only" and kwargs.get("observation_repair") is False:
            continue
        if name == "full_unfitted_optics" and "observation_checkpoint" not in kwargs:
            continue
        if str(Path(kwargs.get("observation_checkpoint", "missing")).resolve()) != optical_path:
            raise ValueError("repaired variant omits the explicit fitted observation checkpoint: " + name)
        kwargs["observation_checkpoint"] = optical_path
    return variants


def freeze(config, output):
    """Bind explicit final choices before any fresh root payload is opened.

    Config keys: old_manifest, candidates, dynamics_checkpoint, population_prior,
    observation_checkpoint, event_checkpoints (named paths), primary_event,
    primary_variant, and variants (same format as repair_evaluation --variants).
    """
    config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    output = _research_output_path(output)
    required = ("old_manifest", "candidates", "dynamics_checkpoint", "population_prior",
        "observation_checkpoint", "event_checkpoints", "primary_event", "primary_variant", "variants")
    if any(key not in config for key in required):
        raise ValueError("fresh freeze requires every explicit configuration field")
    refs = {key: _reference(config[key]) for key in required[:5]}
    candidates = _read_reference(refs["candidates"])
    audit = audit_candidates(candidates, _read_reference(refs["old_manifest"]))
    event_refs = {name: _reference(path) for name, path in config["event_checkpoints"].items()}
    if config["primary_event"] not in event_refs:
        raise ValueError("primary native event checkpoint is not frozen")
    variants = config["variants"]
    if isinstance(variants, str):
        variants = json.loads(Path(variants).read_text())
    variants = _validate_variants(variants, refs["observation_checkpoint"]["path"])
    if config["primary_variant"] not in variants:
        raise ValueError("primary variant not present")
    value = {"schema": DESIGN_SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "FROZEN_BEFORE_FRESH_PAYLOAD_READ", "references": refs,
        "event_checkpoints": event_refs, "primary_event": config["primary_event"],
        "primary_variant": config["primary_variant"], "variants": variants,
        "source_sha256": _sources(), "metadata_audit": audit,
        "evaluation": {"horizon_seconds": 3., "worlds": 16, "forecast_seed": 41,
            "event_seed": 31, "conditioning_seed": 107, "survival_seed": 91,
            "bootstrap_seed": 73, "conditioning_margin_m": .020,
            "conditioning_target_horizons": list(CONDITIONING_HORIZONS), "primary_conditioning_horizon": 2.,
            "selection": "exact 15 metadata-selected roots; no replacement, filtering, tuning or fitting",
            "broad_scientific_claim": "INCONCLUSIVE: 15 roots / 11 scenes"}}
    _json(output, value)
    os.chmod(output, 0o444)
    return value


def validate_design(path):
    design = json.loads(Path(path).read_text())
    if design.get("schema") != DESIGN_SCHEMA or design.get("state") != "FROZEN_BEFORE_FRESH_PAYLOAD_READ":
        raise ValueError("explicit frozen fresh design required")
    if design["source_sha256"] != _sources():
        raise ValueError("source code differs from fresh design freeze")
    values = {key: _read_reference(ref) if key in ("old_manifest", "candidates") else None
              for key, ref in design["references"].items()}
    for ref in [*design["references"].values(), *design["event_checkpoints"].values()]:
        if _sha(ref["path"]) != ref["sha256"]:
            raise ValueError("frozen checkpoint/reference changed: " + ref["path"])
    audit_candidates(values["candidates"], values["old_manifest"])
    return design, values["candidates"]


def _campaign(root):
    for parent in Path(root).parents:
        if (parent / "campaign.sqlite3").is_file():
            return parent
    raise ValueError("candidate root has no identifiable collection journal")


def snapshot(design_path, destination):
    """Call only after explicit design freeze; source access is always read-only."""
    design, candidates = validate_design(design_path)
    campaigns = sorted({_campaign(row["source"]) for row in candidates["roots"]})
    resolved_destination = Path(destination).resolve()
    if any(resolved_destination.is_relative_to(campaign.resolve()) for campaign in campaigns):
        raise ValueError("fresh snapshot destination must be outside every production campaign")
    destination = _research_output_path(destination)
    destination.mkdir(exist_ok=False)
    sealed = {}
    journals = []
    for index, campaign in enumerate(campaigns):
        path = destination / ("journal_" + str(index) + ".sqlite3")
        source = sqlite3.connect("file:" + str(campaign / "campaign.sqlite3") + "?mode=ro", uri=True)
        target = sqlite3.connect(path)
        try:
            source.backup(target, pages=256, sleep=.05)
            if target.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise ValueError("fresh copied journal integrity failure")
            for scene, root_id, folder in target.execute("SELECT scene,json_extract(result,'$.root_id'),json_extract(result,'$.folder') FROM jobs WHERE state='sealed'"):
                if root_id in sealed and sealed[root_id] != (scene, folder):
                    raise ValueError("conflicting sealed root identities")
                sealed[root_id] = (scene, folder)
        finally:
            source.close()
            target.close()
        os.chmod(path, 0o444)
        journals.append({"file": path.name, "sha256": _sha(path), "source": str(campaign)})
    rows = []
    for candidate in candidates["roots"]:
        root_id = candidate["root_id"]
        src = Path(candidate["source"]).resolve()
        if sealed.get(root_id) != (candidate["scene_id"], candidate["source"]):
            raise ValueError("preselected root is not sealed in the consistent journal backup")
        manifest_path = src / "manifest.json"
        if _sha(manifest_path) != candidate["manifest_sha256"]:
            raise ValueError("root metadata changed after candidate selection")
        native = json.loads(manifest_path.read_text())
        if not native["sealed"] or native["split"] != "train" or native["root_id"] != root_id:
            raise ValueError("invalid sealed source manifest")
        if native["files"] != candidate["files"] or native["query_ids"] != candidate["query_ids"]:
            raise ValueError("source payload identity changed after metadata selection")
        target = destination / "roots" / root_id
        target.mkdir(parents=True, exist_ok=False)
        hashes = {**candidate["files"], "manifest.json": candidate["manifest_sha256"]}
        for name, expected in hashes.items():
            if Path(name).name != name:
                raise ValueError("unsafe payload relative path")
            if _sha(src / name) != expected:
                raise ValueError("source checksum mismatch before copy: " + name)
            shutil.copyfile(src / name, target / name)
            if _sha(target / name) != expected or _sha(src / name) != expected:
                raise ValueError("payload changed during snapshot copy: " + name)
            os.chmod(target / name, 0o444)
        if _sha(manifest_path) != candidate["manifest_sha256"]:
            raise ValueError("sealed manifest changed during snapshot copy")
        rows.append({**candidate, "split": "fresh", "files": hashes})
        os.chmod(target, 0o555)
    validate_design(design_path)
    value = {"schema": SNAPSHOT_SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split": "fresh", "design_sha256": _sha(design_path), "journals": journals,
        "candidate_root_id_set_sha256": candidates["candidate_root_id_set_sha256"],
        "rows": rows, "selection": design["evaluation"]["selection"]}
    _json(destination / "manifest.json", value)
    os.chmod(destination / "manifest.json", 0o444)
    os.chmod(destination / "roots", 0o555)
    os.chmod(destination, 0o555)
    return value


def fresh_rows(design_path, snapshot_path, *, verify_payloads=True):
    design, candidates = validate_design(design_path)
    snapshot_path = Path(snapshot_path).resolve()
    value = json.loads((snapshot_path / "manifest.json").read_text())
    if value.get("schema") != SNAPSHOT_SCHEMA or value.get("split") != "fresh":
        raise ValueError("explicit fresh snapshot schema/split required")
    if value.get("design_sha256") != _sha(design_path):
        raise ValueError("fresh snapshot bound to a different design")
    permitted = {row["root_id"]: row for row in candidates["roots"]}
    ids = [row["root_id"] for row in value["rows"]]
    if len(ids) != len(set(ids)) or set(ids) != set(permitted):
        raise ValueError("fresh snapshot differs from the exact preselected cohort")
    for row in value["rows"]:
        candidate = permitted[row["root_id"]]
        if row["split"] != "fresh" or row["scene_id"] != candidate["scene_id"]:
            raise ValueError("fresh row split or scene mismatch")
        expected = {**candidate["files"], "manifest.json": candidate["manifest_sha256"]}
        if row["files"] != expected or row["query_ids"] != candidate["query_ids"]:
            raise ValueError("fresh row inventory mismatch")
        folder = snapshot_path / "roots" / row["root_id"]
        if folder.is_symlink() or not folder.resolve().is_relative_to(snapshot_path):
            raise ValueError("fresh root resolves outside its frozen snapshot")
        if verify_payloads:
            for name, wanted in expected.items():
                if (folder / name).is_symlink() or _sha(folder / name) != wanted:
                    raise ValueError("fresh payload checksum mismatch")
    return design, value, [(row, snapshot_path / "roots" / row["root_id"]) for row in value["rows"]]


def _population_root(model, data, meta, name, verify):
    """Same per-root calculation as repair_evaluation.run; helpers are shared."""
    legal = data["legal"]
    target_ids = visible_ids(data)
    base = model.initialize(legal["history"], legal["partial_graph"])
    belief = tag_root_ancestry(base, data["truth"][0])
    identity = {"root_id": meta["root_id"], "scene_id": meta["scene_id"], "variant": name}
    invariance = None
    if verify:
        a = model.predict_shared(base, .096, 16, seed=211, times=[0., .048, .096])
        b = model.predict_shared(belief, .096, 16, seed=211, times=[0., .048, .096])
        if not (np.array_equal(a.positions, b.positions, equal_nan=True)
                and np.array_equal(a.velocities, b.velocities, equal_nan=True)
                and np.array_equal(a.weights, b.weights) and a.unresolved_mass == b.unresolved_mass):
            raise AssertionError("offline ancestry tags changed prediction")
        invariance = {**identity, "tagged_untagged_bitwise_identical": True}
    score = population_score(belief, data["truth"].shape[1])
    score.update(expected_set_scores(belief.hypotheses,
        [h.mean[:, :2] for h in belief.hypotheses], [h.weight for h in belief.hypotheses], data["truth"][0], target_ids))
    visible = {r["track_id"] for r in legal["history"][-1]["tracks"] if r["visible_now"]}
    previous = {r["track_id"] for row in legal["history"][:-1] for r in row["tracks"] if r["visible_now"]}
    last = {r["track_id"] for r in legal["history"][-2]["tracks"] if r["visible_now"]} if len(legal["history"]) > 1 else set()
    score.update(identity, aged_tracks=any(np.any(belief.timestamp - h.last_seen > .5) for h in belief.hypotheses if h.n),
        association_ambiguous=association_ambiguous(belief), reappearance_proxy=bool((visible - last) & previous),
        root_visible_targets=len(target_ids),
        exact_ancestry_slots_conditional=float(sum(h.weight * len(h.diagnostic_exact) for h in belief.hypotheses) / max(1e-30, 1 - belief.unresolved_mass)),
        aged_oracle_ancestry_slots_conditional=float(sum(h.weight * len(h.diagnostic_aged) for h in belief.hypotheses) / max(1e-30, 1 - belief.unresolved_mass)))
    future = model.predict_shared(belief, 3., 16, seed=41, times=[0., *HORIZONS])
    score.update(sampled_population_scores(future, 0, data["truth"][0]))
    forecasts = []
    for horizon in HORIZONS:
        target = truth_at(data, belief.timestamp + horizon)
        if target is None:
            continue
        index = int(np.argmin(abs(future.times - belief.timestamp - horizon)))
        forecast = ancestry_energy(future, target, target_ids, belief.timestamp + horizon)
        forecast.update(expected_set_scores(future.worlds, future.positions[:, index], future.weights, target, target_ids))
        forecast.update(sampled_population_scores(future, index, target))
        forecasts.append({**identity, **forecast, "horizon": horizon, "unresolved_mass": float(future.unresolved_mass)})
    pairs = []
    for horizon in CONDITIONING_HORIZONS:
        pair = conditioning_pair(model, belief, data, target_ids, target_horizon=horizon)
        if pair is not None:
            pairs.append({**identity, **pair})
    return base, score, forecasts, pairs, invariance


def _event_records(model, belief, data, labels, meta):
    """Full planned legal features; identical masks/labels to repair_events.prepare."""
    steps = [planned_steps(query) for query in data["queries"]]
    if not steps:
        raise ValueError("fresh root has no candidate queries")
    times = np.arange(max(steps) + 1, dtype=float) * STEP_SECONDS
    future = model.predict_shared(belief, float(times[-1]), 16, seed=31, times=times)
    hidden = np.array([h.expected_hidden for h in future.worlds])
    total = hidden + np.array([h.n for h in future.worlds])
    allowed = np.stack([total > 0, total > 0, hidden > 0], -1)
    age = np.mean([max(0., belief.timestamp - h.last_seen.mean()) if h.n else 0. for h in future.worlds])
    primary = max(belief.hypotheses, key=lambda h: h.weight)
    known = set()
    if primary.n and data["truth"].shape[1]:
        distances = np.linalg.norm(primary.mean[:, None, :2] - data["truth"][None, 0], axis=-1)
        ii, jj = linear_sum_assignment(distances)
        known = {int(j) for i, j in zip(ii, jj) if distances[i, j] < 1.5}
    result = []
    for query, outcome, planned in zip(data["queries"], data["outcomes"], steps):
        poses = np.asarray(query["compiled"]["planned_poses"][:planned + 1], float)
        feature_age, feature_elapsed = event_clock_inputs(model, age, times[:planned + 1])
        features = event_features(future.positions[:, :planned + 1], future.velocities[:, :planned + 1],
            poses, hidden, feature_age, feature_elapsed)[:, 1:]
        observed = min(len(outcome["node_ids"]), planned)
        events = np.zeros((observed, 3), bool)
        actual_nodes = []
        for index, node_id in enumerate(outcome["node_ids"][:observed]):
            label = labels[node_id]
            events[index, :2] = label["human_mesh_contact"], label["psc_violation"]
            obs = data["nodes"][node_id]["observation"]
            actual_nodes.append(obs)
            for track in obs["tracks"]:
                if not track["visible_now"] or not data["truth"].shape[1]:
                    continue
                distance = np.linalg.norm(data["truth"][index + 1] - np.asarray(track["position"])[[0, 2]], axis=1)
                identity = int(distance.argmin())
                events[index, 2] |= distance[identity] < 1e-5 and identity not in known
        first = np.array([int(np.flatnonzero(events[:, head])[0]) if events[:, head].any() else -1 for head in range(3)])
        termination = outcome["summary"]["termination"]
        label_known, calibration, administrative = target_masks(first, observed, planned, termination)
        actual_poses = np.asarray([belief.robot_pose, *[row["robot_pose"] for row in actual_nodes]], float)
        record = {"schema": EVENT_SCHEMA, "split": "fresh", "root_id": meta["root_id"], "scene_id": meta["scene_id"],
            "query_id": query["query_id"], "x": features.astype(np.float32), "allowed": allowed.copy(),
            "weights": future.weights.copy(), "unresolved": float(future.unresolved_mass), "first": first,
            "planned_steps": planned, "observed_steps": observed, "planned_seconds": float(times[planned]),
            "actual_observed_seconds": float(actual_nodes[-1]["timestamp"] - belief.timestamp) if observed else 0.,
            "planned_elapsed": times[:planned + 1].copy(),
            "planned_cumulative_progress": np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1))],
            "actual_progress": float(np.linalg.norm(np.diff(actual_poses[:, :3], axis=0), axis=1).sum()),
            "query_actions_total": len(query["compiled"]["actions"]), "original_candidate_queries": len(data["queries"]),
            "planned_query_complete": planned == len(query["compiled"]["actions"]), "label_known": label_known,
            "calibration_eligible": calibration, "administrative_censoring": administrative,
            "termination": termination, "root_weight": 1. / len(data["queries"])}
        result.append(validate_record(record))
    return result


def evaluate(design_path, snapshot_path, output):
    """One frozen pass; no fitting, tuning, fallback cohort, or difficulty filter."""
    torch.set_num_threads(1)
    design, snapshot_manifest, rows = fresh_rows(design_path, snapshot_path)
    output = _research_output_path(output)
    output.mkdir(exist_ok=False)
    ref = design["references"]
    learned = LearnedDynamics.load(ref["dynamics_checkpoint"]["path"])
    prior = _read_reference(ref["population_prior"])
    coefficients = {name: np.asarray(_read_reference(checkpoint)["coefficients"], float)
                    for name, checkpoint in design["event_checkpoints"].items()}
    event = NativeEventModel(coefficients[design["primary_event"]])
    models = {name: _factory(spec["factory"])(dynamics=learned if spec.get("use_learned", True) else None,
        population_prior=prior, event_model=event, **spec.get("kwargs", {})) for name, spec in design["variants"].items()}
    populations, forecasts, pairs, invariance, events = [], [], [], [], []
    started = time.perf_counter()
    for index, (meta, folder) in enumerate(rows):
        data = root_data(folder, True)
        ids = [query["query_id"] for query in data["queries"]]
        if (len(ids) != meta["queries"] or len(ids) != len(set(ids))
                or set(ids) != set(meta["query_ids"]) or len(data["outcomes"]) != len(ids)):
            raise ValueError("fresh actual query identities differ from frozen metadata")
        for name, model in models.items():
            base, pop, forecast, root_pairs, inv = _population_root(model, data, meta, name, index < 3)
            populations.append(pop)
            forecasts.extend(forecast)
            pairs.extend(root_pairs)
            if inv is not None:
                invariance.append(inv)
            if name == design["primary_variant"]:
                events.extend(_event_records(model, base, data, read(folder / "privileged_step_labels.json.gz"), meta))
        print(json.dumps({"fresh_roots": index + 1, "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    risk = {}
    for name, coef in coefficients.items():
        scored = [scored_query(coef, record) for record in events]
        risk[name] = {"events": {event: event_summary(scored, event) for event in EVENT_NAMES},
            "joint_censored_first_event_nll": _infinite_aware_summary(scored, "joint_censored_nll", "joint_censored_nll_infinite"),
            "decision": decision_summary(scored), "query_rows": scored}
    validate_design(design_path)
    common = {"schema": REPORT_SCHEMA, "split": "fresh", "design_sha256": _sha(design_path),
        "snapshot_manifest_sha256": _sha(Path(snapshot_path) / "manifest.json"),
        "source_sha256": _sources(), "roots": len(rows), "scenes": len({r["scene_id"] for r, _ in rows}),
        "threads": 1, "elapsed_s": time.perf_counter() - started,
        "scientific_status": "INCONCLUSIVE_SMALL_FRESH_COHORT",
        "limitations": "15 roots / 11 scenes; exploratory confirmation only. No refitting or selection. Curriculum candidate sets, not closed-loop navigation. Conditional ancestry Energy Score is distinct from all-population GOSPA."
    }
    population_report = {**common, "variants": design["variants"],
        "resolved_model_configuration": {name: model_configuration(model) for name, model in models.items()},
        "summary": _summaries(populations, forecasts, pairs, design["variants"]),
        "population_rows": populations, "forecast_rows": forecasts, "conditioning_rows": pairs,
        "tag_invariance_checks": invariance}
    _json(output / "population.json", population_report)
    _json(output / "risk.json", {**common, "models": risk, "event_checkpoints": design["event_checkpoints"],
        "comparison": "all coefficient sets see identical repaired full-planned cached futures; old coefficients were fitted on legacy features, so this does not isolate stopping-time correction alone",
        "appearance_target": "inherited <=1.5m root-match proxy, not the repaired observation channel"})
    cache_path = output / "events.npy"
    _publish(cache_path, lambda stream: np.save(stream, np.array(events, dtype=object), allow_pickle=True))
    _json(str(cache_path) + ".manifest.json", {**common, "schema": EVENT_SCHEMA,
        "cache_sha256": _sha(cache_path), "purpose": "evaluation only; strict training/dev loader rejects split=fresh",
        "primary_variant": design["primary_variant"], "primary_event": design["primary_event"]})
    return {**common, "output": str(output)}


def selftest():
    """Only synthetic metadata/cache guards; never opens a real candidate root."""
    from .repair_events import load_cache

    def fixture():
        scenes = []
        i = 0
        while len(scenes) < EXPECTED_SCENES:
            name = "fresh_oracle_scene_" + str(i)
            if scene_split(name) == "train":
                scenes.append(name)
            i += 1
        versions = {key: "fixed" for key in CORE_CONTRACT_KEYS}
        rows = []
        for i in range(EXPECTED_ROOTS):
            rid = hashlib.sha256(str(i).encode()).hexdigest()
            rows.append({"root_id": rid, "scene_id": scenes[i % len(scenes)], "official_source_split": "train",
                "source": "/sealed/" + rid, "schema": ROOT_SCHEMA, "files": {name: "hash" for name in PAYLOAD_FILES},
                "query_ids": ["query_" + str(i)], "queries": 1, "versions": versions})
        return {"roots": rows, "candidate_root_id_set_sha256": _id_digest(r["root_id"] for r in rows)}, {
            "rows": [{"root_id": "old", "scene_id": "old_scene", "versions": versions}]}

    class FreshTests(unittest.TestCase):
        def test_metadata_audit_and_exact_scope(self):
            candidates, old = fixture()
            digest = candidates["candidate_root_id_set_sha256"]
            self.assertEqual(audit_candidates(candidates, old, expected_digest=digest)["roots"], 15)
            candidates["roots"].pop()
            with self.assertRaises(ValueError):
                audit_candidates(candidates, old, expected_digest=digest)

        def test_no_old_scene_leakage(self):
            candidates, old = fixture()
            old["rows"][0]["scene_id"] = candidates["roots"][0]["scene_id"]
            with self.assertRaises(ValueError):
                audit_candidates(candidates, old, expected_digest=candidates["candidate_root_id_set_sha256"])

        def test_no_protected_source_or_changed_contract(self):
            candidates, old = fixture()
            candidates["roots"][0]["official_source_split"] = "confirmation"
            with self.assertRaises(ValueError):
                audit_candidates(candidates, old, expected_digest=candidates["candidate_root_id_set_sha256"])
            candidates, old = fixture()
            candidates["roots"][0]["versions"] = {**candidates["roots"][0]["versions"], "query.py": "different"}
            with self.assertRaises(ValueError):
                audit_candidates(candidates, old, expected_digest=candidates["candidate_root_id_set_sha256"])

        def test_independently_fixed_preselection_rejects_self_consistent_alternative(self):
            candidates, old = fixture()
            self.assertEqual(_id_digest(r["root_id"] for r in candidates["roots"]),
                             candidates["candidate_root_id_set_sha256"])
            with self.assertRaisesRegex(ValueError, "independently recorded preselection"):
                audit_candidates(candidates, old)

        def test_output_boundary_rejects_external_paths_before_directory_creation(self):
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "must_not_create" / "result.json"
                with self.assertRaisesRegex(ValueError, "research worktree"):
                    _research_output_path(target)
                self.assertFalse(target.parent.exists())
            base = Path(__file__).resolve().parents[2] / "output" / "l2_partial_graph_wm_repair_20261021"
            with self.assertRaises(ValueError):
                _research_output_path(base / ".." / "outside_repair" / "result.json")
            with self.assertRaises(ValueError):
                _research_output_path(base)
            with tempfile.TemporaryDirectory(dir=base, prefix=".fresh_guard_") as local, tempfile.TemporaryDirectory() as remote:
                link = Path(local) / "escape"
                link.symlink_to(remote, target_is_directory=True)
                with self.assertRaises(ValueError):
                    _research_output_path(link / "must_not_create" / "result.json")
                self.assertFalse((Path(remote) / "must_not_create").exists())

        def test_fit_loader_rejects_fresh_before_opening_payload(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "unopened.npy"
                Path(str(path) + ".manifest.json").write_text(json.dumps({"schema": EVENT_SCHEMA, "split": "fresh"}))
                for expected in ("train", "dev", "fresh"):
                    with self.assertRaises(ValueError):
                        load_cache(path, expected)

        def test_reference_changes_are_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "metadata.json"
                path.write_text('{"version":1}')
                ref = _reference(path)
                self.assertEqual(_read_reference(ref)["version"], 1)
                path.write_text('{"version":2}')
                with self.assertRaises(ValueError):
                    _read_reference(ref)

        def test_fitted_and_explicit_unfitted_variants(self):
            variants = copy.deepcopy(DEFAULT_VARIANTS)
            path = str(Path("oracle_optics.json").resolve())
            for name in ("full", "full_no_maneuver", "full_kinematic"):
                variants[name]["kwargs"]["observation_checkpoint"] = path
            variants["full_unfitted_optics"] = copy.deepcopy(DEFAULT_VARIANTS["full"])
            self.assertEqual(len(_validate_variants(variants, path)), 6)
            variants["full"]["kwargs"].pop("observation_checkpoint")
            with self.assertRaises(ValueError):
                _validate_variants(variants, path)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FreshTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("config")
    p.add_argument("output")
    p = commands.add_parser("snapshot")
    p.add_argument("design_path")
    p.add_argument("destination")
    p = commands.add_parser("evaluate")
    p.add_argument("design_path")
    p.add_argument("snapshot_path")
    p.add_argument("output")
    commands.add_parser("selftest")
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"freeze": freeze, "snapshot": snapshot, "evaluate": evaluate, "selftest": selftest}[command](**args)
    if result is not None:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
