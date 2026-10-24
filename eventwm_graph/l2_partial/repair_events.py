"""Bounded train/dev repair of native event fitting and planned-horizon calibration.

Features cover a horizon chosen from legal query actions, never a realized stop.
Likelihood masks cover only observed risk sets.  Calibration uses terminal-query
outcomes (contact is absorbing) or fully observed planned intervals.  All early
administrative censoring is excluded from endpoint Brier calibration, including
censored paths on which an event has already occurred.  Such paths still enter
the right-censored first-event likelihood.  The calibration estimand is therefore
the complete-followup cohort, not an IPCW estimate under arbitrary censoring.

The third head intentionally retains the release's approximate untracked-person
appearance target.  It is not the repaired generative observation channel.
Only trusted local feature archives are loaded with allow_pickle=True.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import expit
import torch

from .data import rows, root_data, read
from .neural import LearnedDynamics
from .queries import FEATURE_NAMES, NativeEventModel, event_features

SCHEMA = "partial_graph_event_repair_1"
EVENT_NAMES = ("collision", "psc", "untracked_appearance_proxy")
STEP_SECONDS = .048
RESIDUAL_HAZARD = .005
DEFAULT_FACTORY = "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _output_path(path):
    path = Path(path).resolve()
    if "l2_partial_graph_wm_explore_20261021" in path.parts:
        raise ValueError("historical exploratory outputs are immutable")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _publish(path, writer):
    """Exclusive publication: never replace an existing research artifact."""
    path = _output_path(path)
    fd, name = tempfile.mkstemp(prefix=".event_repair_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)  # Fails if another writer has published this identity.
    finally:
        os.unlink(name)
    return path


def _json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    return _publish(path, lambda stream: stream.write(payload))


def _sources():
    package = Path(__file__).parent
    names = ("repair_events.py", "data.py", "model.py", "state.py", "probability.py",
             "neural.py", "queries.py", "maneuver.py", "observation.py",
             "repair_filter.py", "repair_model.py", "hidden.py")
    return {name: _sha(package / name) for name in names if (package / name).exists()}


def _factory(factory):
    if callable(factory):
        return factory
    module, name = (factory or DEFAULT_FACTORY).split(":", 1)
    return getattr(importlib.import_module(module), name)


def _factory_name(factory):
    if callable(factory):
        return factory.__module__ + ":" + factory.__qualname__
    return factory or DEFAULT_FACTORY


def event_clock_inputs(model, initial_age, endpoint_times, action_cursor=0):
    """Match the executor's versioned feature clocks without changing legacy."""
    version = getattr(model, "event_clock_version", "legacy_relative_v1")
    endpoint_times = np.asarray(endpoint_times, dtype=float)
    if version == "native_elapsed_age_v2":
        return initial_age + endpoint_times, action_cursor * STEP_SECONDS + endpoint_times
    if version != "legacy_relative_v1":
        raise ValueError("unsupported native event feature clock: " + version)
    return initial_age, endpoint_times


def model_configuration(model):
    """Record resolved defaults as well as requested factory arguments."""
    names = ("kind", "max_hypotheses", "observation_repair", "maneuver_rate",
             "reset_std", "maneuver_retention", "event_clock_version", "model_version")
    result = {name: getattr(model, name) for name in names if hasattr(model, name)}
    law = getattr(model, "observation_law", None)
    if law is not None:
        result["observation_law_version"] = getattr(law, "version", type(law).__name__)
        result["observation_parameters"] = {name: getattr(law, name) for name in
            ("in_view", "out_view", "regime_prior") if hasattr(law, name)}
    return result


def planned_steps(query, horizon=3., max_actions=120):
    """Depends only on the legal query and a declared computation budget."""
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("positive finite horizon required")
    actions = query["compiled"]["actions"]
    poses = query["compiled"]["planned_poses"]
    if len(poses) != len(actions) + 1:
        raise ValueError("compiled query action/pose cardinality mismatch")
    return min(len(actions), int(math.floor((horizon + 1e-9) / STEP_SECONDS)), max_actions)


def target_masks(first, observed_steps, planned, termination):
    """Separate ascertainment from calibration eligibility and at-risk exposure."""
    first = np.asarray(first, dtype=int)
    if first.shape != (3,) or not 0 <= observed_steps <= planned:
        raise ValueError("invalid event target shape or observation extent")
    if np.any(first < -1) or np.any(first >= observed_steps):
        raise ValueError("first event lies outside observed prefix")
    contact = first[0] >= 0
    if contact and first[0] + 1 != observed_steps:
        raise ValueError("post-contact observations violate absorbing event law")
    if termination == "human_collision" and not contact and observed_steps < planned:
        raise ValueError("native collision termination has no matching contact label")
    complete_followup = observed_steps == planned or contact
    administrative = observed_steps < planned and not contact
    # Early events are logically known positives under censoring, but including
    # only those censored labels in Brier fitting would select on the outcome.
    label_known = (first >= 0) | complete_followup
    calibrate = np.full(3, complete_followup, dtype=bool)
    return label_known, calibrate, bool(administrative)


def _cohort_rows(snapshot, split, cohort):
    if split not in ("train", "dev"):
        raise ValueError("repair permits train/dev only; holdout is prohibited")
    available = rows(snapshot, split)
    if cohort is None:
        return available
    value = json.loads(Path(cohort).read_text()) if isinstance(cohort, (str, Path)) else cohort
    selected = value.get("rows", value.get("root_ids")) if isinstance(value, dict) else value
    if selected is None:
        raise ValueError("cohort must contain rows or root_ids")
    ids = {r["root_id"] if isinstance(r, dict) else r for r in selected}
    permitted = {r["root_id"] for r, _ in available}
    if not ids <= permitted:
        raise ValueError("cohort contains roots outside the permitted split")
    return [(r, p) for r, p in available if r["root_id"] in ids]


def prepare(snapshot, split, checkpoint, population, output, *, cohort=None,
            model_factory=None, model_kwargs=None, horizon=3., num_worlds=16, seed=31):
    """Prepare full planned features and separately censored native supervision."""
    torch.set_num_threads(1)
    source_hashes = _sources()
    output = _output_path(output)
    manifest_path = _output_path(str(output) + ".manifest.json")
    selected = _cohort_rows(snapshot, split, cohort)
    population_values = json.loads(Path(population).read_text())
    model = _factory(model_factory)(dynamics=LearnedDynamics.load(checkpoint),
        population_prior=population_values, **(model_kwargs or {}))
    records = []
    start = time.perf_counter()
    counts = collections.Counter()
    for root_index, (meta, folder) in enumerate(selected):
        data = root_data(folder, True)
        labels = read(folder / "privileged_step_labels.json.gz")
        legal = data["legal"]
        belief = model.initialize(legal["history"], legal["partial_graph"])
        steps = [planned_steps(q, horizon) for q in data["queries"]]
        maximum = max(steps, default=0)
        if maximum == 0:
            counts["zero_action_roots"] += 1
        if not steps:
            raise ValueError("root contains no candidate queries")
        times = np.arange(maximum + 1, dtype=float) * STEP_SECONDS
        future = model.predict_shared(belief, float(times[-1]), num_worlds,
                                      seed=seed, times=times)
        hidden = np.array([h.expected_hidden for h in future.worlds])
        total = hidden + np.array([h.n for h in future.worlds])
        allowed = np.stack([total > 0, total > 0, hidden > 0], axis=-1)
        age = np.mean([max(0., belief.timestamp - h.last_seen.mean())
                       if h.n else 0. for h in future.worlds])
        # This is deliberately the inherited offline proxy, not a true lineage.
        primary = max(belief.hypotheses, key=lambda h: h.weight)
        known = set()
        if primary.n and data["truth"].shape[1]:
            distance = np.linalg.norm(primary.mean[:, None, :2] - data["truth"][None, 0], axis=-1)
            ii, jj = linear_sum_assignment(distance)
            known = {int(j) for i, j in zip(ii, jj) if distance[i, j] < 1.5}
        root_records = []
        for query, outcome, planned in zip(data["queries"], data["outcomes"], steps):
            if planned == 0:
                counts["zero_action_queries"] += 1
            poses = np.asarray(query["compiled"]["planned_poses"][:planned + 1], dtype=float)
            feature_age, feature_elapsed = event_clock_inputs(model, age, times[:planned + 1])
            features = event_features(future.positions[:, :planned + 1],
                future.velocities[:, :planned + 1], poses, hidden, feature_age, feature_elapsed)[:, 1:]
            observed = min(len(outcome["node_ids"]), planned)
            events = np.zeros((observed, 3), dtype=bool)
            for j, node_id in enumerate(outcome["node_ids"][:observed]):
                label = labels[node_id]
                events[j, 0] = label["human_mesh_contact"]
                events[j, 1] = label["psc_violation"]
                for track in data["nodes"][node_id]["observation"]["tracks"]:
                    if not track["visible_now"] or not data["truth"].shape[1]:
                        continue
                    distances = np.linalg.norm(data["truth"][j + 1] - np.asarray(track["position"])[[0, 2]], axis=1)
                    identity = int(distances.argmin())
                    if distances[identity] < 1e-5 and identity not in known:
                        events[j, 2] = True
            first = np.array([int(np.flatnonzero(events[:, k])[0]) if events[:, k].any() else -1
                              for k in range(3)])
            termination = outcome["summary"]["termination"]
            known_labels, calibration, administrative = target_masks(first, observed, planned, termination)
            actual_nodes = [data["nodes"][nid]["observation"] for nid in outcome["node_ids"][:observed]]
            actual_poses = np.asarray([belief.robot_pose, *[row["robot_pose"] for row in actual_nodes]], dtype=float)
            actual_elapsed = float(actual_nodes[-1]["timestamp"] - belief.timestamp) if observed else 0.
            planned_progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1))]
            actual_progress = float(np.linalg.norm(np.diff(actual_poses[:, :3], axis=0), axis=1).sum())
            record = {"schema": SCHEMA, "split": split, "root_id": meta["root_id"],
                "scene_id": meta["scene_id"], "query_id": query["query_id"],
                "x": features.astype(np.float32), "allowed": allowed.copy(),
                "weights": future.weights.copy(), "unresolved": float(future.unresolved_mass),
                "first": first, "planned_steps": planned, "observed_steps": observed,
                "planned_seconds": float(times[planned]), "actual_observed_seconds": actual_elapsed,
                "planned_elapsed": times[:planned + 1].copy(),
                "planned_cumulative_progress": planned_progress,
                "actual_progress": actual_progress,
                "query_actions_total": len(query["compiled"]["actions"]),
                "original_candidate_queries": len(data["queries"]),
                "planned_query_complete": planned == len(query["compiled"]["actions"]),
                "label_known": known_labels, "calibration_eligible": calibration,
                "administrative_censoring": administrative, "termination": termination,
                "root_weight": 0.}
            validate_record(record)
            root_records.append(record)
            counts["queries"] += 1
            counts["planned_endpoints"] += planned
            counts["observed_endpoints"] += observed
            counts["administrative_censoring"] += administrative
            counts["early_contact"] += bool(first[0] >= 0 and observed < planned)
        for record in root_records:
            record["root_weight"] = 1. / len(root_records)
        records.extend(root_records)
        if (root_index + 1) % 100 == 0:
            print(json.dumps({"split": split, "roots": root_index + 1,
                "queries": len(records), "elapsed_s": round(time.perf_counter() - start, 2)}), flush=True)
    if _sources() != source_hashes:
        raise RuntimeError("runtime sources changed during preparation; cache not published")
    _publish(output, lambda stream: np.save(stream, np.array(records, dtype=object), allow_pickle=True))
    metadata = {"schema": SCHEMA, "split": split, "roots": len(selected),
        "scenes": len({r["scene_id"] for r, _ in selected}), "counts": dict(counts),
        "checkpoint_sha256": _sha(checkpoint), "population_sha256": _sha(population),
        "snapshot_manifest_sha256": _sha(Path(snapshot) / "manifest.json"),
        "cache_sha256": _sha(output), "source_sha256": source_hashes,
        "model_factory": _factory_name(model_factory),
        "model_kwargs": model_kwargs or {}, "model_configuration": model_configuration(model),
        "seed": seed, "worlds": num_worlds,
        "horizon_seconds": horizon, "endpoint_clock": "legal nominal 0.048s action schedule; native actual timestamps retained as diagnostics",
        "event_clock_version": getattr(model, "event_clock_version", "legacy_relative_v1"),
        "feature_semantics": {"tracked_count": "legacy name: finite sampled spatial-person count /4, including private hidden paths when supplied by repaired dynamics",
                              "hidden_count": "remaining unlocalized count /4; separate covariate, not a second population increment"},
        "appearance_target": "legacy <=1.5m root-track matching proxy, not an exact ancestry or observation-channel target",
        "calibration_estimand": "complete planned exposure or absorbing contact; all administrative truncations excluded",
        "elapsed_s": time.perf_counter() - start, "threads": 1}
    _json(manifest_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def validate_record(record):
    x = np.asarray(record["x"])
    planned = int(record["planned_steps"])
    observed = int(record["observed_steps"])
    if x.ndim != 3 or x.shape[1:] != (planned, len(FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError("invalid full-planned feature array")
    weights = np.asarray(record["weights"])
    unresolved = float(record["unresolved"])
    if weights.shape != (len(x),) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("invalid world weights")
    if not 0 <= unresolved <= 1 or abs(weights.sum() + unresolved - 1) > 1e-6:
        raise ValueError("world/residual probability mass mismatch")
    if np.asarray(record["allowed"]).shape != (len(x), 3):
        raise ValueError("invalid structural event mask")
    masks = target_masks(record["first"], observed, planned, record["termination"])
    if not np.array_equal(masks[0], record["label_known"]) or not np.array_equal(masks[1], record["calibration_eligible"]):
        raise ValueError("label/calibration eligibility mismatch")
    return record


def load_cache(path, expected_split):
    metadata = json.loads(Path(str(path) + ".manifest.json").read_text())
    if expected_split not in ("train", "dev") or metadata.get("split") != expected_split or metadata.get("schema") != SCHEMA:
        raise ValueError("cache split/schema mismatch; old holdout is prohibited")
    if _sha(path) != metadata["cache_sha256"]:
        raise ValueError("cache checksum mismatch")
    records = np.load(path, allow_pickle=True).tolist()
    for record in records:
        if record.get("split") != expected_split or record.get("schema") != SCHEMA:
            raise ValueError("record split/schema mismatch")
        validate_record(record)
    return records, metadata


def batch(records):
    count = len(records)
    if not count:
        raise ValueError("empty batch")
    worlds = max(len(r["weights"]) for r in records)
    observed_max = max(1, max(r["observed_steps"] for r in records))
    x = np.zeros((count, worlds, observed_max, len(FEATURE_NAMES)), np.float32)
    mask = np.zeros((count, observed_max, 3), np.float32)
    first = np.zeros_like(mask)
    weights = np.zeros((count, worlds), np.float64)
    allowed = np.zeros((count, worlds, 3), np.float32)
    rw = np.zeros(count, np.float64)
    unresolved = np.zeros(count, np.float64)
    for i, record in enumerate(records):
        n = record["observed_steps"]
        w = len(record["weights"])
        x[i, :w, :n] = record["x"][:, :n]
        weights[i, :w] = record["weights"]
        allowed[i, :w] = record["allowed"]
        rw[i] = record["root_weight"]
        unresolved[i] = record["unresolved"]
        for k, at in enumerate(record["first"]):
            mask[i, :at + 1 if at >= 0 else n, k] = 1
            if at >= 0:
                first[i, at, k] = 1
    return [torch.from_numpy(v) for v in (x, mask, first, weights, allowed, rw, unresolved)]


def likelihood(coefficients, tensors):
    """Joint first-event likelihood marginalized over common latent worlds."""
    x, mask, y, weights, allowed, rw, unresolved = tensors
    logits = (x @ coefficients.T).clamp(-30., 30.)  # Same native sigmoid guard.
    supported = allowed[:, :, None, :] > 0
    log_event = torch.where(supported, torch.nn.functional.logsigmoid(logits),
                            torch.full_like(logits, -torch.inf))
    log_survival = torch.where(supported, torch.nn.functional.logsigmoid(-logits),
                               torch.zeros_like(logits))
    # where avoids 0*(-inf): an impossible positive remains EXACTLY impossible,
    # while an impossible event correctly gives unit non-event probability.
    logs = torch.where(y[:, None] > 0, log_event, log_survival)
    conditional = torch.where(mask[:, None] > 0, logs, torch.zeros_like(logs)).sum((2, 3))
    negative_infinity = torch.full_like(weights, -torch.inf)
    logweights = torch.where(weights > 0, weights.log(), negative_infinity)
    joint = conditional + logweights
    possible = torch.isfinite(joint).any(-1)
    # An all-impossible retained mixture has zero derivative. Avoid the NaN
    # softmax derivative of logsumexp([-inf,...]) when residual mass explains it.
    safe_joint = torch.where(possible[:, None], joint, torch.zeros_like(joint))
    retained = torch.where(possible, torch.logsumexp(safe_joint, -1), torch.full_like(unresolved, -torch.inf))
    residual_log = ((y * math.log(RESIDUAL_HAZARD) + (1 - y) * math.log1p(-RESIDUAL_HAZARD)) * mask).sum((1, 2))
    residual_log = residual_log + torch.where(unresolved > 0, unresolved.log(), torch.full_like(unresolved, -torch.inf))
    mixture = torch.logaddexp(retained, residual_log)
    return -(mixture * rw).sum() / rw.sum()


def probabilities(coefficients, record):
    """Full-planned probabilities of each event before absorbing contact."""
    x = np.asarray(record["x"], dtype=float)
    hazards = expit(np.clip(x @ np.asarray(coefficients).T, -30., 30.)) * np.asarray(record["allowed"])[:, None, :]
    survival_before = np.concatenate([np.ones((len(x), 1)), np.cumprod(1 - hazards[:, :, 0], axis=1)[:, :-1]], axis=1)
    event_before = np.concatenate([np.ones((len(x), 1, 3)), np.cumprod(1 - hazards, axis=1)[:, :-1]], axis=1)
    first = survival_before[:, :, None] * event_before * hazards
    first[:, :, 0] = survival_before * hazards[:, :, 0]
    result = np.asarray(record["weights"]) @ first.sum(axis=1)
    n = record["planned_steps"]
    residual_first = (1 - RESIDUAL_HAZARD) ** np.arange(n) * RESIDUAL_HAZARD
    residual_competing = (1 - RESIDUAL_HAZARD) ** (2 * np.arange(n)) * RESIDUAL_HAZARD
    result += record["unresolved"] * np.array([residual_first.sum(), residual_competing.sum(), residual_competing.sum()])
    return result


def _root_groups(records):
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["root_id"]].append(record)
    return list(groups.values())


def _nll(coefficients, groups, batch_roots=4):
    numerator = denominator = 0.
    with torch.no_grad():
        for i in range(0, len(groups), batch_roots):
            current = [r for group in groups[i:i + batch_roots] for r in group]
            tensors = batch(current)
            mass = float(tensors[5].sum())
            numerator += float(likelihood(coefficients, tensors)) * mass
            denominator += mass
    return numerator / denominator


def _calibration_weights(records, head):
    eligible = [r for r in records if r["calibration_eligible"][head]]
    counts = collections.Counter(r["root_id"] for r in eligible)
    return eligible, np.array([1. / counts[r["root_id"]] for r in eligible])


def _calibrate(coefficients, records):
    coefficients = coefficients.copy()
    offsets = []
    diagnostics = []
    for head in range(3):
        eligible, weights = _calibration_weights(records, head)
        if not eligible:
            offsets.append(0.)
            diagnostics.append({"event": EVENT_NAMES[head], "status": "NO_COMPLETE_FOLLOWUP"})
            continue
        pieces = []
        for r in eligible:
            x = np.asarray(r["x"], dtype=float)
            base = x @ coefficients[head]
            if head:
                hc = expit(np.clip(x @ coefficients[0], -30., 30.)) * r["allowed"][:, None, 0]
                before = np.c_[np.ones(len(x)), np.cumprod(1 - hc, axis=1)[:, :-1]]
            else:
                before = None
            residual_power = (2 if head else 1) * np.arange(r["planned_steps"])
            residual = r["unresolved"] * float(np.sum((1 - RESIDUAL_HAZARD) ** residual_power * RESIDUAL_HAZARD))
            pieces.append((base, r["allowed"][:, None, head], before, r["weights"], residual, float(r["first"][head] >= 0)))

        def objective(offset):
            errors = []
            for base, allowed, contact_before, world_weights, residual, target in pieces:
                hazard = expit(np.clip(base + offset, -30., 30.)) * allowed
                before = np.c_[np.ones(len(hazard)), np.cumprod(1 - hazard, axis=1)[:, :-1]]
                first = before * hazard
                if contact_before is not None:
                    first *= contact_before
                probability = float(world_weights @ first.sum(1)) + residual
                errors.append((probability - target) ** 2)
            return float(weights @ errors / weights.sum())

        initial = objective(0.)
        found = minimize_scalar(objective, bounds=(-2., 2.), method="bounded", options={"xatol": .01})
        offset = float(found.x) if found.fun < initial else 0.
        coefficients[head, 0] += offset
        offsets.append(offset)
        diagnostics.append({"event": EVENT_NAMES[head], "queries": len(eligible),
            "roots": len({r["root_id"] for r in eligible}), "prefit_brier": initial,
            "postfit_brier": objective(offset), "offset": offset,
            "excluded_administrative_queries": len(records) - len(eligible),
            "estimand": "root-balanced complete-followup planned-horizon Brier"})
    return coefficients, offsets, diagnostics


def fit(train, dev, output, *, epochs=10, initial_checkpoint=None,
        calibrate_only=False, batch_roots=4):
    """At most ten epochs of the 36-parameter event head; separate prefit artifact."""
    if not 0 <= epochs <= 10 or not 1 <= batch_roots <= 8:
        raise ValueError("bounded repair requires epochs<=10 and batch_roots<=8")
    torch.set_num_threads(1)
    torch.manual_seed(59)
    output = _output_path(output)
    prefit_path = _output_path(output.with_name(output.stem + "_prefit.json"))
    train_records, train_metadata = load_cache(train, "train")
    dev_records, dev_metadata = load_cache(dev, "dev")
    if not train_records or not dev_records:
        raise ValueError("nonempty train and development caches required")
    for key in ("checkpoint_sha256", "population_sha256", "snapshot_manifest_sha256",
                "model_factory", "model_kwargs", "model_configuration", "horizon_seconds", "worlds", "seed",
                "endpoint_clock", "event_clock_version", "source_sha256"):
        if train_metadata[key] != dev_metadata[key]:
            raise ValueError("train/dev feature-generation contract mismatch: " + key)
    if {r["root_id"] for r in train_records} & {r["root_id"] for r in dev_records}:
        raise ValueError("root leakage between train and development caches")
    if {r["scene_id"] for r in train_records} & {r["scene_id"] for r in dev_records}:
        raise ValueError("scene leakage between train and development caches")
    initial = NativeEventModel().coefficients
    if initial_checkpoint:
        initial = np.asarray(json.loads(Path(initial_checkpoint).read_text())["coefficients"], dtype=float)
    if initial.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(initial).all():
        raise ValueError("invalid initial event coefficients")
    coefficients = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
    optimizer = torch.optim.Adam([coefficients], lr=.025)
    train_groups = _root_groups(train_records)
    dev_groups = _root_groups(dev_records)
    rng = np.random.default_rng(59)
    history = []
    best = float("inf")
    best_coefficients = initial.copy()
    started = time.perf_counter()
    for epoch in range((0 if calibrate_only else epochs) + 1):
        training = []
        if epoch:
            permutation = rng.permutation(len(train_groups))
            for j in range(0, len(permutation), batch_roots):
                current = [r for i in permutation[j:j + batch_roots] for r in train_groups[i]]
                optimizer.zero_grad()
                loss = likelihood(coefficients, batch(current)) + .0005 * coefficients[:, 1:].square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite repair event likelihood")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([coefficients], 10.)
                optimizer.step()
                training.append(float(loss.detach()))
        dev_loss = _nll(coefficients, dev_groups, batch_roots)
        row = {"epoch": epoch, "train_objective": float(np.mean(training)) if training else None,
               "dev_censored_nll": dev_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_loss < best:
            best = dev_loss
            best_coefficients = coefficients.detach().numpy().copy()
    common = {"schema": SCHEMA, "features": FEATURE_NAMES, "history": history,
        "training_queries": len(train_records), "dev_queries": len(dev_records),
        "training_roots": len(train_groups), "dev_roots": len(dev_groups),
        "seed": 59, "threads": 1, "source_sha256": _sources(),
        "train_cache_sha256": train_metadata["cache_sha256"],
        "dev_cache_sha256": dev_metadata["cache_sha256"],
        "initial_checkpoint_sha256": _sha(initial_checkpoint) if initial_checkpoint else None,
        "calibrate_only": calibrate_only, "appearance_target": train_metadata["appearance_target"],
        "feature_semantics": train_metadata.get("feature_semantics", {}),
        "event_clock_version": train_metadata["event_clock_version"],
        "semantics": "root-balanced joint first-event mixture NLL; observed-prefix censoring; full-planned competing-contact endpoint calibration",
        "censoring_limitation": "Brier calibration targets complete-followup queries; no claim of IPCW correction for arbitrary informative administrative censoring"}
    _json(prefit_path, {**common, "coefficients": best_coefficients.tolist(),
                       "selected_dev_censored_nll": best, "calibrated": False})
    final, offsets, calibration = _calibrate(best_coefficients, dev_records)
    value = {**common, "coefficients": final.tolist(), "intercept_offsets": offsets,
        "calibration": calibration, "prefit_sha256": _sha(prefit_path),
        "selected_dev_censored_nll": best,
        "final_dev_censored_nll": _nll(torch.tensor(final, dtype=torch.float32), dev_groups, batch_roots),
        "elapsed_s": time.perf_counter() - started, "calibrated": True}
    _json(output, value)
    print(json.dumps({"output": str(output), "prefit": str(prefit_path), "calibration": calibration}, indent=2), flush=True)
    return value


def selftest():
    """Executable stopping-time, censoring and mixture-likelihood oracles."""
    def record(planned=4, observed=1, first=(0, -1, -1), termination="human_collision", root="r"):
        known, eligible, administrative = target_masks(first, observed, planned, termination)
        x = np.zeros((2, planned, len(FEATURE_NAMES)), np.float32)
        x[:, :, 0] = 1.
        return {"x": x, "weights": np.array([.25, .75]), "unresolved": 0.,
            "allowed": np.ones((2, 3), bool), "first": np.array(first),
            "planned_steps": planned, "observed_steps": observed,
            "termination": termination, "label_known": known,
            "calibration_eligible": eligible, "administrative_censoring": administrative,
            "root_id": root, "root_weight": 1.}

    class RepairTests(unittest.TestCase):
        def test_early_contact_does_not_shorten_probability_horizon(self):
            r = record()
            c = np.zeros((3, len(FEATURE_NAMES)))
            c[:, 0] = np.log(np.array([.2, .1, .3]) / (1 - np.array([.2, .1, .3])))
            p = probabilities(c, r)
            self.assertAlmostEqual(p[0], 1 - .8 ** 4)
            self.assertAlmostEqual(p[1], .1 * sum(.72 ** i for i in range(4)))
            self.assertAlmostEqual(p[2], .3 * sum(.56 ** i for i in range(4)))
            self.assertTrue(r["calibration_eligible"].all())

        def test_administrative_positive_is_not_selected_for_brier(self):
            r = record(observed=2, first=(-1, 0, -1), termination="route_invalidated")
            self.assertEqual(r["label_known"].tolist(), [False, True, False])
            self.assertFalse(r["calibration_eligible"].any())

        def test_censored_likelihood_ignores_unobserved_suffix(self):
            r = record(observed=2, first=(-1, -1, -1), termination="route_invalidated")
            c = torch.ones((3, len(FEATURE_NAMES))) * .1
            before = float(likelihood(c, batch([r])))
            r["x"][:, 2:] = 100.
            self.assertEqual(float(likelihood(c, batch([r]))), before)
            expected = -6 * math.log1p(-float(torch.sigmoid(torch.tensor(.1))))
            self.assertAlmostEqual(before, expected, places=5)

        def test_joint_world_mixture_not_independent_event_marginals(self):
            r = record()
            r["x"][1, :, 1] = 1.
            c = np.zeros((3, len(FEATURE_NAMES)))
            c[:, 0] = [-1., -.5, -.2]
            c[:, 1] = [2., 1., -.8]
            hz = expit(r["x"][:, 0] @ c.T)
            expected = -math.log(float(r["weights"] @ (hz[:, 0] * (1 - hz[:, 1]) * (1 - hz[:, 2]))))
            actual = float(likelihood(torch.tensor(c, dtype=torch.float32), batch([r])))
            self.assertAlmostEqual(actual, expected, places=6)

        def test_native_terminal_endpoint_includes_psc(self):
            r = record(first=(0, 0, -1))
            c = torch.zeros((3, len(FEATURE_NAMES)))
            self.assertAlmostEqual(float(likelihood(c, batch([r]))), -math.log(.125), places=6)

        def test_eligible_root_weights_are_renormalized(self):
            records = [record(root="a"), record(root="a"), record(root="b"),
                record(observed=1, first=(-1, -1, -1), termination="route_invalidated", root="b")]
            eligible, weights = _calibration_weights(records, 0)
            self.assertEqual(len(eligible), 3)
            np.testing.assert_allclose(weights, [.5, .5, 1.])

        def test_structural_zero_and_residual_mass(self):
            r = record()
            r["allowed"][:] = False
            r["weights"] *= .8
            r["unresolved"] = .2
            p = probabilities(np.zeros((3, len(FEATURE_NAMES))), r)
            self.assertAlmostEqual(p[0], .2 * (1 - .995 ** 4))
            self.assertGreater(p[1], 0.)
            self.assertLess(p[1], p[0])

        def test_structurally_impossible_positive_has_zero_likelihood(self):
            r = record()
            r["allowed"][:] = False
            c = torch.zeros((3, len(FEATURE_NAMES)))
            self.assertTrue(torch.isinf(likelihood(c, batch([r]))))
            no_event = record(observed=4, first=(-1, -1, -1), termination="query_complete")
            no_event["allowed"][:] = False
            self.assertAlmostEqual(float(likelihood(c, batch([no_event]))), 0., places=12)

        def test_residual_only_event_has_finite_zero_parameter_gradient(self):
            r = record()
            r["allowed"][:] = False
            r["weights"] *= .8
            r["unresolved"] = .2
            c = torch.zeros((3, len(FEATURE_NAMES)), requires_grad=True)
            loss = likelihood(c, batch([r]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(c.grad).all())
            self.assertEqual(float(c.grad.abs().max()), 0.)

        def test_tiny_supported_world_mass_is_not_clamped_or_deleted(self):
            r = record()
            r["weights"] = np.array([1e-100, 1.])
            r["allowed"][1] = False
            c = torch.zeros((3, len(FEATURE_NAMES)))
            value = float(likelihood(c, batch([r])))
            self.assertAlmostEqual(value, -math.log(1e-100 * .125), places=6)

        def test_split_guard_and_horizon_independent_of_outcomes(self):
            with self.assertRaises(ValueError):
                _cohort_rows("unopened", "holdout", None)
            q = {"compiled": {"actions": ["forward"] * 100, "planned_poses": [[0, 0, 0, 0]] * 101}}
            self.assertEqual(planned_steps(q), 62)
            with self.assertRaises(ValueError):
                target_masks([0, -1, -1], 3, 4, "human_collision")

        def test_repaired_clock_composes_across_query_cursor(self):
            class Repaired:
                event_clock_version = "native_elapsed_age_v2"
            first_age, first_elapsed = event_clock_inputs(Repaired(), 1., np.array([.048, .096]))
            next_age, next_elapsed = event_clock_inputs(Repaired(), 1.048, np.array([.048]), 1)
            self.assertAlmostEqual(first_age[-1], next_age[0])
            self.assertAlmostEqual(first_elapsed[-1], next_elapsed[0])
            legacy_age, legacy_elapsed = event_clock_inputs(object(), 1., np.array([.048]), 1)
            self.assertEqual(legacy_age, 1.)
            self.assertAlmostEqual(legacy_elapsed[0], .048)

    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("snapshot", "split", "checkpoint", "population", "output"):
        p.add_argument(name)
    p.add_argument("--cohort")
    p.add_argument("--model-factory", default=DEFAULT_FACTORY)
    p.add_argument("--model-kwargs", type=json.loads, default={})
    p.add_argument("--horizon", type=float, default=3.)
    p.add_argument("--num-worlds", type=int, default=16)
    p.add_argument("--seed", type=int, default=31)
    f = commands.add_parser("fit")
    for name in ("train", "dev", "output"):
        f.add_argument(name)
    f.add_argument("--epochs", type=int, default=10)
    f.add_argument("--initial-checkpoint")
    f.add_argument("--calibrate-only", action="store_true")
    f.add_argument("--batch-roots", type=int, default=4)
    commands.add_parser("selftest")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "fit": fit, "selftest": selftest}[command](**args)


if __name__ == "__main__":
    main()
