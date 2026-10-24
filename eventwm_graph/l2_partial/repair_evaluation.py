"""Dev-only population, fixed-ancestry and conditioning repair diagnostics.

The diagnostic Hypothesis subclass carries offline identity assignments through
dataclasses.replace.  No online feature or decision reads those tags.  Assignment
is solved ONCE, on unsampled root means; exact root detections are anchored first,
and remaining aged tracks have a separately reported <=2m oracle alignment.
No forecast world or posterior is relabeled using a future target.  Standard
unlabeled set GOSPA is reported separately because its endpoint assignment is
part of that metric, not a correction to the predictive distribution.

Only train/dev metadata and the fixed development payloads may be read.  This
module rejects holdout cohorts and never replaces existing research artifacts.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import time
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .data import root_data
from .evaluate import truth_at, visible_ids, weighted_energy
from .neural import LearnedDynamics
from .queries import NativeEventModel, event_features
from .state import Hypothesis
from .repair_events import (_cohort_rows, _factory, _factory_name, _json,
                            _output_path, _sha, _sources, event_clock_inputs, model_configuration)


SCHEMA = "partial_graph_population_repair_evaluation_1"
HORIZONS = (.5, 1., 2., 3.)
CONDITIONING_HORIZONS = (2., 1.)  # Primary first; secondary support diagnostic.
CONDITIONING_MARGIN_M = .020
DEFAULT_VARIANTS = {
    "legacy": {"factory": "eventwm_graph.l2_partial.model:GraphBeliefWorldModel", "kwargs": {}},
    "association_only": {"factory": "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel",
                         "kwargs": {"observation_repair": False, "maneuver_rate": 2., "reset_std": 2., "maneuver_retention": 1.}},
    "full": {"factory": "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel",
             "kwargs": {"observation_repair": True, "maneuver_rate": 2., "reset_std": 2., "maneuver_retention": 1.}},
    "full_no_maneuver": {"factory": "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel",
             "kwargs": {"observation_repair": True, "maneuver_rate": 0., "reset_std": 2., "maneuver_retention": 1.}},
    "full_kinematic": {"factory": "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel",
             "kwargs": {"observation_repair": True, "maneuver_rate": 2., "reset_std": 2., "maneuver_retention": 1.},
             "use_learned": False},
}


@dataclass
class DiagnosticHypothesis(Hypothesis):
    # Offline scoring metadata; original online input arrays remain unchanged.
    diagnostic_origin: int = -1
    diagnostic_ancestry: tuple = ()
    diagnostic_exact: tuple = ()
    diagnostic_aged: tuple = ()


def _metric_sources():
    return {**_sources(), "repair_evaluation.py": _sha(__file__),
            "evaluate.py": _sha(Path(__file__).with_name("evaluate.py")),
            "data.py": _sha(Path(__file__).with_name("data.py"))}


def _finite_log(probability):
    return float(-math.log(probability)) if probability > 0 else None


def population_score(belief, true_count):
    """Resolved count score and rigorous bounds for unspecified residual mass.

    The unresolved event law does not define a count distribution. It would be
    false precision to assign that mass to the true count or a guessed prior.
    P(N=true) is therefore bounded by [resolved mass, resolved mass + residual].
    Conditional resolved metrics and unsupported cases are always visible.
    """
    size = max([true_count + 1] + [h.n + len(h.hidden_pmf) for h in belief.hypotheses])
    pmf = np.zeros(size)
    localized = hidden = 0.
    for h in belief.hypotheses:
        pmf[h.n:h.n + len(h.hidden_pmf)] += h.weight * h.hidden_pmf
        localized += h.weight * h.n
        hidden += h.weight * h.expected_hidden
    mass = float(pmf.sum())
    unresolved = float(belief.unresolved_mass)
    if abs(mass + unresolved - 1) > 1e-6:
        raise ValueError("population probability accounting mismatch")
    probability = float(pmf[true_count])
    mean = float(pmf @ np.arange(len(pmf)) / mass) if mass else None
    return {"true_count": int(true_count), "unresolved_mass": unresolved,
        "resolved_count_pmf": pmf.tolist(), "resolved_mass": mass,
        "true_count_probability_lower": probability,
        "true_count_probability_upper": min(1., probability + unresolved),
        "count_nll_lower": _finite_log(min(1., probability + unresolved)),
        "count_nll_upper": _finite_log(probability),
        "count_nll_upper_unbounded": probability == 0.,
        "count_nll_lower_unbounded": probability + unresolved == 0.,
        "count_nll_resolved": _finite_log(probability / mass) if mass else None,
        "conditional_expected_count": mean,
        "count_absolute_error": abs(mean - true_count) if mean is not None else None,
        "conditional_localized_count": localized / mass if mass else None,
        "conditional_hidden_count": hidden / mass if mass else None,
        "truth_excluded_from_resolved_support": probability == 0.}


def gospa(prediction, target, cutoff=2.):
    """GOSPA p=2, alpha=2 with localization/false/missed decomposition."""
    a = np.asarray(prediction, float).reshape(-1, 2)
    b = np.asarray(target, float).reshape(-1, 2)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("GOSPA requires finite localized coordinates")
    matched = 0
    localization = 0.
    if len(a) and len(b):
        distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
        ii, jj = linear_sum_assignment(np.minimum(distance, cutoff) ** 2)
        keep = distance[ii, jj] < cutoff
        matched = int(keep.sum())
        localization = float(np.square(distance[ii[keep], jj[keep]]).sum())
    false = len(a) - matched
    missed = len(b) - matched
    cardinality = .5 * cutoff ** 2 * (false + missed)
    return {"gospa": math.sqrt(localization + cardinality),
            "localization_squared": localization,
            "false_localized": float(false), "missed_localized": float(missed),
            "cardinality_squared": cardinality}


def fixed_ancestry_cost(h, positions, target, target_ids, cutoff=2., known_only=False):
    """A labeled GOSPA diagnostic: identity switches cannot be rematched away."""
    mapping = dict(h.diagnostic_ancestry)
    target_ids = set(int(i) for i in target_ids)
    matched = set()
    localization = 0.
    false = 0
    for i, slot in enumerate(h.slot_ids):
        if known_only and int(slot) not in h.diagnostic_exact:
            continue
        identity = mapping.get(int(slot))
        if identity not in target_ids:
            # Unknown ancestry is not silently dropped from full-population loss.
            false += 1
            continue
        d = float(np.linalg.norm(positions[i] - target[identity]))
        if identity in matched or d >= cutoff:
            false += 1
        else:
            matched.add(identity)
            localization += d * d
    missed = len(target_ids - matched)
    cardinality = .5 * cutoff ** 2 * (false + missed)
    return {"gospa": math.sqrt(localization + cardinality),
            "localization_squared": localization,
            "false_localized": float(false), "missed_localized": float(missed),
            "cardinality_squared": cardinality}


def tag_root_ancestry(belief, root_truth, aged_cutoff=2.):
    """One-to-one offline root alignment; no future sample or target is read."""
    truth = np.asarray(root_truth)
    tagged = []
    for origin, h in enumerate(belief.hypotheses):
        mapping = {}
        exact = []
        aged = []
        available = set(range(len(truth)))
        visible = np.flatnonzero(np.isclose(h.last_seen, belief.timestamp, atol=1e-7, rtol=0))
        if len(visible) and len(truth):
            distance = np.linalg.norm(h.mean[visible, None, :2] - truth[None], axis=-1)
            ii, jj = linear_sum_assignment(distance)
            for i, j in zip(ii, jj):
                if distance[i, j] < 1e-5:
                    slot = int(h.slot_ids[visible[i]])
                    mapping[slot] = int(j)
                    exact.append(slot)
                    available.remove(int(j))
        candidates = [i for i, slot in enumerate(h.slot_ids) if int(slot) not in mapping]
        available = sorted(available)
        if candidates and available:
            distance = np.linalg.norm(h.mean[candidates, None, :2] - truth[None, available], axis=-1)
            ii, jj = linear_sum_assignment(distance)
            for i, j in zip(ii, jj):
                if distance[i, j] < aged_cutoff:
                    slot = int(h.slot_ids[candidates[i]])
                    mapping[slot] = int(available[j])
                    aged.append(slot)
        original = {f.name: getattr(h, f.name) for f in fields(Hypothesis)}
        tagged.append(DiagnosticHypothesis(**original, diagnostic_origin=origin,
            diagnostic_ancestry=tuple(sorted(mapping.items())),
            diagnostic_exact=tuple(exact), diagnostic_aged=tuple(aged)))
    result = belief.fork(hypotheses=tagged)
    if result.fingerprint() != belief.fingerprint():
        raise AssertionError("offline ancestry metadata changed runtime fingerprint")
    return result


def expected_set_scores(hypotheses, positions, weights, truth, known_ids=None):
    """Resolved mixture metric; unlocalized humans are missed spatial targets."""
    weights = np.asarray(weights, float)
    mass = float(weights.sum())
    if mass <= 0:
        return {"set_scored_mass": 0., "gospa": None, "ancestry_gospa": None}
    result = collections.defaultdict(float)
    for h, p, weight in zip(hypotheses, positions, weights / mass):
        for key, value in gospa(p[:h.n], truth).items():
            result[key] += weight * value
        for key, value in fixed_ancestry_cost(h, p[:h.n], truth, range(len(truth))).items():
            result["ancestry_" + key] += weight * value
        if known_ids is not None:
            for key, value in fixed_ancestry_cost(h, p[:h.n], truth, known_ids, known_only=True).items():
                result["known_ancestry_" + key] += weight * value
    return {**result, "set_scored_mass": mass}


def sampled_population_scores(future, index, truth):
    """Secondary metric over every finite sampled slot, including private hidden.

    The legacy variants have no spatial law for their remaining hidden counts.
    Their finite-slot score is therefore not a complete population score on
    such worlds. Coverage is explicit, and no replacement positions are drawn
    on their behalf. The comparable primary metric remains localized tracks
    against ALL truth targets, with missing hidden people penalized as misses.
    """
    mass = float(future.weights.sum())
    result = collections.defaultdict(float)
    complete_mass = complete_error = 0.
    if mass <= 0:
        return {"sampled_population_gospa": None, "sampled_population_scored_mass": 0.,
                "sampled_population_complete_mass": 0., "sampled_population_complete_conditional_gospa": None}
    for w, h in enumerate(future.worlds):
        if future.weights[w] <= 0:
            continue
        positions = future.positions[w, index]
        positions = positions[np.isfinite(positions).all(axis=-1)]
        values = gospa(positions, truth)
        for key, value in values.items():
            result["sampled_population_" + key] += float(future.weights[w] / mass) * value
        expected = h.n + int(round(h.expected_hidden))
        if len(positions) == expected:
            complete_mass += float(future.weights[w])
            complete_error += float(future.weights[w]) * values["gospa"]
    return {**result, "sampled_population_scored_mass": mass,
            "sampled_population_complete_mass": complete_mass,
            "sampled_population_complete_conditional_gospa": complete_error / complete_mass if complete_mass else None}


def ancestry_energy(future, target, target_ids, absolute_time):
    """Finite-distribution proper ES conditional on each represented ancestry.

    Targets are a fixed legal root-visible subset. Missing ancestry mass is
    measured per target and never repaired using future position matching.
    """
    index = int(np.argmin(abs(future.times - absolute_time)))
    if abs(float(future.times[index]) - absolute_time) > 1e-6:
        raise ValueError("ancestry score target timestamp mismatch")
    values = []
    masses = []
    coverage = []
    radii = []
    missing = 0
    for identity in target_ids:
        samples = []
        weights = []
        for w, h in enumerate(future.worlds):
            if future.weights[w] <= 0:
                continue
            if not isinstance(h, DiagnosticHypothesis):
                raise ValueError("positive forecast mass lost its fixed ancestry metadata")
            mapping = dict(h.diagnostic_ancestry)
            slots = [i for i, slot in enumerate(h.slot_ids) if mapping.get(int(slot)) == identity]
            if len(slots) == 1 and future.weights[w] > 0:
                samples.append(future.positions[w, index, slots[0]])
                weights.append(future.weights[w])
        mass = float(sum(weights))
        masses.append(mass)
        if not samples or mass <= 0:
            missing += 1
            continue
        p = np.asarray(samples)
        weight = np.asarray(weights) / mass
        y = target[identity]
        values.append(weighted_energy(p[:, None, :], y[None, :], weight))
        mean = weight @ p
        distance = np.linalg.norm(p - mean, axis=-1)
        order = np.argsort(distance)
        radius = distance[order[min(len(order) - 1, int(np.searchsorted(np.cumsum(weight[order]), .9)))]]
        coverage.append(float(np.linalg.norm(y - mean) <= radius))
        radii.append(float(radius))
    return {"ancestry_energy_score": float(np.mean(values)) if values and not missing else None,
        "supported_target_only_energy_score": float(np.mean(values)) if values else None,
        "ancestry_coverage90": float(np.mean(coverage)) if coverage else None,
        "ancestry_radius90": float(np.mean(radii)) if radii else None,
        "ancestry_scored_mass_mean": float(np.mean(masses)) if masses else 0.,
        "ancestry_scored_mass_min": float(min(masses)) if masses else 0.,
        "ancestry_target_count": len(target_ids), "ancestry_missing_targets": missing,
        "ancestry_all_targets_supported": bool(target_ids) and missing == 0}


def scene_ci(records, key, seed=73, replicates=1000):
    """Equal-scene mean; roots averaged first, siblings never independent."""
    roots = collections.defaultdict(list)
    for r in records:
        value = r.get(key)
        if value is not None and np.isfinite(value):
            roots[(r["scene_id"], r["root_id"])].append(float(value))
    scenes = collections.defaultdict(list)
    for (scene, _), values in roots.items():
        scenes[scene].append(float(np.mean(values)))
    values = np.array([np.mean(v) for v in scenes.values()])
    if not len(values):
        return {"mean": None, "lo": None, "hi": None,
                "one_sided_upper95": None, "roots": 0, "scenes": 0}
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(bootstrap, 2.5)),
        "hi": float(np.percentile(bootstrap, 97.5)),
        "one_sided_upper95": float(np.percentile(bootstrap, 95)),
        "roots": len(roots), "scenes": len(values), "estimand": "equal-scene root mean"}


def _signature(h):
    # The Gaussian process state is intentionally excluded: survival is projected
    # onto the same analytic component, retaining its continuous covariance.
    return (h.diagnostic_origin, tuple(h.slot_ids), tuple(np.asarray(h.intent).ravel()),
            tuple(np.asarray(h.mode).ravel()), h.visibility_mode)


def association_ambiguous(belief):
    """Exclude duplicate optical/motion hypotheses from association strata."""
    layouts = set()
    for h in belief.hypotheses:
        if h.weight > 1e-10:
            layouts.add((tuple(h.slot_ids), tuple(np.round(h.mean[:, :2].ravel(), 7)),
                         tuple(np.round(h.last_seen, 7))))
    return len(layouts) > 1


def legacy_ancestry_merge_risk(model, prior):
    """Fail closed where legacy moment merging can erase offline ancestry.

    Its merge key omits diagnostic metadata. Different parent assignments may
    consequently become one Gaussian carrying only its first parent's tags.
    We do not change that online operation merely to improve an offline score.
    This conservative eligibility guard retains all population/set diagnostics.
    """
    if getattr(model, "observation_repair", False):
        return False
    groups = collections.defaultdict(set)
    for h in prior.hypotheses:
        key = (tuple(h.slot_ids), tuple(np.asarray(h.mode).ravel()),
               tuple(np.round(np.asarray(h.intent).ravel(), 7)) if h.relaxation else ())
        groups[key].add(tuple(h.diagnostic_ancestry))
    return any(len(values) > 1 for values in groups.values())


def survival_conditioned_prior(model, belief, data, outcome, action_count=10):
    """Identical actual-path evidence for both arms, with NO synthetic sensor row.

    Same 16-world event law as query risk. Unrepresented quadrature mass follows
    the declared residual event law. Within retained mode/count components we
    retain analytic Gaussian moments: this is a documented survival projection,
    not exact survival-truncated filtering.
    """
    nodes = [data["nodes"][nid]["observation"] for nid in outcome["node_ids"][:action_count]]
    times = np.array([belief.timestamp] + [float(o["timestamp"]) for o in nodes])
    poses = np.array([belief.robot_pose] + [o["robot_pose"] for o in nodes])
    if len(nodes) != action_count or np.any(np.diff(times) <= 0):
        raise ValueError("conditioning requires a strictly timed actual action prefix")
    future = model.predict_shared(belief, times[-1] - belief.timestamp, 16,
                                  seed=91, times=times - belief.timestamp)
    hidden = np.array([h.expected_hidden for h in future.worlds])
    age = np.mean([max(0., belief.timestamp - h.last_seen.mean()) if h.n else 0.
                   for h in future.worlds])
    feature_age, feature_elapsed = event_clock_inputs(model, age, times - belief.timestamp)
    features = event_features(future.positions, future.velocities, poses, hidden,
                             feature_age, feature_elapsed)
    event = model.event_model or NativeEventModel()
    hazards = event.hazards(features)[:, 1:, 0]
    total = hidden + np.array([h.n for h in future.worlds])
    hazards[total == 0] = 0.
    survival = np.prod(1 - hazards, axis=1)
    prior = model.propagate(belief, float(times[-1]))
    components = {_signature(h): h for h in prior.hypotheses}
    count_mass = {}
    for h, weight, s in zip(future.worlds, future.weights, survival):
        if weight <= 0:
            continue
        key = _signature(h)
        if key not in components:
            raise ValueError("survival projection lost an analytic ancestor component")
        if key not in count_mass:
            count_mass[key] = np.zeros_like(components[key].hidden_pmf)
        count_mass[key] += float(weight * s) * h.hidden_pmf
    residual = float(future.unresolved_mass * .995 ** action_count)
    normalizer = residual + sum(float(p.sum()) for p in count_mass.values())
    if normalizer <= 0:
        raise ValueError("impossible survival observation under entire event law")
    hypotheses = []
    for key, pmf in count_mass.items():
        mass = float(pmf.sum())
        if mass:
            hypotheses.append(components[key].fork(weight=mass / normalizer, hidden_pmf=pmf / mass))
    evidence = dict(prior.evidence)
    evidence["evaluation_survival_projection"] = "native conditional hazard; analytic component moments; quadrature count/mode support"
    prior = prior.fork(hypotheses=hypotheses, unresolved_mass=residual / normalizer,
                       robot_pose=poses[-1], evidence=evidence).validate()
    return prior, nodes[-1], {"survival_probability": normalizer,
        "quadrature_unresolved_before_survival": float(future.unresolved_mass),
        "survival_projection": evidence["evaluation_survival_projection"]}


def _exact_visible_identity_set(observation, truth):
    """Offline diagnostic only; reject nonunique or unmatched exact centers."""
    detections = [np.asarray(row["position"])[[0, 2]] for row in observation["tracks"] if row["visible_now"]]
    if not detections:
        return set()
    if truth is None or not len(truth):
        return None
    distance = np.linalg.norm(np.asarray(detections)[:, None] - np.asarray(truth)[None], axis=-1)
    matches = distance < 1e-5
    if not np.all(matches.sum(1) == 1) or np.any(matches.sum(0) > 1):
        return None
    return set(np.flatnonzero(matches.any(0)).tolist())


def physical_reappearance(data, outcome, action_count=10):
    """Actual identities visible, missing, then visible within the root prefix.

    No pre-root privileged identity tape is opened. Previously seen people who
    were already invisible at the root are outside this diagnostic's scope.
    Missing detections classify visibility events; they do not certify free space.
    """
    sequence = [_exact_visible_identity_set(data["legal"]["history"][-1], data["truth"][0])]
    for node_id in outcome["node_ids"][:action_count]:
        obs = data["nodes"][node_id]["observation"]
        sequence.append(_exact_visible_identity_set(obs, truth_at(data, float(obs["timestamp"]))))
    supported = len(sequence) == action_count + 1 and all(ids is not None for ids in sequence)
    if supported:
        seen = set().union(*sequence[:-1])
        reappeared = (sequence[-1] & seen) - sequence[-2]
        first_seen = sequence[-1] - seen
    else:
        reappeared, first_seen = set(), set()
    return {"reappearance_physical_supported": supported,
        "reappearance_physical": bool(reappeared), "reappearance_physical_ids": sorted(reappeared),
        "first_seen_since_root_physical_ids": sorted(first_seen),
        "physical_visibility_scope": "exact offline identities over root plus10 native arrived endpoints; ambiguous/unmatched centers unsupported"}


def conditioning_pair(model, belief, data, target_ids, target_horizon=2.):
    if not np.isfinite(target_horizon) or target_horizon <= 0:
        raise ValueError("positive finite absolute conditioning target horizon required")
    index = max(range(len(data["queries"])), key=lambda i:
                (len(data["queries"][i]["compiled"]["actions"]), data["queries"][i]["query_id"]))
    outcome = data["outcomes"][index]
    target_time = belief.timestamp + float(target_horizon)
    if len(outcome["node_ids"]) <= 10 or data["times"][-1] < target_time - 1e-8:
        return None
    prior, observation, survival = survival_conditioned_prior(model, belief, data, outcome)
    if prior.timestamp >= target_time:
        return None
    ancestry_merge_risk = legacy_ancestry_merge_risk(model, prior)
    posterior = model.observe(prior, observation)
    duration = target_time - prior.timestamp
    a = model.predict_shared(prior, duration, 16, seed=107, times=[0., duration])
    b = model.predict_shared(posterior, duration, 16, seed=107, times=[0., duration])
    target = truth_at(data, target_time)
    sa = ancestry_energy(a, target, target_ids, target_time)
    sb = ancestry_energy(b, target, target_ids, target_time)
    if ancestry_merge_risk:
        sb = {**sb, "ancestry_energy_score": None,
              "supported_target_only_energy_score": None,
              "ancestry_all_targets_supported": False,
              "legacy_merged_ancestry_unscorable": True}
    full_a = expected_set_scores(a.worlds, a.positions[:, -1], a.weights, target, target_ids)
    full_b = expected_set_scores(b.worlds, b.positions[:, -1], b.weights, target, target_ids)
    sampled_a = sampled_population_scores(a, -1, target)
    sampled_b = sampled_population_scores(b, -1, target)
    supported = (sa["ancestry_all_targets_supported"] and sb["ancestry_all_targets_supported"]
                 and not ancestry_merge_risk)
    visible = [r for r in observation["tracks"] if r["visible_now"]]
    current_truth = truth_at(data, prior.timestamp)
    informative = any(np.min(np.linalg.norm(current_truth[target_ids] - np.asarray(r["position"])[[0, 2]], axis=-1)) < 1e-5
                      for r in visible) if target_ids else False
    root_visible = {r["track_id"] for r in data["legal"]["history"][-1]["tracks"] if r["visible_now"]}
    reappearance = any(r["track_id"] not in root_visible for r in visible)
    aged = any(np.any(prior.timestamp - h.last_seen > .5) for h in prior.hypotheses if h.n)
    count_a = population_score(prior, data["truth"].shape[1])
    count_b = population_score(posterior, data["truth"].shape[1])
    return {"observation_time": prior.timestamp - belief.timestamp,
        "target_time": float(target_horizon), "target_absolute_time": target_time,
        "conditioning_role": "primary" if target_horizon == 2. else "secondary",
        "prior": sa["ancestry_energy_score"],
        "posterior": sb["ancestry_energy_score"] if not ancestry_merge_risk else None,
        "delta": sb["ancestry_energy_score"] - sa["ancestry_energy_score"] if supported else None,
        "unsupported_pair": not supported,
        "legacy_ancestry_merge_risk": ancestry_merge_risk,
        "prior_ancestry": sa, "posterior_ancestry": sb,
        "prior_population": count_a, "posterior_population": count_b,
        "prior_all_population_gospa": full_a["gospa"],
        "posterior_all_population_gospa": full_b["gospa"],
        "all_population_gospa_delta": full_b["gospa"] - full_a["gospa"] if full_a["gospa"] is not None and full_b["gospa"] is not None else None,
        "prior_sampled_population": sampled_a, "posterior_sampled_population": sampled_b,
        "sampled_population_gospa_delta": sampled_b["sampled_population_gospa"] - sampled_a["sampled_population_gospa"]
            if sampled_a["sampled_population_gospa"] is not None and sampled_b["sampled_population_gospa"] is not None else None,
        "positive": informative, "any_detection": bool(visible),
        "occluded_or_ambiguous": not visible, "reappearance_proxy": reappearance,
        "aged_tracks": aged, "dense": data["truth"].shape[1] >= 4,
        "association_ambiguous": association_ambiguous(belief),
        "prior_unresolved": a.unresolved_mass, "posterior_unresolved": b.unresolved_mass,
        **physical_reappearance(data, outcome),
        **survival}


def _conditioning_summary(records):
    categories = ("overall", "positive", "any_detection", "occluded_or_ambiguous",
                  "reappearance_physical", "reappearance_proxy", "aged_tracks", "dense", "association_ambiguous")
    result = {}
    for category in categories:
        selected = [r for r in records if category == "overall" or r.get(category)]
        summary = scene_ci(selected, "delta")
        summary["eligible_pairs"] = len(selected)
        summary["unsupported_pairs"] = sum(r["unsupported_pair"] for r in selected)
        summary["legacy_ancestry_merge_risk_pairs"] = sum(r.get("legacy_ancestry_merge_risk", False) for r in selected)
        summary["physical_reappearance_supported_pairs"] = sum(r.get("reappearance_physical_supported", False) for r in selected)
        summary["noninferiority_margin_m"] = CONDITIONING_MARGIN_M
        if summary["roots"] < 100 or summary["scenes"] < 30:
            summary["status"] = "INCONCLUSIVE_INDEPENDENT_SUPPORT"
        elif summary["one_sided_upper95"] <= CONDITIONING_MARGIN_M:
            summary["status"] = "SUPPORTED_NONINFERIORITY_ON_REPRESENTED_ANCESTRY"
        else:
            summary["status"] = "NONINFERIORITY_NOT_ESTABLISHED"
        summary["full_population_gospa_delta"] = scene_ci(selected, "all_population_gospa_delta")
        summary["sampled_population_gospa_delta"] = scene_ci(selected, "sampled_population_gospa_delta")
        result[category] = summary
    return result


def _summaries(populations, forecasts, pairs, variants):
    result = {}
    for variant in variants:
        population = [r for r in populations if r["variant"] == variant]
        psummary = {key: scene_ci(population, key) for key in
            ("count_absolute_error", "count_nll_lower", "count_nll_upper",
             "count_nll_resolved", "conditional_expected_count", "conditional_localized_count",
             "conditional_hidden_count", "unresolved_mass", "gospa", "false_localized",
             "missed_localized", "ancestry_gospa", "ancestry_false_localized", "ancestry_missed_localized",
             "sampled_population_gospa", "sampled_population_complete_mass")}
        for bound in ("lower", "upper"):
            unbounded = sum(r["count_nll_" + bound + "_unbounded"] for r in population)
            psummary["count_nll_" + bound]["unbounded_roots"] = unbounded
            if unbounded:
                psummary["count_nll_" + bound]["finite_only_mean"] = psummary["count_nll_" + bound]["mean"]
                psummary["count_nll_" + bound]["finite_only_ci95"] = [psummary["count_nll_" + bound]["lo"], psummary["count_nll_" + bound]["hi"]]
                psummary["count_nll_" + bound]["mean"] = None
                for endpoint in ("lo", "hi", "one_sided_upper95"):
                    psummary["count_nll_" + bound][endpoint] = None
                psummary["count_nll_" + bound]["status"] = "UNBOUNDED_LOG_SCORE_PRESENT"
        psummary["truth_excluded_roots"] = sum(r["truth_excluded_from_resolved_support"] for r in population)
        if psummary["truth_excluded_roots"]:
            psummary["count_nll_resolved"]["finite_only_mean"] = psummary["count_nll_resolved"]["mean"]
            psummary["count_nll_resolved"]["finite_only_ci95"] = [psummary["count_nll_resolved"]["lo"], psummary["count_nll_resolved"]["hi"]]
            psummary["count_nll_resolved"]["mean"] = None
            for endpoint in ("lo", "hi", "one_sided_upper95"):
                psummary["count_nll_resolved"][endpoint] = None
            psummary["count_nll_resolved"]["status"] = "UNBOUNDED_LOG_SCORE_PRESENT"
            psummary["count_nll_resolved"]["unbounded_roots"] = psummary["truth_excluded_roots"]
        strata = {}
        for label, predicate in (
            ("count_0", lambda r: r["true_count"] == 0),
            ("count_1_3", lambda r: 1 <= r["true_count"] <= 3),
            ("count_4_plus", lambda r: r["true_count"] >= 4),
            ("aged", lambda r: r["aged_tracks"]),
            ("ambiguous", lambda r: r["association_ambiguous"]),
            ("reappearance", lambda r: r["reappearance_proxy"])):
            selected = [r for r in population if predicate(r)]
            strata[label] = {key: scene_ci(selected, key) for key in
                            ("count_absolute_error", "gospa", "false_localized", "missed_localized", "unresolved_mass")}
        fsummary = {}
        for horizon in HORIZONS:
            selected = [r for r in forecasts if r["variant"] == variant and r["horizon"] == horizon]
            fsummary[str(horizon)] = {key: scene_ci(selected, key) for key in
                ("ancestry_energy_score", "ancestry_coverage90", "ancestry_radius90",
                 "ancestry_scored_mass_mean", "ancestry_missing_targets", "gospa",
                 "ancestry_gospa", "known_ancestry_gospa", "false_localized", "missed_localized", "unresolved_mass",
                 "sampled_population_gospa", "sampled_population_complete_mass")}
        variant_pairs = [r for r in pairs if r["variant"] == variant]
        by_horizon = {str(horizon): _conditioning_summary([r for r in variant_pairs
            if r.get("target_time", 2.) == horizon]) for horizon in CONDITIONING_HORIZONS}
        result[variant] = {"population": psummary, "population_strata": strata,
            "forecast": fsummary, "conditioning": by_horizon["2.0"],
            "conditioning_by_horizon": by_horizon}
    return result


def run(snapshot, checkpoint, population, event_checkpoint, cohort, output, *,
        variants=None, limit=0, verify_invariance_roots=3, smoke_roots=None):
    """Fixed full dev cohort; --limit is explicitly a smoke test, not a release."""
    torch.set_num_threads(1)
    variants = variants or DEFAULT_VARIANTS
    if isinstance(variants, (str, Path)):
        variants = json.loads(Path(variants).read_text())
    output = _output_path(output)
    source_hashes = _metric_sources()
    all_rows = _cohort_rows(snapshot, "dev", cohort)
    permitted = _cohort_rows(snapshot, "dev", None)
    if {m["root_id"] for m, _ in all_rows} != {m["root_id"] for m, _ in permitted}:
        raise ValueError("evaluation cohort must contain every development root")
    if smoke_roots:
        requested = json.loads(Path(smoke_roots).read_text()) if isinstance(smoke_roots, (str, Path)) else smoke_roots
        ids = set(requested)
        if not ids <= {m["root_id"] for m, _ in all_rows}:
            raise ValueError("smoke root selection contains a non-development root")
        rootrows = [(m, p) for m, p in all_rows if m["root_id"] in ids]
    else:
        rootrows = all_rows[:limit] if limit else all_rows
    smoke = bool(limit or smoke_roots)
    learned = LearnedDynamics.load(checkpoint)
    prior = json.loads(Path(population).read_text())
    event = NativeEventModel(json.loads(Path(event_checkpoint).read_text())["coefficients"])
    models = {name: _factory(config["factory"])(dynamics=learned if config.get("use_learned", True) else None,
        population_prior=prior, event_model=event, **config.get("kwargs", {}))
        for name, config in variants.items()}
    populations = []
    forecasts = []
    pairs = []
    invariance = []
    started = time.perf_counter()
    for ri, (meta, folder) in enumerate(rootrows):
        data = root_data(folder, True)
        legal = data["legal"]
        target_ids = visible_ids(data)
        for name, model in models.items():
            base = model.initialize(legal["history"], legal["partial_graph"])
            belief = tag_root_ancestry(base, data["truth"][0])
            identity = {"root_id": meta["root_id"], "scene_id": meta["scene_id"], "variant": name}
            if ri < verify_invariance_roots:
                a = model.predict_shared(base, .096, 16, seed=211, times=[0., .048, .096])
                b = model.predict_shared(belief, .096, 16, seed=211, times=[0., .048, .096])
                identical = (np.array_equal(a.positions, b.positions, equal_nan=True)
                    and np.array_equal(a.velocities, b.velocities, equal_nan=True)
                    and np.array_equal(a.weights, b.weights) and a.unresolved_mass == b.unresolved_mass)
                if not identical:
                    raise AssertionError("offline tags changed sampled predictive behavior")
                invariance.append({**identity, "tagged_untagged_bitwise_identical": True})
            score = population_score(belief, data["truth"].shape[1])
            score.update(expected_set_scores(belief.hypotheses,
                [h.mean[:, :2] for h in belief.hypotheses], [h.weight for h in belief.hypotheses], data["truth"][0], target_ids))
            visible_root = {r["track_id"] for r in legal["history"][-1]["tracks"] if r["visible_now"]}
            previous_visible = {r["track_id"] for row in legal["history"][:-1] for r in row["tracks"] if r["visible_now"]}
            last_visible = {r["track_id"] for r in legal["history"][-2]["tracks"] if r["visible_now"]} if len(legal["history"]) > 1 else set()
            score.update(identity, aged_tracks=any(np.any(belief.timestamp - h.last_seen > .5) for h in belief.hypotheses if h.n),
                association_ambiguous=association_ambiguous(belief),
                reappearance_proxy=bool((visible_root - last_visible) & previous_visible),
                root_visible_targets=len(target_ids),
                exact_ancestry_slots_conditional=float(sum(h.weight * len(h.diagnostic_exact) for h in belief.hypotheses) / max(1e-30, 1 - belief.unresolved_mass)),
                aged_oracle_ancestry_slots_conditional=float(sum(h.weight * len(h.diagnostic_aged) for h in belief.hypotheses) / max(1e-30, 1 - belief.unresolved_mass)))
            populations.append(score)
            future = model.predict_shared(belief, 3., 16, seed=41, times=[0., *HORIZONS])
            score.update(sampled_population_scores(future, 0, data["truth"][0]))
            for horizon in HORIZONS:
                target = truth_at(data, belief.timestamp + horizon)
                if target is None:
                    continue
                index = int(np.argmin(abs(future.times - belief.timestamp - horizon)))
                forecast = ancestry_energy(future, target, target_ids, belief.timestamp + horizon)
                forecast.update(expected_set_scores(future.worlds, future.positions[:, index], future.weights, target, target_ids))
                forecast.update(sampled_population_scores(future, index, target))
                forecasts.append({**identity, **forecast, "horizon": horizon,
                                  "unresolved_mass": float(future.unresolved_mass)})
            for target_horizon in CONDITIONING_HORIZONS:
                pair = conditioning_pair(model, belief, data, target_ids, target_horizon=target_horizon)
                if pair is not None:
                    pairs.append({**identity, **pair})
        if (ri + 1) % 25 == 0:
            print(json.dumps({"roots": ri + 1, "elapsed_s": round(time.perf_counter() - started, 2),
                              "conditioning_pairs": len(pairs)}), flush=True)
    ending_sources = _metric_sources()
    source_changed = ending_sources != source_hashes
    if source_changed and not smoke:
        raise RuntimeError("evaluation sources changed during run; result not published")
    result = {"schema": SCHEMA, "split": "dev", "roots": len(rootrows),
        "scenes": len({m["scene_id"] for m, _ in rootrows}), "fixed_cohort_roots": len(all_rows),
        "smoke_test_only": smoke, "source_sha256": source_hashes,
        "source_sha256_end": ending_sources, "source_changed_during_smoke": source_changed,
        "snapshot_manifest_sha256": _sha(Path(snapshot) / "manifest.json"),
        "cohort_sha256": _sha(cohort), "dynamics_checkpoint_sha256": _sha(checkpoint),
        "population_prior_sha256": _sha(population), "event_checkpoint_sha256": _sha(event_checkpoint),
        "variants": variants, "resolved_model_configuration": {name: model_configuration(model) for name, model in models.items()},
        "seeds": {"forecast": 41, "survival": 91, "conditioning": 107, "bootstrap": 73},
        "threads": 1, "elapsed_s": time.perf_counter() - started,
        "metric_contract": {
            "population": "all truth counts; residual unspecified count law reported as log-score bounds; conditional resolved MAE",
            "gospa": "unlabeled all-population set GOSPA p2 cutoff2m alpha2, separate from fixed-ancestry labeled diagnostic",
            "sampled_population_gospa": "secondary score uses all finite forecast positions, including private never-localized hidden samples; complete sampled-population mass is reported because legacy has no spatial law for hidden counts; no fabricated positions fill its missing support",
            "ancestry": "root exact visible anchors then <=2m aged one-to-one oracle alignment ONCE on unsampled means; fixed slot identities thereafter; root-born missing ancestry never rematched",
            "forecast_energy": "per-root-visible-target finite-distribution Energy Score conditional on represented ancestry mass; target support and mass reported",
            "conditioning": "primary same absolute root+2s target; separately reported predeclared secondary root+1s target; both use10th actual action observation and identical manual native survival projection, no pseudoempty observation",
            "reappearance": "physical identities diagnosed only over root plus10 actual native endpoints using unique exact center matching; pre-root invisible history unavailable. Source track-ID reappearance_proxy is kept separately and does not establish physical identity",
            "uncertainty": "equal-scene root-mean percentile bootstrap1000 seed73; +0.020m predeclared noninferiority margin; minimum100roots30scenes",
            "selection": "fixed development cohort only; this is not fresh held-out confirmation",
            "ablations": "legacy->association_only changes maneuver-aware association; association_only->full jointly changes the set observation law, persistent spatial hidden prior, first-detection velocity prior, and history transition from CV to structured OU; it is not an observation-only isolation. full_no_maneuver and full_kinematic are fixed mechanism ablations, not parameter search",
            "controlled_revision": "v1 zero-mean resets damped learned OU intent to approximately0.273 stationary mean. v2 keeps fixed2Hz/sigma2 but retains velocity mean (retention1); historical v1 smoke/oracle artifacts remain failed diagnostic witnesses. This preserves the learned mean semantics but adds forecast uncertainty, whose quality must still be measured",
            "limitations": "root aged alignment is an offline oracle diagnostic; ancestry energy remains conditional; residual has no point count law; GOSPA is a set-error metric, not a proper probabilistic score"},
        "summary": _summaries(populations, forecasts, pairs, variants),
        "population_rows": populations, "forecast_rows": forecasts, "conditioning_rows": pairs,
        "tag_invariance_checks": invariance}
    _json(output, result)
    print(json.dumps({"output": str(output), "roots": len(rootrows), "elapsed_s": result["elapsed_s"]}), flush=True)
    return result


def selftest():
    from .state import BeliefState, ObservedGraph
    from .model import GraphBeliefWorldModel

    def belief(mean, hidden=(1.,), unresolved=0.):
        mean = np.asarray(mean, float).reshape(-1, 4)
        h = Hypothesis(1 - unresolved, mean, np.tile(np.eye(4) * .1, (len(mean), 1, 1)),
            np.zeros(len(mean)), np.arange(len(mean)), np.asarray(hidden))
        graph = ObservedGraph("oracle", 0, .1, 0., {})
        return BeliefState(graph, [h], 0., np.zeros(4), unresolved)

    class EvaluationTests(unittest.TestCase):
        def test_gospa_penalizes_missing_and_false_people(self):
            score = gospa([[0, 0], [10, 0]], [[.5, 0], [4, 0]])
            self.assertAlmostEqual(score["gospa"], math.sqrt(.25 + 4.))
            self.assertEqual(score["false_localized"], 1.)
            self.assertEqual(score["missed_localized"], 1.)

        def test_no_fabricated_hidden_spatial_positions(self):
            b = belief([[0, 0, 0, 0]], hidden=(0., 1.))
            c = population_score(b, 2)
            self.assertEqual(c["conditional_expected_count"], 2.)
            self.assertEqual(c["count_nll_lower"], 0.)
            self.assertEqual(gospa(b.hypotheses[0].mean[:, :2], [[0, 0], [3, 0]])["missed_localized"], 1.)

        def test_sampled_population_includes_actual_private_hidden_slots(self):
            from .state import SharedFuture
            b = tag_root_ancestry(belief([[0, 0, 0, 0]], hidden=(0., 1.)), [[0, 0], [3, 0]])
            h = b.hypotheses[0]
            positions = np.array([[[[0., 0.], [3., 0.], [np.nan, np.nan]]]])
            future = SharedFuture(np.array([0.]), positions, positions.copy(), np.array([1.]), [h], b, 0., 1)
            score = sampled_population_scores(future, 0, [[0, 0], [3, 0]])
            self.assertEqual(score["sampled_population_gospa"], 0.)
            self.assertEqual(score["sampled_population_complete_mass"], 1.)
            future.positions = positions[:, :, :1]
            incomplete = sampled_population_scores(future, 0, [[0, 0], [3, 0]])
            self.assertEqual(incomplete["sampled_population_complete_mass"], 0.)
            self.assertAlmostEqual(incomplete["sampled_population_gospa"], math.sqrt(2.))

        def test_residual_count_score_is_an_interval(self):
            b = belief([[0, 0, 0, 0]], hidden=(.75, .25), unresolved=.2)
            c = population_score(b, 2)
            self.assertAlmostEqual(c["true_count_probability_lower"], .2)
            self.assertAlmostEqual(c["true_count_probability_upper"], .4)
            self.assertAlmostEqual(c["count_nll_lower"], -math.log(.4))
            self.assertAlmostEqual(c["count_nll_upper"], -math.log(.2))

        def test_ghost_population_cannot_get_finite_resolved_log_score(self):
            b = belief([[0, 0, 0, 0], [1, 0, 0, 0]], unresolved=.01)
            c = population_score(b, 1)
            self.assertTrue(c["count_nll_upper_unbounded"])
            self.assertIsNone(c["count_nll_upper"])
            self.assertAlmostEqual(c["count_nll_lower"], -math.log(.01))

        def test_ancestry_persists_and_switch_is_not_rematched(self):
            b = tag_root_ancestry(belief([[0, 0, 0, 0], [1, 0, 0, 0]]), [[0, 0], [1, 0]])
            h = b.hypotheses[0].fork(mean=np.array([[1., 0, 0, 0], [0, 0, 0, 0]]))
            target = np.array([[0., 0], [1., 0]])
            self.assertEqual(gospa(h.mean[:, :2], target)["gospa"], 0.)
            self.assertAlmostEqual(fixed_ancestry_cost(h, h.mean[:, :2], target, [0, 1])["gospa"], math.sqrt(2.))

        def test_diagnostic_tags_do_not_change_model_predictions(self):
            b = belief([[0, 0, 1, 0]])
            tagged = tag_root_ancestry(b, [[0, 0]])
            model = GraphBeliefWorldModel(kind="memory_cv")
            a = model.predict_shared(b, .1, 4, seed=11, times=[0., .1])
            c = model.predict_shared(tagged, .1, 4, seed=11, times=[0., .1])
            self.assertEqual(b.fingerprint(), tagged.fingerprint())
            np.testing.assert_array_equal(a.positions, c.positions)
            np.testing.assert_array_equal(a.weights, c.weights)
            self.assertTrue(all(isinstance(h, DiagnosticHypothesis) for h in c.worlds))

        def test_scene_ci_counts_only_supported_independent_roots(self):
            rows = [{"root_id": "a", "scene_id": "s", "score": 1.},
                    {"root_id": "a", "scene_id": "s", "score": 3.},
                    {"root_id": "b", "scene_id": "t", "score": None}]
            c = scene_ci(rows, "score")
            self.assertEqual(c["roots"], 1)
            self.assertEqual(c["scenes"], 1)
            self.assertEqual(c["mean"], 2.)

        def test_empty_targets_do_not_produce_fake_zero_energy(self):
            model = GraphBeliefWorldModel(kind="memory_cv")
            b = tag_root_ancestry(belief([]), np.empty((0, 2)))
            future = model.predict_shared(b, .1, 4, seed=3, times=[0., .1])
            score = ancestry_energy(future, np.empty((0, 2)), [], .1)
            self.assertIsNone(score["ancestry_energy_score"])
            self.assertFalse(score["ancestry_all_targets_supported"])

        def test_survival_projection_does_not_call_sensor_update(self):
            class SensorMustNotRun(GraphBeliefWorldModel):
                def observe(self, *args, **kwargs):
                    raise AssertionError("survival may not fabricate a sensor row")
            coefficients = np.zeros((3, 12))
            model = SensorMustNotRun(kind="memory_cv", event_model=NativeEventModel(coefficients))
            original = tag_root_ancestry(belief([[0, 0, 0, 0]]), [[0, 0]])
            data = {"nodes": {str(i): {"observation": {"timestamp": i * .048,
                    "robot_pose": [0., 0., 0., 0.], "tracks": []}}
                    for i in (1, 2)}}
            prior, _, details = survival_conditioned_prior(model, original, data,
                {"node_ids": ["1", "2"]}, action_count=2)
            self.assertAlmostEqual(details["survival_probability"], .25)
            self.assertAlmostEqual(prior.timestamp, .096)
            self.assertEqual(prior.hypotheses[0].n, 1)
            self.assertNotIn("last_observation_key", prior.evidence)

        def test_legacy_ancestry_merge_is_explicitly_unsupported(self):
            model = GraphBeliefWorldModel(kind="memory_cv")
            b = tag_root_ancestry(belief([[0, 0, 0, 0], [1, 0, 0, 0]]), [[0, 0], [1, 0]])
            b = model.prepare(b)
            h = b.hypotheses[0]
            other = h.fork(weight=.5, diagnostic_ancestry=((0, 1), (1, 0)))
            b = b.fork(hypotheses=[h.fork(weight=.5), other])
            self.assertTrue(legacy_ancestry_merge_risk(model, b))
            identical = h.fork(weight=.5, diagnostic_origin=9)
            b = b.fork(hypotheses=[h.fork(weight=.5), identical])
            self.assertFalse(legacy_ancestry_merge_risk(model, b))

        def test_conditioning_default_keeps_primary_absolute_target(self):
            class NoAdditionalSensorInformation(GraphBeliefWorldModel):
                def observe(self, prior, observation, *args, **kwargs):
                    return prior
            model = NoAdditionalSensorInformation(kind="memory_cv", event_model=NativeEventModel(np.zeros((3, 12))))
            original = tag_root_ancestry(belief([[0, 0, 0, 0]]), [[0, 0]])
            track = {"track_id": 7, "position": [0., 0., 0.], "visible_now": True}
            root_obs = {"timestamp": 0., "robot_pose": [0., 0., 0., 0.], "tracks": [track]}
            data = {"times": np.arange(64) * .048, "truth": np.zeros((64, 1, 2)),
                "queries": [{"query_id": "q", "compiled": {"actions": ["wait"] * 11}}],
                "outcomes": [{"node_ids": [str(i) for i in range(1, 12)]}],
                "legal": {"history": [root_obs]},
                "nodes": {str(i): {"observation": {**root_obs, "timestamp": i * .048}}
                          for i in range(1, 12)}}
            primary = conditioning_pair(model, original, data, [0])
            explicit = conditioning_pair(model, original, data, [0], target_horizon=2.)
            secondary = conditioning_pair(model, original, data, [0], target_horizon=1.)
            self.assertEqual(primary, explicit)
            self.assertEqual(primary["target_time"], 2.)
            self.assertEqual(secondary["target_time"], 1.)
            self.assertEqual(primary["observation_time"], secondary["observation_time"])
            self.assertEqual(primary["survival_probability"], secondary["survival_probability"])
            self.assertEqual(primary["delta"], 0.)
            self.assertEqual(secondary["delta"], 0.)
            short = {**data, "times": data["times"][:26], "truth": data["truth"][:26]}
            self.assertIsNone(conditioning_pair(model, original, short, [0]))
            self.assertIsNotNone(conditioning_pair(model, original, short, [0], target_horizon=1.))

        def test_conditioning_horizons_are_not_pooled(self):
            records = [{"variant": "v", "root_id": "r", "scene_id": "s", "target_time": h,
                        "delta": score, "unsupported_pair": False}
                       for h, score in ((2., .3), (1., -.8))]
            summary = _summaries([], [], records, ["v"])["v"]
            self.assertEqual(summary["conditioning"]["overall"]["mean"], .3)
            self.assertEqual(summary["conditioning_by_horizon"]["1.0"]["overall"]["mean"], -.8)
            self.assertEqual(summary["conditioning"]["overall"]["roots"], 1)

        def test_physical_reappearance_uses_identity_not_source_track_id(self):
            def observed(source_id):
                return {"timestamp": 0., "tracks": [{"track_id": source_id,
                        "position": [0., 0., 0.], "visible_now": True}]}
            data = {"times": np.arange(11) * .048, "truth": np.zeros((11, 1, 2)),
                "legal": {"history": [observed(1)]},
                "nodes": {str(i): {"observation": {**observed(100 + i), "timestamp": i * .048}}
                          for i in range(1, 11)}}
            outcome = {"node_ids": [str(i) for i in range(1, 11)]}
            self.assertFalse(physical_reappearance(data, outcome)["reappearance_physical"])
            data["nodes"]["9"]["observation"]["tracks"] = []
            result = physical_reappearance(data, outcome)
            self.assertTrue(result["reappearance_physical_supported"])
            self.assertEqual(result["reappearance_physical_ids"], [0])
            duplicate_truth = np.zeros((11, 2, 2))
            self.assertFalse(physical_reappearance({**data, "truth": duplicate_truth}, outcome)["reappearance_physical_supported"])

    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    for name in ("snapshot", "checkpoint", "population", "event_checkpoint", "cohort", "output"):
        execute.add_argument(name)
    execute.add_argument("--variants")
    execute.add_argument("--limit", type=int, default=0)
    execute.add_argument("--verify-invariance-roots", type=int, default=3)
    execute.add_argument("--smoke-roots")
    commands.add_parser("selftest")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"run": run, "selftest": selftest}[command](**args)


if __name__ == "__main__":
    main()
