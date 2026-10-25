"""Secondary full-call latency of an external, selected-threshold adapter.

The released model is unmodified and still constructs its original argmax map.
The adapter additionally prepares/constructs a thresholded future map inside the
timer. This is deliberately conservative redundant work, not a production patch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from .common import ROOT, OUT, digest, save
from .decision_overhead import construct_future, decision_states, _case_digest, _snapshot
from .overhead_admission import assert_quiet, accepted_primary_identity


BASE = OUT / "visibility"


def external_predict(model, *, partial_graph, trajectory, threshold):
    start = time.perf_counter()
    original = model.predict(partial_graph=partial_graph, trajectory=trajectory)
    predicted = time.perf_counter()
    # This public-cache lookup rehashes the supplied graph and stays timed.
    prepared = model.prepare_graph(partial_graph)
    ready = time.perf_counter()
    converted, components = construct_future(partial_graph, original, threshold,
                                              getattr(prepared, "future_base", None))
    return converted, dict(original_complete_prediction_seconds=predicted - start,
                           additional_graph_lookup_seconds=ready - predicted,
                           additional_conversion=components,
                           total_seconds=time.perf_counter() - start)


def protocol_path():
    return BASE / "full_adapter_protocol_v3.json"


def identity():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), Path(__file__).with_name("decision_overhead.py"), Path(__file__).with_name("common.py"),
             Path(__file__).with_name("overhead_admission.py"))}


def write_protocol():
    summary_path = OUT / "probability/summary.json"
    summary = json.loads(summary_path.read_text())
    selected = summary["diagnostic_scene_validation"]["selected_rules"]
    if any(not rule.startswith("threshold_") for rule in selected.values()):
        raise ValueError("unexpected selected-rule format")
    thresholds = {objective: float(rule.split("_")[1]) for objective, rule in selected.items()}
    value = dict(schema="full_external_threshold_adapter_latency_v3", source=identity(),
                 supersedes_before_timings="visibility/full_adapter_protocol_v2.json",
                 guard_addendum="visibility/overhead_guard_addendum_v3.json",
                 guard_addendum_sha256=digest(BASE / "overhead_guard_addendum_v3.json"),
                 probability_summary_sha256=digest(summary_path), diagnostic_protocol_sha256=digest(OUT / "protocol.json"),
                 precision_contract_sha256=digest(OUT / "precision_contract.json"),
                 thresholds_selected_on_70_scenes=thresholds, repeats=10,
                 selection="First8fixed representative and first8fixed dense primary saved examples;16cohort examples, report unique count and overlap",
                 cache="Retained graph cache; warm every rule per case; alternate rule order; normal unchanged GC",
                 boundary="Unmodified complete model.predict for baseline; external adapter calls same complete predict (including redundant argmax graph), additional public graph lookup, threshold decision+complete new future graph, in-process output",
                 specificity="Only selected IoU/F1 thresholds; deduplicate equal thresholds and report aliases; no new threshold selection from timings",
                 scope="Secondary16example practical latency diagnostic; not the full original185example suite, not a substitute primary gate or production promotion",
                 outputs="P50/P95/P99 per representative/dense cohort, rawcalls, CPUtime, GC events, complete futuregraph counts; startup separately",
                 launch="Parent C/D release plus read-only absence of active sustained_service.py, benchmark and diagnostic processes; no historical completion-marker requirement",
                 production_changed=False, checkpoint_changed=False)
    if protocol_path().exists():
        raise FileExistsError("external-adapter protocol already exists")
    save(protocol_path(), value)
    print(protocol_path())


def execute(priority_released):
    assert_quiet(priority_released)
    config = json.loads(protocol_path().read_text())
    if config["source"] != identity() or config["probability_summary_sha256"] != digest(OUT / "probability/summary.json"):
        raise ValueError("external-adapter source/selected thresholds changed")
    if not gc.isenabled():
        raise RuntimeError("normal GC required")
    from .common import make_model
    from eventwm_graph.stage2_segment.benchmark import inputs, GC, gc_event
    accepted = accepted_primary_identity(); thresholds = config["thresholds_selected_on_70_scenes"]
    unique_thresholds = sorted(set(thresholds.values()))
    rules = [("released_argmax", None)] + [(f"threshold_{theta:.2f}", theta) for theta in unique_thresholds]
    workloads = inputs("single")
    cases = [(cohort, case) for cohort in ("representative", "dense") for case in workloads[cohort][:8]]
    if len(cases) != 16:
        raise ValueError("frozen16example selection unavailable")
    dest = BASE / "full_adapter_latency_v3"; dest.mkdir(exist_ok=False)
    load = time.perf_counter(); model = make_model(); load = time.perf_counter() - load
    gc.callbacks.append(gc_event); initial_gc = gc.get_threshold()
    records = []; startup = []; proofs = []; holders = {}; seen = set()
    for index, (cohort, case) in enumerate(cases):
        graph = case["graph"]; path = case["trajectory"]; before = _case_digest(graph, path); seen.add(before)
        model.clear_cache(); start = time.perf_counter(); baseline = model.predict(partial_graph=graph, trajectory=path)
        startup.append(dict(cohort=cohort, input_sha256=before, rule="released_argmax", seconds=time.perf_counter() - start))
        for name, theta in rules[1:]:
            start = time.perf_counter(); delta, _ = external_predict(model, partial_graph=graph, trajectory=path, threshold=theta)
            startup.append(dict(cohort=cohort, input_sha256=before, rule=name, seconds=time.perf_counter() - start))
            np.testing.assert_array_equal(delta["probabilities"], baseline["probabilities"])
            np.testing.assert_array_equal(delta["states"], decision_states(baseline["probabilities"], theta))
            if index == 0:
                owned, _ = construct_future(graph, baseline, theta, None)
                assert _snapshot(owned["future_partial_graph"]) == _snapshot(delta["future_partial_graph"])
            proofs.append(dict(cohort=cohort, input_sha256=before, rule=name, probabilities_unchanged=True,
                               states_match_frozen_rule=True, future_graph_materialized=True))
        for repeat in range(config["repeats"]):
            ordered = rules if (index + repeat) % 2 == 0 else list(reversed(rules))
            for name, theta in ordered:
                begin_gc = len(GC); cpu = time.process_time(); start = time.perf_counter()
                if theta is None:
                    holders[name] = model.predict(partial_graph=graph, trajectory=path)
                    profile = None
                else:
                    holders[name], profile = external_predict(model, partial_graph=graph, trajectory=path, threshold=theta)
                end = time.perf_counter(); process_cpu = time.process_time() - cpu
                records.append(dict(cohort=cohort, root_id=case["root_id"], input_sha256=before, rule=name, repeat=repeat,
                                    total_seconds=end - start, process_cpu_seconds=process_cpu, adapter_components=profile,
                                    materialized_output_cells=len(holders[name]["future_partial_graph"]["cells"]),
                                    updates=len(holders[name]["static_updates"]),
                                    gc_events=[event for event in GC[begin_gc:] if event["start"] < end and event["end"] > start]))
        if _case_digest(graph, path) != before:
            raise ValueError("external adapter changed supplied input")
    if accepted_primary_identity() != accepted or not gc.isenabled() or gc.get_threshold() != initial_gc:
        raise ValueError("runtime/checkpoint/GC changed")
    assert_quiet(priority_released)
    summary = {}
    for cohort in ("representative", "dense"):
        summary[cohort] = {}
        for name, _ in rules:
            rows = [record for record in records if record["cohort"] == cohort and record["rule"] == name]
            q = np.quantile([row["total_seconds"] * 1000 for row in rows], [.5, .95, .99])
            summary[cohort][name] = dict(calls=len(rows), p50_ms=float(q[0]), p95_ms=float(q[1]), p99_ms=float(q[2]),
                                        mean_process_cpu_ms=1000 * float(np.mean([row["process_cpu_seconds"] for row in rows])))
    save(dest / "calls.json", records); save(dest / "proofs.json", proofs)
    save(dest / "metrics.json", dict(workloads=summary, cohort_examples=16, unique_inputs=len(seen),
                                      selected_thresholds=thresholds, equivalent_rule_aliases=thresholds,
                                      checkpoint_load_seconds=load, startup_calls=startup,
                                      joint_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                      gc_enabled=True, actual_future_graphs_inside_timer=True,
                                      accepted_primary=accepted, protocol_sha256=digest(protocol_path()),
                                      scope=config["scope"], memory_scope="Joint process/input/JIT/outputs lifetime RSS, not model-only",
                                      probabilities_changed=False, production_changed=False, primary_gates_redefined=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-protocol", action="store_true"); group.add_argument("--execute", action="store_true")
    parser.add_argument("--priority-c-d-released", action="store_true")
    args = parser.parse_args()
    write_protocol() if args.write_protocol else execute(args.priority_c_d_released)


if __name__ == "__main__":
    main()
