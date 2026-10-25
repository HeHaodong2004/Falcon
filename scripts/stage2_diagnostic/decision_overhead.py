"""Separate post-inference decision/output cost; never a full-inference gate.

One unchanged model prediction is cached per fixed saved workload. Timed calls
then validate cached probabilities, decide labels, construct sparse updates and
materialize a complete hypothetical graph. No target labels are used.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .common import ROOT, OUT, THRESHOLDS, digest, save
from .overhead_admission import assert_quiet, accepted_primary_identity


def decision_states(probabilities, threshold=None):
    """Preserve raw argmax or apply the frozen diagnostic precision contract.

    Revelation sum is computed F32, explicitly promoted to F64, then compared
    with the F64 threshold. FREE wins a FREE/BLOCKED probability tie.
    """
    p = np.asarray(probabilities)
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("cached probabilities must be float32[N,3]")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or np.any(abs(p.sum(1, dtype=np.float64) - 1) > 1e-6)):
        raise ValueError("invalid cached probability distribution")
    if threshold is None:
        return p.argmax(1).astype(np.uint8)
    theta = np.float64(threshold)
    if not np.isfinite(theta) or not 0 < theta <= 1:
        raise ValueError("threshold must be finite and in (0,1]")
    reveal = np.add(p[:, 1], p[:, 2], dtype=np.float32).astype(np.float64)
    states = np.zeros(len(p), np.uint8)
    selected = reveal >= theta
    states[selected] = np.where(p[selected, 1] >= p[selected, 2], 1, 2)
    return states


def construct_future(graph, prediction, threshold=None, native_base=None):
    """Convert one authenticated cached model output, preserving its probabilities.

    ``native_base`` is the same owned immutable graph preparation already paid
    for by the original model inference. This measures post-inference output
    cost only. The standalone owned updater is used if the native path declines.
    """
    start = time.perf_counter()
    states = decision_states(prediction["probabilities"], threshold)
    keys = prediction["candidate_cells"]
    heights = prediction["heights"]
    if np.shape(keys) != (len(states), 2) or np.shape(heights) != (len(states),) or not np.isfinite(heights).all():
        raise ValueError("invalid cached cell/height alignment")
    decided = time.perf_counter()
    delta = {key: value for key, value in prediction.items()
             if key not in ("states", "static_updates", "unknown_cells", "future_partial_graph", "provenance")}
    now = float(delta["timestamp"])
    delta["states"] = states
    delta["unknown_cells"] = keys[states == 0]
    delta["static_updates"] = [dict(cell=keys[i].tolist(), state=int(states[i]), height=float(heights[i]),
                                    timestamp=now, source="predicted_stage2_research")
                               for i in np.flatnonzero(states)]
    delta["provenance"] = copy.deepcopy(prediction.get("provenance", {}))
    delta["provenance"].update(diagnostic_only=True, hypothetical=True,
                               decision="raw_softmax_argmax" if threshold is None else "diagnostic_revelation_threshold",
                               revelation_threshold=None if threshold is None else float(threshold),
                               probability_values_unchanged=True,
                               threshold_precision="F32_class_sum_then_F64_comparison")
    sparse = time.perf_counter()
    future = native_base.update(delta) if native_base is not None else None
    if future is None:
        from eventwm_graph.stage2_segment.update import apply_graph_delta
        future = apply_graph_delta(graph, delta)
    delta["future_partial_graph"] = future
    delta["future_format"] = "native_columns" if hasattr(future, "cell_columns") else "snapshot"
    end = time.perf_counter()
    return delta, dict(validate_and_decide_seconds=decided - start, sparse_delta_seconds=sparse - decided,
                       complete_graph_seconds=end - sparse, total_seconds=end - start,
                       future_cells=len(future["cells"]), updates=len(delta["static_updates"]))


def protocol_path():
    return OUT / "visibility/decision_protocol_v3.json"


def write_protocol():
    value = dict(schema="cached_decision_output_overhead_v3", source_sha256=digest(__file__),
                 common_sha256=digest(Path(__file__).with_name("common.py")),
                 admission_sha256=digest(Path(__file__).with_name("overhead_admission.py")),
                 supersedes_before_timings="visibility/decision_protocol_v2.json",
                 guard_addendum="visibility/overhead_guard_addendum_v3.json",
                 guard_addendum_sha256=digest(OUT / "visibility/overhead_guard_addendum_v3.json"),
                 selection="First8 representative and first8 dense frozen primary single examples, deduplicated by graph/trajectory digest; no labels or outcomes",
                 maximum_cases=16, thresholds=THRESHOLDS.tolist(), repeats=5,
                 precision="F32 pFREE+pBLOCKED, then F64 comparison with F64 threshold; FREE wins conditional class tie",
                 timer="Validated cached p/keys/heights -> decision -> sparse delta -> complete native future graph; cached original graph preparation reused",
                 exclusions="Model/map preprocessing already completed once per case; not a complete inference latency or replacement acceptance gate",
                 gc="Normal enabled unchanged GC; no disable/collect/freeze/threshold adjustment",
                 launch="Explicit parent C/D release plus read-only absence of active sustained_service.py, benchmark and diagnostic processes; no historical completion-marker requirement",
                 reference="Raw-argmax output map exactly equals original cached full map; native/owned updater equivalence for every rule on first fixed case",
                 checkpoint_changed=False, model_rerun_per_threshold=False, probabilities_changed=False)
    path = protocol_path()
    if path.exists():
        raise FileExistsError("decision protocol already frozen")
    save(path, value)
    print(path)


def _case_digest(graph, trajectory):
    value = hashlib.sha256(pickle.dumps(graph, protocol=5))
    path = np.asarray(trajectory, dtype="<f8")
    value.update(str(path.shape).encode()); value.update(path.tobytes(order="C"))
    return value.hexdigest()


def _snapshot(value):
    return value.to_snapshot() if hasattr(value, "to_snapshot") else value


def execute(priority_released=False):
    # Imports that may initialize Torch/JIT are deliberately behind explicit run.
    if not (OUT / "protocol.json").exists():
        raise RuntimeError("freeze diagnostic protocol first")
    assert_quiet(priority_released)
    config = json.loads(protocol_path().read_text())
    if (config["source_sha256"] != digest(__file__) or config["common_sha256"] != digest(Path(__file__).with_name("common.py"))
            or config["admission_sha256"] != digest(Path(__file__).with_name("overhead_admission.py"))):
        raise ValueError("decision helper changed after protocol freeze")
    if not gc.isenabled():
        raise RuntimeError("normal GC required")
    from .common import make_model
    from eventwm_graph.stage2_segment.benchmark import inputs
    accepted = accepted_primary_identity()
    workloads = inputs("single")
    cases = {}; selection = workloads["representative"][:8] + workloads["dense"][:8]
    for case in selection:
        cases.setdefault(_case_digest(case["graph"], case["trajectory"]), case)
    dest = OUT / "visibility/decision_overhead_v3"; dest.mkdir(exist_ok=False)
    model = make_model(); rules = [("argmax", None)] + [(f"threshold_{theta:.2f}", theta) for theta in THRESHOLDS]
    calls = []; witnesses = []; holders = {}; preparation = []
    initial_thresholds = gc.get_threshold()
    for case_number, (identity, case) in enumerate(cases.items()):
        graph = case["graph"]; path = case["trajectory"]
        start = time.perf_counter(); original = model.predict(partial_graph=graph, trajectory=path)
        prepared = model.prepare_graph(graph); base = getattr(prepared, "future_base", None)
        preparation.append(dict(input_sha256=identity, root_id=case["root_id"], model_and_cached_base_seconds=time.perf_counter()-start,
                                native_base_available=base is not None))
        p_sha = hashlib.sha256(original["probabilities"].tobytes()).hexdigest()
        raw, _ = construct_future(graph, original, native_base=base)
        assert _snapshot(raw["future_partial_graph"]) == _snapshot(original["future_partial_graph"])
        if case_number == 0:
            for name, theta in rules:
                native, _ = construct_future(graph, original, theta, base)
                owned, _ = construct_future(graph, original, theta, None)
                assert _snapshot(native["future_partial_graph"]) == _snapshot(owned["future_partial_graph"])
                witnesses.append(dict(rule=name, full_native_owned_map_exact=True))
        for repeat in range(config["repeats"]):
            ordered = rules if repeat % 2 == 0 else list(reversed(rules))
            for name, theta in ordered:
                start = time.perf_counter(); holders[name], profile = construct_future(graph, original, theta, base)
                calls.append(dict(root_id=case["root_id"], input_sha256=identity, rule=name, repeat=repeat,
                                  total_seconds=time.perf_counter()-start, profile=profile,
                                  candidate_cells=len(original["states"]), future_format=holders[name]["future_format"]))
        assert _case_digest(graph, path) == identity
        assert hashlib.sha256(original["probabilities"].tobytes()).hexdigest() == p_sha
    assert gc.isenabled() and gc.get_threshold() == initial_thresholds and accepted_primary_identity() == accepted
    assert_quiet(priority_released)
    summary = {}
    for name, _ in rules:
        rows = [row for row in calls if row["rule"] == name]
        values = np.asarray([row["total_seconds"] for row in rows]) * 1000
        summary[name] = dict(calls=len(rows), p50_ms=float(np.quantile(values, .5)), p95_ms=float(np.quantile(values, .95)),
                             p99_ms=float(np.quantile(values, .99)),
                             mean_components_ms={key: float(np.mean([row["profile"][key] for row in rows])) * 1000
                                                 for key in ("validate_and_decide_seconds", "sparse_delta_seconds", "complete_graph_seconds")})
    save(dest / "calls.json", calls); save(dest / "proofs.json", witnesses)
    save(dest / "metrics.json", dict(cases=len(cases), rules=summary, preparation=preparation,
                                      protocol_sha256=digest(protocol_path()), global_protocol_sha256=digest(OUT / "protocol.json"),
                                      accepted_primary=accepted, gc_enabled=True, input_immutability_pass=True,
                                      scope=config["timer"], exclusions=config["exclusions"], primary_gates_redefined=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-protocol", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--priority-c-d-released", action="store_true")
    args = parser.parse_args()
    write_protocol() if args.write_protocol else execute(args.priority_c_d_released)


if __name__ == "__main__":
    main()
