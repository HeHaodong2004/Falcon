"""CPU-only full-call latency and bounded repeated-planner profiling.

Only legal_root.json.gz and queries.json.gz from the original development
snapshot are opened. No future, outcome, or fresh-cohort payload is needed.
This module does not run a benchmark on import. Outputs are published once.

Example (coordinate the two-thread research allowance before launching):
  nice -n 19 ionice -c 3 python -m eventwm_graph.l2_partial.repair_profile \
    SNAPSHOT NEW_OUTPUT.json --checkpoint CHECKPOINT --population PRIOR.json \
    --event-checkpoint EVENTS.json --limit 60 --repeats 10

Use --factory eventwm_graph.l2_partial.model:GraphBeliefWorldModel and a
different output to measure the retained baseline with identical accounting.
That is the current source baseline, not a replay of a historical timing run.
"""
from __future__ import annotations

import os

# Set before NumPy/Torch load when invoked through python -m. threadpoolctl below
# additionally limits already loaded BLAS pools when used as an imported runner.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import collections
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import numpy as np

DEFAULT_FACTORY = "eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel"
MODES = ("fully_cold", "graph_revalidation", "warm")
PERCENTILES = (50, 95, 99)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _sources():
    return {p.name: _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _metadata():
    repo = Path(__file__).resolve().parents[2]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout
    status = Path("/proc/self/status").read_text()
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    return {"git_revision": git, "git_status": dirty.splitlines(),
            "timestamp_unix": time.time(), "load_average": list(os.getloadavg()),
            "logical_cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "rss_kib": int(fields["VmRSS"].strip().split()[0]),
            "process_threads": int(fields["Threads"].strip()),
            "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def _rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS unavailable")


def _output_path(path):
    path = Path(path).resolve()
    if "l2_partial_graph_wm_explore_20261021" in path.parts:
        raise ValueError("historical exploratory outputs are immutable")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _publish(path, value):
    data = json.dumps(value, indent=2, allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix=".repair_profile_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _clear_caches(model, footprints):
    model.graph_cache.clear()
    if footprints and hasattr(model, "support_footprint_cache"):
        model.support_footprint_cache.clear()


def _cache_state(model):
    footprints = getattr(model, "support_footprint_cache", {})
    return {"support_results": len(model.graph_cache),
            "support_footprints": len(footprints),
            "footprint_array_bytes": sum(sum(a.nbytes for a in value)
                                         for value in footprints.values())}


def _percentiles(values):
    if not values:
        return None
    q = np.percentile(values, PERCENTILES)
    return {**dict(zip(("p50", "p95", "p99"), map(float, q))),
            "n": len(values), "maximum": float(max(values))}


def _summarize(records):
    keys = sorted({key for record in records for key in record["timings_ms"]})
    return {"calls": len(records),
            "roots": len({r["root_id"] for r in records}),
            "scenes": len({r["scene_id"] for r in records}),
            "timings_ms": {key: _percentiles([r["timings_ms"][key] for r in records
                                               if key in r["timings_ms"]])
                           for key in keys}}


def _bucket(value, edges):
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right:
            return str(left) + "-" + (str(right - 1) if math.isfinite(right) else "plus")
    raise ValueError("stratum edges do not cover value")


def _strata(records):
    definitions = {"tracked_slots_max": (0, 1, 3, 6, 10, math.inf),
                   "graph_cells": (0, 1000, 2000, 4000, 8000, math.inf),
                   "distinct_candidates": (0, 2, 5, 13, 25)}
    result = {}
    for field, edges in definitions.items():
        groups = collections.defaultdict(list)
        for record in records:
            groups[(record["mode"], _bucket(record[field], edges))].append(record)
        result[field] = {mode + ":" + bucket: _summarize(values)
                         for (mode, bucket), values in sorted(groups.items())}
    return result


def _selected_rows(snapshot, limit, cohort):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    rows = [row for row in manifest["rows"] if row["split"] == "dev"]
    if cohort:
        spec = json.loads(Path(cohort).read_text())
        requested = spec.get("rows", spec.get("root_ids")) if isinstance(spec, dict) else spec
        if requested is None:
            raise ValueError("cohort must contain rows or root_ids")
        ids = {r["root_id"] if isinstance(r, dict) else r for r in requested}
        if not ids <= {r["root_id"] for r in rows}:
            raise ValueError("profiling cohort must be entirely original development roots")
        rows = [row for row in rows if row["root_id"] in ids]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("empty development cohort")
    return rows


def _causal_input(model, snapshot, row):
    # A deliberately narrow reader: never call data.root_data, which opens truth.
    folder = snapshot / "roots" / row["root_id"]
    legal = _read(folder / "legal_root.json.gz")
    queries = _read(folder / "queries.json.gz")
    if not queries:
        raise ValueError("root has no legal queries: " + row["root_id"])
    history = legal["history"]
    if not history:
        raise ValueError("root has no history")
    began = time.perf_counter()
    if len(history) > 1:
        from ..partial.map import PartialGraph
        graph = PartialGraph(legal["partial_graph"]["resolution"],
                             legal["partial_graph"]["height"])
        for observation in history[:-1]:
            graph.update(observation)
        previous = model.initialize(history[:-1], graph.snapshot())
    else:
        previous = model.initialize(history, legal["partial_graph"])
    initialization_ms = (time.perf_counter() - began) * 1000
    workload = (queries * math.ceil(24 / len(queries)))[:24]
    return previous, history[-1], workload, queries, initialization_ms, len(history)


def _invoke(model, previous, actual, workload, seed):
    # No graph materialization, source-file reads, quantiles or cache inspection
    # in this interval. The public call includes observations and all successors.
    model.timings.clear()
    began = time.perf_counter()
    belief, batch = model.online_call(previous, actual, workload,
                                     horizon=3., num_worlds=16, seed=seed)
    elapsed = (time.perf_counter() - began) * 1000
    timings = {key: float(value) for key, value in batch.timings_ms.items()}
    timings["outer_complete"] = elapsed
    return belief, batch, timings


def _description(row, belief, previous, actual, queries, workload, history_length):
    distinct = len({q["query_id"] for q in workload})
    return {"root_id": row["root_id"], "scene_id": row["scene_id"],
            "tracked_slots_max": max((h.n for h in belief.hypotheses), default=0),
            "tracked_slots_weighted": float(sum(h.weight * h.n for h in belief.hypotheses)),
            "hypotheses": len(belief.hypotheses),
            "unresolved_mass": float(belief.unresolved_mass),
            "visible_detections": sum(bool(r.get("visible_now") and r.get("valid", True))
                                      for r in actual.get("tracks", [])),
            "graph_cells": len(belief.graph.materialize()),
            "graph_depth": belief.graph.depth,
            "graph_namespace_changed": belief.graph.namespace != previous.graph.namespace,
            "new_observation_time": float(actual["timestamp"] - previous.timestamp),
            "has_incremental_observation": history_length > 1,
            "available_candidates": len(queries), "bench_candidates": len(workload),
            "distinct_candidates": distinct, "duplicated_candidate_rows": len(workload) - distinct,
            "worlds": 16, "refined_queries": 4, "physical_horizon_s": 3.}


def _branch_description(batch):
    return {"branches_per_refined_query": [len(p.observation_branches) for p in batch.predictions[:4]],
            "successor_beliefs": sum(len(p.observation_branches) for p in batch.predictions),
            "evaluated_intervals": [list(p.evaluated_interval) for p in batch.predictions[:4]],
            "completed_queries": sum(bool(p.query_complete) for p in batch.predictions),
            "route_invalidated_queries": sum(p.termination_probabilities.get("route_invalidated", 0.) > 0
                                              for p in batch.predictions),
            "horizon_censored_queries": sum(p.termination_probabilities.get("horizon_censored", 0.) > 0
                                             for p in batch.predictions),
            "timeout_queries": sum(p.termination_probabilities.get("query_timeout", 0.) > 0
                                    for p in batch.predictions)}


def profile(snapshot, output, checkpoint=None, population=None, event_checkpoint=None,
            factory=DEFAULT_FACTORY, model_kwargs=None, cohort=None, limit=60,
            repeats=10, cold_repeats=1, rl_repeats=100, rl_roots=3, seed=27):
    if min(limit, rl_repeats, rl_roots) < 0 or repeats < 1 or cold_repeats < 1:
        raise ValueError("invalid benchmark counts")
    output = _output_path(output)
    snapshot = Path(snapshot).resolve()
    rows = _selected_rows(snapshot, limit, cohort)
    source_before = _sources()
    before = _metadata()
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    module, name = factory.split(":", 1)
    constructor = getattr(importlib.import_module(module), name)
    kwargs = dict(model_kwargs or {})
    if "dynamics" in kwargs or "event_model" in kwargs or "population_prior" in kwargs:
        raise ValueError("pass checkpoint, event checkpoint and population through their named arguments")
    began = time.perf_counter()
    if checkpoint:
        from .neural import LearnedDynamics
        kwargs["dynamics"] = LearnedDynamics.load(checkpoint)
    if event_checkpoint:
        from .queries import NativeEventModel
        kwargs["event_model"] = NativeEventModel(json.loads(Path(event_checkpoint).read_text())["coefficients"])
    if population:
        kwargs["population_prior"] = json.loads(Path(population).read_text())
    model = constructor(**kwargs)
    load_ms = (time.perf_counter() - began) * 1000
    if checkpoint and any(p.device.type != "cpu" for p in model.dynamics.network.parameters()):
        raise ValueError("this runner is CPU-only; GPU synchronization is not implemented")
    records = []
    initialization = []
    root_descriptions = []
    repeated = []
    checkpoints = {name: {"path": str(Path(path).resolve()), "sha256": _sha(path)}
                   for name, path in (("checkpoint", checkpoint), ("population", population),
                                      ("event_checkpoint", event_checkpoint)) if path}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for root_index, row in enumerate(rows):
            previous, actual, workload, queries, init_ms, nhistory = _causal_input(model, snapshot, row)
            initialization.append({"root_id": row["root_id"], "prefix_initialization_ms": init_ms})
            # Root-specific priming is excluded and reported as preparation. It
            # creates the same belief and all support footprints used in repeats.
            _clear_caches(model, footprints=True)
            belief, primed, _ = _invoke(model, previous, actual, workload, seed)
            description = _description(row, belief, previous, actual, queries, workload, nhistory)
            root_descriptions.append(description)
            del primed
            # Rotate modes across roots to avoid confounding every cold run with
            # the same scheduler/thermal position. Warm is explicitly re-primed.
            modes = MODES[root_index % 3:] + MODES[:root_index % 3]
            for mode in modes:
                if mode == "warm":
                    _, primed, _ = _invoke(model, previous, actual, workload, seed)
                    del primed
                for repetition in range(repeats if mode == "warm" else cold_repeats):
                    if mode != "warm":
                        _clear_caches(model, footprints=mode == "fully_cold")
                    cache_before = _cache_state(model)
                    belief, batch, timings = _invoke(model, previous, actual, workload, seed)
                    records.append({**description, **_branch_description(batch),
                                    "mode": mode, "repetition": repetition, "seed": seed,
                                    "timings_ms": timings, "cache_before": cache_before,
                                    "cache_after": _cache_state(model)})
                    del batch
            if (root_index + 1) % 10 == 0 or root_index + 1 == len(rows):
                print(json.dumps({"profile_roots_finished": root_index + 1, "roots": len(rows)}), flush=True)

        # Separate planner-loop test: same arrived evidence, newly generated
        # shared worlds on each call. This is not simulated physical execution.
        ranked = sorted(root_descriptions, key=lambda r: (r["tracked_slots_max"], r["graph_cells"]))
        picks = []
        if ranked:
            ordered = [ranked[-1], max(ranked, key=lambda r: r["graph_cells"]), ranked[len(ranked)//2]] + ranked
            for choice in ordered:
                if choice["root_id"] not in picks:
                    picks.append(choice["root_id"])
                if len(picks) >= rl_roots:
                    break
        if not rl_roots:
            picks = []
        row_by_id = {row["root_id"]: row for row in rows}
        for root_id in picks if rl_repeats else []:
            row = row_by_id[root_id]
            previous, actual, workload, queries, init_ms, nhistory = _causal_input(model, snapshot, row)
            belief, batch, _ = _invoke(model, previous, actual, workload, seed)
            description = _description(row, belief, previous, actual, queries, workload, nhistory)
            del batch
            fingerprint = belief.fingerprint()
            memory = [{"call": 0, "rss_kib": _rss_kib(), **_cache_state(model)}]
            samples = []
            for repetition in range(rl_repeats):
                successor, batch, timings = _invoke(model, belief, actual, workload, seed + repetition)
                unchanged = successor.fingerprint() == fingerprint
                samples.append({**description, "mode": "repeated_planner_no_new_evidence",
                                "repetition": repetition, "seed": seed + repetition,
                                "timings_ms": timings, "belief_unchanged": unchanged})
                del batch, successor
                if (repetition + 1) % 10 == 0 or repetition + 1 == rl_repeats:
                    memory.append({"call": repetition + 1, "rss_kib": _rss_kib(), **_cache_state(model)})
            repeated.append({"root_id": root_id, "scene_id": row["scene_id"],
                             "selection": "dense/largest-graph/median development cases; diagnostic only",
                             "prefix_initialization_ms": init_ms, "records": samples,
                             "summary": _summarize(samples), "memory_samples": memory,
                             "rss_change_kib": memory[-1]["rss_kib"] - memory[0]["rss_kib"],
                             "all_beliefs_unchanged": all(r["belief_unchanged"] for r in samples)})

    source_after = _sources()
    result = {"schema": "partial_graph_repair_full_path_profile_1",
              "factory": factory, "model_kwargs": model_kwargs or {}, "artifacts": checkpoints,
              "snapshot": str(snapshot), "snapshot_manifest_sha256": _sha(snapshot / "manifest.json"),
              "cohort": {"split": "dev", "selection": "manifest order, optional explicit cohort, then limit",
                         "limit": limit, "root_ids": [r["root_id"] for r in rows],
                         "roots": len(rows), "scenes": len({r["scene_id"] for r in rows}),
                         "cohort_file_sha256": _sha(cohort) if cohort else None},
              "workload": {"horizon_s": 3., "worlds": 16, "candidate_rows": 24, "refined_queries": 4,
                           "observation_branches": "actual count recorded; no branch removal by profiler",
                           "mode_order": "rotated by root; each warm block explicitly primed",
                           "fully_cold": "clear result and footprint caches before each timed call",
                           "graph_revalidation": "clear result cache; retain pose footprints; assimilate actual legal graph delta",
                           "graph_change_qualification": "record actual namespace change; forced revalidation also covers unchanged geometry",
                           "warm": "repeat genuine observation update from the same causal preceding prefix",
                           "cold_excludes": "checkpoint loading and initialization, reported separately",
                           "repeated_planner": "same arrived belief/evidence; vary forecast seed; no physical rollout claim"},
              "resource_policy": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                                  "torch_interop_threads": torch.get_num_interop_threads(),
                                  "threadpools": pools, "nice": 19, "io_class": "idle",
                                  "production_coexistence": "not established by this runner; qualify using independent collector status audit"},
              "metadata_before": before, "metadata_after": _metadata(),
              "sources_before": source_before, "sources_after": source_after,
              "source_changed_during_measurement": source_before != source_after,
              "model_load_ms": load_ms, "initialization": initialization,
              "initialization_summary_ms": _percentiles([r["prefix_initialization_ms"] for r in initialization]),
              "records": records, "root_descriptions": root_descriptions,
              "summary": {mode: _summarize([r for r in records if r["mode"] == mode]) for mode in MODES},
              "strata": _strata(records), "repeated_planner": repeated,
              "limitations": ["P95/P99 are empirical invocation quantiles; roots and repeated calls are not independent trials.",
                              "Real density is legal visible/inferred slot count; no privileged count payload was opened.",
                              "Repeated candidate rows are computational load, not distinct-route evidence.",
                              "RSS changes combine Python/NumPy allocator retention and live state; short flat traces cannot prove absence of leaks.",
                              "Performance gates apply to this declared CPU workload and collector/load condition only."]}
    _publish(output, result)
    print(json.dumps({mode: result["summary"][mode]["timings_ms"]["outer_complete"] for mode in MODES}, indent=2))
    if source_before != source_after:
        raise RuntimeError("source changed during benchmark; output preserved but cannot serve as a release timing")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--population")
    parser.add_argument("--event-checkpoint")
    parser.add_argument("--factory", default=DEFAULT_FACTORY)
    parser.add_argument("--model-kwargs", help="JSON file of factory keyword arguments")
    parser.add_argument("--cohort", help="JSON development-only root list or rows; no fresh/holdout cohort")
    parser.add_argument("--limit", type=int, default=60, help="manifest-order roots; 0 selects all development roots")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cold-repeats", type=int, default=1)
    parser.add_argument("--rl-repeats", type=int, default=100)
    parser.add_argument("--rl-roots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=27)
    args = vars(parser.parse_args())
    if args["model_kwargs"]:
        args["model_kwargs"] = json.loads(Path(args["model_kwargs"]).read_text())
    profile(**args)


if __name__ == "__main__":
    main()
