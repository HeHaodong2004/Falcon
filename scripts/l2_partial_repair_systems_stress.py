"""Bounded SYNTHETIC systems stress; never reads collection root payloads.

Run only in an available one-thread research slot. This does not replace the
real development-root profile or measure predictive accuracy. The count pmf and
legal certificates below are explicitly constructed stress-fixture assumptions.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import pathlib
import resource
import signal
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eventwm_graph.l2_partial.hidden import prepare_prior
from eventwm_graph.l2_partial.neural import LearnedDynamics
from eventwm_graph.l2_partial.queries import NativeEventModel
from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
from eventwm_graph.l2_partial.state import BeliefState, Hypothesis, ObservedGraph, EXECUTOR_VERSION


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def sources():
    return {p.name: sha(p) for p in sorted((REPO / "eventwm_graph/l2_partial").glob("*.py"))}


def collector():
    path = pathlib.Path("/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1/status.json")
    raw = json.loads(path.read_text())
    keys = ("status", "status_reason", "outcomes", "roots", "active_worker_count",
            "collector_restart_count", "supervisor_pid", "validated_outcomes_per_second_recent")
    return {"time_unix": time.time(), "path": str(path), "status": {k: raw.get(k) for k in keys}}


def fixture(model, n, certificate):
    graph = ObservedGraph("synthetic-certified-square-61x61-v1", 1, .25, 0.,
                          {(x, z): (1, 0.) for x in range(-30, 31) for z in range(-30, 31)})
    means = np.zeros((n, 4))
    means[:, 0] = 3. + .6 * np.arange(n)
    means[:, 1] = np.where(np.arange(n) % 2, -1.4, 1.4)
    means[:, 2] = -.2
    hm, hc = prepare_prior(np.zeros(4))
    hypotheses = [Hypothesis(float(weight), means.copy(), np.tile(np.diag([.04, .04, .25, .25]), (n, 1, 1)),
                            np.zeros(n), np.arange(n), np.array([.25, 0., .5, 0., .25]),
                            height=np.zeros(n), visibility_mode=regime,
                            hidden_mean=hm.copy(), hidden_covariance=hc.copy())
                  for regime, weight in enumerate(model.observation_law.regime_prior)]
    belief = BeliefState(graph, hypotheses, 0., np.zeros(4), model_version=model.model_version).validate()
    belief = model.propagate(belief, 0.).validate()
    row = {"timestamp": .048, "robot_pose": [0., 0., 0., 0.], "static_updates": [], "tracks": []}
    if certificate:
        row["certified_visible_empty_boxes"] = [[[-1., -1.], [1., 1.]]]
    else:
        row["tracks"] = [{"track_id": i, "position": [float(p[0] - .2 * .048), 0., float(p[1])],
                          "visible_now": True, "valid": True, "last_seen_time": .048}
                         for i, p in enumerate(means)]
    queries = []
    for qi in range(24):
        count = 40 + qi
        poses = [np.zeros(4)]
        actions = []
        for index in range(count):
            pose = poses[-1].copy()
            if index < 12:
                pose[0] += 25 / 120
                actions.append("forward")
            else:
                pose[3] += 10 / 120
                actions.append("left")
            poses.append(pose)
        queries.append({"query_id": "synthetic_distinct_" + str(qi), "executor_version": EXECUTOR_VERSION,
                        "compiled": {"planned_poses": np.asarray(poses).tolist(), "actions": actions, "graph_revision": 1},
                        "waypoints": [[2.5, 0., 0.]], "final_yaw": float(poses[-1][3])})
    return belief, row, queries


def validate(belief, batch, row, initial_count):
    belief.validate()
    expected = {tuple(np.asarray(t["position"])[[0, 2]]) for t in row["tracks"]}
    allowed_counts = {initial_count, initial_count + 2, initial_count + 4}
    beliefs = [belief] + [branch.successor for query in batch.predictions for branch in query.observation_branches]
    for b in beliefs:
        b.validate()
        if b.graph.namespace != "synthetic-certified-square-61x61-v1":
            raise AssertionError("synthetic observation changed certified geometry")
        for h in b.hypotheses:
            if h.weight <= 0:
                continue
            if any(h.n + int(k) not in allowed_counts for k in np.flatnonzero(h.hidden_pmf > 1e-12)):
                raise AssertionError("population support changed without a population event")
    for h in belief.hypotheses:
        if h.weight > 0 and expected and not expected <= {tuple(p) for p in h.mean[:, :2]}:
            raise AssertionError("arrived exact detected position missing in retained posterior")
    branch_errors = []
    for query in batch.predictions:
        if not np.isclose(sum(query.termination_probabilities.values()), 1., atol=1e-8):
            raise AssertionError("termination probability mass")
        if not np.isclose(query.duration_weights.sum(), 1., atol=1e-8):
            raise AssertionError("duration probability mass")
        elapsed = query.evaluated_interval[1] - query.evaluated_interval[0]
        if elapsed > 3. + 1e-8 or not np.isclose(elapsed / .048, round(elapsed / .048), atol=1e-6):
            raise AssertionError("native evaluated endpoint clock")
        if query.observation_branches:
            error = abs(sum(br.probability for br in query.observation_branches) - 1.)
            branch_errors.append(error)
            if error > 1e-8:
                raise AssertionError("branch probability mass")
    return {"exact_positions_checked": bool(expected), "resolved_hypotheses": len(belief.hypotheses),
            "unresolved_mass": belief.unresolved_mass, "population_support": sorted(allowed_counts),
            "validated_successors": len(beliefs) - 1, "maximum_branch_mass_error": max(branch_errors, default=0.),
            "posterior_expected_count_resolved_mass": float(sum(h.weight * (h.n + h.expected_hidden) for h in belief.hypotheses)),
            "observation_evidence": belief.evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "population", "event_checkpoint", "optical", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--seconds", type=float, default=45.)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    base = REPO / "output/l2_partial_graph_wm_repair_20261021"
    if not output.is_relative_to(base) or output.exists():
        raise ValueError("new artifact inside research repair output required")
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = sources()
    refs = {name: {"path": getattr(args, name), "sha256": sha(getattr(args, name))}
            for name in ("checkpoint", "population", "event_checkpoint", "optical")}
    before = collector()
    results = []
    stopped = None
    failure = None
    def expired(signum, frame):
        raise TimeoutError("bounded synthetic stress wall deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1), pathlib.Path(str(output) + ".jsonl").open("x") as log:
            model = RepairedGraphBeliefWorldModel(dynamics=LearnedDynamics.load(args.checkpoint),
                population_prior=json.loads(pathlib.Path(args.population).read_text()),
                event_model=NativeEventModel(json.loads(pathlib.Path(args.event_checkpoint).read_text())["coefficients"]),
                observation_checkpoint=args.optical)
            for n, certificate in ((6, False), (12, False), (6, True), (12, True)):
                setup_started = time.perf_counter()
                belief, row, queries = fixture(model, n, certificate)
                setup_ms = (time.perf_counter() - setup_started) * 1000
                fingerprint = belief.fingerprint()
                for mode in ("fully_cold", "warm", "warm"):
                    if mode == "fully_cold":
                        model.graph_cache.clear()
                        if hasattr(model, "support_footprint_cache"):
                            model.support_footprint_cache.clear()
                    label = {"tracks": n, "explicit_certificate": certificate, "mode": mode}
                    log.write(json.dumps({**label, "started": True}) + "\n")
                    log.flush()
                    wall = time.perf_counter()
                    cpu = time.process_time()
                    posterior, batch = model.online_call(belief, row, queries, horizon=3., num_worlds=16, seed=73)
                    cpu_ms = (time.process_time() - cpu) * 1000
                    wall_ms = (time.perf_counter() - wall) * 1000
                    checks = validate(posterior, batch, row, n)
                    if belief.fingerprint() != fingerprint:
                        raise AssertionError("input belief changed during counterfactual evaluation")
                    record = {**label, "outer_wall_ms": wall_ms, "process_cpu_ms": cpu_ms,
                              "components_ms": batch.timings_ms, "setup_ms_excluded": setup_ms,
                              "checks": checks, "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    results.append(record)
                    log.write(json.dumps(record, allow_nan=False, default=json_default) + "\n")
                    log.flush()
    except TimeoutError as exc:
        stopped = str(exc)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
    result = {"schema": "synthetic_repair_systems_stress_1", "synthetic": True,
              "workload": {"candidate_queries": 24, "distinct_queries": 24, "worlds": 16, "refined": 4,
                           "horizon_seconds": 3., "native_dt": .048, "hidden_count_values": [0, 2, 4],
                           "hidden_count_probabilities": [.25, .5, .25], "graph_cells": 3721},
              "claims": "synthetic computation/correctness only; no predictive or real-population claim; no collection payload read",
              "source_sha256": source, "sources_unchanged": source == sources(), "script_sha256": sha(__file__),
              "references": refs, "collector_before": before, "collector_after": collector(),
              "records": results, "deadline_stop": stopped, "failure": failure,
              "elapsed_seconds": time.perf_counter() - started,
              "quantile_limit": "two warm repeats per case are descriptive timings, not tail-latency evidence"}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=json_default)
    print(json.dumps({"records": len(results), "deadline_stop": stopped, "failure": failure,
                      "elapsed_seconds": result["elapsed_seconds"]}))
    if failure:
        raise RuntimeError(failure)
    if not result["sources_unchanged"]:
        raise RuntimeError("package source changed during probe")


if __name__ == "__main__":
    main()
