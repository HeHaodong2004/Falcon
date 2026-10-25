"""Revelation-only legal-map proxy audit; no neural inference or occupancy labels.

The fixed three-category calibration uses only the already declared 70 scene
selection partition. Its binary probabilities are evaluated on the other 38
development scenes, which are not an untouched confirmation set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from .common import ROOT, OUT, CHECKPOINT_SHA, EDGES, digest, save, labels
from .statistics import rank_metrics
from .visibility import visibility, frustum_opportunity


BASE = OUT / "visibility"
RULES = ("strict_frontier", "frustum_opportunity", "stage2_raw_argmax", "stage2_revelation_0.5")
WORKER_ROOT = None
WORKER_IDENTITY = None


def _input_digest(graph, trajectory):
    value = hashlib.sha256(pickle.dumps(graph, protocol=5))
    path = np.asarray(trajectory, dtype="<f8")
    value.update(str(path.shape).encode()); value.update(path.tobytes(order="C"))
    return value.hexdigest()


def protocol_path():
    return BASE / "calibration_protocol_v2.json"


def source_identity():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), Path(__file__).with_name("visibility.py"), Path(__file__).with_name("common.py"),
             Path(__file__).with_name("statistics.py"))}


def write_protocol():
    global_protocol = json.loads((OUT / "protocol.json").read_text())
    value = dict(schema="legal_visibility_revelation_diagnostic_v1", source=source_identity(),
                 supersedes_pre_execution="visibility/calibration_protocol_v1.json",
                 revision_reason="Authenticate and restrict summary to current successful requested Root results; preserve failures without mixing stale resume artifacts. No scientific rule/fit/domain change; no C experiment has run.",
                 diagnostic_protocol_sha256=digest(OUT / "protocol.json"),
                 precision_contract_sha256=digest(OUT / "precision_contract.json"),
                 checkpoint_sha256=CHECKPOINT_SHA, expected_roots=2110, expected_windows=47128,
                 candidates="Unchanged Features.candidates('both') from starting legal graph and saved path; assert exact A domain SHA and labels",
                 sensor="Checksummed legal_root sensor mapping only; no future articulated camera state",
                 controls=list(RULES), categories=["outside_frustum", "frustum_only", "strict_frontier"],
                 calibration="Three category revelation frequencies using window-cell-normalized counts, then equal Root within scene and equal selection scene; absent category uses selection prevalence",
                 selection_scenes=global_protocol["threshold_selection_scenes"],
                 validation_scenes=global_protocol["diagnostic_validation_scenes"],
                 split_caveat=global_protocol["split_caveat"],
                 raw_metrics="Per-window binary2x2 counts -> mean withinRoot -> mean withinScene -> mean scene -> precision/recall/F1/revelationIoU; no labeled GraphDeltaIoU",
                 calibrated_metrics="Frozen3category binaryBrier/logloss and expected-count MAE/bias on38validation scenes vs SAME cached Stage2 rows; float64 scores from F32 Stage2 revelation sum, logclip1e-9",
                 rank="Calibrated geometry exact tied-score weighted AUROC/AP; Stage2 same-subset histogram estimate+binning bounds; window-cell-normalized -> Root -> scene",
                 uncertainty="2000 paired validation scene-cluster bootstrap draws seed2526; category fit frozen on70scenes; conditional on that fit",
                 raw_boolean_caveat="Both bool controls are geometric opportunity decisions, not sensor probabilities; label-aware map IoU unavailable because no FREE/BLOCKED labels are assigned",
                 limits="2D base-yaw and C-space frontier surrogate; no exact optical visibility or real-sensor bound; no hidden support/humans/camera articulation",
                 pilot="First2existingdevRoots only; preserve pilot separately; use measured cost before full8CPUrun",
                 launch="No empirical launch until Priority B released by parent; --run requires explicit --priority-b-released; fullA cache must be complete",
                 production_changed=False, neural_weight_fitting=False,
                 statistical_category_calibration=True, neural_inference=False)
    if protocol_path().exists():
        raise FileExistsError("visibility calibration protocol already exists")
    save(protocol_path(), value)
    print(protocol_path())


def binary_confusion(truth, predicted):
    return np.bincount(2 * np.asarray(truth, np.int64) + np.asarray(predicted, np.int64), minlength=4).reshape(2, 2)


def binary_rates(cm):
    tn, fp, fn, tp = np.asarray(cm, np.float64).ravel()
    return dict(precision=float(tp / max(tp + fp, 1e-30)), recall=float(tp / max(tp + fn, 1e-30)),
                f1=float(2 * tp / max(2 * tp + fp + fn, 1e-30)),
                revelation_iou=float(tp / max(tp + fp + fn, 1e-30)),
                false_revelation_rate=float(fp / max(tn + fp, 1e-30)))


def calibrated_window(category_counts, category_rates):
    """Return LL, Brier, count MAE/bias from legal-target diagnostic counts."""
    counts = np.asarray(category_counts, np.float64)
    negative = counts[:, 0]; positive = counts[:, 1:].sum(1)
    rate = np.asarray(category_rates, np.float64)
    n = counts.sum()
    if n <= 0:
        raise ValueError("empty diagnostic candidate domain")
    ce = -np.sum(positive * np.log(np.clip(rate, 1e-9, 1))
                 + negative * np.log(np.clip(1 - rate, 1e-9, 1))) / n
    brier = np.sum(positive * (1 - rate) ** 2 + negative * rate ** 2) / n
    expected = float(np.sum(counts.sum(1) * rate)); actual = float(positive.sum())
    return np.array([ce, brier, abs(expected - actual), expected - actual], np.float64)


def exact_category_rank(category_counts, category_rates):
    """Exact weighted noninterpolated AP/AUC; equal score categories stay tied."""
    counts = np.asarray(category_counts, np.float64)
    masses = defaultdict(lambda: np.zeros(2, np.float64))
    for row, rate in zip(counts, category_rates):
        masses[float(rate)] += [row[0], row[1:].sum()]
    ordered = np.array([masses[rate] for rate in sorted(masses)])
    result = rank_metrics(ordered[:, 0], ordered[:, 1])
    result["auroc_binning_bounds"] = [result["auroc"], result["auroc"]]
    result["auprc_binning_bounds"] = [result["auprc_ap"], result["auprc_ap"]]
    result["method"] = "exact weighted category-score ties; noninterpolated AP"
    return result


def initialize(destination, identity):
    global WORKER_ROOT, WORKER_IDENTITY
    WORKER_ROOT = Path(destination); WORKER_IDENTITY = identity
    from numba import set_num_threads
    set_num_threads(1)


def root_worker(row):
    from eventwm_graph.stage2_segment.data import replay, read
    from eventwm_graph.stage2_quality.features import Features, PreparedGraph
    result_path = WORKER_ROOT / "roots" / (row["root_id"] + ".json.gz")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    a_path = OUT / "probability/roots" / (row["root_id"] + ".json.gz")
    start = time.perf_counter()
    try:
        with gzip.open(a_path, "rt") as stream:
            cached = json.load(stream)
        if cached["checkpoint_sha256"] != CHECKPOINT_SHA:
            raise ValueError("cached checkpoint changed")
        p_path = a_path.with_name(row["root_id"] + ".probabilities.npz")
        h_path = a_path.with_name(row["root_id"] + ".hist.npz")
        if digest(p_path) != cached["probability_cache_sha256"] or digest(h_path) != cached["histogram_sha256"]:
            raise ValueError("Priority A cache hash mismatch")
        source = dict(a_records_sha256=digest(a_path), probability_sha256=digest(p_path), histogram_sha256=digest(h_path))
        if result_path.exists():
            with gzip.open(result_path, "rt") as stream:
                old = json.load(stream)
            if old["identity"] != WORKER_IDENTITY or old["cache_source"] != source:
                raise ValueError("incompatible visibility resume")
            return dict(root_id=row["root_id"], windows=len(old["records"]), seconds=0., resumed=True)
        with np.load(p_path) as arrays:
            p = arrays["p"]; truth = arrays["y"]; offsets = arrays["offsets"]
        if p.dtype != np.float32 or truth.dtype != np.uint8 or len(offsets) != len(cached["records"]) + 1 or offsets[0] != 0 or offsets[-1] != len(p) or len(p) != len(truth):
            raise ValueError("malformed probability cache")
        lookup = {(r["start_node"], r["end_node"]): (index, r) for index, r in enumerate(cached["records"])}
        if len(lookup) != len(cached["records"]):
            raise ValueError("duplicate A window identity")
        sensor = read(row, "legal_root.json.gz")["sensor"]
        prepared = OrderedDict(); records = []; seen = set(); audit_seen = False
        for ex in replay(row, "mixed"):
            if "audit" in ex:
                audit_seen = True; continue
            key = (ex["start_node"], ex["end_node"])
            index, expected = lookup[key]; lo, hi = map(int, offsets[index:index + 2])
            if key in seen:
                raise ValueError("duplicate replay window")
            seen.add(key)
            if ex["start_node"] not in prepared:
                prepared[ex["start_node"]] = PreparedGraph(ex["graph"])
                while len(prepared) > 4:
                    prepared.popitem(last=False)
            prep = prepared[ex["start_node"]]; prepared.move_to_end(ex["start_node"])
            context = Features(ex["graph"], ex["trajectory"], prep)
            keys = context.candidates("both")
            if len(keys) != expected["candidates"] or hashlib.sha256(keys.astype("<i4").tobytes()).hexdigest() != expected["domain_sha256"]:
                raise ValueError("candidate domain changed")
            y = labels(keys, ex["target"])
            np.testing.assert_array_equal(y, truth[lo:hi])
            if not len(y):
                raise ValueError("empty candidate domain")
            initial_input = _input_digest(ex["graph"], ex["trajectory"])
            strict = visibility(ex["graph"], ex["trajectory"], keys, sensor)
            frustum = frustum_opportunity(ex["graph"], ex["trajectory"], keys, sensor)
            if _input_digest(ex["graph"], ex["trajectory"]) != initial_input:
                raise ValueError("geometry baseline mutated supplied graph/path")
            if np.any(strict & ~frustum):
                raise ValueError("frontier is not a subset of opportunity")
            category = frustum.astype(np.int64) + strict.astype(np.int64)
            counts = np.bincount(category * 3 + y, minlength=9).reshape(3, 3)
            probabilities = p[lo:hi]; reveal = (probabilities[:, 1] + probabilities[:, 2]).astype(np.float64)
            observed = y > 0; actual = int(observed.sum()); expected_count = float(reveal.sum())
            stage_scores = np.array([-np.log(np.clip(np.where(observed, reveal, 1 - reveal), 1e-9, 1)).mean(),
                                     np.mean((reveal - observed) ** 2), abs(expected_count - actual), expected_count - actual])
            masks = (strict, frustum, probabilities.argmax(1) > 0, reveal >= np.float64(.5))
            records.append(dict(start_node=ex["start_node"], end_node=ex["end_node"], query_id=ex["query_id"],
                                query_aliases=ex["query_aliases"], candidates=len(y), domain_sha256=expected["domain_sha256"],
                                binary_confusions=[binary_confusion(observed, mask).tolist() for mask in masks],
                                category_label_counts=counts.tolist(), true_count=actual,
                                raw_opportunity_counts=[int(strict.sum()), int(frustum.sum())], stage2_expected_count=expected_count,
                                stage2_scores=stage_scores.tolist(), duration=expected["duration"], groups=expected["groups"],
                                binary_score_outside_unit_interval=int(np.count_nonzero((reveal < 0) | (reveal > 1)))))
        if not audit_seen or len(records) != len(cached["records"]) or seen != set(lookup):
            raise ValueError("incomplete legal replay")
        result = dict(root_id=row["root_id"], scene_id=row["scene_id"], records=records, identity=WORKER_IDENTITY,
                      cache_source=source, legal_member_sha256={name: row["files"][name] for name in
                      ("legal_root.json.gz", "queries.json.gz", "outcomes.json.gz", "future_observations.json.gz")},
                      sensor=sensor, seconds=time.perf_counter() - start, neural_inference=False)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump(result, stream, allow_nan=False)
        temporary.replace(result_path)
        return dict(root_id=row["root_id"], windows=len(records), seconds=result["seconds"], resumed=False)
    except BaseException:
        error = WORKER_ROOT / "failures" / (row["root_id"] + ".json")
        save(error, dict(root_id=row["root_id"], error=traceback.format_exc(), identity=WORKER_IDENTITY))
        return dict(root_id=row["root_id"], windows=0, failed=True, seconds=time.perf_counter() - start)


def summarize(destination):
    protocol = json.loads(protocol_path().read_text())
    completion = json.loads((destination / "complete.json").read_text())
    requested = set(completion["requested_root_ids"])
    failed = set(completion["failed_root_ids"])
    grouped = defaultdict(list); receipts = []
    for path in sorted((destination / "roots").glob("*.json.gz")):
        if path.name.removesuffix(".json.gz") not in requested - failed:
            continue
        with gzip.open(path, "rt") as stream:
            root = json.load(stream)
        if root["identity"] != completion["identity"]:
            raise ValueError("visibility summary encountered stale source/protocol result")
        grouped[root["scene_id"]].append(root)
        receipts.append(dict(path=str(path), sha256=digest(path), windows=len(root["records"])))
    if {root["root_id"] for roots in grouped.values() for root in roots} != requested - failed:
        raise ValueError("missing successful requested C Root")
    if not grouped:
        raise ValueError("no valid C Roots")
    scenes = {}; total_windows = 0; unit_interval_violations = 0
    for scene, roots in grouped.items():
        scene_cm = []; category_mass = []
        for root in roots:
            records = root["records"]; total_windows += len(records)
            scene_cm.append(np.mean([r["binary_confusions"] for r in records], axis=0))
            category_mass.append(np.mean([np.asarray(r["category_label_counts"], np.float64) / r["candidates"] for r in records], axis=0))
            unit_interval_violations += sum(r["binary_score_outside_unit_interval"] for r in records)
        scenes[scene] = dict(cm=np.mean(scene_cm, axis=0), category=np.mean(category_mass, axis=0))
    selection = [scene for scene in protocol["selection_scenes"] if scene in scenes]
    validation = [scene for scene in protocol["validation_scenes"] if scene in scenes]
    if not selection or not validation:
        save(destination / "summary_pending.json", dict(roots=len(receipts), windows=total_windows,
             selection_scenes=len(selection), validation_scenes=len(validation), calibration="PENDING until both partitions represented"))
        return
    fit = np.mean([scenes[scene]["category"] for scene in selection], axis=0)
    mass = fit.sum(1); positives = fit[:, 1:].sum(1); prevalence = float(positives.sum() / fit.sum())
    rates = np.divide(positives, mass, out=np.full(3, prevalence), where=mass > 0)
    validation_scores = []; validation_hist = []
    for scene in validation:
        root_scores = []; root_hist = []
        for root in grouped[scene]:
            records = root["records"]
            geom = np.mean([calibrated_window(r["category_label_counts"], rates) for r in records], axis=0)
            stage = np.mean([r["stage2_scores"] for r in records], axis=0)
            root_scores.append(np.stack([stage, geom]))
            hist_path = OUT / "probability/roots" / (root["root_id"] + ".hist.npz")
            if digest(hist_path) != root["cache_source"]["histogram_sha256"]:
                raise ValueError("Stage2 histogram changed before C summary")
            with np.load(hist_path) as hist:
                h = hist["histogram"][3]
                root_hist.append(np.stack([h[0], h[1] + h[2]]))
        validation_scores.append(np.mean(root_scores, axis=0)); validation_hist.append(np.mean(root_hist, axis=0))
    score_array = np.asarray(validation_scores); mean_scores = score_array.mean(0)
    val_category = np.mean([scenes[scene]["category"] for scene in validation], axis=0)
    val_hist = np.mean(validation_hist, axis=0)
    rng = np.random.default_rng(2526); differences = []
    for _ in range(2000):
        draw = score_array[rng.integers(0, len(validation), len(validation))].mean(0)
        differences.append(draw[1] - draw[0])
    names = ("binary_logloss", "binary_brier", "count_mae", "count_bias")
    result = dict(roots=len(receipts), scenes=len(scenes), windows=total_windows, failures=len(failed),
                  failed_root_ids=sorted(failed), historical_failure_artifacts=len(list((destination / "failures").glob("*.json"))),
                  raw_binary={rule: binary_rates(cm) for rule, cm in zip(RULES, np.mean([s["cm"] for s in scenes.values()], axis=0))},
                  label_aware_graphdelta_iou="UNAVAILABLE: revelation-only baselines assign no FREE/BLOCKED labels",
                  calibration=dict(selection_scenes=len(selection), validation_scenes=len(validation), category_rates=rates.tolist(),
                      selection_category_mass=mass.tolist(), absent_category_fallback_prevalence=prevalence,
                      stage2={name: float(value) for name, value in zip(names, mean_scores[0])},
                      geometry={name: float(value) for name, value in zip(names, mean_scores[1])},
                      paired_scene_geometry_minus_stage2_ci95={name: np.quantile(np.asarray(differences)[:, i], [.025, .975]).tolist() for i, name in enumerate(names)},
                      geometry_rank=exact_category_rank(val_category, rates), stage2_rank=rank_metrics(val_hist[0], val_hist[1]),
                      caveat=protocol["split_caveat"]),
                  binary_stage2_score_outside_unit_interval_cells=unit_interval_violations,
                  proper_score_precision=protocol["calibrated_metrics"], no_probability_change=True,
                  neural_inference=False, occupancy_labels_assigned=False, protocol_sha256=digest(protocol_path()), artifacts=receipts)
    save(destination / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-protocol", action="store_true"); group.add_argument("--run", action="store_true")
    group.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--priority-b-released", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        write_protocol(); return
    config = json.loads(protocol_path().read_text())
    if config["source"] != source_identity() or config["diagnostic_protocol_sha256"] != digest(OUT / "protocol.json"):
        raise ValueError("visibility code/protocol changed after freeze")
    destination = BASE / ("pilot_v1" if args.limit else "full_v1")
    if args.summarize:
        summarize(destination); return
    if not args.priority_b_released or not (OUT / "probability/complete.json").exists():
        raise RuntimeError("wait until Priority A complete and parent releases Priority B")
    from eventwm_graph.stage2.data import source_rows
    rows, source = source_rows("dev")
    if args.limit:
        rows = rows[:args.limit]
    if args.workers not in range(1, 9):
        raise ValueError("use1..8 bounded CPU workers")
    if args.limit not in (0, 2):
        raise ValueError("predeclared C cohort is first2Root pilot or complete original development set")
    identity = dict(source=config["source"], protocol_sha256=digest(protocol_path()),
                    a_complete_sha256=digest(OUT / "probability/complete.json"), source_manifest_sha256=digest(source))
    destination.mkdir(parents=True, exist_ok=True); start = time.perf_counter(); finished = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(destination), identity)) as pool:
        jobs = [pool.submit(root_worker, row) for row in rows]
        for future in as_completed(jobs):
            finished.append(future.result())
            if len(finished) % 20 == 0 or len(finished) == len(rows):
                print(json.dumps(dict(roots=len(finished), windows=sum(r["windows"] for r in finished),
                                      failures=sum(r.get("failed", False) for r in finished), seconds=time.perf_counter() - start)), flush=True)
    if config["source"] != source_identity():
        raise ValueError("visibility source changed during execution")
    failures = sum(r.get("failed", False) for r in finished)
    windows = sum(r["windows"] for r in finished)
    if not args.limit and not failures and (len(finished) != config["expected_roots"] or windows != config["expected_windows"]):
        raise ValueError("incomplete frozen full visibility cohort")
    save(destination / "complete.json", dict(roots=len(finished), windows=windows,
         failures=sum(r.get("failed", False) for r in finished), workers=args.workers, seconds=time.perf_counter() - start,
         requested_root_ids=[row["root_id"] for row in rows],
         failed_root_ids=[row["root_id"] for row in finished if row.get("failed", False)],
         identity=identity, neural_inference=False, checkpoint_changed=False))
    summarize(destination)


if __name__ == "__main__":
    main()
