"""Fixed non-neural soft opportunity control on the frozen recorded B pairs.

Only starting G, keys, recorded sensor configuration and supplied poses enter
the scorer. Targets enter the separate calibration/evaluation statistics only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT, OUT, EDGES, digest, save
from .statistics import rank_metrics

BASE = OUT / "visibility/soft_paired_v1"
B = OUT / "trajectory/primary_v1"
SCORE_EDGES = np.linspace(0., 6., 33)


def opportunity_score(graph, trajectory, keys, sensor):
    """Float64 dt-weighted distance/bearing opportunity; no occupancy output."""
    path = np.asarray(trajectory, np.float64); keys = np.asarray(keys)
    if path.ndim != 2 or path.shape[1] != 5 or not len(path) or not np.isfinite(path).all():
        raise ValueError("finite saved [t,x,y,z,yaw] path required")
    if np.any(np.diff(path[:, 0]) <= 0) or abs(path[0, 0] - graph["timestamp"]) > 1e-9:
        raise ValueError("invalid causal saved boundaries")
    if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype.kind not in ("i", "u"):
        raise ValueError("fixed integer[N,2] keys required")
    resolution = float(graph["resolution"]); hfov = float(sensor["horizontal_fov_deg"])
    if not np.isfinite([resolution, hfov]).all() or resolution <= 0 or not 0 < hfov < 180 or abs(float(sensor["max_range_m"]) - 10.) > 1e-9:
        raise ValueError("recorded10m range and valid horizontal FOV required")
    centers = (keys.astype(np.float64) + .5) * resolution
    score = np.zeros(len(keys), np.float64); cutoff = np.cos(np.deg2rad(hfov / 2))
    for index in range(1, len(path)):
        pose = path[index]; dx = centers[:, 0] - pose[1]; dz = centers[:, 1] - pose[3]
        distance = np.hypot(dx, dz)
        facing = np.divide(dx * np.cos(pose[4]) - dz * np.sin(pose[4]), distance,
                           out=np.zeros(len(keys), np.float64), where=distance > 0)
        facing = np.clip(facing, -1., 1.)
        keep = (distance > 0) & (distance <= 10.) & (facing >= cutoff)
        score[keep] += (path[index, 0] - path[index - 1, 0]) * np.maximum(0., 1 - distance[keep] / 10.) * np.maximum(0., (facing[keep] - cutoff) / (1 - cutoff))
    return score


def score_bins(score):
    score = np.asarray(score, np.float64)
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("invalid geometric score")
    bins = np.minimum(np.searchsorted(SCORE_EDGES[1:], score, side="left") + 1, 32)
    bins[score == 0] = 0
    return bins.astype(np.uint8)


def fit_pav(counts):
    """Weighted PAV; unsupported bins interpolate fitted supported frequencies."""
    counts = np.asarray(counts, np.float64); mass = counts.sum(1); support = np.flatnonzero(mass > 0)
    if not len(support):
        raise ValueError("no calibration mass")
    blocks = []
    for index in support:
        blocks.append([int(index), int(index), mass[index], counts[index, 1]])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop(); left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(support), np.float64)
    for lo, hi, total, positive in blocks:
        fitted[(support >= lo) & (support <= hi)] = positive / total
    rates = np.interp(np.arange(len(mass)), support, fitted)
    if np.any(np.diff(rates) < -1e-14) or np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("PAV probability constraint failed")
    return rates


def binary_scores(p, y):
    p = np.asarray(p, np.float64); y = np.asarray(y, bool)
    return np.array([-np.log(np.clip(np.where(y, p, 1 - p), 1e-9, 1)).mean(), np.mean((p - y) ** 2)])


def histogram(bins, y, length=33):
    return np.bincount(2 * np.asarray(bins, np.int64) + np.asarray(y, np.int64), minlength=2 * length).reshape(length, 2)


def histogram_scores(hist, rates):
    hist = np.asarray(hist, np.float64); rates = np.asarray(rates, np.float64); n = hist.sum()
    return np.array([-np.sum(hist[:, 1] * np.log(np.clip(rates, 1e-9, 1)) + hist[:, 0] * np.log(np.clip(1 - rates, 1e-9, 1))) / n,
                     np.sum(hist[:, 1] * (1 - rates) ** 2 + hist[:, 0] * rates ** 2) / n])


def exact_calibrated_rank(hist, rates):
    groups = defaultdict(lambda: np.zeros(2))
    for row, rate in zip(hist, rates):
        groups[float(rate)] += row
    values = np.array([groups[rate] for rate in sorted(groups)])
    result = rank_metrics(values[:, 0], values[:, 1]); result["method"] = "exact tied PAV-bin score AP/AUC"
    result["auroc_binning_bounds"] = [result["auroc"], result["auroc"]]
    result["auprc_binning_bounds"] = [result["auprc_ap"], result["auprc_ap"]]
    return result


def protocol_path():
    return BASE / "protocol.json"


def write_protocol():
    p = json.loads((OUT / "protocol.json").read_text())
    value = dict(schema="soft_paired_geometry_v1", source_sha256=digest(__file__),
        statistics_sha256=digest(Path(__file__).with_name("statistics.py")), common_sha256=digest(Path(__file__).with_name("common.py")),
        diagnostic_protocol_sha256=digest(OUT / "protocol.json"), b_complete_sha256=digest(B / "complete.json"),
        b_pair_manifest_sha256=digest(OUT / "trajectory/availability_v1/pairs.json"), expected_pairs=1705,
        selection_scenes=p["threshold_selection_scenes"], validation_scenes=p["diagnostic_validation_scenes"],
        expected_selection_pairs=1136, expected_validation_pairs=569,
        formula="sum_i>0 dt_i*max(0,1-planar_distance/10)*max(0,(cosbearing-cos(HFOV/2))/(1-cos(HFOV/2))) with0<distance<=10 and HFOV mask; only supplied saved poses, no interpolation",
        zero_distance="Undefined bearing at zero planar range contributes0, a conservative fixed convention",
        source="Current legalG from authenticated Bmapcache; sensor only from checksummed legal_root; fixedU keys and A/B paths from B; no target passed to scorer",
        bins="Exactzero separate;32rightclosed positive bins over(0,6]; overflow saturates final calibration bin and audited; no hyperparameter search",
        calibration="Branchmean of percellnormalized bin counts perRoot, RootmeanwithinScene, equal70selectionScene; weightedPAV; emptybins monotone linear interpolation in binindex and constanttails",
        evaluation="Frozen38validationScenes on same fixedU vs cached Stage2direct_U(primary, extended-support caveat) and production-padded_U; binaryLL/Brier/countMAE/bias, exact PAVscore rank, raw continuous score/6 highresolutionhist rank withbinbounds; paired own/cross binaryLLgain",
        uncertainty="2000pairedvalidationScene bootstrapdraws seed2527, bothbranches staytogether; conditionalonfixed70scene calibration",
        raw_score_range="Scores>6 clipped for rawrank histogram only with audited count/max; calibrated proper/count metrics use finalbin; no claim of uncapped rank if overflow occurs",
        split_caveat=p["split_caveat"], approximation="2D baseyaw opportunity, no optical/camera articulation/occluder/native support model; UNKNOWN is never called FREE; no occupancy labels",
        outputs="Perpair F64softscore/bin caches + proper/hist stats + A/B predicted revelation counts for matched Ddiagnostics",
        max_extra_cpu_workers=2, neural_inference=False, production_changed=False, checkpoint_changed=False)
    if protocol_path().exists():
        raise FileExistsError("soft protocol exists")
    save(protocol_path(), value); print(protocol_path())


def pair_worker(job):
    row, path = job; path = Path(path); root_id = row["root_id"]; output = BASE / "roots" / (root_id + ".json")
    record = json.loads(path.read_text()); cache = path.with_suffix(".npz"); maps = path.with_name(root_id + ".maps.json.gz")
    source = dict(record_sha256=digest(path), cache_sha256=digest(cache), maps_sha256=digest(maps))
    if source["cache_sha256"] != record["cache_sha256"] or source["maps_sha256"] != record["maps_cache_sha256"]:
        raise ValueError("B cache identity changed")
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["source"] != source or previous["protocol_sha256"] != digest(protocol_path()) or digest(output.with_suffix(".npz")) != previous["score_cache_sha256"]:
            raise ValueError("incompatible soft cache resume")
        return root_id
    with gzip.open(maps, "rt") as stream:
        mapped = json.load(stream)
        if mapped["input_graph_sha256"] != record["graph_sha256"]:
            raise ValueError("B current graph identity mismatch")
        graph = mapped["current_legal_graph"]
    from eventwm_graph.stage2_segment.data import read
    sensor = read(row, "legal_root.json.gz")["sensor"]
    if row["files"]["legal_root.json.gz"] != record["pair"]["source_member_sha256"]["legal_root.json.gz"]:
        raise ValueError("sensor source identity mismatch")
    with np.load(cache, allow_pickle=False) as arrays:
        keys = arrays["keys"]; ys = [arrays["y_a"] > 0, arrays["y_b"] > 0]
        ps = {name: [(arrays[name + "_a"][:, 1] + arrays[name + "_a"][:, 2]).astype(np.float64),
                     (arrays[name + "_b"][:, 1] + arrays[name + "_b"][:, 2]).astype(np.float64)] for name in ("direct", "production")}
    known = {tuple(c["cell"]) for c in graph["cells"]}
    if any(tuple(key) in known for key in keys):
        raise ValueError("fixedU contains known cells")
    scores = [opportunity_score(graph, record["pair"]["trajectory_" + branch], keys, sensor) for branch in ("a", "b")]
    bins = [score_bins(score) for score in scores]; counts = [histogram(b, y) for b, y in zip(bins, ys)]
    crossed = [histogram(bins[0], ys[1]), histogram(bins[1], ys[0])]
    histograms = {"raw_geometry": sum(histogram(np.clip(np.searchsorted(EDGES, np.clip(score / 6, 0, 1), side="right") - 1, 0, len(EDGES) - 2), y, len(EDGES) - 1) for score, y in zip(scores, ys))}
    stage = {}
    for name, values in ps.items():
        own = .5 * (binary_scores(values[0], ys[0]) + binary_scores(values[1], ys[1]))
        cross = .5 * (binary_scores(values[0], ys[1]) + binary_scores(values[1], ys[0]))
        expected = [float(v.sum()) for v in values]; actual = [int(y.sum()) for y in ys]
        stage[name] = dict(own_scores=own.tolist(), cross_scores=cross.tolist(), logloss_gain=float(cross[0] - own[0]),
                           expected_counts=expected, count_mae=float(np.mean(abs(np.asarray(expected) - actual))), count_bias=float(np.mean(np.asarray(expected) - actual)))
        histograms[name] = sum(histogram(np.clip(np.searchsorted(EDGES, value, side="right") - 1, 0, len(EDGES) - 2), y, len(EDGES) - 1) for value, y in zip(values, ys))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), score_a=scores[0], score_b=scores[1], bin_a=bins[0], bin_b=bins[1], **histograms)
    result = dict(root_id=root_id, scene_id=record["scene_id"], split=record["pair"]["diagnostic_scene_split"], candidates=len(keys),
                  counts=[c.tolist() for c in counts], crossed_counts=[c.tolist() for c in crossed], stage2=stage,
                  true_counts=[int(y.sum()) for y in ys], maximum_score=max(float(s.max()) for s in scores),
                  score_overflow_count=sum(int((s > 6).sum()) for s in scores), source=source,
                  scoring_domain="fixed_U_only", out_of_union_truth_counts=record["out_of_union_truth_counts"],
                  stage2_binary_score_outside_unit_interval=sum(int(((v < 0) | (v > 1)).sum()) for values in ps.values() for v in values),
                  score_cache_sha256=digest(output.with_suffix(".npz")), protocol_sha256=digest(protocol_path()), sensor=sensor)
    save(output, result); return root_id


def summarize():
    protocol = json.loads(protocol_path().read_text()); records = [json.loads(p.read_text()) for p in sorted((BASE / "roots").glob("*.json"))]
    scenes = defaultdict(list)
    for record in records:
        if record["protocol_sha256"] != digest(protocol_path()):
            raise ValueError("soft result protocol mismatch")
        scenes[record["scene_id"]].append(record)
    selection = [s for s in protocol["selection_scenes"] if s in scenes]; validation = [s for s in protocol["validation_scenes"] if s in scenes]
    fit = np.mean([np.mean([np.sum(r["counts"], axis=0) / (2 * r["candidates"]) for r in scenes[s]], axis=0) for s in selection], axis=0)
    rates = fit_pav(fit); scene_scores = []; hist_scenes = []; calibrated_scenes = []; count_table = []
    for scene in selection + validation:
        root_scores = []; root_hist = []; calibrated_hist = []
        for r in scenes[scene]:
            h = np.asarray(r["counts"]); ch = np.asarray(r["crossed_counts"])
            own = np.mean([histogram_scores(a, rates) for a in h], axis=0); cross = np.mean([histogram_scores(a, rates) for a in ch], axis=0)
            expected = [float(np.dot(a.sum(1), rates)) for a in h]; errors = np.asarray(expected) - r["true_counts"]
            geom = np.r_[own, np.mean(abs(errors)), np.mean(errors), cross[0] - own[0]]
            root_scores.append(np.stack([np.r_[r["stage2"][name]["own_scores"], r["stage2"][name]["count_mae"], r["stage2"][name]["count_bias"], r["stage2"][name]["logloss_gain"]] for name in ("direct", "production")] + [geom]))
            count_table.append(dict(root_id=r["root_id"], scene_id=scene, split=r["split"], true_counts=r["true_counts"],
                                    geometry_expected_counts=expected, stage2_direct_expected_counts=r["stage2"]["direct"]["expected_counts"], stage2_production_expected_counts=r["stage2"]["production"]["expected_counts"]))
            if scene in validation:
                cache = BASE / "roots" / (r["root_id"] + ".npz")
                if digest(cache) != r["score_cache_sha256"]:
                    raise ValueError("soft score cache changed")
                with np.load(cache) as arrays:
                    root_hist.append(np.stack([arrays[name] / (2 * r["candidates"]) for name in ("direct", "production", "raw_geometry")]))
                calibrated_hist.append(h.sum(0) / (2 * r["candidates"]))
        if scene in validation:
            scene_scores.append(np.mean(root_scores, axis=0)); hist_scenes.append(np.mean(root_hist, axis=0)); calibrated_scenes.append(np.mean(calibrated_hist, axis=0))
    values = np.asarray(scene_scores); mean = values.mean(0); hist = np.mean(hist_scenes, axis=0)
    draws = np.random.default_rng(2527).integers(0, len(validation), size=(2000, len(validation)))
    diff = values[draws].mean(1)[:, 2] - values[draws].mean(1)[:, 0]
    fields = ["binary_logloss", "binary_brier", "count_mae", "count_bias", "own_cross_logloss_gain"]
    result = dict(pairs=len(records), selection_pairs=sum(len(scenes[s]) for s in selection), validation_pairs=sum(len(scenes[s]) for s in validation),
                  selection_scenes=len(selection), validation_scenes=len(validation), rates=rates.tolist(), selection_bin_mass=fit.sum(1).tolist(),
                  metrics={name: dict(zip(fields, map(float, row))) for name, row in zip(("stage2_direct", "stage2_production", "calibrated_geometry"), mean)},
                  paired_scene_geometry_minus_direct_ci95={name: np.quantile(diff[:, i], [.025, .975]).tolist() for i, name in enumerate(fields)},
                  rank={name: rank_metrics(h[:, 0], h[:, 1]) for name, h in zip(("stage2_direct", "stage2_production", "raw_continuous_geometry"), hist)},
                  calibrated_geometry_rank=exact_calibrated_rank(np.mean(calibrated_scenes, axis=0), rates),
                  maximum_score=max(r["maximum_score"] for r in records), score_overflow_count=sum(r["score_overflow_count"] for r in records),
                  out_of_union_truth_counts=sum(np.asarray(r["out_of_union_truth_counts"]) for r in records).tolist(),
                  scoring_domain="fixed_U_only; out-of-union target counts audited separately",
                  stage2_binary_score_outside_unit_interval=sum(r["stage2_binary_score_outside_unit_interval"] for r in records),
                  protocol_sha256=digest(protocol_path()), no_occupancy_labels=True, no_neural_inference=True,
                  caveats=[protocol["approximation"], protocol["split_caveat"], "Stage2directU includes extended-support diagnostic rows; production-paddedU separately reported"])
    if (result["pairs"], result["selection_pairs"], result["validation_pairs"]) != (1705, 1136, 569):
        raise ValueError("incomplete fixed B cohort")
    save(BASE / "summary.json", result); save(BASE / "branch_expected_counts.json", dict(records=count_table, calibration_scope="70selectionfit; frozen38validation", protocol_sha256=digest(protocol_path())))


def main():
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-protocol", action="store_true"); group.add_argument("--run", action="store_true"); group.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=2); args = parser.parse_args()
    if args.write_protocol:
        write_protocol(); return
    protocol = json.loads(protocol_path().read_text())
    if (protocol["source_sha256"] != digest(__file__) or protocol["b_complete_sha256"] != digest(B / "complete.json")
            or protocol["statistics_sha256"] != digest(Path(__file__).with_name("statistics.py"))
            or protocol["common_sha256"] != digest(Path(__file__).with_name("common.py"))
            or protocol["diagnostic_protocol_sha256"] != digest(OUT / "protocol.json")):
        raise ValueError("soft source/B cohort changed")
    if args.summarize:
        summarize(); return
    if not 1 <= args.workers <= 2:
        raise ValueError("use at most2extra CPU workers")
    from eventwm_graph.stage2.data import source_rows
    rows, _ = source_rows("dev"); lookup = {r["root_id"]: r for r in rows}; paths = sorted((B / "roots").glob("*.json"))
    if len(paths) != 1705:
        raise ValueError("fixed1705B pairs required")
    start = time.perf_counter(); finished = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(pair_worker, (lookup[path.stem], str(path))) for path in paths]
        for future in as_completed(jobs):
            finished.append(future.result())
            if len(finished) % 100 == 0 or len(finished) == len(paths):
                print(json.dumps(dict(pairs=len(finished), seconds=time.perf_counter() - start)), flush=True)
    if protocol["source_sha256"] != digest(__file__):
        raise ValueError("soft source changed during execution")
    save(BASE / "complete.json", dict(pairs=len(finished), seconds=time.perf_counter() - start, workers=args.workers, protocol_sha256=digest(protocol_path())))
    summarize()


if __name__ == "__main__":
    main()
