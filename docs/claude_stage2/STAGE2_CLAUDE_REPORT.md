# Stage 2 redesign — legal voxel simulation + masked BEV U-Net (Claude, 2026-10-26)

Status: DRAFT — final accuracy/latency tables are filled in from the frozen
evaluation and benchmark receipts (section 1).

## 1. Results

### 1.0 Current best: frozen model `final/v11.pt` (checkpoint `training/v11_ext/final.pt` + frozen operating point), 256 px sensing

Lineage: v7 (2 epochs) -> v7_restart_iou (1-epoch warm restart + soft IoU loss) -> v11_ext (1-epoch warm
restart with three new legal input channels, zero-initialized coarse-level capacity, EMA, +2,914 TRAIN
Roots). Operating point selected on 400 scene-disjoint TRAIN-validation Roots (`select.py`): class rule,
FREE if r(1-b) > 0.475, BLOCKED if rb > 0.575 (validation 0.4750; v7r 0.4711; v7 0.4628). Sensor
resolution selected on the 199 TRAIN-validation Roots (22 scenes) that have both packet resolutions
(`sensor_select.py`): 256 px 0.4621 vs 128 px 0.4597. ONE original full-development evaluation
(`evaluate.py`, `--sensor-directory full_range_branches_256`, original assertions, `summarize()`):

| Model | sensing | full-dev label-aware IoU | revelation IoU | pooled IoU | bootstrap 95 % |
| --- | --- | ---: | ---: | ---: | --- |
| graph-only baseline (original receipts) | — | 0.2177 | 0.2870 | 0.2240 | |
| previous best A (handoff) | 128 px | 0.4171 | 0.4667 | 0.3992 | |
| v7 (section 1.1) | 128 px | 0.4563 | 0.4903 | 0.4423 | [0.4438, 0.4687] |
| **v11 (this section)** | 256 px | **0.4688** | **0.5026** | **0.4552** | [0.4557, 0.4815] |

| metric | graph-only baseline | v11 |
| --- | ---: | ---: |
| hierarchical label_aware_delta_iou | 0.217733 | 0.468801 |
| hierarchical revelation_iou | 0.286985 | 0.502619 |
| hierarchical revelation_precision | 0.407199 | 0.688567 |
| hierarchical revelation_recall | 0.492926 | 0.650496 |
| hierarchical free precision / recall | 0.3665 / 0.4197 | 0.6043 / 0.6546 |
| hierarchical blocked precision / recall | 0.3137 / 0.3868 | 0.6790 / 0.6090 |
| pooled label_aware_delta_iou | 0.223950 | 0.455169 |

Roots 2110, scenes 108, windows 47128; zero-new windows {"windows": 2561, "candidate_exact_empty": 1687, "baseline_exact_empty": 736}; out-of-domain positives 0; paired delta vs the graph-only baseline [0.239, 0.2631].

Pooled confusion (rows true U/F/B, cols predicted): [[1703774272, 1512853, 2469478], [1501076, 2918772, 132025], [3622569, 384696, 5552023]]

| stratum | windows | v7 (128 px) | v11 (256 px) |
| --- | ---: | ---: | ---: |
| duration:<=0.5 | 30397 | 0.4805 | 0.4917 |
| duration:0.5-1 | 9894 | 0.4467 | 0.4586 |
| duration:1-2 | 5793 | 0.4126 | 0.4259 |
| duration:>2 | 1044 | 0.4192 | 0.4372 |
| human:present | 40140 | 0.4390 | 0.4520 |
| human:zero | 6988 | 0.4608 | 0.4726 |
| motion:mixed | 23987 | 0.4742 | 0.4888 |
| motion:rotation | 20534 | 0.3862 | 0.3967 |
| motion:translation | 2607 | 0.5200 | 0.5396 |
| new_cells:1-99 | 18807 | 0.4537 | 0.4539 |
| new_cells:>=100 | 25760 | 0.4418 | 0.4554 |

Per-Root comparison with v7 (same 2,110 Roots, per-Root mean confusion): label-aware IoU improved on
69.9 % of Roots (worse on 28.8 %), mean +0.0140 / median +0.0128; per-scene mean improved in 94 of 108
scenes (range -0.024 ... +0.066).

Complete latency of the same frozen `final/v11.pt` (original 32 workloads with 256 px packets
`range_benchmark_256`, K=1/4/6, 3 repetitions, `lanes=3`, BatchNorm folded; physical GPU0 with this
session's own training paused by `quiet.py`; another session's Stage-1 jobs kept running on physical
GPU1 and the CPU) — `benchmark/v11_bench256.json`, `benchmark/quiet_v11_256.json`:

| cohort / cache | K=1 P50 / P95 | K=4 P50 / P95 | K=6 P50 / P95 |
| --- | ---: | ---: | ---: |
| dense, cold | 11.610 / 25.327 | 19.798 / 29.855 | **24.905 / 34.200** |
| dense, map_warm_new_queries | 7.172 / 9.380 | 16.656 / 20.231 | 21.298 / 25.428 |
| dense, warm | 7.150 / 9.244 | 16.692 / 19.654 | 21.459 / 25.235 |
| representative, cold | 10.075 / 11.787 | 16.956 / 21.286 | 21.531 / 23.623 |
| representative, map_warm_new_queries | 6.297 / 8.275 | 14.245 / 16.599 | 18.924 / 21.499 |
| representative, warm | 6.280 / 7.947 | 14.388 / 19.128 | 18.903 / 21.978 |

startup: constructor 6168 ms (CUDA kernel compilation + graph capture), first complete K=6 call 147 ms.
Median dense-cold K=6 stage times: graph preparation 1.8 ms, legal map 2.9 ms (256 px), query
features 4.9 ms, network 11.3 ms (10.4M parameters; v7 with 3.6M parameters: 12.3 ms without lanes/folding),
decode + sparse increment + complete native hypothetical graphs 3.0 ms. Contract tests on `final/v11.pt`
(`test_api.py`): exact domain, zero horizon, no mutation, batched == single bit-identical,
lanes=3 == lanes=1 bit-identical — all pass.

The required accuracy (> 0.50) is **not** reached; see sections 9-10. The latency requirement is met
(dense K=6 cold P95 34.2 ms < 50 ms, inside the preferred 20-30 ms range at P50).

### 1.1 Frozen model `final/v7.pt` (checkpoint `training/past_v7/final.pt` + frozen operating point)

Operating point selected on 400 scene-disjoint TRAIN-validation Roots (`select.py`): class rule,
FREE if r(1-b) > 0.425, BLOCKED if rb > 0.525 (validation label-aware IoU 0.4628; single-threshold
rule 0.4597). Original full development set, 128 px sensing (`full_range_branches`), original
evaluator assertions (`evaluate.py`):

| Model | full-dev label-aware IoU | revelation IoU | pooled IoU | dense K=6 cold P95 |
| --- | ---: | ---: | ---: | ---: |
| previous best A (handoff, 128 px) | 0.4171 | 0.4667 | 0.3992 | 58.6 ms (256 px workloads) |
| **this work, v7 (128 px)** | **0.4563** | **0.4903** | **0.4423** | **29.0–35.9 ms** (below) |

| metric | previous baseline | this model |
| --- | ---: | ---: |
| hierarchical label_aware_delta_iou | 0.217733 | 0.456265 |
| hierarchical revelation_iou | 0.286985 | 0.490326 |
| hierarchical revelation_precision | 0.407199 | 0.679549 |
| hierarchical revelation_recall | 0.492926 | 0.637797 |
| hierarchical free precision / recall | 0.3665 / 0.4197 | 0.5903 / 0.6383 |
| hierarchical blocked precision / recall | 0.3137 / 0.3868 | 0.6709 / 0.5967 |
| pooled label_aware_delta_iou | 0.223950 | 0.442303 |

Roots 2110, scenes 108, windows 47128; zero-new windows {"windows": 2561, "candidate_exact_empty": 1596, "baseline_exact_empty": 736}; out-of-domain positives 0; bootstrap 95% candidate [0.4438, 0.4687], paired delta vs baseline [0.227, 0.2499]

Pooled confusion (rows true U/F/B, cols predicted): [[1703743207, 1525941, 2487455], [1604469, 2838314, 109090], [3744407, 407953, 5406928]]

| stratum | windows | label-aware IoU (pooled) |
| --- | ---: | ---: |
| duration:0.5-1 | 9894 | 0.4467 |
| duration:1-2 | 5793 | 0.4126 |
| duration:<=0.5 | 30397 | 0.4805 |
| duration:>2 | 1044 | 0.4192 |
| human:present | 40140 | 0.4390 |
| human:zero | 6988 | 0.4608 |
| motion:mixed | 23987 | 0.4742 |
| motion:rotation | 20534 | 0.3862 |
| motion:translation | 2607 | 0.5200 |
| new_cells:0 | 2561 | 0.0000 |
| new_cells:1-99 | 18807 | 0.4537 |
| new_cells:>=100 | 25760 | 0.4418 |

### 1.2 Complete latency (original 32 workloads, 128 px packets, K=1/4/6, 3 repetitions)

Frozen `final/v7.pt`, physical GPU0 (RTX 3090 with the desktop session), this session's own jobs
paused (`quiet_final_v7_128_gpu0.json`); an external Stage-1 training job (another session, not
paused) occupied physical GPU1 and ~9 CPU processes during this run:

| cohort / cache | K=1 P50 / P95 | K=4 P50 / P95 | K=6 P50 / P95 |
| --- | ---: | ---: | ---: |
| dense, cold | 12.312 / 18.040 | 23.236 / 27.500 | 31.088 / 35.913 |
| dense, map_warm_new_queries | 8.223 / 10.204 | 19.278 / 23.931 | 25.472 / 29.911 |
| dense, warm | 7.953 / 10.715 | 19.464 / 23.393 | 25.481 / 31.222 |
| representative, cold | 10.141 / 14.346 | 19.277 / 22.237 | 25.619 / 29.247 |
| representative, map_warm_new_queries | 7.219 / 9.934 | 17.106 / 19.263 | 23.354 / 27.100 |
| representative, warm | 7.076 / 10.602 | 17.068 / 21.124 | 23.689 / 27.592 |

startup: {"constructor_ms": 5470.571909099817, "first_complete_K6_ms": 177.11907299235463}; load average before/after: [12.240234375, 7.8291015625, 10.171875] / [12.9833984375, 8.35791015625, 10.28564453125]; device cuda:0 (NVIDIA GeForce RTX 3090)

Same architecture and features (v7 step 20000), physical GPU1 at 15:12 UTC before the external job
started, own jobs paused (`quiet_v7s20000.json`):

| cohort / cache | K=1 P50 / P95 | K=4 P50 / P95 | K=6 P50 / P95 |
| --- | ---: | ---: | ---: |
| dense, cold | 9.686 / 15.835 | 19.066 / 23.894 | 24.642 / 28.980 |
| dense, map_warm_new_queries | 6.195 / 8.025 | 15.864 / 19.063 | 21.628 / 25.393 |
| dense, warm | 6.226 / 8.169 | 15.775 / 18.881 | 21.750 / 25.681 |
| representative, cold | 8.426 / 9.897 | 16.113 / 17.832 | 21.542 / 23.365 |
| representative, map_warm_new_queries | 5.680 / 7.286 | 14.194 / 15.638 | 19.724 / 21.570 |
| representative, warm | 5.584 / 7.117 | 14.200 / 15.878 | 19.827 / 21.523 |

startup: {"constructor_ms": 4289.102727081627, "first_complete_K6_ms": 143.9595310948789}; load average before/after: [11.8017578125, 12.42626953125, 13.6103515625] / [9.439453125, 11.8466796875, 13.39404296875]; device cuda:1 (NVIDIA GeForce RTX 3090)

On physical GPU1 *shared with the external Stage-1 training* the same benchmark measured dense K=6
cold P50 51.8 / P95 63.4 ms (`final_v7_bench128_r2.json`): GPU time-sharing with another training
job roughly doubles the GPU stages; this is reported, not selected away.


## 2. What the label is (read from the environment observer)

`eventwm_graph/partial/observer.py` / `fast_observer.py` publish, at every observation pose:

* **FREE** for a cell iff (a) navmesh support (center snaps, clearance ≥ r/√2+0.008 on the
  0.25 m-inflated navmesh, 9 sample points step-reachable) and (b) all **63 probes**
  (center + 4 corners at 0.49·r, 16 halo points at 0.25+r/√2, at +0.025/+0.7/+1.4 m) are inside
  the 90°×90°, 10 m frustum and have a clear line of sight (humans occlude).
* **BLOCKED** for every cell whose center lies within 0.25 m (xz) of the first hit of one of
  **35×3 wall rays** (x∈[−0.95,0.95], y∈{−0.5,0,0.5}) if the hit is static and its height is in
  (base+0.2, base+1.4). A human hit consumes the ray without a certificate.

Verified: rebuilding BLOCKED targets from the *actual* future depth at the wall-ray pixels
reproduces 95.4 % of BLOCKED targets with 91 % precision (`blocked_debug.py`); the legal
proxy future camera (current rigid camera/base extrinsic applied to the supplied path)
matches recorded future cameras to 0.0000 m / 0.0000 (`cam_check.py`).

## 3. Information-budget diagnostics (TRAIN roots unless noted)

| Finding | Evidence |
| --- | --- |
| Most new cells are in never-measured space | only 59 % of BLOCKED targets have any measured wall-band hit within 0.25 m; simulating the exact wall rays on legal geometry recovers 24 % of new BLOCKED cells (95 % with the real future depth) |
| Rays enter unknown space next to the robot | median unknown-entry distance 1.33 m from the future camera; real hit median 1.5 m beyond entry (`entry_debug.py`). Windows start early in episodes (≈0.6 s history) and the robot turns into unseen directions |
| The previous model and this one underfit | train ≈ validation IoU (0.34 vs 0.37 at the same checkpoint, `fitcheck.py`); old A: train 0.4195 vs dev 0.4171 |
| Error strata (v2) | 53 % of targets are cells whose probes are possible but cross unknown space (IoU 0.31, 58 % of misses, 51 % of false, 90 % of wrong-type); simulation-determined strata reach IoU 0.71–0.89 (`strata.py`) |
| Simulation fidelity | fraction of actually certified FREE cells reproduced as known-visible from the very poses that certified them: 72.8 % (10 cm voxels, 96² renders, splatting) → 82.1 % (192² renders) → 85.8 % (5 cm voxels) → 87.1 % (+ unknown voxels within 0.5 m of the camera treated as the robot's own volume); 0.3 % false certification throughout (`sim_check2.py`) |
| Past visibility is decisive for short windows | among cells the future poses would fully see, FREE rate is 1.7 % if a legal PAST camera already saw them fully (not certifiable) vs 29.9 % otherwise (short windows; 11.5 % vs 48.2 % long) (`check_past.py`) |
| Splatting hurts | measured-surface splatting increases spurious early wall-ray hits (9.8 % → 12.2 %); human ghost filtering removes ~1 % (`ray_debug.py`) |
| 256 px sensing is not a lever here | same 128-trained model: pilot 0.3679 (128) vs 0.3705 (256) |
| The frozen v7 does not overfit | 120 random TRAIN Roots 0.4329 vs 120 TRAIN-validation Roots 0.4506 (same code, threshold 0.6; `fitcheck.py`) |
| Where v7's errors are (120 validation Roots, `strata.py`) | cells whose probes are possible but cross unknown space hold 56 % of targets at IoU 0.35 and 64 % of misses, 66 % of false revelations, 98 % of wrong types; known-wall cells (18 % of targets) reach 0.87; probes mostly known 0.89; long windows are hardest (<=0.5 s 0.53, 0.5-1 s 0.48, 1-2 s 0.41, >2 s 0.37 pooled) |
| Humans near the path cost accuracy (`human_diag.py`) | v7: windows with a tracked human within 2 m of the path: 28 % of targets, IoU 0.425; 2-4 m: 11 %, 0.434; none within 8 m: 55 %, 0.463. v11 (with the `human_future` sweep channel, same thresholds): 0.437 / 0.455 / 0.476 (overall 0.4507 -> 0.4643) — the < 2 m gap is not closed |
| What the warm restart + IoU loss improved (v7 -> v7r, same 120 validation Roots, `strata.py`) | pooled 0.4463 -> 0.4535; walls behind known geometry (`frustum_blocked`) 0.268 -> 0.289, rays only traversing unknown space 0.283 -> 0.306, no-evidence cells 0.146 -> 0.161, uncertain-probe cells 0.346 -> 0.349 with fewer false FREE (66 % -> 61 % of false revelations); known-geometry strata unchanged (0.87-0.90) |
| Unseen walls are hard to imagine (`aux_eval.py`, auxiliary head of aux_human_v3) | wall-band occupancy in never-observed columns that later get observed (16 % positive): best IoU 0.21 (precision 0.22 / recall 0.80 at 0.1; 0.65 / 0.05 at 0.5) |

## 4. Method

1. **Legal tri-state voxel map** (`geometry.py`, `kernels.py`): 5 cm voxels (2×2 per graph cell,
   5 cm height bins, base−0.4 … base+2.0 m). OCCUPIED from measured hits (hits on legally tracked,
   visible humans are skipped), FREE by projective carving (margin 4.5 cm), else UNKNOWN. Currently
   tracked humans are stamped as dynamic occluders (block lines of sight; consume wall rays).
2. **Certificate simulation along each supplied trajectory**: proxy future cameras; per unique pose
   a 192×192 render (distance to first OCCUPIED/human and first UNKNOWN voxel); the exact 63-probe
   geometry is tested against the renders; the exact 35×3 wall rays are DDA-marched with the 0.25 m
   inflation rule and the (base+0.2, base+1.4) band.
3. **Inputs per query** (exact original candidate crop frame): 26 map channels (graph FREE/BLOCKED,
   per-height-band occupied/free/unknown fractions, current humans, and probe/wall statistics from
   up to 16 legal PAST camera poses) + query channels (probe statistics incl. unknown depth and
   pose fractions, known/after-unknown wall hits, hit distance, unknown traversal/entry, path raster,
   window length/duration/distance, frame validity; v7 uses 24 of them). Added after v7: the
   constant-velocity sweep of the currently tracked humans over the window (`human_future`) and the
   expected first-hit density of the exact wall rays inside unknown space (`wall_unknown_hit`,
   survival exp(-unknown length/1 m)).
4. **Masked U-Net** (`model.py`): full cell resolution, widths 32-64-96-128-192; activations outside
   the frame are zeroed at every layer and the global context is a masked mean, so predictions do
   not depend on padding or batch composition. Heads: revelation r and BLOCKED|revealed b.
5. **Training** (`train.py`): TRAIN roots only, scene-disjoint validation (4 % of TRAIN scenes);
   dense per-cell BCE (pos-weight 3) + 2×BCE on b over revealed cells, window weights =
   scene factor / windows-in-Root (the evaluation hierarchy); AdamW 2e-3, cosine, AMP, dihedral
   augmentation inside the frame.
   Continued training (after v7) uses warm restarts (1 epoch, peak 4-5e-4, cosine) with
   0.5 × a batch-level soft label-aware IoU loss (window-weighted like the metric), an EMA of the
   weights (0.999, validated/saved), a 256-window shuffle pool (windows of more Roots per batch),
   legal features built on a side CUDA stream/thread overlapping the training step, and optional
   distillation toward a frozen teacher ensemble (logit average; soft-target BCE mixed 50/50 with the
   hard loss). Warm starts can add input channels (`--extend`, zero stem weights) and coarse-level
   capacity (`--side`: 1×1-projected residual branches at encoder levels 3/4 whose output BatchNorm
   scale/shift start at zero), both function-preserving at step 0.
6. **Decision**: frozen operating point selected on TRAIN validation (`select.py`), stored in the
   checkpoint; development data are only used for the final evaluation.
7. **Inference** (`api.py`): exact original domain on GPU, per-query batch-1 CUDA graphs (bit-identical
   single vs batched), one pinned device→host transfer, original sparse increment and native
   hypothetical graph materialization. BatchNorm is folded into the convolutions of the F16
   inference copy, and with `lanes>1` the per-query graphs of one call are replayed on separate
   streams (independent graph copies; results bit-identical to one lane, tested).

## 5. Interface

See `eventwm_graph/stage2_claude/README.md` (unchanged `predict_segments` contract).

## 6. Accuracy protocol

`evaluate.py` mirrors `stage2_repair/range_full_evaluate.py`: original replay/windows, legal
ancestor range frames and tracks only, exact domain SHA-256 and target counts asserted against the
frozen original baseline receipts, labels asserted equal to the frozen label arrays, every
materialized hypothetical graph verified, original `summarize()` (equal scene/Root/window mean
confusion → rates; pooled; bootstrap). 128 px sensing (`full_range_branches`, complete for all 2,110
Roots).

## 7. Latency protocol

`benchmark.py` mirrors `stage2_repair/range_benchmark.py`: the original 32 workloads/paths, K=1/4/6,
cold (graph AND legal-map caches cleared), warm, reusable-map/new-queries, 3 repetitions, dense =
upper half by known cells. Timed: validation, content identities, graph preparation, candidate
universe, legal map, features, transfers, network, decode, sparse increment, complete native
hypothetical graphs. Excluded: deserialization, serialization, sensor acquisition. CUDA graph capture
is part of the constructor (startup), like kernel compilation.

## 8. Experiment log

| Run | Change | val IoU @4k steps |
| --- | --- | ---: |
| unet_v1 | unmasked U-Net, 256 crops | (framing-inconsistent; abandoned) |
| masked_v2 | masked U-Net, exact frames | 0.347 (0.357 @8k, 0.366 @12k) |
| aux_human_v3 | + label-side geometry aux loss + human ghost filter | 0.338 (0.349 @8k) |
| big_v4 | + 2× width | 0.350 |
| past_v7 | 5 cm voxels, 192² renders, no splat, human filter + current-human occluders, past-pose channels, query scalars | **0.383** (0.391 @8k, 0.412 @12k, 0.414 @16k, 0.415 @20k) |
| near_v9 | v7 + near-camera fix + wall-hit distance channel + 1.7k extra TRAIN Roots, 3 epochs | 0.303 (0.386 @8k, 0.398 @12k) |

Decision-rule selection on identical TRAIN-validation Roots (v7 @8k, 160 Roots): single revelation
threshold 0.391 vs class-specific thresholds (FREE if r(1-b)>0.40, BLOCKED if rb>0.50) 0.402.
Dihedral test-time augmentation (frame + transpose): +0.001 (v7 @12k) — not adopted.

Pilot comparison with the previous model A (same 666 windows, per-window confusion receipts,
`compare_pilot.py`; used diagnostically, never for selection): v2 @4k trailed A mainly on short
(<=0.5 s) and small-reveal windows (false revelations 0.69 per target vs 0.29), which led to the
past-pose visibility channels; v7 @4k: pooled 0.402 vs A 0.407, better than A on windows > 1 s.

Later experiments (TRAIN validation, 160 Roots, single-threshold rule unless noted):

| Run | Change | val IoU |
| --- | --- | ---: |
| past_v7 final | (above), 2 epochs | **0.4454** (400 Roots, class rule: **0.4628**) |
| cascade_v7s24 | refiner (24-48-64-96-128) on frozen v7 @24k (0.4222) | 0.4285 / 0.4248 / 0.4255 / 0.4317 / 0.4264 @2k..10k |
| cascade_v7f | refiner on frozen final v7 (0.4454) | 0.4435 @4k, 0.4451 @8k — no gain; stopped |
| ensemble | v7 @20k + v9 @12k (weaker member) | 0.4218 vs 0.4279 single (class rule) — worse |
| 256 px input | final v7, no retraining, 38 TRAIN-val Roots with both resolutions | 0.4454 vs 0.4430 (128 px), class rule |
| 256 px input | v7r (frozen class rule), 199 TRAIN-val Roots / 22 scenes with both resolutions (161 new 256 px packets, unchanged environment replay) | 0.4567 vs 0.4546 (128 px) hierarchical; pooled 0.4509 vs 0.4497 |
| v7_restart_iou | final v7 + 1 epoch warm restart (peak 5e-4) + 0.5 x soft label-aware IoU loss | 0.4351 @4k, 0.4450 @8k, 0.4510 @12k, 0.4517 @16k, 0.4546 @20k, **final 0.4569** (v7 0.4454) |
| v11_ext | warm start from v7_restart_iou with three new zero-initialized input channels (`human_future`, `wall_hit_distance`, `wall_unknown_hit`), zero-initialized coarse-level side branches (level 3: 192 x 3 blocks, level 4: 256 x 4; 3.55M -> 10.42M parameters), near-camera fix (0.5 m), +2,914 new TRAIN Roots, EMA 0.999, shuffle pool 16 batches, overlapped feature production, 1 epoch, IoU loss 0.5 | 0.4573 @4k, 0.4597 @8k, 0.4583 @12k, 0.4629 @16k, 0.4622 @20k, 0.4627 @24k, **final 0.4629** |
| ens_v7r_v11 | logit ensemble of v7_restart_iou + v11_ext (teacher candidate) | 0.4608 (< v11_ext alone; not used) |
| v12_distill | warm start from v11_ext, 1 epoch (peak 4e-4), self-distillation from frozen v11_ext (soft targets 50 %), seed 2, otherwise as v11_ext | (running) |

An equal-strength second member for an ensemble (v7 config, seed 1, +2,914 new TRAIN Roots) was
started but stopped: from 17:21 UTC an external Stage-1 training job (another session) occupies
physical GPU1, which would have made the run take ~9 h and slowed the other job.

## 9. Negative results

* Auxiliary geometry supervision (walls/floor from all Root frames) did not improve the main metric;
  unseen-wall completion stayed poor (precision 0.55 / recall 0.08 at 4k steps).
* 2× model width: +0.003 at equal steps.
* Measured-surface splatting, 256 px sensing, sub-voxel hit refinement: no material gain.

* Cascade refinement helps an under-trained base but not the converged one.
* Window-duration-dependent class thresholds (4 duration buckets, coordinate ascent; `bucket_select.py`):
  no held-out gain in two-fold cross-validation over 400 validation Roots (0.4667 vs 0.4666 global;
  0.4604 vs 0.4609) — one global operating point is kept.
* Ensembling with a weaker member reduces accuracy; within the warm-restart lineage, neither the logit
  ensemble of v7 + v7r (20 validation Roots: 0.5106 vs v7r 0.5109) nor the weight average of v7 and v7r
  (160 Roots: 0.4506 vs v7r 0.4569) beats the newest member.

## 10. Limitations

* Labels in never-observed space are only partially predictable from legal inputs (sections 3, 9).
* The range sensor extension is the same ideal simulator sensing used by the previous model; a real
  front end must supply equivalent calibrated ranges, timestamps and tracks.
* Remaining error is dominated by never-observed space (wall bands at the far boundary of newly
  revealed free regions, FREE support behind unseen obstacles): cells whose probes are possible but
  cross unknown space hold ~56 % of the targets at IoU ~0.35, while every simulation-determined stratum
  is at 0.71-0.90 (section 3). Rotation windows (44 % of windows, the robot turning into unseen
  directions) stay at 0.40 on full development.
* What still helps is more optimization of the same information: no model overfits (v11: 120 TRAIN
  Roots 0.4455 vs 120 validation Roots 0.4668), and each warm-restart cycle still added validation
  accuracy, but with diminishing returns (TRAIN-validation 400 Roots, class rule: v7 0.4628 -> v7r 0.4711
  -> v11 0.4750). Ensembles or weight averages of lineage members with a weaker member do not help;
  3x coarse-level capacity, new legal inputs and EMA together gave +0.004 (400 Roots) / +0.006 (160).
* Most promising next steps: (1) explicit, sampled geometry imagination — generate plausible completions
  of the unknown space (rooms continue, walls close), re-run the exact wall-ray/probe simulation on each
  sample and average the certificates, instead of asking one network to output marginals directly
  (the auxiliary wall head alone did not learn unseen walls well: best IoU 0.21); (2) many more
  training Roots: 105,838 eligible TRAIN Roots exist but only 18,424 have range packets (the
  environment replay costs ~1 CPU-minute per Root); (3) per-pose human motion inside the simulation
  (windows with a human within 2 m of the path: 28 % of targets, IoU 0.425 vs 0.463 without);
  (4) sub-voxel surface geometry for the remaining ~13 % visibility-simulation infidelity.

## 11. Reproduction

Environment as in the handoff (`conda activate eventwm-falcon`; `output -> $DEV/output`; data link).
All commands from the repository root; `CUDA_VISIBLE_DEVICES=0,1` (the conda activation otherwise
exports `CUDA_VISIBLE_DEVICES=1`).

```bash
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
R="$PY scripts/stage2_segment_run.py"
# 1. train (TRAIN Roots with 128 px range packets; scene-disjoint validation)
$R eventwm_graph.stage2_claude.train --name past_v7 --device cuda:0 --epochs 2 --workers 7 \
   --val-every 4000 --val-roots 160 --human-filter --human-occluders --splat 0 --sub 2 --render 192
# 1b. warm restart + soft IoU loss (v7r)
$R eventwm_graph.stage2_claude.train --name v7_restart_iou --device cuda:0 --epochs 1 --workers 7 --val-every 4000 \
   --val-roots 160 --warm output/claude_stage2/training/past_v7/final.pt --lr 5e-4 --warmup 200 --iou-weight 0.5
# 1c. new zero-initialized input channels + coarse-level side branches + EMA + extra TRAIN Roots (v11)
$R eventwm_graph.stage2_claude.train --name v11_ext --device cuda:0 --epochs 1 --workers 7 --val-every 4000 --val-roots 160 \
   --warm output/claude_stage2/training/v7_restart_iou/final.pt --extend --near 0.5 --lr 5e-4 --warmup 200 \
   --iou-weight 0.5 --ema 0.999 --extra-sensor claude_train_range_128 --pool-factor 16 --overlap --side 3:192:3,4:256:4
# 1d. self-distillation restart (v12)
$R eventwm_graph.stage2_claude.train --name v12_distill --device cuda:0 --epochs 1 --workers 7 --val-every 4000 \
   --val-roots 160 --warm output/claude_stage2/training/v11_ext/final.pt --extend --near 0.5 --lr 4e-4 --warmup 200 \
   --iou-weight 0.5 --ema 0.999 --extra-sensor claude_train_range_128 --pool-factor 16 --overlap --side 3:192:3,4:256:4 \
   --seed 2 --teacher output/claude_stage2/training/v11_ext/final.pt --distill 0.5
# optional refinement stage on a frozen trained base
$R eventwm_graph.stage2_claude.train --name cascade --device cuda:0 --epochs 1 --workers 7 \
   --cascade-from output/claude_stage2/training/past_v7/final.pt --widths 24,48,64,96,128 --blocks 1,1,2,2,2 --lr 1e-3
# 2. operating point on TRAIN validation (400 Roots) and freezing into the deployable checkpoint
$R eventwm_graph.stage2_claude.select --checkpoint <trained.pt> --device cuda:0 --roots 400 \
   --out <name> --freeze-to output/claude_stage2/final/<name>.pt
# 2b. sensor resolution on the TRAIN-validation Roots with both packet resolutions (256 px packets for
#     more validation Roots: unchanged environment replay, `val256_expand.py <part> <parts> <count>`)
$R eventwm_graph.stage2_claude.sensor_select output/claude_stage2/final/<name>.pt cuda:0
# 3. original full development evaluation (2,110 Roots / 47,128 windows); 256 px: --sensor-directory full_range_branches_256
$R eventwm_graph.stage2_claude.evaluate --checkpoint output/claude_stage2/final/<name>.pt \
   --name full_<name> --split full --workers 6 --device cuda:0 [--sensor-directory full_range_branches_256]
# 4. contract tests (exact domain, zero horizon, no mutation, single == batched)
$R eventwm_graph.stage2_claude.test_api output/claude_stage2/final/<name>.pt cuda:1
# 5. complete K=1/4/6 latency on the original 32 workloads, own background jobs paused
$PY -m eventwm_graph.stage2_claude.quiet --receipt output/claude_stage2/benchmark/quiet_<name>.json \
   --pattern "stage2_claude.train" -- $R eventwm_graph.stage2_claude.benchmark \
   --checkpoint output/claude_stage2/final/<name>.pt --name <name>_bench128 --repetitions 3 --device cuda:1 \
   [--sensor-directory range_benchmark_256] [--lanes 3]
```

## 12. Process, data and evaluation-integrity notes

* With the user's explicit permission, the previous A epoch-2 training (+selector) and the
  sensor-256 TRAIN preparation workers were stopped at 11:35 UTC; the sensor-256 DEV preparation
  workers stopped at the same time and were later resumed from their frozen worktree
  (`eventwm-falcon-stage2-sensor256-20261026-0924`, same source identity, existing packets reused) and
  completed all 2,110 Roots. No checkpoints, datasets or results were deleted.
* New TRAIN range packets (128 px) for 3,082 additional Roots (`source_rows('train')[16384:]`) were
  produced with the unchanged environment replay (`range_train_expand.py`) into
  `$OUT/claude_train_range_128 -> /var/tmp/claude-stage2/train128`.
* Model development used TRAIN Roots only; operating points were selected on scene-disjoint TRAIN
  validation Roots. The fixed 31-Root development pilot was used for diagnostic comparisons with the
  previous model (window-level receipts) and as a regression check of the public API; these
  diagnostics informed feature design (e.g. past-pose visibility). The full development set was
  evaluated only for frozen candidates (section 1 lists every full-development evaluation run).
* Latency was measured on the RTX 3090 used by the previous benchmarks (physical GPU1) with this
  session's own background jobs paused by `quiet.py` (receipts list the paused PIDs/command lines and
  load averages); no external process was paused.
