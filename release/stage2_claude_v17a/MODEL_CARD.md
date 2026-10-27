# Stage 2 final model — `stage2_claude_v17a.pt`

Trajectory-conditioned future-observation model of EventWM-Falcon Stage 2: given the current legal partial graph
G_t, the legal past/current range frames and legal human tracks, and a supplied hypothetical robot trajectory, it
predicts which initially UNKNOWN static cells become observed FREE or BLOCKED along that trajectory (`Delta G_k`),
with a complete hypothetical future graph per query.

## Use

```python
from eventwm_graph.stage2_claude.api import LegalVoxelPredictor
predictor = LegalVoxelPredictor('release/stage2_claude_v17a/stage2_claude_v17a.pt', device='cuda:0', lanes=3)
deltas = predictor.predict_segments(partial_graph=G_t, trajectories=[tau_1, ..., tau_K],
                                    range_observations=packet, track_history=tracks)   # one delta per trajectory
```

Interface, inputs and every output field: `eventwm_graph/stage2_claude/README.md` (unchanged `predict_segments`
contract of the previous range API). Use 256 px range packets (selected on TRAIN validation). `states` apply the
frozen operating point; `probabilities` are uncalibrated, overconfident scores (see Limitations). Runtime: PyTorch 1.12 /
CUDA 11.3, RTX 3090; kernels are compiled with NVRTC at construction (6-11 s incl. CUDA-graph capture, depending on load).

## Results (development split, original evaluator / benchmark)

Checkpoint `stage2_claude_v17a.pt` (41.8 MB, 10,434,072 parameters, sha256 in `manifest.json`); it is the
TRAIN-validation-selected frozen checkpoint `output/claude_stage2/final/v17a.pt` without its raw (non-EMA) training
weights, and its predictions are identical (verified on 72 benchmark queries).

**Map accuracy** — original full-development evaluation: 2,110 Roots, 108 scenes, 47,128 windows; original
assertions and `summarize()`; receipts `receipts/full_dev_*.json`.

| model | label-aware increment IoU | revelation IoU | pooled IoU |
| --- | ---: | ---: | ---: |
| graph-only baseline | 0.2177 | 0.2870 | 0.2240 |
| previous model A (handoff) | 0.4171 | 0.4667 | 0.3992 |
| **v17_a (this model)** | **0.4831** [0.4701, 0.4961] | **0.5216** | **0.4695** |

Further detail for v17_a:
- FREE precision / recall 0.624 / 0.665; BLOCKED 0.683 / 0.628.
- By motion: rotation 0.411, mixed 0.504, translation 0.557.
- By duration: up to 0.5 s 0.510, 0.5–1 s 0.472, 1–2 s 0.439, over 2 s 0.451.

Every earlier frozen candidate evaluated on development: v7 0.4563, v11 0.4688, v12 0.4716, v15 0.4784,
v16_b 0.4809. The required > 0.50 is **not** met.

**Latency** — complete dense K=6 cold, original 32 workloads, 256 px packets, `lanes=3`, physical GPU0
(RTX 3090), this session's own jobs paused. All complete-call stages are timed (validation, graph preparation, legal
map, features, network, decode, sparse increment, complete native hypothetical graphs). Receipts `receipts/latency_*.json`.

| run | load average | P50 | P95 |
| --- | ---: | ---: | ---: |
| v17_a (final pipeline) | ~27 (another session's ~20 CPU-bound processes) | 34.8 ms | 41.1 ms |
| v17_a re-run | ~27 | 36.2 ms | 40.6 ms |
| v15, same code and architecture, run back-to-back with the v17_a re-run | ~27 | 34.7 ms | 38.8 ms |
| v15, same code and architecture, earlier the same day | ~13-15 | 24.1 ms | 27.4 ms |

The < 50 ms gate is met in every condition measured. The preferred 20–30 ms band is reached only when the machine is
not heavily loaded. v17_a and v15 share architecture and code, so their latency is the same under equal conditions.
Contract tests (`test_api.py`) all pass: exact domain, zero horizon, no mutation, batched == single and
lanes=3 == lanes=1 (bit-identical). The cache contract (`test_cache.py`) also passes.

**Trajectory-wise (query-ranking) metrics** — diagnostic. Recorded alternative Query Outcomes of the same Root at a
common 1.0 s horizon: 885 development Roots, 93 scenes, 3,658 decisive pairs. Full method in
`docs/claude_stage2/QUERY_BENCH_REPORT.md`; receipt `receipts/query_bench_report.json`.

| metric | v17_a | v17_a + TRAIN-val recalibration | best legal-map heuristic | random |
| --- | ---: | ---: | ---: | ---: |
| pairwise ranking accuracy (Root-macro) | 92.9 % [91.2, 94.4] | 93.7 % [92.1, 95.2] | 80.2 % | 50 % |
| top-1 hit (predicted best = actual best) | 81.8 % [79.0, 84.6] | 82.9 % | 68.5 % | 34.7 % |
| mean top-1 regret (cells; mean gain 526) | 17.2 [11.3, 23.3] | 15.2 | 49.3 | 185.4 |
| information-gain MAE / bias (cells) | 133.3 / +88.8 (+17 %) | 108.2 / +27.9 (+5 %) | — | — |

Further detail at 1.0 s:
- v17_a minus the visibility heuristic: +12.7 points [10.2, 15.3] pairwise and −32.1 cells [−42.3, −22.1] regret.
- Per-outcome label-aware IoU: mean 0.560, median 0.571.
- Previously observed cells reach IoU 0.721; never-observed cells 0.393.
- Of predicted-FREE cells, 6.4 % are observed BLOCKED and 28.7 % stay UNKNOWN.

Other horizons: 0.5 s 93.1 % / 84.3 % / 7.1 cells; 2.0 s 92.9 % / 81.9 % / 28.2 cells; full outcomes (non-nested
pairs) 95.2 % / 85.7 % / 18.5 cells.


## Method and training

* Legal tri-state voxel map (5 cm voxels) from all legal range frames, projective carving (exact frame culling),
  current tracked humans as dynamic occluders.
* Exact simulation of the original 63-probe FREE certificates and 35x3 wall rays from the future camera poses of
  the supplied trajectory (192 px visibility renders).
* Masked full-resolution BEV U-Net (widths 32-64-96-128-192, coarse side branches at levels 3/4, 10.4M parameters),
  with 53 legal input channels. It outputs revelation and BLOCKED-given-revelation, followed by the frozen class rule
  FREE if r(1-b) > 0.475, BLOCKED if rb > 0.575, chosen on 400 scene-disjoint TRAIN-validation Roots.
* Training on TRAIN Roots only (24,258 Roots with range packets for the last cycle):
  * v7 from scratch;
  * successive 1-epoch warm restarts with a soft label-aware IoU loss;
  * zero-initialized new channels and side capacity;
  * EMA 0.999;
  * self-distillation from the previous model;
  * more TRAIN Roots;
  * soft-IoU weight 1.0 in the last cycle (v17_a, from v16_a).

  Lineage, every experiment and every negative result: `docs/claude_stage2/STAGE2_CLAUDE_REPORT.md`.

## Limitations

* The required full-development label-aware IoU > 0.50 is not met. About 56 % of targets lie where future visibility
  crosses never-observed space (IoU ~0.39 there vs ~0.72 in observed space).
* Probabilities are overconfident. Expected gains over-predict by ~17 % unless recalibrated (a binned recalibration
  fitted on TRAIN validation is shown in the query benchmark).
* Inputs are ideal simulator range packets (the legal sensor extension); a real front end must supply equivalent
  calibrated ranges, timestamps and tracks.
* The query-ranking results are for recorded executed Queries, not hypothetical Sandbox trajectories, and they
  measure information gain, not navigation utility.
