> Source-publication note: this is a frozen local research report. Model weights, datasets and detailed generated evidence are deliberately excluded; reported results were not rerun for publication. Location references are redacted. See [SOURCE_REVIEW_GUIDE.md](../../../SOURCE_REVIEW_GUIDE.md).

# Stage 2: Trajectory-Conditioned GraphDelta Predictor

**Stage 2 is the primary research development objective.** Its loadable trained checkpoint is checkpoint.pt (local artifact; excluded from source publication) (SHA256 `2c77ef1c6821f4c9dc3fac8d6128d1052f5bbdc2a449789b84e73c716a646ed4`).

Inputs are the initial legal PartialGraph and executed [absolute_time,x,y,z,yaw] rows. The network predicts newly observed FREE/BLOCKED cells, explicit UNKNOWN/no-update probabilities and height residuals. The existing deterministic PartialGraph.update constructs the hypothetical future graph. The original map stays unchanged; predicted provenance is explicit. No Stage1 prediction is an input.

## Data and fixed offline training

Training reuses all 105,838 final eligible independent Roots / 769,349 Query branches / 476 fitting scenes. Actual legal future-observation DAG ancestry determines executed duration/termination and observation-limited cell targets. No privileged geometry, future jaw-camera transform, pedestrian state, new simulation, planner-in-the-loop training or RL is used.

Candidate cells are initially UNKNOWN grid cells within 11m of the executed trajectory. Each receives 47 legal map/trajectory features: local FREE/BLOCKED support, distances and height, four nearest pose bearings/times, and segment summaries. The neural architecture is `47 → Linear48 → SiLU → Linear32 → SiLU → 4`, producing three class logits and one height residual.

The shared cell MLP has 4,004 parameters, width48/32, UNKNOWN/FREE/BLOCKED softmax and a height head. Six fixed epochs, seed2410, AdamW lr0.001/decay0.0001, clip5 and batch8192. Up to64 new-information and64 remaining-UNKNOWN cells per Query use exact inverse-inclusion weights, with equal Root and within-Root Query weights. The sampled log score estimates full-domain risk; it does not count cells or siblings as independent trajectories. No development threshold or hyperparameter search was performed.

Completed 69,516 optimizer steps in 522.4s; peak CUDA allocation 8,344,576 bytes. Exact config: config.json (local artifact; excluded from source publication). Immutable extraction/training source: manifest.json (local artifact; excluded from source publication). Optimized inference/evaluation source: manifest.json (local artifact; excluded from source publication).

Data extraction: [data.py](../../../eventwm_graph/stage2/data.py). Independent exact replay audits: REVIEW_STAGE2_LEGAL_TARGETS.json (local artifact; excluded from source publication) and STAGE2_CORRECTNESS_REVIEW.json (local artifact; excluded from source publication).

## GraphDelta quality

Every original development Query is evaluated: 2,110 Roots / 108 scenes / 14,546 outcomes. Raw-probability argmax and all scene/bootstrap rules were frozen before scoring. Candidate-domain misses: 0. Deterministic graph-update correctness: 1.000000.

| New information | Precision | Recall | F1 | 95% scene-bootstrap F1 interval |
|---|---:|---:|---:|---|
| new_free | 0.483252 | 0.151433 | 0.230604 | [0.2123637823072555, 0.2509934001444055] |
| new_blocked | 0.559248 | 0.055800 | 0.101475 | [0.09111185071826927, 0.11244627920481177] |

**The fixed-argmax output has low recall: 15.1% for FREE and 5.6% for BLOCKED.** This first supervised baseline leaves many true revelations as UNKNOWN. Improving Stage2 recall remains a primary research objective. Deterministic update correctness measures application of a supplied delta; prediction accuracy is measured separately above.

Scene-balanced label-aware GraphDelta IoU: 0.074700; reveal-set IoU: 0.096515. Root-balanced multiclass Brier: 0.01586960, versus persist-UNKNOWN 0.02014556. Multiclass log score: 0.032696 (saved numerical floor1e-9). Height MAE on observed cells: 0.015610m.

The substantial-information cohort (≥100 true new cells) contains 9,880 Queries /1,858 Roots /107 scenes. Its pooled results are:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| new_free | 0.505565 | 0.159151 | 0.242091 |
| new_blocked | 0.564151 | 0.058306 | 0.105688 |

Per-class Root-weighted calibration error:

| Class | ECE |
|---|---:|
| unknown | 0.00045102 |
| new_free | 0.00022403 |
| new_blocked | 0.00026135 |

The large UNKNOWN region influences aggregate calibration scores. Full probability-bin masses and frequencies are retained for inspection. Complete pooled and scene-balanced metrics, support, confidence intervals, every Query confusion matrix and graph-update checks are retained in evaluation.json (local artifact; excluded from source publication) and optimized_dev (local artifact; excluded from source publication).

## Complete API performance

| Workload | Predict P50/P95/P99 ms | Predict + deterministic update + sparse JSON P50/P95/P99 ms |
|---|---|---|
| representative | 48.085 / 58.892 / 71.824 | 55.887 / 70.347 / 103.700 |
| dense | 53.046 / 68.231 / 71.960 | 68.320 / 89.330 / 216.196 |

| Batch | P50 ms | P95 ms | P99 ms | Trajectories/s |
|---|---:|---:|---:|---:|
| 1 | 43.426 | 43.829 | 43.860 | 23.028 |
| 4 | 213.746 | 300.705 | 337.629 | 18.714 |
| 16 | 793.762 | 807.873 | 813.208 | 20.157 |

Peak benchmark-process RSS: 737,017,856 bytes; checkpoint initialization 0.011118s. All representative/dense cold process observations, workload sizes, raw calls and memory data remain in performance.json (local artifact; excluded from source publication). These timings include domain/feature construction, neural work and delta construction; update/wire rows include deterministic graph copying/application and sparse serialization. Batch tails use eight observations and are descriptive. The overall integrated Layer2 P95≤100ms target is preserved; integration has not been benchmarked.

Dense complete-call P99 remains **216.2ms**. Batch sizes use different prefixes of the input pool, so these values do not establish a matched batch speedup. Peak benchmark RSS includes retained workload examples, libraries and scratch arrays. Optimized cold predict/apply/wire observations conservatively include import-witness hashing overhead.

The original complete-call P95 failures (111.7ms representative /143.8ms dense) and original repository-namespace cold results are preserved in final (local artifact; excluded from source publication). Execution-only optimization reuses exact neighbor/pose features, tiles independent neural rows and transfers updater-owned data without redundant copies. Model architecture, trained checkpoint, candidate support and scoring protocol are unchanged. The optimized cold subprocess uses the frozen namespace from /tmp and records its actual import paths/hashes. Independent equivalence proof: STAGE2_RUNTIME_EQUIVALENCE_REVIEW.json (local artifact; excluded from source publication). Complete original/optimized scientific-summary comparison: scientific_summary_comparison.json (local artifact; excluded from source publication).

## Limitations and API

This first predictor models new information at initially UNKNOWN cells. The training ledger retains 15,966,501 per-branch known-cell change events as an unsupported scope; final known-cell correction is not predicted. Four nearest actual pose features can lose turn headings in tied-position rotations. Requested inputs omit articulated jaw-camera state and dynamic-human occluders, which leave irreducible reveal uncertainty. Observation-limited C-space certificates are not hidden material geometry or a safety certificate.

Executable API: [api.py](../../../eventwm_graph/stage2/api.py). Loading, batch inference, delta application and exact reproduction: [REPRODUCE.md](REPRODUCE.md). Independent final model/performance audit: STAGE2_OPTIMIZED_FINAL_INDEPENDENT_REVIEW.json (local artifact; excluded from source publication). Stage1, L2.5, L3 and L4 implementations remain intact; no L3 RL training was started.
