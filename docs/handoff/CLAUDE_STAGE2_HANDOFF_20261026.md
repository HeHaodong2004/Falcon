# Falcon Stage 2 — Claude Code Opus handoff, 2026-10-26

This is a handoff of executed research and its source, not a release claiming acceptance. Model development and new experiments stopped for this handoff. Existing jobs were left running. Read this document before older progress reports: their status and selected models can be obsolete.

**Latest verified legal-input full-development label-aware increment IoU is 0.4171172210**, revelation IoU 0.4666754111, on all 2,110 development Roots / 108 scenes / 47,128 windows. The best verified fixed-pilot result is 0.4532446764 / 0.5061800299 using 256-pixel measured ranges. These are different sensor resolutions, not two evaluations of an identical input configuration. Latest paired complete dense K=6 cold P95 is 58.573 ms for strict geometry and 52.539 ms for the measured 3 mm approximation. Neither current acceptance requirement has been met.

The original development branch is preserved. The dedicated GitHub branch is `research/stage2-claude-handoff-20261026` in [HeHaodong2004/Falcon](https://github.com/HeHaodong2004/Falcon). Its implementation snapshot is `e83edef0e9b329c3758302ef189a7c0208032e9c`; the final handoff adds documentation and a publication audit, without model changes. See section I for exact revision verification and the final local publication receipt.

## A. Project overview

The problem is partially observed PointGoal Social Navigation in Habitat/Falcon. A robot receives a public goal and legal observations, navigates around static structure and humans, and must reason about the consequences of possible actions. **The online robot has no oracle complete static map.** Scene collision geometry available inside the simulator is not automatically an admissible world-model input.

The intended composition is:

```text
real legal observations -> current PartialGraph G_t and positive human history
                         -> Stage 1 human world distribution
current pose/goal/G_t -> sandbox action/trajectory proposals tau_1 ... tau_K
(G_t, supplied tau_k, current legal sensing) -> Stage 2 hypothetical delta G_k
(human worlds, delta G_k, queries) -> Graph Compiler -> L3 choice
L3 choice -> native executor -> real transition -> new actual observations
```

| Component | Responsibility and actual status |
| --- | --- |
| Stage 1 | History-conditioned joint human/population world distribution, including uncertainty about unlocalized humans; frozen and outside Stage 2 development. `eventwm_graph/l2_two_stage/conditioned.py::HistoryConditionedWorldModel.predict(history, obs, partial_graph=None, horizon=3., worlds=32, seed=None, actions=(), executed_action=None)`. Supporting `model.py`, `neural.py`, `population.py`, `motion.py`, `context.py`, `positive_history.py`, `association.py`. Tracker IDs are not guaranteed physical identities. A research checkpoint exists; historical population/calibration gates were not all met. Do not present it as a fully accepted system. |
| Stage 2 | Predict newly observed legal static certificates along a supplied robot trajectory. Implemented and trained; this handoff concerns its quality and complete cost. Stage 1 future predictions are not inputs. |
| Sandbox / robot graph | Propose and execute hypothetical native robot trajectories from legal current state and goal. See `partial/query.py`, `partial/execution.py`, `robotmap/`, `l2_partial/planning.py`, `queries.py`, `structural_adapter.py`. It must not grant hidden-map connectivity to the online predictor. |
| Graph Compiler / L2.5 | `eventwm_graph/l25/{compiler,schema,graph,numeric}.py`; `QueryOutcomeCompiler.build_context` and `evaluate_candidates`. Builds typed spatial/query/outcome graph features, provenance/support masks and one legal-root query score, reuses a shared human future bank, and evaluates prefix collision/social/observation outcomes. Unsupported true contact, episode completion and geodesic quantities remain unsupported, rather than fabricated. Historical implementation targets the frozen `cf197122` belief interface. Integration with the new range-backed Stage 2 API is still work. |
| L3 | Intended high-level query/goal selection and RL policy learning. Example message-passing and Falcon policy code exist, but there is no verified substantial L3 RL run using this new Stage 2. Do not infer integration from individual module tests. |
| Native execution | The existing `eventwm_graph/l4/` namespace contains executor/search/forecast code (`Executor.start/act/on_transition/feedback`); it is not a new prediction task. Uses real transitions and observations. Some older benchmark profiles use certified full static geometry and therefore do not establish legal partially observed execution. |

Stage 2's precise original contract is `(G_t, tau) -> predicted increment`, with trajectory rows `[absolute_time, x, y, z, yaw]`. Training trajectories are recorded executed paths; deployment trajectories are planner trials supplied by the sandbox. Only cells UNKNOWN at the start are predicted: UNKNOWN remains UNKNOWN, UNKNOWN becomes FREE, or UNKNOWN becomes BLOCKED. The graph interface also carries height. Already-known cells are retained exactly. Known-cell corrections in recordings are counted separately.

The current best implementation explicitly extends this input with **actual current/past range observations and their actual observed camera poses**. This was a research change after the user authorized broader approaches, while retaining the explicit later prohibition on privileged online inputs. It is not an unannounced graph-only baseline improvement. The sensor extension needs a real online front end and integration agreement; its local simulator replay does not by itself establish that integration.

## B. Current Stage 2 implementation

### Observation and privilege boundary

`eventwm_graph/stage2_repair/current_range_sensor.py::CurrentRangeSensor` is an **environment-side** ideal range camera: 90° horizontal and vertical FOV, 10 m maximum Euclidean range, 128×128 or 256×256 pixels. It uses `NativeRayBatch.nearest_hits` on the simulator; robot and inactive bodies are ignored, **active humans occlude**. The predictor receives measured range arrays, origins, rotations, timestamps and already-observed track history, not collision meshes, scene IDs, navmesh, future cameras or future humans.

`range_branch_replay.py`, `range_train_expand.py`, and `range_full_replay.py` use privileged offline initialization and recorded simulator states to regenerate physical sensing at recorded current/past observation times. Arrays for different branches share an archive, but `range_history.py::ancestor_frames` selects only the legal ancestry at the query start. Future archive frames are not legal inputs. Targets are attached only after legal feature construction. This is replay of existing recordings, not a new data-collection campaign.

Measured range images are ideal first-hit sensing, not validated real RGB-D observations. Reconstructed surfaces are **partial observed geometry**, not a loaded whole-scene mesh. Filtering uses legally observed tracks; missed human detections can leave apparent static surface ghosts. Do not replace it with perfect human masks or make humans transparent. A deployment front end must supply the same calibration and current-time packet. The latest sensor timestamp must agree with the graph's query-start timestamp. A recursively imagined future graph does not license a future real sensor packet.

`legal_camera.py::current_extrinsic` recovers a rigid camera/base transform from the current legal camera/base pair and applies it to the supplied base-position/yaw path. A diagnostic on 128 training Roots / 10,690 frames found maximum origin residual about 5.52e-6 m. Yaw is present; the actual cardinal checks did not establish a sign reversal. Future recorded camera poses are diagnostic/label-side only.

### Graph, certificates, candidates and updates

`eventwm_graph/partial/map.py::PartialGraph` stores sparse `(floor(x/resolution), floor(z/resolution))` keys. Missing=0 UNKNOWN, 1 FREE, 2 BLOCKED. Negative coordinates use floor, x/z ordering is explicit, world y is height. Actual Root resolutions must be read, not assumed from a single scene.

The audited data use certificate semantics, not a physical occupancy segmentation target. A FREE publication requires all **63 certificate probes** (three heights × 21 square/halo positions) to satisfy native visibility/support checks. BLOCKED publications arise from 105 wall rays (35 horizontal × three vertical) and 0.25 m inflation. Within an observation BLOCKED takes precedence; later explicit FREE certificates can correct known BLOCKED cells. `occupied_ego_base_support` is skipped as a grid certificate: the robot's occupied-base support is separate metadata and does not prove an entire cell FREE. `last_seen` for already-published FREE certificates is publication recency, not complete camera visibility recency.

`stage2_segment/data.py::replay` constructs branch-start graphs and later-observation targets. `stage2_repair/diagnose.py::reference_state/reference_update` independently replay dictionary state for alignment checks. A starting-pose-only path produces no reveal. A stationary path with later observation timestamps can legitimately reveal information. Do not replace this with a displacement-zero heuristic.

The evaluation candidate universe is the original initially-UNKNOWN cell centers within a closed 11 m union around supplied poses. The original keys/order, missing-positive accounting and empty windows remain in evaluation. Positives outside the candidate domain count as misses, not exclusions. The 192-cell internal canvas does not shrink this output universe.

`range_api.py::RangeEvidencePredictor.predict_segments(partial_graph=..., trajectories=[...], range_observations=..., track_history=...)` returns per-query candidate keys, three-state probabilities, hard states, heights, concrete sparse `static_updates`, and normally `future_partial_graph`. `increment.py::SparseIncrement`, `materialize.py::future_from_columns`, and `stage2_segment/native.py::NativePartialGraph` construct full hypothetical graph columns and metadata. Known cells, their heights and observation metadata are retained; predictions have hypothetical provenance and do not mutate/certify the real graph. `materialize=False` is a diagnostic option, not the complete inference benchmark.

### Selected learned model

`range_spatial.py::SpatialEvidenceNet` is the selected `legal_range_spatial_1` architecture:

- Shared **14×192×192** top-down canvas: six legal graph channels (FREE/BLOCKED fractions, height, two distances, known fraction), six measured clear/hit channels at three heights, two relative-position anchors. Each internal cell aggregates 2×2 original cells.
- Four two-convolution SiLU stages, widths 16/32/64/96 with 2× average pooling; residual dilated context at dilation 2 and 4. Upsampled concatenation has 208 channels, mixed to a 24-channel local field; pooled deepest context supplies 16 global channels.
- Per original candidate cell: 88 legal graph/trajectory features + 16 measured range/viewing features + 24 local + 16 global = 144; MLP 144→128→64→2 produces additive revelation and conditional-BLOCKED logits.
- Frozen `range_evidence.py::EvidenceNet` adds a 104→128→64→2 range residual to the frozen `stage2_hf/time_model.py::TimeNet` graph/time prior. TimeNet's recency inputs are zero for these models. Existing frozen prior supplies height.
- `r = sigmoid(raw_revelation - log(8))`, `b = sigmoid(raw_blocked)`; probabilities are `[1-r, r*(1-b), r*b]`. Selected A uses **argmax**. A stored threshold 0.2 is not an argmax operating point. Do not apply both rules.

Original `SpatialNet`'s learned field was approximately 48×48 original cells (about 4.88 m in the audited resolution), despite a large input. It also had nonlocal distance/line summaries. The current FPN/global model is different; that old receptive-field calculation is not a measurement of this model.

### Data and training

`range_online_data.py::inventory/OnlineRangeGroups` freezes eligible source Roots, ancestry groups, source identities and deterministic seeds. `range_online_features.py::LegalFeatureBuilder` consumes named legal graph/sensor/path fields. `fresh_data.py::draw` samples fresh strata each pass (128 positives, 384 near-UNKNOWN, 128 other UNKNOWN when available), with inverse-inclusion, per-window/Root and scene weights. The candidate universe is constructed before sampling. Padding/empty handling and weights have real-batch tests. Old fixed 96-cell samples were a material fitting limitation.

`range_online_train.py::make_batch/run` trains revelation BCE with positive weight 8, plus `8 * is_revealed * conditional-BLOCKED BCE`, normalized using sampling/scene/window weights; adds 0.1×typed soft Dice and 0.25×auxiliary future-published-certificate-class-union loss. Auxiliary future arrays are targets, never deployed inputs. AdamW lr 0.0002, weight decay 0.0001, 100-update warmup and cosine schedule, gradient clipping 5, seed 5511, typical batch 4 / four data workers. AMP applies to spatial encoder/head; point prior stays F32 with TF32 off. Heights remain frozen. `--train-range-head` is an optional tested control, not selected A.

Earlier `support_teacher.py` / `support_graph_prepare.py` use privileged installed-navmesh support as an explicitly training-only auxiliary target. `range_online_surface_train.py` similarly investigates measured-surface training targets. Neither is a license to pass hidden support or future surfaces to inference. Labels remain published certificates.

### Runtime implementation map

All following names are under `eventwm_graph/stage2_repair/` unless stated otherwise.

| Files / functions | Purpose and important limits |
| --- | --- |
| `range_fast.py`, `range_exact_fast.py`, `device_range_features.py` | Original 63 LOS probes, wall-ray forecast, nine floor probes, supplied-path exposure/frustum features and observed obstacle distance capped at 2 m. No hidden complete occupancy. |
| `device_pipeline.py::ColumnPreparedPipelinePredictor` | Reference fast-columns execution used by completed full-development evaluation: shared map preparation, exact GPU features, concurrent work, complete graph materialization. |
| `device_query.py`, `device_point_features.py`, `gpu_edt.py` | Candidate ordering, original 88 features, exact distance/height/tie behavior. CPU boundary fallback remains. Whole-batch x88 storage is already preallocated; do not propose that as new. |
| `range_planar.py::PlanarRangeMesh` | Full-pixel measured surface mesh, keyframe motion thresholds 0.07 m / 0.07 rotation-matrix difference; merges valid 4×4 planar patches, with holes and 0.3 m edge safeguards. Native mode skips redundant software BVH/Morton construction. |
| `native_triangle_api.py::NativeTrianglePredictor`, `optix_backend.py`, `optix_native_bridge.cpp` | Native hardware triangle traversal. Earlier `optix_bridge.cpp` custom AABB / padded intersection implementation is retained. Watertight boundary semantics differ, so this is not automatically bitwise equivalent geometry. Measured pilot/real-case tests support the selected use. |
| `shared_sensor_api.py::{SensorBuffers,SharedSensorPredictor,SharedFastBuildPredictor}` | Share one host-to-device sensor copy between mesh and canvas; cache descriptors only after full-content identity checks; in-place input mutation invalidates reuse; noncontiguous inputs supported. Fast build switches native acceleration preference to PREFER_FAST_BUILD. |
| `shared_sensor_api.py::Plane3mmPredictor` | Approximation: measured point-to-reference-plane tolerance 0.003 m instead of strict 0.00005 m. Keeps hole/edge safeguards. Pilot delta is measured below; a 3 mm point-plane bound is not a universal 3 mm ray-hit distance bound at grazing angles. |
| `range_identity.py` | XXH3_128 full-content identity (`libxxhash`, SHA256 fallback); separate SHA256 for files/checkpoints. Already optimized; never replace content checks with pointer identity alone. |
| `cuda_module.py` | Cache immutable compiled `_CodeModule` by source/device; independent wrappers retain mutable OptiX state. Fixes a demonstrated module/resource leak. |
| `decode.py::decode_columns`, `increment.py`, `materialize.py` | Fused Numba decode, concrete sparse increment, normal full hypothetical graph. No deferred output shortcut. |
| `range_implementations.py` | Explicit predictor selection. Pilot supports `plane3mm`; the current full-evaluation CLI does not expose that option. Do not invent a full plane3mm command without wiring/testing it. |
| `range_sensor_cache.py` | A sensor change needs BOTH new 16-feature and 14-channel canvas caches. Checks original keys, labels and x88 exactly; preserves old caches. |

Important preserved alternatives: `bilinear_context.py`, `ray_residual_model.py`, `initial_model.py`, `cardinal_augment.py`, `increment_loss.py`, `tile8_api.py`, `full_history_api.py`, `captured_head.py`. Their presence does not make them selected production candidates. Native-only traversal explicitly rejects `fast_features=False`; normal supported out-of-domain/boundary feature fallback is tested. Portable CPU-only inference is not established for the selected accelerated model.

## C. Current experimental status

Use the absolute paths in section I and abbreviations in section D. Values here were read from actual completed summaries on 2026-10-26. Original pilot: 31 development Roots / 16 scenes / 666 windows. Original full development: 2,110 / 108 / 47,128. Already-inspected confirmation data are not a new untouched holdout.

| Evaluation / input | Label-aware increment IoU | Revelation IoU | Evidence under `$OUT` |
| --- | ---: | ---: | --- |
| Historical graph-only baseline, fixed pilot | 0.2174127202 | 0.3016100480 | Historical packed pilot/baseline receipts; source snapshot `15644c6` |
| Historical best new graph-only pilot | 0.2188987313 | See old report | `$HF/training/` checkpoint SHA `8293cd10…`; not a range model |
| Original baseline, full development | 0.2177327982 | 0.2869848059 | `full_evaluation/legal_range_online16384_epoch01_columns/summary.json`, baseline field |
| Repaired graph-only point prior, full development | 0.2195680605 | 0.28666122 | `full_evaluation/selected_global_point/summary.json` |
| B: 4,096-Root E2, 128 sensor, full development | 0.4082849606 | 0.4548212895 | `full_evaluation/legal_range_online4096_epoch02_columns/summary.json` |
| **A: 16,384-Root E1, 128 sensor, full development, completed** | **0.4171172210** | **0.4666754111** | `full_evaluation/legal_range_online16384_epoch01_columns/summary.json` |
| A, 128 sensor, fixed pilot | 0.4247093503 | 0.4781501774 | `range_runtime_online16384_epoch01_columns/summary.json` |
| **A, 256 sensor, strict/shared, fixed pilot** | **0.4532446764** | **0.5061800299** | `range_runtime_online16384_epoch01_sensor256_shared_fast/summary.json` |
| A, 256 sensor, plane3mm, fixed pilot | 0.4530660768 | 0.5059950465 | `range_runtime_online16384_epoch01_sensor256_plane3mm/summary.json` |
| Frozen bilinear-context A diagnostic, 256 pilot | 0.4534557632 | 0.5058246688 | Diagnostic only; tiny +0.000211, no fitted bilinear model selected |

The new completed full A evaluation replaces the user's historical 0.4083 reference: **+0.0088322604** label-aware IoU versus B, with identical full universe and 128 sensing. Full A pooled IoU is **0.3991845393** (B 0.3904051684). Its manifest SHA256 is `4187cf65a73c4eafdedbc7b862b01a9bc4820ece12b9e980e5305aa6559bc128`, evaluator SHA256 `1b107549af59482e05dc77a727af51dad0c886e6e5d4fdf2957523afdf27e17e`. Frozen evaluator worktree is `stage2-joint-20261026-0656`, Git `31a508239d4a4c52ef86581c88b7236ec91a9379`. Completed log: `$OUT/range_full_online16384_epoch01_columns.log` (2,110/2,110, no pending, about 6,181 s). No 256-sensor full-development evaluation has been executed at handoff.

A full hierarchical FREE precision/recall: **0.526250 / 0.586600**; BLOCKED: **0.753799 / 0.503186**. Pooled confusion rows=true, columns=predicted, state order UNKNOWN/FREE/BLOCKED:

```text
[[1704715220, 1660701, 1380682],
 [   1727814, 2607254,  216805],
 [   4549420,  488532, 4521336]]
```

This separates 6,277,234 missed new observations, 3,041,383 false new observations and 705,337 wrong FREE/BLOCKED classifications. Of 2,561 no-new-information windows, A predicts exactly empty on 2,143, versus baseline 736. Dominant residual error is missed revelation, especially BLOCKED, rather than merely wrong FREE/BLOCKED typing. Do not substitute high UNKNOWN accuracy for reconstruction.

### Exact selected checkpoints

| Name | Path under `$OUT/training` | SHA256 |
| --- | --- | --- |
| A, best current selected weights | `range_online16384_streamed/selected_epoch01_argmax.pt` | `41602272b9b5a4a7355f97508de8585c0465ff59d82282cf8a50a8a2038a5218` |
| B, previous completed full best | `range_online4096/selected_epoch02_argmax.pt` | `7ff203237eec551e28a8ba0eb62fc72759df5ccef15766f40279ce5c8162f605` |
| B's warm model | `support_pretrained256_control/selected.pt` | `6e6879c1e679e51231d46961492eee99be877a0f0302038d1b842a19fa0fddf4` |
| Constructor dependency 1 | `range_evidence64/selected.pt` | `057eeb69e23a077845ea7c549b026fbd5bc67d71e790a283d3027a6a3697565b` |
| Constructor dependency 2 | `point_residual_full/epoch06.pt` | `64be876a5f63611c51f25bb0e33ffc00ebc96939fc002974e903c6b59fc3e7e8` |

Constructor dependency 3 is ignored local `docs/l2/stage2_segment_20261024/artifacts/segment_ray104.pt`, SHA256 `1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd`. Even though top-level checkpoints carry weights, constructors currently load these linked predecessor files. Copying only A is insufficient. Full chain/configs are in `STAGE2_HANDOFF_RESOURCES_20261026.json`.

A trains on 16,384 eligible Roots / 470 scenes / 385,430 windows / 250,318 map groups. Frozen fit source `6478a1d8`; snapshot SHA256 `f358230c5ee3468870d33f36cc8c7bb60f8c20d0a29b50c92e3b12ab704f82dc`; source training manifest SHA256 `dc5d93bef00db38fb47ca3450d144af8565c3039bf365c3340bc82256cd6131e`. A's old config lacks `training_source_commit`: use its frozen worktree and snapshot, not an inherited prior-model source field. B used 4,096 Roots / 463 scenes / 96,186 windows / 62,338 groups; original source `01c600fb`, selected E2 preceded later migration to module-leak-fixed `11fa958e`. The point prior had full 105,838-Root training. Do not describe A as end-to-end full 105,838-Root range fitting.

### Complete latency

Latest matched source includes native triangles, one shared sensor copy, fast native BVH build. Same A weights and original 32 workload maps, three repetitions, all K=1/4/6, dense and representative; 48 calls per cohort/cache/K. Sensor256 full dense K=6 contains about 203,542–253,792 candidate rows. Below: **P50 / P95 ms**, selected range model only.

| Implementation / cohort / cache | K=1 | K=4 | K=6 |
| --- | ---: | ---: | ---: |
| Strict, dense cold | 19.664 / 42.901 | 33.700 / 51.489 | 41.135 / 58.573 |
| Strict, dense reusable map / new queries | 9.268 / 14.457 | 23.083 / 32.432 | 31.327 / 42.884 |
| Strict, dense identical-query warm | 8.938 / 13.431 | 23.062 / 32.906 | 30.098 / 42.149 |
| Strict, representative cold | 18.043 / 20.301 | 27.689 / 33.113 | 35.240 / 41.888 |
| Strict, representative reusable map / new queries | 8.312 / 11.384 | 19.579 / 24.986 | 26.136 / 32.876 |
| Plane3mm, dense cold | 19.002 / 36.213 | 33.272 / 44.843 | 42.055 / 52.539 |
| Plane3mm, dense reusable map / new queries | 9.305 / 14.953 | 22.926 / 33.102 | 31.655 / 42.899 |
| Plane3mm, dense identical-query warm | 8.821 / 13.010 | 22.535 / 32.406 | 30.527 / 42.254 |
| Plane3mm, representative cold | 16.688 / 19.828 | 26.811 / 31.049 | 33.966 / 38.826 |
| Plane3mm, representative reusable map / new queries | 8.050 / 10.789 | 19.225 / 23.150 | 26.026 / 31.278 |

Evidence: `$OUT/benchmark_online16384_sensor256_plane_comparison_shared-fast-build.json` and `..._plane3mm.json`. Full summaries, including baseline in the same run and remaining warm rows, are copied into the compact handoff resource manifest. Strict pilot/native/shared variants retained exact per-window confusion and graph checks over all 666 pilot windows; real mutation/noncontiguous/single-vs-batch tests cover more than matching class counts. This is not a claim of bitwise probability equality over every pilot row.

Timer includes input normalization/content identity, graph preparation, candidate construction, features, CPU/GPU transfers, synchronization, network, probabilities/hard decode, concrete sparse update and full normal hypothetical graph. Cold clears both graph and mesh caches. Reusable-map/new-query mode primes only the starting pose; identical-query warm repeats the same paths. **Excludes input deserialization, output serialization and environment sensor acquisition**. Graph outputs are not removed. RTX3090, one CPU thread for BLAS/Torch, AMP spatial branch/F32 prior, TF32 off, F64 camera/frustum geometry.

Prior shared-fast run measured 53.8798 ms cold P95 and 37.0123 ms reusable-map P95 (the user's 53.9/37.0 references). Latest paired strict run was noisier and is reported above, rather than selected away. External TeamViewer activity remained untouched. Benchmark wrappers temporarily paused only recorded owned research jobs and resumed all at `2026-10-26T10:45:41.666Z`; receipts include birth-time checks. This is not an uncontended machine acceptance guarantee. **No processes were paused or terminated for this handoff.**

Startup is separate: in the latest paired run, first strict constructor ~173 ms and first complete K6 ~1,497 ms. Plane3mm ran second with backend/JIT warmed (~92 ms constructor / 37 ms first call), which is not a comparable clean fresh-process startup result. The 3 mm approximation reduced median dense triangles from ~1,009,065 to ~685,714, but its measured pilot label IoU loss is 0.0001786. Full-development approximation impact remains untested.

### Privileged geometry diagnostics — never accepted online performance

`geometry/native_geometry_pilot/summary.json`: 0.9108944845 pilot IoU; GPU counterpart 0.9123244672. `geometry/legal_current_camera_oracle.json` uses a legal current rigid camera transform but still **privileged complete geometry**, and reports 0.9123244672. Legal camera poses do not legalize a full mesh.

There is also a **completed full-universe oracle diagnostic**, not just a strong pilot: `$OUT/full_evaluation/geometry_selected_f32/summary.json`, all 2,110 Roots / 108 scenes / 47,128 windows, label/revelation IoU **0.9278161035**, pooled 0.9368052123. Its explicit role is `OFFLINE PRIVILEGED-GEOMETRY/FUTURE-CAMERA ORACLE DIAGNOSTIC ONLY`. `geometry_full_evaluate.py` marks `oracle_only=True`, `acceptance_eligible=False`. It uses complete static geometry/support and future camera information; static-only geometry also omits active-human visibility effects. It helps locate an information/geometry gap but does not satisfy the WM contract or either real-WM acceptance requirement. See `docs/l2/stage2_repair_20261025/GEOMETRY_CONTRACT.json`.

## D. Reproduction instructions

These commands reference actual existing entry points and local resources. Training/pilot/full/preparation CLI help was validated during handoff without launching those jobs. The numerical results above were executed before/during the already-running work, not reproduced anew for publication. GPU tests/benchmarks were not rerun just to improve a handoff number. **Do not duplicate the running jobs in section I.** Use new output names for new identities; receipt checks reject stale incompatible caches. Publication contains no weights/data and is not a self-contained fresh-machine benchmark package.

### 1. Environment and existing local artifacts

```bash
source /home/haodong/miniforge3/etc/profile.d/conda.sh
conda activate eventwm-falcon
cd /home/haodong/workspace/projects/eventwm-falcon-stage2-claude-handoff-20261026
export PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
export DEV=/home/haodong/workspace/projects/eventwm-falcon-stage2-repair-20261025
export OUT="$DEV/output/stage2_repair_20261025"
export HF="$DEV/output/stage2_hf_20261025"
export CP_A="$OUT/training/range_online16384_streamed/selected_epoch01_argmax.pt"
export CP_B="$OUT/training/range_online4096/selected_epoch02_argmax.pt"
export EVENTWM_NATIVE_BATCH_BUILD=/home/haodong/workspace/projects/eventwm-falcon/output/stage01_acceleration_20261021/native_batch/build
export FALCON_TEST_DEVICE=cuda:0
```

The handoff worktree has ignored `output -> $DEV/output`, `data -> /home/haodong/workspace/baselines/Falcon/data`, and the segment-checkpoint link described above. A fresh clone needs these resources restored/relocated deliberately. `scripts/stage2_segment_run.py` sets research-root imports, one-thread BLAS settings, Numba workqueue/cache, and hardcodes the old `eventwm-falcon-l2-two-stage-20261023` base and its locally supplied orjson dependency. Use this runner rather than bare module imports. Merely exporting `EVENTWM_STAGE2_BASE` does not override its hardcoded assignment. See section I before porting.

Optional missing-header preparation for a relocated installation (already present here):

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.optix_setup
```

This fetches pinned external build headers into ignored `$OUT/deps`, not a new CUDA/system install. OptiX 7.7.0 repository commit `7b5c4e8608b8b4b601729f6240fc3fd53cb36d23`; CUDA runtime header wheel 11.8.89 SHA256 `92d04069a987e1fbc9213f8376d265df0f7bb42617d44f5eda1f496acea7f2d1`. `FALCON_OPTIX_INCLUDE` / `FALCON_CUDA_INCLUDE` can override include paths. Native bridges compile locally; compiled libraries and PTX are excluded from Git.

### 2. Dataset preparation

Reuse existing prepared packets first. The following are the actual single-worker equivalents of current 256 preparation. They do not start the collector. Do not run over a directory with live writers; inspect section I and receipts before resuming a missing range of Roots.

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_train_expand \
  --roots 4096 --offset 0 --split train --sensor-size 256 \
  --destination expanded_train_range_branches_256
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_full_replay \
  --roots 2110 --offset 0 --split dev --sensor-size 256 \
  --destination full_range_branches_256
```

128 preparation uses the same modules with `--sensor-size 128` and destinations `expanded_train_range_branches` / `full_range_branches`. Source Root identities and sensor checksums are enforced. `range_sensor_cache` creates BOTH matching 256 pilot caches from already prepared `range_branches_256`:

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_sensor_cache \
  --device cuda:0 --name sensor256 --sensor-directory range_branches_256
```

Those caches are already complete (31 Roots / 666 windows). Do not mix 256 point features with old 128 canvases. Original packed data and replay dependencies are also local; see inventory paths in I. On a fresh machine, relocating manifests/old source paths is necessary before these commands can run.

### 3. Training and selecting completed checkpoints

The recipe that produced A is below; it is **already running its second epoch**, so this is a reproduction recipe rather than an instruction to start it again:

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_online_train \
  --roots 16384 --epochs 2 --name range_online16384_streamed \
  --batch-size 4 --workers 4 --device cuda:1 --warm "$CP_B" \
  --device-canvas --device-edt --wait --stream-preparation
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.online_select \
  --names range_online16384_streamed --epochs 2 --device cuda:1 --watch \
  --output-name online16384_selections
```

New fitting requires a new `--name` (the trainer refuses overwriting an existing destination). `--resume <epoch.pt>` is for optimizer/scaler-preserving continuation in the matching existing name/snapshot. A different sensor directory requires both `--dev-feature-manifest` and `--dev-map-manifest`. Current bounded256 control used 256 Roots / four epochs, warm A, `expanded_train_range_branches_256` and `$OUT/sensor_cache_dev_sensor256_{point,maps}/manifest.json`; best 0.4487702902 at E1, below warm A. Do not substitute this result for A or assume finishing four epochs improves selection.

### 4. Fixed pilot via the public API

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_runtime_evaluate \
  --checkpoint "$CP_A" --device cuda:0 --spatial \
  --implementation shared-fast-build --sensor-directory range_branches_256 \
  --name claude_reproduce_A_sensor256_shared_fast
```

Expected original 666-window result: label IoU 0.4532446764. For the measured approximation use `--implementation plane3mm` and another name; expected 0.4530660768. For A128 use `--implementation fast-columns --sensor-directory range_branches`, expected 0.4247093503. Actual candidate keys, training labels and hypothetical graph preservation are checked in this evaluator.

### 5. Full development

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_full_evaluate \
  --checkpoint "$CP_A" --device cuda:0 --implementation fast-columns \
  --sensor-directory full_range_branches --name claude_reproduce_A128_full
```

Expected 2,110 Roots / 47,128 windows, 0.4171172210. This evaluator additionally requires original full baseline `.json.gz` / `.probabilities.npz` receipts and `selected_global_point` Root receipts under `$OUT/full_evaluation`; they preserve exact original candidate-domain accounting. They are local large artifacts, not GitHub files. Reusing the existing original evaluation name with matching identity resumes verified per-Root receipts; a new name recomputes. Do not pass `--limit` for acceptance. `--wait` supports pending prepared packets; it does not establish completion.

After 256 full preparation finishes, the same entry point accepts `--sensor-directory full_range_branches_256 --implementation shared-fast-build` with a **new** name. That experiment has not been run. Freeze the checkpoint/operating point before it. Full CLI currently does not expose plane3mm or bilinear; extending it is future implementation work, not a result delivered here.

### 6. Complete K=6 dense latency (also K=1/4)

The actual paired original256-workload benchmark entry point is:

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.plane_tolerance_speed
```

It uses A, `workloads('range_benchmark_256')`, three repetitions, complete materialization, and produces the two files listed in C. **Read `quiet_owned_jobs.py` first:** this historical benchmark wrapper briefly SIGSTOPs only its listed owned jobs, checks process birth identity and resumes in `finally`. During this handoff it was not executed. Claude should update/review its names or invoke `range_benchmark.run(..., _workloads=workloads('range_benchmark_256'))` in an explicitly chosen load setting. Never pause external services/collector to obtain a passing number. Generic `range_benchmark` CLI defaults to original128 workloads; do not accidentally label that sensor256.

Standalone original128 complete benchmark command:

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_benchmark \
  --checkpoint "$CP_A" --device cuda:0 --spatial --implementation fast-columns \
  --repetitions 3 --map-only-warm --name claude_A128_complete
```

### 7. Regression tests and publication validation

Run after choosing an available GPU, not alongside latency acceptance measurement:

```bash
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_cuda_module_lifetime
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_fast_prepared_pipeline
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_shared_sensor
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_native_triangle
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_range_online
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_bilinear_context
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_increment_loss
"$PY" scripts/stage2_segment_run.py eventwm_graph.stage2_repair.test_history_pruning
"$PY" scripts/audit_stage2_publication.py --repo . --ref HEAD \
  --output "$OUT/claude_publication_audit.json"
```

Existing evidence: `shared_sensor_tests.log` (3 passed), `module_lifetime_parity_tests_v2.log` (5 passed including online batch targets/weights and real pipeline parity), `bilinear_context_tests.log` (2 passed), plus `native_triangle_tests.log`, `range_online_parity_tests.log`, `fast_prepared_pipeline_tests_v2.log`. These are prior test executions, not a claim that the entire historical test tree has just passed on this source. Some old logs deliberately retain failures later repaired; read the named latest receipts. For CPU-only test selection, inspect each test's imports before assuming no CUDA initialization.

## E. Deficiencies, evidence and research opportunities

Confirmed implementation defects are distinct from representation limitations and unresolved hypotheses:

| Finding / experiment | Evidence and consequence |
| --- | --- |
| CUDA module lifetime leak — fixed | Repeated actual modules: 1,024 instances grew host ~443.78 MiB and GPU ~92 MiB; immutable module cache reduced growth to ~0.043 MiB host / zero GPU. Real kernel, threads/device identity, mutable-state isolation and public-output parity tests exist. Earlier jobs failing `cuModuleGetFunction` status 2 were preserved and resumed with receipts. |
| Evaluation/API zero-start contract — fixed | Light evaluation lacked the zero-horizon UNKNOWN override. Unified starting-pose-only behavior. Zero affected fixed-pilot windows; does not explain the best spatial model's plateau. Later stationary observations are not suppressed. |
| Static sampled fitting limitation — changed | Same fixed 24 TRAIN examples, full original candidate reconstruction: old fixed96 sampling ~0.25581 versus fresh-stratum fit **0.9009389**. This is a fitting result, not generalization. Optimizer updates and encoder gradients were observed, not inferred from declining loss. |
| Replay/ordering/alignment | Independent dictionary reference traced 24 real training windows across motion/target cases; exact keys, boundaries, ancestry, metadata and labels checked. No single global misalignment or yaw-sign bug was established. Source audit's synthetic failures were leads, not dataset prevalence estimates. |
| Measured legal sensing plus spatial context | Graph-only repaired full ~0.2196; explicit range/spatial model B ~0.4083 and A **0.4171**. This is a major measured improvement with an input-contract extension, not a pure architecture-only causal ablation. |
| Sensor resolution | Same A pilot, 128→256: **0.424709→0.453245**. Both point evidence and canvas caches rebuilt; public API/cached pilot agree and original keys/labels checked. Latest 256Root refit E1–E4: 0.448770 / 0.447037 / 0.437838 / 0.442921; none exceeds the unchanged warm model. Full256 generalization unmeasured. |
| Human occlusion | Actual source provenance matters: old `scripts/collector/observation.py::HabitatHumanObserver` differs from `partial/{observer,fast_observer}.py`; acceleration binds the fast observer. Cohorts used metadata human presence, not no tracks. Descriptor-controlled full E1 zero-human ~0.3948 versus human-present ~0.3795 was descriptive, not a causal experiment. Zero-human errors remain substantial. Do not attribute all errors to humans. |
| Camera/base | Current rigid extrinsic residual measured above; no missing-heading explanation. Current input uses legal calibration. No evidence that variable camera motion alone explains remaining error. |
| Recency | FREE publication age is not full visibility recency. Earlier recency ablation did not establish improvement. Adding temporal features blindly is not a demonstrated remedy. |
| Bounded architectural/training controls | 256Root controls: control best .410799, cardinal .406870, ray-residual .412078, initial-visibility .412087, micro-Dice .413351, range-head unfreezing .408539; warm B .412756. These did not establish material improvement. Full point probe details remain in `BOUNDED_TRAINING_COMPARISONS.json`. |
| Ray residual capacity | Same24-train fit .8757085, but no material development gain. More fitting capacity alone did not solve generalization. |
| Bilinear context | Frozen A pilot .453456, +.000211; no learned bilinear fit, full evaluation or deployable speed assessment. Preserved hypothesis, not selected success. |
| Internal coarse cells | In 62 archived examples, mixed-label2×2 groups accounted for ~27.3% of errors. This is association, not proof that pooling causes those errors; exact per-cell geometry/features already remain. |
| Speed changes | Native triangles/shared H2D/fast build materially reduced complete cost. 8×8 tile merging was slower (~70.8 ms dense cold P95) and rejected. Existing hashes and x88 batch allocation are already optimized. Plane3mm trades small measured pilot loss for lower triangle count; full impact pending. |

Small same-model train-cohort probe: 32 TRAIN Roots / 848 windows, A128 IoU .4195038993 and A256 .4451514978. This narrow inspected training cohort is not a full training metric, but its errors resemble development and argue against calling the gap solely scene overfitting. A128 pilot .4247 versus full .4171 is a matched-sensor gap; subtracting full128 from pilot256 conflates sensor resolution with generalization.

Representative coordinate-aligned visualizations are local `$OUT/figures_online4096_epoch02` and use actual certificate labels, graph, path/headings, prediction and FP/FN/wrong-class overlays. They are B128, not new A256 predictions. Each runtime pilot stores two full key/label/path/graph/prediction examples per Root (62 total), including native/shared/plane3mm output directories; use those for new-model plotting rather than relabeling old figures. Detailed initial replay/fit and failure analyses remain in `$OUT/WORKING_STATE.md`, `$OUT/stage2_audit_15644c6/`, and compact tracked `docs/l2/stage2_repair_20261025/*.json` reports.

Remaining hypotheses include fine spatial feature aliasing, inadequate learned use of observed surfaces, certificate-support uncertainty behind unseen geometry, missed-human surface artifacts, dataset coverage and multiscale inductive bias. Complete geometry oracle results show that predicting certificate visibility is much easier given hidden geometry, but do not identify which legal-information model can close that gap. Neither a few identical-input collisions nor oracle performance proves the original legal task impossible or solved.

## F. Performance requirements

**Acceptance: full-development label-aware increment IoU > 0.50 and complete K=6 dense cold P95 < 50 ms**, under the original universe/accounting and legal deployed inputs. A revelation IoU above 0.50, pilot-only result, warm latency, or privileged oracle does not substitute. Current status: **below target / research integration candidate only**.

Preferred objectives: full IoU around 0.55–0.65 or higher, cold P95 around 20–30 ms, robust across scenes. These are goals, not claims of attainability and not permission to change metrics. The earlier 0.35 milestone / 0.45 target are historical records superseded by the user's later >0.50 requirement. An older total-Layer2 100 ms budget is not the current Stage2 budget.

L3 should ideally finish a substantial RL run within 48 hours, with 72 hours the upper planning target. For illustration, one million serial K6 batches at 50 ms consume ~13.9 hours of Stage2 service time; 20–30 ms consume ~5.6–8.3 hours. Actual end-to-end time also includes Stage1, environment/sensor acquisition, graph compilation, rollout/learner synchronization, data transfer/serialization, policy optimization and contention. Parallelism can change this arithmetic. No substantial integrated L3 throughput measurement establishes the requested schedule yet.

## G. Research and scientific constraints

These constraints come from the user's Stage2 contract, later clarification that a real WM cannot access privileged data, and current handoff/acceptance instructions; they are not new algorithm restrictions imposed by this document. The user authorized wider redesign. Claude may redesign Stage2 when evidence warrants it, while keeping legal inputs, an honest protocol and the two requirements.

1. The real graph contains only legal observations. Predicted updates remain hypothetical. Already-known corrections are separate from the initially-UNKNOWN target. Preserve heights and all required outputs.
2. FREE/BLOCKED are the actual published configuration-space certificates, not interchangeable with occupancy, raw depth hits or navmesh support. Preserve target semantics and the full evaluation universe, including no-new windows and out-of-domain positives.
3. Static scene geometry is allowed inside the environment sensor and for clearly marked offline labels/diagnostics. A complete scene mesh/navmesh, hidden connectivity, future sensor/human truth or full static occupancy is not a deployed predictor input. Partially reconstructed past measured surfaces are allowed only through the explicit sensor extension. Do not silently compare that extension as graph-only.
4. Stage1 human forecasting is separate from Stage2 static information prediction. Legal already-observed tracks may filter current range evidence; future humans/predictions are not currently Stage2 inputs. Stage1 stays frozen for this work.
5. Train/development scene manifests stay intact. The previously opened 222 confirmation Roots are excluded from fitting/model selection and are already inspected, so cannot be called untouched confirmation. Do not select on newly designated test data after inspecting it.
6. Supplied trajectories, including yaw/time and the true starting pose, condition prediction. Planner trials do not imply future real observations. Preserve single-query vs batched equivalence for unequal lengths/extents and different cache states.
7. Primary metrics use the existing equal-scene / equal-Root / equal-window **mean confusion then rates** aggregation, not a convenient new averaging scheme. Also report pooled confusion/IoU. Use `scripts/stage2_diagnostic/common.py::extended_rates` and `full_evaluate.py::summarize`; wrong-class, missed and false revelations must remain distinguishable. Keep operating-point selection explicit and freeze it before genuinely untouched evaluation.
8. Complete speed includes all required materialization/decoding/transfer work as in C. Report startup and exclusions separately, original workload identities and load. No dropping difficult candidates/windows or benchmarking only a forward pass. Numerical/geometry approximations need matched reconstruction evidence.

## H. Recommended next actions for Claude

1. **First concrete task:** read this document and the resource manifest, inspect the still-running A epoch2 and 256 preparation jobs, and verify the completed A128 full summary/identity/checkpoint hash. Avoid duplicate writers or killing jobs. Then independently reproduce the unchanged A256 public-API pilot (command D4) before changing its representation. This anchors legal input, reconstruction, output semantics and runtime.
2. Essential integration decision: establish how the actual online robot will supply current/past range packets (or retain a graph-only variant), calibration, timestamps, human occlusion, cache ownership and hypothetical-vs-real provenance. Connect the new native graph/range API to the historical compiler/sandbox explicitly. Ideal simulator range replay is not a finished deployment front end.
3. When existing256 full packets finish, evaluate **frozen A** on the original full development set. This is the most important missing matched accuracy result. Check A epoch2 selector output when it actually finishes; do not call an unfinished epoch better. Bounded sensor256Roots256 fitting is now complete and below warm A, so repeating the same short recipe without a new hypothesis has weak support.
4. Investigate remaining misses with coordinate-aligned train/dev examples and measured surface/LOS/support evidence. Separate absent legal information from poor learned use, fine-cell aliasing and sampling/loss problems. Repeat an independent small-set reconstruction fit for any redesigned representation before scaling. Preserve the old full universe for comparison.
5. Promising alternatives, not mandates: learned multiscale/fine-cell surface context, more direct ray/surface reasoning, query-conditioned field heads, improved representation of support uncertainty, or a simpler distilled deployable model using legal sensing. Bilinear context is a cheap bounded hypothesis; its +.000211 frozen result does not establish a useful training gain. Complete-geometry oracles can remain training/diagnostic teachers, never production inputs.
6. Optimize measured cold map/mesh preparation and feature/graph materialization, not just network FLOPs. Shared copies, whole-batch buffers and fast identity hashes already exist. Recheck 3 mm approximation on full development before adopting; measure cold tail under a documented load rather than choosing the nicest prior run. Do not repeat the rejected slow8tile merge as an established win.
7. Postpone broad unrelated Stage1/policy/calibration sweeps and full L3 RL until a legal interface and stable quality/cost candidate exist. A small integration smoke and throughput accounting are useful before committing to a 48–72 hour RL run. Architectural freedom does not remove the need for an honest matched baseline.

## I. Exact handoff metadata and live state

### Git and preservation

- Repository: `https://github.com/HeHaodong2004/Falcon`.
- Published dedicated branch: `research/stage2-claude-handoff-20261026`.
- Verified implementation commit: `e83edef0e9b329c3758302ef189a7c0208032e9c`; this exact implementation was pushed and remote-verified before writing this document.
- Final publication is the documentation/audit child of that implementation. A commit cannot contain its own hash. The exact final SHA, remote match and URL are written after pushing to local `$OUT/CLAUDE_STAGE2_PUBLICATION_20261026.json` and supplied in the final user handoff. Resolve the final published documentation revision with the command below; its parent/code-source identity is unambiguous.
- Handoff checkout: `/home/haodong/workspace/projects/eventwm-falcon-stage2-claude-handoff-20261026`.
- Original Stage2 checkout: `/home/haodong/workspace/projects/eventwm-falcon-stage2-repair-20261025`, branch `research/stage2-pipeline-repair-20261025`, unchanged HEAD `e83edef0e9b329c3758302ef189a7c0208032e9c`. It had **no modified or untracked files** at takeover. All Stage2 implementations/tests/prototypes were already committed; nothing untracked there was omitted.
- Original live checkout: `/home/haodong/workspace/projects/eventwm-falcon`, branch `perf/stage01-acceleration-20261021`, HEAD `93f41d5dde3b94d4399ffbaa16f16d5f8eea8e08`. It has unrelated historical modified/untracked Stage0/Stage1/LDM/audit work. Complete status inventory is in the resource manifest; all are left intact, not swept into the Stage2 branch or reset. In particular `.gitignore`, `EVENTWM_SR_DATA_AUDIT.md`, `scripts/audit_eventwm_sr.py` have local edits, and many legacy modules/configs are untracked there.
- Remote `origin` points to `Zeying-Gong/Falcon`; `hehaodong` points to the requested owner. Publication uses explicit `git@github.com:HeHaodong2004/Falcon.git`. HTTPS lacked credentials, SSH worked. No upstream/Fibonaccirabbit push, force push or shared-history rewrite.
- This repair branch descends from sanitized root `28b8512e` and public snapshot `15644c6a35cf3081dbf1eb65738966267c08d933`, rather than inheriting dirty live checkpoint history. Full inherited reachable-history audit at e83 found 2,109 objects / 1,739 blobs / 13,245,247 blob bytes, largest 556,270-byte `PUBLICATION_MANIFEST.json`, no >10 MiB blobs, prohibited asset extensions, NUL binaries or recognized credential patterns. Final history is audited again. Pattern scanning is not absolute proof of secret absence. Historical original branches are unchanged.
- No applicable `AGENTS.md` was found in the relevant checkout or its ancestors. Existing repository reports/source guides were consulted; there is no hidden approval requirement to resume the documented work.

```bash
git ls-remote git@github.com:HeHaodong2004/Falcon.git \
  refs/heads/research/stage2-claude-handoff-20261026
git -C /home/haodong/workspace/projects/eventwm-falcon-stage2-claude-handoff-20261026 rev-parse HEAD
```

### Hardware and environment

Conda `/home/haodong/miniforge3/envs/eventwm-falcon`, Python **3.9.19**, PyTorch **1.12.1**, Torch CUDA runtime **11.3**, cuDNN **8302**, NumPy **1.23.5**, SciPy **1.13.1**, Numba **0.59.1**, Habitat-sim/lab **0.3.1**. Local orjson **3.10.18** is supplied from the old two-stage checkout's `output/l2_two_stage_20261023/deps`, not conda package metadata. CPU i9-10980XE (18 cores / 36 threads), RAM ~251 GiB, **two RTX3090 ~24 GB**, compute capability 8.6. Build compiler g++11.4.0. Optional OptiX/CUDA header pins are in D1.

Preexisting NVIDIA kernel driver **595.91.07** vs NVML userspace **595.99.02** mismatch makes `nvidia-smi` fail; Torch/CUDA and OptiX still run. No driver/package changes were made. Do not mistake failed NVML for no active GPU jobs. At initial handoff capture `/home` had ~83 GB free, root filesystem ~553 GB; evolving caches consume space. 256 large preparation was placed on `/var/tmp` to avoid exhausting `/home`.

### Data, caches and excluded artifacts

| Resource | Exact location / status |
| --- | --- |
| Full eligible source training | `/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/final_collection_20261024/manifest.json`: 105,838 Roots / 476 scenes / 2,482,285 mixed windows |
| Original full development source | `/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/manifest.json`: 2,110 / 108 / 47,128; Root payloads under sibling `roots/` |
| Scene/Habitat resources | `/home/haodong/workspace/baselines/Falcon/data`, linked by ignored `data`; includes simulator resources needed only by sensor producers/diagnostics |
| Common repair artifacts | `/home/haodong/workspace/projects/eventwm-falcon-stage2-repair-20261025/output/stage2_repair_20261025` (`$OUT`) |
| Original high-fidelity packed caches | `/home/haodong/workspace/projects/eventwm-falcon-stage2-repair-20261025/output/stage2_hf_20261025` (`$HF`); `protocol.json`, `packed/{full,pilot_fit,dev_pilot}/manifest.json`, `light_eval_cache/manifest.json` |
| Sensors128 | `$OUT/expanded_train_range_branches` (16,384 prepared training Roots); `full_range_branches` (2,110); `range_branches` (31 pilot); `range_benchmark` (32 original benchmark maps) |
| Sensors256 completed | `$OUT/range_branches_256` (31 pilot), `range_benchmark_256` (32 original benchmark maps) |
| Sensors256 training, ongoing | `$OUT/expanded_train_range_branches_256` -> `/var/tmp/falcon-stage2-range256-20261026/train`, requested first 4,096 Roots |
| Sensors256 development, ongoing | `$OUT/full_range_branches_256` -> `/var/tmp/falcon-stage2-range256-20261026/dev`, requested all 2,110 |
| Matching cached pilot128 | `$OUT/range_evidence_dev_v2/manifest.json` and `range_bev_dev/manifest.json` |
| Matching cached pilot256 | `$OUT/sensor_cache_dev_sensor256_point/manifest.json` and `sensor_cache_dev_sensor256_maps/manifest.json`, complete, exact keys/labels/x88 checked |
| Checkpoints/optimizers | `$OUT/training/`, selected/dependencies in C; `$OUT/sensor256_starting_model_check.pt` is config-only A copy for new caches; `bilinear_context_warm.pt` is diagnostic, not selected |
| Evaluations / prediction archives | `$OUT/full_evaluation/`, `range_runtime_*`, `geometry/`, benchmark JSON paths in C |
| Audit attachment | Original live checkout `stage2_audit_15644c6.zip`, extracted `$OUT/stage2_audit_15644c6/{AUDIT_ZH.md,CODEX_DIAGNOSTIC_TASK.txt,reproduce_checks.py,reproduction_results.json}` |
| Detailed notes / tests | `$OUT/WORKING_STATE.md`, named `*.log`, figures and per-Root receipts; compact conclusions tracked under `docs/l2/stage2_repair_20261025/` |
| Build deps/cache | `$OUT/deps/` and ignored output JIT/native/PTX files; environment native build path in D1 |

Datasets, raw observations, checkpoints/optimizers, large cache arrays, prediction archives, simulator assets, optional SDK/wheels/binaries and unnecessary logs are **intentionally excluded from GitHub**, preserved in these local paths. Historical large caches (e.g. `range_dense_train`, `pose_prepared`) also remain. There are no credentials in the intended publication. A fresh clone without local resources cannot reproduce training simply by installing Python requirements.

### Process status at handoff

The machine-readable manifest captures live state at **2026-10-26T11:11:51Z**: 34 Stage2 processes, 641 training256 packet files/receipts and 710 full-development256 packet files/receipts. Counts are live filesystem inventory, not additional verified evaluation results. It records exact `/proc` PID, parent, start identity, cwd, command, worktree revision, prepared-packet counts and capture UTC. PIDs can exit naturally; check identity before acting. No running experiments/preparation/services were killed for this transfer.

| Job | Frozen source / process and status |
| --- | --- |
| A range128 full16,384 fit | `eventwm-falcon-stage2-online16384-20261026-0520`, `6478a1d8`; PID3302010 plus four workers, GPU1. Epoch1 selected A exists; epoch2 still active (~126,800 / 250,318 groups at last log inspection). `$OUT/range_online16384_streamed.log`; selector PID3302049 watches two epochs, `online16384_selections.json`. Do not call epoch2 finished. |
| A128 full development | `eventwm-falcon-stage2-joint-20261026-0656`, `31a50823`; former PID3309811 GPU0. **Completed naturally**, 2,110/2,110, summary in C. |
| Sensor256 first256Roots fit | `eventwm-falcon-stage2-fit256-20261026-0931`, `a4e45bd3`; former trainer PID3312351 / selector3312350. **All four epochs and selection complete naturally.** Best .448770 E1, below warm A. `$OUT/online_sensor256_roots256.log`, `sensor256_roots256_selections.json`. |
| Sensor256 data preparation | `eventwm-falcon-stage2-sensor256-20261026-0924`, `97cb19cb`: 12 disjoint full-dev workers plus 16 additional train workers (offsets256,496,…3856, 240 Roots each). Earlier first256 producer workers finished. Historical 40 PID records in `$OUT/sensor256_preparation_jobs.json`; continuing workers read existing recordings, not a collector restart. |
| External load | TeamViewer/external desktop activity remains; no clean-uncontended promise. See benchmark load receipts rather than current CPU load alone. |

Known blockers/limits: full256 data still preparing; no full256 evaluation; no final A epoch2 result; current best legal-input accuracy/cold latency below target; ideal sensor extension and compiler/L3 integration unfinished; NVML mismatch; local absolute-path/checkpoint constructor dependencies; no verified L3 total runtime. These are measured gaps, not a request to resume Codex research during handoff.

The final publication receipt and this branch are the transfer boundary. Claude should preserve original worktrees and existing jobs, reproduce the current legal result, then choose its own evidence-driven next approach.
