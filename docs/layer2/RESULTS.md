# Unified Layer 2: integration results (2026-10-27)

Branch `research/l2-integration-20261027`. Interface: [INTERFACE.md](INTERFACE.md). Receipts: [receipts/](receipts/).

## Status in one paragraph

Layer 2 works end to end on the frozen development split.

* Stage 1 (`hwm_lite_e12`) and Stage 2 (`v17_a`) are integrated without retraining or modifying either stage.
* The shared decision state, multiple candidates, alignment, legality and replaceability are all validated by genuine
  inference on all 2,110 development Roots: 0 errors, every check passed.
* Steady per-decision latency at K=6 is 37 ms P50 / 45 ms P95.
* One RTX 3090 sustains 65 K=6 decisions/s with 6 processes.

**The system is not accepted as a whole.** Stage 2's own accuracy requirement (full-development label-aware IoU
> 0.50) is still unmet at 0.4831. Integration did not change that number: Stage 2 outputs are bit-identical to the
released model.

## 1. What was integrated (provenance)

| component | source | checkpoint (SHA-256) | status carried over |
|---|---|---|---|
| Stage 1 human world model | `research/stage1-claude-20261026` @ `cd7f3761` (merged, no conflicts; both branch from `2a876fe8`) | `release/stage1_hwm_lite_e12/hwm_lite_e12.pt` `53d44d85…` (byte copy of the `SELECTED.json` checkpoint, 2.3 MB) | selected under the Stage 1 speed rule; beats legacy Stage 1 on its quality metrics (`docs/stage1_hwm/RESULTS.md`) |
| Stage 2 future-observation model | `research/stage2-claude-exploration-20261026` @ `70fae3cf` (branch base) | `release/stage2_claude_v17a/stage2_claude_v17a.pt` `bef7f5fd…` | full-dev IoU 0.4831 (**gate > 0.50 not met**); dense K=6 cold P95 41.1 ms (gate < 50 ms met) |
| Planner / dynamics / Query generator | querygen worktree (in development) | — | not integrated; only its rollout format (`[T+1,4]` poses at 0.048 s) is supported via `trajectory_from_poses` |

New code: `eventwm_graph/layer2/`:

* `state.py`: decision state and candidates.
* `backends.py`: replaceable stage adapters.
* `model.py`: `Layer2WorldModel`.
* `reduction.py`: bit-identical, GIL-free form of the L2.5 joint reduction (3× faster).
* `devdata.py`: frozen-data loader.
* `validate.py`, `benchmark.py`.

Also: `tests/layer2/test_layer2.py` (13 tests), and the runner scripts `scripts/layer2_run.py` and
`scripts/layer2_validate_shards.sh`.

## 2. Integrated validation on the frozen development split

`fulldev_final` (`receipts/validation_fulldev_final.json`): 2,110 Roots, 108 scenes, final code, 4 shards on GPU0.

Decision states:

* **Root decisions:** the legal history, the legal Root graph and 256 px range frames.
* **Later decisions on a recorded branch:** history plus the arrived branch frames, with the graph replayed and every
  recorded `graph_delta` verified.

Candidates are the recorded executions of the alternative Queries issued at the same Root. Recorded futures were read
only to build labels.

| check | result |
|---|---|
| Roots / errors / CUDA faults | 2,110 / 0 / 0 |
| Stage 1 shared bank == standalone `HumanWorldModel.predict` (bit-identical) | 2,110 / 2,110 |
| Stage 2: Roots whose candidates all stay inside the default legal-map extent | 2,081 Roots, 13,216 / 13,216 executions bit-identical to the released v17_a query-benchmark records (hard states, candidate domain, label-aware confusion, expected counts) |
| Stage 2: Roots with candidates leaving the default extent (§3) | 29 Roots, 66 such candidates: 286 / 286 candidates equal a cold single-candidate Stage 2 call. The batch-dependent reference differs on 7 hard counts and 67 expected sums (~1e-5 relative). |
| Shared state: batched K == K single calls on a rebuilt state object | 211 Roots, 1,336 candidates: banks and maps bit-identical |
| Incremental Stage 1 stream (later branch decision) == fresh stream | 2,016 / 2,016 decisions |
| Visible tracks: bank position at t=0 vs legal track position | max 6.9e-7 m (1,209 tracks) |
| Candidate poses on the 0.048 s clock vs recorded poses | max 8.2e-15 |
| Stage 2 grid vs world frame: predicted cells within the 11 m domain of the path | max 10.77 m |
| Clocks: bank / increments vs decision time | 0.0 s |
| Hypothetical future graph flagged, and rejected as a decision-state input | 2,110 / 2,110 |
| Legal input graph unmutated | 2,110 / 2,110 |
| Yaw convention: forward = (cos yaw, −sin yaw) in (x, z) | 1,328 / 1,328 recorded translation steps |

Accuracy, for context only (Stage 2 is unchanged):

* On the recorded full Query executions from each Root, the original Stage 2 hierarchical reduction gives
  label-aware IoU **0.4821** and revelation IoU 0.5256. Root-macro pairwise ranking of predicted vs actual gain is
  94.2 % over 2,073 Roots.
* The official Stage 2 number remains **0.4831** on its 47,128 evaluation windows, below the required 0.50.
* Later branch decisions: pooled label-aware IoU 0.431.

Test suites on this branch (`receipts/stage_tests.log`):

* Layer 2: 13 / 13 pass.
* Stage 1: 7 / 7 pass (online/offline equivalence, batched API, sandbox candidates, sampler, likelihood, selected
  model, legacy context input).
* Stage 2: `test_api` passes (batched == single, lanes 3 == 1, exact domain, no mutation). `test_cache` passes; it
  was run with its hard-coded v15 checkpoint path pointed at the v17_a release.

## 3. Integration issues found and resolved

1. **Stage 2 depends on batch composition for long paths.**
   * **Cause:** v17_a builds its legal voxel map over start ± 19 m. A trajectory whose 11 m candidate crop leaves
     that extent enlarges the map to the union of all crops in the call, or reuses a larger cached map. The map
     extent changes the outputs slightly: ~1e-5 relative on expected gains and rare hard-state flips. As a result,
     one candidate's increment depended on its batch-mates and on the cache.
   * **Scope:** 29 of 2,110 Roots (66 candidates). Stage 2's own tests only used short benchmark paths.
   * **Fix (adapter only, Stage 2 untouched):** Layer 2 defines each candidate's increment as a *cold single-candidate
     call*. In-extent candidates are batched; they share the default map, which is bit-identical to a single call.
     Each out-of-extent candidate runs alone, and its enlarged map is evicted from the cache.
   * **Verification:** `test_long_paths_keep_single_call_semantics`, and 286 / 286 candidates in the full run.
   * **Cost:** one extra legal-map build per out-of-extent candidate. That is rare for short planner Queries.
2. **One CUDA "illegal memory access" in an earlier full run** (`receipts/cuda_fault_fulldev_v1.json`).
   * **Where:** shard 2 faulted on Root `ee60ad4a…` after 211 Roots (about 450 Layer 2 calls in that process). The
     poisoned CUDA context then failed every later Root in the process.
   * **Not reproducible:**
     * The same Root passed 4 / 4 times in fresh processes, overlapped, sequential and with `CUDA_LAUNCH_BLOCKING=1`.
     * The final full run (about 6,000 Layer 2 calls) and the latency and throughput runs (about 13,000 calls) had no
       fault.
   * **Status:** unresolved. It may be a rare race in Stage 2's custom kernels or multi-stream lanes, or a transient
     fault on a shared machine whose NVML reports a driver/library mismatch.
   * **Mitigation:** `layer2_validate_shards.sh` restarts a shard that hits a CUDA fault and retries the Root once.
     Layer 2.5 / Layer 3 workers should do the same: treat a CUDA error as fatal to the process and restart it.
3. **The conda environment exports `CUDA_VISIBLE_DEVICES=1`.** Physical GPU1 runs other sessions' training. All
   Layer 2 jobs pin `0,1` explicitly and use `cuda:0` (physical GPU0). One launch that inherited the default died at
   construction without affecting the other job.
4. **The human-interaction reduction cost ~8 ms of CPU per K=6 decision.** It is now a numba kernel (1.5 ms,
   bit-identical; tested) and runs on the Stage 1 worker thread, overlapped with Stage 2.

## 4. Latency (complete outputs, K = 1/4/6/8)

`receipts/latency_final_v1.json`; command in §7.

* **Workloads:** the original 32 Stage 2 multi-query Roots and their recorded paths (first K). Dense = upper half by
  known cells.
* **Setup:** 3 repetitions in alternating order, RTX 3090 (physical GPU0), 1 Torch thread. Load average 22–25 from
  other sessions (querygen ~20 CPU-bound processes, Stage 2 training on GPU1).
* **What is timed:** the whole `predict` call: validation, Stage 1, Stage 2 with complete hypothetical future
  graphs, candidate reductions and output assembly.

**Steady per-decision cost inside an episode** (`online`: the Stage 1 stream holds the pre-arrival state, so one new
frame is ingested; Stage 2 caches are cold for the new state), Stage 1 overlapped:

| K | all P50 | all P95 | representative P50 / P95 | dense P50 / P95 |
|---|---:|---:|---:|---:|
| 1 | 20.5 ms | 27.9 ms | 20.1 / 22.6 | 21.4 / 28.1 |
| 4 | 30.7 ms | 38.2 ms | 30.7 / 34.1 | 30.6 / 39.5 |
| 6 | **37.4 ms** | **45.0 ms** | 37.5 / 43.1 | 33.7 / 46.4 |
| 8 | 45.4 ms | 52.9 ms | 45.7 / 51.1 | 43.8 / 54.3 |

The same calls run sequentially (no thread): K=6 44.7 / 53.4 ms.

Other modes (all cohorts, overlapped, P50 / P95):

| mode | K=1 | K=4 | K=6 | K=8 |
|---|---:|---:|---:|---:|
| same decision state, new candidates (bank and legal map reused) | 11.7 / 16.1 | 23.5 / 28.3 | 29.7 / 36.1 | 37.0 / 43.4 |
| cold: first decision of a stream (full legal-history replay in Stage 1) | 38.0 / 101.8 | 48.5 / 110.9 | 55.5 / 117.8 | 63.6 / 130.0 |
| Stage 2 alone, cold (same load; released figure: 34.8 / 41.1 at load ~27) | 13.9 / 21.1 | 24.0 / 30.0 | 29.5 / 37.3 | 36.6 / 45.1 |
| Stage 1 alone, online | 11.8 / 13.6 | | | |

Notes:

* The cold P95 comes from long histories (up to 82 frames) replayed once per stream. The dense cold P95 reaches
  157 ms at K=6. It is paid only at the first decision of an episode, or after `reset`.
* Overlapping hides most of Stage 1. Both stages still share one Python interpreter (GIL), so Stage 2 slows from
  29 to 36 ms P50 when Stage 1 runs beside it.
* Construction takes 20 s (NVRTC kernels and CUDA-graph capture). Each process holds 1.2–1.5 GB of GPU memory.

## 5. Throughput and Layer 3 feasibility

`receipts/throughput_final_v1.json`: online mode, K = 6, 60 s per setting, N independent processes on one RTX 3090,
load 26–32 from other sessions.

| processes | decisions / s | per-call P50 / P95 | decisions / day |
|---:|---:|---:|---:|
| 1 | 17.3 | 39.8 / 45 ms | 1.5 M |
| 2 | 31.8 | 45.5 / 55 ms | 2.7 M |
| 4 | 55.2 | 55.1 / 72 ms | 4.8 M |
| 6 | 65.0 | 75.6 / 93 ms | 5.6 M |

* **Layer 2 budget:** in a 2–3 day Layer 3 run, one GPU gives about 11–17 M K=6 decisions, if Layer 2 were the only
  cost. A second GPU (GPU1 is currently used by another session) roughly doubles that, since the processes are
  independent.
* **Real cost includes Layer 2.5:** a full training step also runs the simulator, the Query generator and K
  planner/dynamics rollouts, which these numbers exclude. If those cost as much as Layer 2, the budget halves.
* **Feasibility:** a Layer 3 run needing up to roughly 5–10 M decisions fits in 2–3 days on the current hardware.
  More than that needs more GPUs, or reuse of Layer 2 outputs.
* **Reuse is valid:** Layer 2 is frozen and deterministic per decision state (seeded bank). Off-policy or offline
  Layer 3 training can compute Layer 2 once per (state, candidate set) and reuse it across epochs. Store compact
  summaries, not full future graphs.
* **Scaling beyond 6 processes** is limited by the CPU: each process uses about 1.5 cores (Stage 1, Stage 2 host
  work, Python), and the machine is shared.

## 6. Unresolved and open items

1. **Stage 2 accuracy gate not met:** 0.4831 < 0.50. This is inherited and unchanged. About 56 % of targets lie where
   future visibility crosses never-observed space (IoU ~0.39 there). Stage 2 probabilities are uncalibrated
   (expected gains run about 17 % high).
2. **The one unreproduced CUDA fault** (§3.2). Workers must restart on CUDA errors.
3. **Human-horizon cap:** the human bank covers 3 s. Interaction scores of longer candidates are truncated at
   2.976 s and flagged; about 1 % of recorded Query executions are longer.
4. **No coupling between the stages:** Stage 2 uses its own constant-velocity occluders for tracked humans, not
   Stage 1 worlds. Coupling them would require Stage 2 retraining.
5. **Chained look-ahead from a predicted state is not legal** and not supported (INTERFACE.md §5).
6. **The range packet is the simulator's legal sensor extension;** a real front end must supply equivalent calibrated
   ranges.
7. **Validated inputs only:** recorded executions of recorded Queries were validated, not trajectories from the new
   planner/dynamics stack. Integration tests should be repeated once Layer 2.5 produces rollouts.

## 7. Reproduce

```bash
cd /home/haodong/workspace/projects/eventwm-falcon-l2-integration-20261027
export CUDA_VISIBLE_DEVICES=0,1          # the conda env sets 1; physical GPU0 = cuda:0
python scripts/layer2_run.py tests/layer2/test_layer2.py                                   # 13 integration tests (~1 min)
scripts/layer2_validate_shards.sh fulldev_final 4                                          # full development validation (~65 min)
python scripts/layer2_run.py eventwm_graph.layer2.benchmark latency final_v1 --repetitions 3
python scripts/layer2_run.py eventwm_graph.layer2.benchmark throughput final_v1 --procs 1 2 4 6 --seconds 60 --k 6
```

Inputs (read-only, outside this worktree):

* development snapshot: `eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot`;
* 256 px range packets and v17_a reference records: `eventwm-falcon-stage2-claude-handoff-20261026/output/...`;
* benchmark workloads: `eventwm-falcon-stage2-segment-20261024/.../benchmark_inputs/multi.json.gz` (SHA-256 checked).

Outputs go to `output/layer2/` (gitignored); compact receipts are committed under `docs/layer2/receipts/`.
