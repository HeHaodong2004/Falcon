# World Model Sandbox Core: technical report

Branch `research/wm-sandbox-20261028`. Base: unified Layer 2 `research/l2-integration-20261027` @ `da161bd7`, merged
with the querygen / Oracle track `research/querygen-oracle-20261026`; neither stage nor the querygen package was
modified. Interface: [INTERFACE.md](INTERFACE.md).
All timings: one RTX 3090 (GPU0), a 36-core host shared with this project's Oracle runs (load 15–25 during the runs;
each summary records its load), branch-simulator episodes unless marked "real".

## 1. Summary

* **What it is.** `eventwm_graph/sandbox/` evaluates K candidate Queries at one legal decision state:
  * one shared Stage 1 human bank (40 worlds × 3 s);
  * each candidate imagined by its own planner, executor and dynamics family, on the frozen legal map, against one
    nominal future of that bank;
  * one batched Stage 2 call for the K trajectories (map increments and complete hypothetical future graphs);
  * collision / social risk of every trajectory scored against all 40 worlds;
  * structured, JSON-ready, hypothetical outputs for the future Graph Compiler.
* **Correctness.** 18 tests pass on real Falcon episodes. They cover:
  * formats, clocks and coordinates; legality, including rejection of hypothetical inputs; the real legal state
    untouched;
  * batched == individual; one shared bank; failure isolation;
  * every planner × dynamics family;
  * exactness of all accelerations (bit-identical rollouts);
  * the fast range sensor against the native one.
* **Speed** (warm, 8 episodes, branch environment):
  * K=10: sandbox P50 432 ms, P95 1,375 ms. Imagination 355 ms, Stage 2 52 ms, Stage 1 20 ms, risk 7 ms.
  * K=6: P50 286 ms, P95 982 ms.
  * DWA under inertia, K=10: P50 874 ms.
* **Throughput, one GPU, 6 processes:** ~10 K=10 or ~20 K=6 sandbox decisions/s. That is 1.7-2.6 M (K=10) or
  3.5-5.3 M (K=6) calls in 2-3 days, which meets the target.
* **The full RL loop is environment-bound:** ~2.0 decisions/s with 6 workers on one GPU. The sandbox is ~21% of a
  decision; the rest is the environment's sensing (legal sensor ≈ 35-40%, range frames ≈ 25%) and stepping.
* **Optimisations, all measured:**
  * exact per-decision caches plus exact numba helpers: imagination 2.0× faster, bit-identical;
  * fast range sensor: 1.2 ms GPU kernel per 256 × 256 frame, 15.7 ms per frame P50 including native recasts near
    people, against 840 ms native (54×); identical hit masks on 100 frames, ≤ 8.8e-5 m;
  * compact static updates plus GC freezing: no growth of live objects with history, no 150–215 ms collection
    pauses;
  * exact branch-simulator driver: simulator steps 6× cheaper than habitat.Env.
* **Known limits.** Stage 2's own accuracy gate is unmet (0.4831 < 0.50; unchanged). Imagination has no new map
  evidence and no navmesh sliding (fidelity: §4). The fast range sensor is ulp-level, not bit-identical.
  Environment-side sensing dominates the RL budget (§6).

## 2. What was built

| file | role |
|---|---|
| `context.py` | `LegalStream`: legal rows -> Layer 2 frames (dataset format; `CompactUpdates`), causal partial graph (`PartialGraph.update`, identical to the development graphs), append-only range-packet views; clock / pose / protocol / order checks |
| `imagine.py` | `NominalHumans` (the tracked people's nominal future from the shared bank), `ImaginedLegal` (the executor's legal view on the frozen map), `advance` (family kinematics), `imagine` -> `ImaginedRollout` |
| `core.py` | `WorldModelSandbox.evaluate`, `Candidate`, `SandboxConfig`, `CandidateOutcome`, `SandboxResult` (+ JSON `record()`) |
| `accel.py` | exact per-decision memoisation on the frozen map + exact numba helpers (installed only for imagination) |
| `fastrange.py` | `FastRangeSensor`: GPU Bullet-emulating stage raycast + exact native recast of human / ambiguous rays |
| `episode.py` | `RealEpisode` (unchanged habitat.Env), `BranchEpisode` (exact branch simulator), frozen generator configurations, GC helpers |
| `benchmark.py`, `validate.py`, `report.py` | latency / throughput benchmarks, fidelity and range validation, report tables |
| `tests/sandbox/test_sandbox.py` | 18 tests |

Per decision: Stage 1 once (`l2.predict_humans`, cached for the state object) -> nominal future of the perceivable
tracked people (present-weighted bank mean; tracks Stage 1 does not model fall back to constant velocity) -> every
candidate's `QueryExecution` (deep copy for continued queries) stepped on `ImaginedLegal` for its horizon (16 steps per
movement decision by default) -> one `Layer2WorldModel.predict` on the K trajectories (Stage 2 batched; the bank is
reused; reductions vs all 40 worlds) -> outcomes.  Candidate failures (executor cannot start, planner exception,
non-finite pose) are recorded per candidate with a stationary trajectory; contract violations raise.

## 3. Correctness

Tests (`tests/sandbox/test_sandbox.py`, real episode hm3d/val/hkr2MGpHD6B with up to 6 tracked people, genuine Layer 2):

| group | tests |
|---|---|
| formats & clocks | frames have exactly the dataset keys and reference update dicts; the stream's graph equals plain `PartialGraph.update` over the same frames (JSON-identical) and the executor's GridMap; range packet aligned; candidate row 0 = start pose, times t + k * 0.048 s, Stage 2 / risk clocks |
| legality | out-of-order / predicted rows, misaligned / wrong-protocol / out-of-range range frames rejected; hypothetical future graphs rejected as decision-state inputs; the real legal map, pose, tracks and running execution unchanged bit for bit; the frozen alias shares arrays but not the object |
| sandbox semantics | batched evaluation == K single evaluations (rollouts and maps bit-identical, risk 1e-12); one Stage 1 bank per decision; the imagined planner perceives every perceivable tracked person at the bank's present-weighted mean; maps off keeps risk; summary without people; JSON record |
| heterogeneity | rollouts for ORCA / SFM / DWA x native / slowturn / inertia / slowturn_inertia / heavy_brake / slowturn_heavy respect the family (no dynamics violation, speed <= v_max, turn <= turn_rad, translate-or-turn, straight ahead) |
| exactness of accelerations | memoised == plain rollouts bit for bit across 6 planner x dynamics combinations (>100 rollouts); numba helpers == originals on 4,000 random poses |
| failure handling | empty candidate set, STOP, an executor that cannot start (recorded), invalid candidates and mismatched legal state (raise) |
| fast range sensor | identical mask, <= 1e-4 m vs the native frame (validation: 100 frames of 4 episodes, see section 5) |

Exactness evidence beyond the tests: 140 rollouts of 14 decision contexts (4 episodes) bit-identical with and without
the caches; 84,000 body-axis and 42,000 blocked-sector checks identical; prefetching Stage 2 changes no output
(10 decisions).

## 4. Fidelity of the imagined executions (privileged validation only)

`validate.py fidelity` covers 4 episodes (Nfvx, hkr2, JeFG MP3D, an unseen3 HM3D), 6 decision states each, all 240
K=10 candidates. Each candidate is executed for real from the same state in querygen's exact branch simulator; those
true poses and outcomes are labels only. It is also imagined three ways: with the Stage 1 nominal future (what the
sandbox uses), with constant-velocity people, and with the true future of the tracked people.

| imagined with | mean deviation P50 / mean / P90 (m) | end-point deviation mean / P90 (m) | mean yaw deviation (rad) | same length |
|---|---|---|---|---|
| Stage 1 nominal (default) | 0.00 / 0.099 / 0.30 | 0.26 / 0.86 | 0.044 | 95.8% |
| constant velocity | 0.00 / 0.091 / 0.29 | 0.26 / 0.86 | 0.035 | 95.8% |
| true future (privileged) | 0.00 / 0.083 / 0.26 | 0.23 / 0.65 | 0.037 | 95.4% |

* Most imagined executions coincide with the real one (median deviation 0). The same executor with exact family
  kinematics repeats the real decisions.
* Deviations come from three sources:
  * new map evidence during the real interval, which makes the real executor re-plan;
  * navmesh sliding and contacts near walls (largest in hkr2's confined spawn: 0.16-0.26 m);
  * the people's motion.
* Human prediction explains only ~0.016 m of the 0.099 m mean deviation (Stage 1 nominal vs the true future).
  Constant velocity is as good as the Stage 1 mean for the planner's perception, because the real planner itself
  reacts to constant-velocity extrapolations of its observed tracks.
* Decision-relevant quantities agree:
  * imagined vs real goal progress: Pearson r 0.90;
  * within-decision ranking: Kendall τ 0.79;
  * the best-progress candidate is also the real best (within 0.05 m) in 67% of decisions.
* Risk is informative. The 6 candidates whose real execution touched a person (human collision or body overlap)
  had a predicted collision probability of 0.55 on average, against 0.30 for the others.

**DWA with inertia dynamics** (velocity commands, acceleration limits; 3 episodes, 10 decisions, 94 candidates):

| imagined with | mean deviation P50 / mean / P90 (m) | end-point deviation mean / P90 (m) | same length |
|---|---|---|---|
| Stage 1 nominal | 0.00 / 0.014 / 0.028 | 0.075 / 0.21 | 91.5% |
| constant velocity / true future | 0.00 / 0.014 / 0.028 | 0.074 / 0.21 | 91.5% |

* Progress correlation is 0.94 and the within-decision Kendall τ is 0.75.
* In this small sample the predicted collision probability did not separate the 8 candidates whose real execution
  touched a person (0.11 vs 0.11). Risk discrimination for momentum dynamics needs a larger study.


## 5. Performance

Warm decisions (not the first of an episode), 8 episodes with people (3 HM3D val, 2 HM3D train, 3 MP3D train), up to
20 decision intervals each; `k10_real` uses the unchanged habitat.Env on 3 of them.  K = generator candidates
(continue + Queries); horizon 16 steps per movement decision.

| run | decisions | sandbox P50 / mean / P95 (ms) | Stage 1 P50 | imagination P50 / P95 | Stage 2 P50 / P95 | risk P50 | first-decision sandbox P50 | end-to-end decision P50 / mean / P95 (ms) |
|---|---|---|---|---|---|---|---|---|
| k10_branch | 107 | 432 / 538 / 1375 | 20.0 | 355 / 1293 | 51.8 / 71.1 | 7.4 | 210 | 2121 / 2751 / 7066 |
| k6_branch | 107 | 286 / 370 / 982 | 19.7 | 217 / 899 | 38.9 / 55.2 | 6.8 | 128 | 1920 / 2552 / 6777 |
| k10_branch_dwa_inertia | 124 | 874 / 1099 / 2329 | 20.0 | 799 / 2260 | 47.6 / 60.1 | 7.2 | 847 | 2133 / 2490 / 5755 |
| k10_real | 41 | 433 / 552 / 1362 | 41.7 | 329 / 1270 | 51.4 / 66.9 | 6.7 | 217 | 2259 / 2782 / 4431 |

Sandbox = Stage 1 + imagination + Layer 2 (Stage 2 + risk) + assembly.  The first decision of an episode (cold Stage 1
stream, small history) is cheaper; process start costs the Layer 2 build (16-20 s) and, with cold JIT caches, ~8 s on
the first call (1-2 s with the on-disk numba caches warm).

Where a decision interval goes (mean ms per 16-step interval; `k10_branch` used querygen's reference legal observer,
and the exact fast sensor (`k10_branch_fs`, identical rows) lowers the legal-sensor column to 920 ms and end-to-end
P50 to 2.04 s):

| run | simulator | legal sensor | range frames | stream update | decision state | generator | sandbox | sandbox share |
|---|---|---|---|---|---|---|---|---|
| k10_branch | 189 | 1039 | 738 | 36 | 17 | 38 | 538 | 21% |
| k6_branch | 180 | 1025 | 731 | 36 | 20 | 37 | 370 | 15% |
| k10_branch_dwa_inertia | 126 | 693 | 286 | 38 | 12 | 28 | 1099 | 48% |
| k10_real | 1034 | 575 | 376 | 74 | 7 | 33 | 552 | 21% |

(mean ms per decision interval of 16 steps; warm decisions)

Stage 2 per K=10 call (P50 ms): graph prepare 5.6, legal voxel map 13.9, query features 5.8, network 19.3 (one batch
for all K), decode / materialize 5.6 -- total 51 (K=6: 38).  With history length:

| frames in history | decisions | Stage 2 P50 (ms) | decision state P50 (ms) | sandbox P50 (ms) |
|---|---|---|---|---|
| 0-99 | 40 | 43.6 | 3.9 | 448 |
| 100-199 | 32 | 52.7 | 9.4 | 405 |
| 200-299 | 21 | 63.4 | 12.2 | 518 |
| 300-399 | 6 | 69.6 | 12.4 | 344 |

Imagination costs ~2.4 ms per imagined step (K=10: 151 imagined steps P50).  The executor's own numba searches dominate
it (space-time plans 36%, body lattice 22%, 4 episodes profiled); with DWA under inertia the executor replans its body
lattice every step (stateful families never reuse a plan) and DWA scores many command sequences, ~2.3x ORCA.

### Optimisations (measured)

| change | effect | exactness |
|---|---|---|
| per-decision memoisation on the frozen map (map variants, views, fields, routes, lattice plans) + numba body-axis / blocked-sector helpers | imagination 2.0x faster (11.8 s -> 5.9 s on 14 contexts) | bit-identical rollouts (tests) |
| fast range sensor (GPU Bullet emulation, native recast near humans / ambiguous) | 100 frames, 4 episodes (2 HM3D, 2 MP3D): frame P50 15.7 ms / P95 53 ms vs native 840 / 1,149 ms (54x); kernel 1.2 ms per 65,536 rays (RTX 3090); 1.7% of rays recast natively (~12 us each; crowds up to ~140 ms per frame) | identical hit / no-hit masks on all 6.55 M pixels; max 8.7e-5 m; 47% bit-equal; all 19,717 human hits recast natively |
| `CompactUpdates` + `gc.freeze` per episode | live objects stay ~314 k instead of growing to 754 k; no 150-215 ms full-collection pauses | identical frame content |
| branch-simulator driver | simulator 189 ms vs 1,034 ms per 16 steps (habitat.Env) | the exact Env.step mirror (querygen) |
| Stage 2 prefetch on a worker thread (`prefetch_maps`) | no gain (337 vs 322 ms): its candidate-independent part is only 3-9 ms | outputs identical; off by default |
| shortened native recast for human rays | rejected: 1.6x cheaper but human-hit distances change up to 2.4 cm | -- |

## 6. Throughput and RL budget

Multi-process, one GPU (GPU0), branch environment, K=10 ORCA, 8 episodes cycled per worker, 180-300 s steady
window; the host also ran this project's niced Oracle searches (load 20-28).

| mode | workers | decisions/s | per worker | sandbox s / decision | environment s / decision (16 steps) | sandbox share |
|---|---|---|---|---|---|---|
| full loop (env + sensing + generator + sandbox) | 1 | 0.27 | 0.27 | 0.59 | 2.47 | 19% |
| full loop | 4 | 0.73 | 0.18 | 0.78 | 3.81 | 17% |
| full loop | 7 (8 launched, 1 GPU OOM) | 1.48 | 0.21 | 0.74 | 3.21 | 18% |
| full loop, exact fast legal sensor (`QG_FAST_SENSE=1`; the runs above used the reference observer -- identical rows) | 6 | 2.00 | 0.33 | 0.54 | 1.98 | 21% |
| sandbox only (re-evaluating 10 stored decision states per worker) | 1 / 4 / 6 | 0.87 / 6.8 / 9.9 | 0.87 / 1.7 / 1.66 | ~0.6 | -- | -- |
| sandbox only, K=6 | 6 | 20.5 | 3.4 | ~0.29 | -- | -- |

* **GPU memory, not compute, caps the processes per GPU.**
  * Each sandbox process holds ~2.4-2.6 GB: Layer 2 with Stage 2's captured CUDA graphs for 3 lanes × 8 input sizes,
    plus the range buffers.
  * The 8th process OOMed beside the other GPU0 users (~2.5 GB). Plan at most 6-7 processes per 24 GB GPU.
  * GPU compute per decision is ~70 ms: Stage 2 ~50 ms plus 16 range kernels × 1.3 ms.
* **The sandbox alone sustains ~10 K=10 or ~20 K=6 decisions/s on one GPU** (6 processes). That is ~0.86 M or ~1.8 M
  decisions per day, i.e. 1.7-2.6 M (K=10) or 3.5-5.3 M (K=6) sandbox calls in 2-3 days. It meets the "millions of
  calls in 2-3 days" target. (The single-worker sandbox-only figure is pessimistic: re-evaluating stored states from one episode
  rebuilds Stage 1's incremental stream each call, and that worker held the heaviest episode.)
* **The full RL inner loop is environment-bound.**
  * One worker spends ~2.5-3.8 CPU-s per decision on the 16 environment steps, mostly the legal sensor and the range
    frames, against ~0.6-0.8 s in the sandbox.
  * Six workers on one GPU give ~2.0 decisions/s with the exact fast legal sensor: ~0.17 M per day, 1 M in ~6 days.
    With the reference observer it was ~1.5 decisions/s on 7 workers.
* **What it takes to reach millions in 2-3 days end to end:**
  * cheaper environment sensing. In the exact fast legal sensor (19-41 ms per step on 3 measured episodes) the
    native support-probe rays are 25-48% of the time. Emulating them on the GPU (exact booleans, native fallback near
    the 0.015 m cutoff or near people) would bound the gain at 1.3-1.9× on the sensor, ~10-20% end to end. The rest
    is the sensor's Python/numpy bookkeeping (frustum tests, probe-index lookups, tracker, wall band), which would
    need a vectorised re-implementation of the observer.
  * the second GPU.
  * a smaller Stage 2 graph footprint, to fit more processes per GPU. A first measurement under concurrent GPU
    load: `lanes=1` reserves 0.86 GB vs 1.30 GB of torch memory per process, at Stage 2 P50 51 vs 45 ms. That is
    about one more process per 24 GB GPU.


## 7. Limits and next steps

* **Environment-side sensing dominates RL cost.**
  * querygen's exact fast legal sensor costs 19-41 ms per step: native certificate rays 25-48%, Python
    bookkeeping the rest.
  * The range frame costs 4.5 ms per step plus ~12 us per native recast near people.
  * Together they are ~65% of a decision.
  * The same GPU emulation of Bullet's triangle callback (with native recast of ambiguous cases) can take over the
    certificate rays. A vectorised observer would remove the bookkeeping. Both are environment-side, outside the
    sandbox.
* **Imagination fidelity.**
  * Imagination runs on the frozen legal map. The real executor re-plans with each new observation; imagination
    cannot.
  * There is no navmesh sliding: nominal kinematics only.
  * One nominal human future drives the planner (Stage 1 mean); the risk is still scored against all 40 worlds.
  * Deviations are measured in section 4.
* **Imagination cost** is the executor's own planning (numba space-time and body-lattice searches, DWA sequence
  scoring). It runs single-threaded and holds the GIL. Releasing the GIL in those searches (`nogil` twins) would
  allow candidate-parallel imagination for latency, not for throughput.
* **Stage 2 and Stage 1 are used as released.** Stage 2's own gate is still unmet (0.4831 < 0.50). Its
  probabilities are uncalibrated (expected gains ~17% high).
* **Range values are ulp-level, not bit-identical.** Bullet's internal ray transform into the stage frame could not
  be reproduced bit for bit. 47% of pixels are bit-equal and the rest are within 8.7e-5 m; the hit pattern is
  identical.


## Reproduction

```bash
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
$PY scripts/sandbox_run.py tests/sandbox/test_sandbox.py                       # tests (~6 min)
$PY scripts/sandbox_run.py eventwm_graph.sandbox.benchmark episodes --out output/bench/k10_branch --config k10 \
    --episodes dev:0,dev:2,dev:7,unseen:31,unseen:32,unseen2:47,unseen3:5,unseen4:4 --decisions 20 --env branch
$PY scripts/sandbox_run.py eventwm_graph.sandbox.benchmark throughput --out output/bench/tp_full_k10_w6 --workers 6 \
    --episodes dev:0,dev:2,dev:7,unseen:31,unseen:32,unseen2:47,unseen3:5,unseen4:4 --seconds 300 --env branch --work full
$PY scripts/sandbox_run.py eventwm_graph.sandbox.validate ranges --out output/validate/ranges \
    --episodes dev:0,dev:7,unseen:31,unseen2:47 --steps 48 --every 2
$PY scripts/sandbox_run.py eventwm_graph.sandbox.validate fidelity --out output/validate/fidelity \
    --episodes dev:0,dev:7,unseen:31,unseen3:5 --decisions 6
$PY scripts/sandbox_run.py eventwm_graph.sandbox.report output/bench output/validate     # the tables of this report
```
