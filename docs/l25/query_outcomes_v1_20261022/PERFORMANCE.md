# Measured performance

**INCREMENTAL LATENCY REVISE; FULL CALL LATENCY REVISE.**

Release source/checkpoint stayed unchanged throughout measurement. All 581 calls completed across 83 development roots. The target is ≤10 ms incremental and ≤100 ms full-call P95. The release cold and dense workloads fail. No training or GPU work occurred.

| Cohort/cache | Calls | Incremental P50/P95/P99 ms | Full P50/P95/P99 ms | Candidate/s P50/P95/P99 |
|---|---:|---|---|---|
| representative:warm | 300 | 11.655/22.994/25.098 | 48.958/87.212/119.434 | 107.855/249.772/297.499 |
| representative:dynamic_cold | 120 | 24.280/45.082/57.311 | 66.600/119.241/158.265 | 84.590/143.771/206.302 |
| dense:warm | 60 | 10.217/25.104/25.620 | 90.551/124.883/126.542 | 69.685/122.481/131.508 |
| dense:dynamic_cold | 24 | 33.651/58.165/60.776 | 132.021/155.246/158.142 | 55.687/81.209/81.940 |
| expanded:warm | 415 | 11.050/22.436/24.414 | 48.668/116.481/124.451 | 101.958/264.404/301.219 |
| expanded:dynamic_cold | 166 | 21.120/45.061/53.232 | 66.020/145.167/156.853 | 78.905/175.570/213.276 |

## Workload and accounting

Historical metadata-only selection: first 60 development roots plus 12 densest
and 12 longest-history roots, deduplicated to 83; 59 scenes. Five warm and two
dynamically cold repeats per root. Up to eight existing native candidates are
requested; actual counts are {3: 147, 6: 56, 8: 203, 5: 28, 4: 84, 7: 63}. Static maps have
738–3,783 arrived cells, 68–535 tiled nodes; tracked component maxima 0–6.
Forecast: 3 s, 16 worlds, 64 timestamps, hidden population included. All core
risk/social/time/progress/optical outputs and typed graph export are enabled.
Branch selection considers one informative event, up to two finite successors;
branch-count frequencies are {2: 438, 1: 45, 0: 98}. Zero branches means no
informative event, not an unreported performance shortcut.

Full call starts before real `observe`, then includes context/readout, bank,
candidate compilation and graph export. It excludes upstream reference
generation, prefix reconstruction, imports/JIT and the untrained score consumer.
Incremental subtracts the measured forecast time and actual update time, and
includes context generation. Route/tile/topology caches are versioned static
work; dynamic cold retains them but clears context, bank, numeric and branch
caches plus inherited route/footprint/kernel caches. Warm bank/numeric hits occur
for repeat zero matching the primed seed (20% of warm calls); dynamic cold banks
are fresh. Actual observation updates are never cached. The fixed-belief cache
probe is separate from these complete calls. Full cumulative cache counters
(including priming) are in `artifacts/workload_summary.json`.

Across 3318 repeated candidate evaluations,
21 (0.63%) are static-unsupported and
56 (1.69%) have censored nominal tails. No root is
dropped, and 0 branch requests are budget-truncated in this gate: the
optional-branch soft budget is deliberately 1e6 ms, so speed does not come from
deadline-driven output omission. Unsupported candidate masks remain explicit.

Peak process RSS 811.2 MiB includes Torch/Numba and the frozen
development metadata cache. Model load after imports: 15.643 ms. Process
and first-call startup are separate: `artifacts/startup_process_time.txt` records
a new demo process with an initially empty isolated Numba cache, including
imports, legal prefix/JIT, first inference and plotting. Its per-stage timings
are in `artifacts/demo.json`; they are not steady-state latency.

## Replication and failures

The pre-hardening run measured representative full P95 75.228/89.792 ms and
dense 82.017/99.945 ms. After immutable nested metadata was added, the release
run above was slower. Both are retained; no fastest-run selection or causal
claim about the metadata patch is made. Runtime varies under the independent
collector and other host activity. The release figures govern acceptance.
The original L2 report's 87.446/88.275 ms and the new dense-enriched 11-root
gate's 117.358/131.940 ms remain separate workloads. The gate used native
`initialize`; this complete benchmark uses the historical seeded legal-prefix
conversion. These are not matched speedup estimates.

Exploratory profiling found Python static-edge construction, route lookup and
fragmented height regions. Vectorized static export and versioned route caches
reduced overhead. Early runs also repaired a public-goal fixture; they are not
a controlled optimization ablation. The remaining cold swept-support checks,
actual observation update, per-query reductions and coarse successor construction
still cost time. The frozen L2 was not retrained or restructured.

## Bounded fallback and sensitivity

`artifacts/ablation.json` retains 192 calls on two fixed legal roots, four repeats,
1/4/8 requested candidates (actual counts retained), 8/16/32/64 worlds and zero
or one event expansion. Those timings include a fresh bank and all candidate
core features, but **exclude** actual update/context creation, so they are not
full-call passes. Increasing worlds is not hidden behind a constant workload.

The soft-budget-zero fallback returns the same supported nominal risk values,
retains all candidates/core features and puts optional unexpanded observation
mass in explicit residuals. The ablation checks equality. It cannot preempt L2
or guarantee a full-call deadline. Reducing worlds is an explicit caller choice
with reduced finite-bank support, not a default silent dense-scene speed fix.
Dense/unbounded deployment therefore remains REVISE.

## Startup, cache probes and selected sensitivity cells

New process with an initially empty isolated Numba cache: demo process wall time
9.23 s (includes imports, prefix reconstruction, inference and plotting); model
load after imports 18.558 ms; first legal-prefix reconstruction/JIT
1559.125 ms; first complete inference
3430.678 ms. The complete `/usr/bin/time -v` ledger is
committed. This startup is substantially slower than loaded-process calls.

Repeated fixed-belief fully cached candidate evaluation, separate from full
calls: P50/P95/P99 6.631/12.934/15.068 ms, 83 probes.
These results do not establish the 10 ms incremental gate.

| Root class | Actual queries | Worlds | Requested event expansions | Actual branch counts over 4 calls | Evaluation P95 ms | Incremental P95 ms |
|---|---:|---:|---:|---|---:|---:|
| representative | 5 | 16 | 0 | {0: 4} | 16.995 | 8.406 |
| representative | 5 | 16 | 1 | {2: 4} | 19.630 | 11.182 |
| representative | 5 | 64 | 0 | {0: 4} | 43.000 | 19.516 |
| representative | 5 | 64 | 1 | {2: 4} | 52.589 | 29.165 |
| dense | 8 | 16 | 0 | {0: 4} | 50.432 | 25.285 |
| dense | 8 | 16 | 1 | {0: 4} | 46.927 | 25.228 |
| dense | 8 | 64 | 0 | {0: 4} | 115.527 | 58.958 |
| dense | 8 | 64 | 1 | {0: 4} | 70.172 | 39.484 |

These are small sequential sensitivity cells under host variability, not causal branch-cost estimates. Zero actual branches reflects the zero-information guard. They cannot establish a full-call speed pass.
