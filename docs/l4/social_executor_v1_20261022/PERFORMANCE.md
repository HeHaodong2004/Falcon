# Control performance

**Decision: REVISE.** Warm open-space fixtures meet the proposed 10 ms P95 /
20 ms P99 targets. The complete native public panel does not. The mesh fallback
is a measured bottleneck and cannot be excluded from an RL-readiness assessment.

## Configuration and hardware

Intel Core i9-10980XE at 3.00 GHz, 36 logical CPUs, Linux 6.8.0-138-generic;
two RTX 3090 cards are installed. L4, frozen L2 and Falcon policy inference run
on CPU with one numerical thread. B3 alone uses native GPU depth rendering.
Research runs use nice 19 and idle I/O priority. The 28-worker collector remained
active, so these are shared-machine measurements, not an isolated hardware lab.

The default beam is 16 wide and 32 native actions deep: 1.536 nominal seconds,
screened using the measured 56 ms maximum action duration (1.792 s). There are
36 retained turn/advance/rejoin templates, plus 19 bounded mesh-fallback
templates when needed. Public B1/B2 candidate-count P95 is 1,873 / 1,868,
P99 2,249 / 2,242, maximum 2,574. Every proposed motion is collision checked;
the enclosing-region shortcut proves support over all enclosing cells rather
than skipping checks. No model was trained for speed.

## Warm core fixtures

Each row is 120 repeated control calls at a fixed legal fixture state, including
candidate generation, geometry checks, time-dependent human checks and scoring.
Forecasts are unavailable in this isolated fallback profile. These rows are
not pooled with native navigation or interpreted as evidence of completion.

| Fixture | Humans | P50 ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|
| Open, no human | 0 | 0.186 | 0.220 | 0.240 | 0.251 |
| Typical | 2 | 3.644 | 3.798 | 3.990 | 4.163 |
| Dense | 6 | 6.091 | 6.266 | 6.629 | 7.041 |

Isolated Python-to-kernel P95 calls take 0.0091 ms for generating the three
native actions, 0.0043 ms for a swept static check, 0.0036 ms for dense human
checking, and 0.0044 ms for candidate scoring. These microbenchmarks are not
an additive decomposition of a complete beam search.

Fresh-cache JIT startup, measured with a new `NUMBA_CACHE_DIR`, costs 1.523 s
for the first no-human call and 5.732 s for the first call reaching the human
search kernels. These are startup observations, not normal control P95s.
The warm fixture process peaks at 142.9 MiB RSS. The observed native navigation
process high-water mark is 1,428,628 KiB, including simulation, maps, mesh/FK,
L2 and trace recording; it is not L4-only memory. A separate B3 memory high-water
mark was not collected.

## Native control distributions

Fresh L2 inference and belief updates are excluded from these control rows and
reported separately below. First calls and mesh fallback are included unless
the row explicitly narrows its scope. Handoff decisions are also retained.

| Workload | N control calls | P50 ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|
| Eight scenarios B1 | 459 | 1.701 | 18.563 | 56.507 | 80.828 |
| Eight scenarios B2 | 453 | 1.806 | 17.881 | 55.368 | 82.845 |
| Public panel B1, all calls | 1,752 | 12.630 | 267.933 | 385.688 | 607.750 |
| Public panel B2, all calls | 1,771 | 15.323 | 350.099 | 453.814 | 614.846 |
| Public B1, grid path only | 643 | 4.934 | 11.082 | 13.696 | 15.897 |
| Public B2, grid path only | 671 | 5.216 | 15.810 | 17.538 | 20.087 |
| Public B1, mesh fallback | 1,101 | 36.177 | 290.346 | 411.929 | 607.750 |
| Public B2, mesh fallback | 1,092 | 38.279 | 409.901 | 464.178 | 614.846 |

Removing each episode's first control call does not fix the tails: public B1
P95/P99 remains 275.164/385.765 ms; B2 remains 352.866/453.814 ms. The grid path
also exceeds 10 ms P95 on some public workloads, despite passing fixed fixtures.

An offline replay profiles the worst retained B1 actual state (HM3D
5cdEh9F2hJL/public_0, observation 61). Five warmed mesh-proposal calls take
387–392 ms each and check 469 candidate transitions each. The compiled
`mesh_free` triangle/link checks consume 1.668 of 1.948 profiled seconds (85.6%).
Native ego-motion prediction contributes 0.137 s (7.0%). This establishes the
bottleneck; simply reducing collision-check coverage would invalidate the
safety assumptions. The profile uses legal actual ego articulation and human
observations, with neutral side preference; it is an isolated replay rather
than a bit-for-bit repeat of the whole recorded 607.750 ms action call.

The exact free-region shortcut repaired the earlier dense-fixture P95 failure
(10.32 ms before optimization). Large-mesh acceleration remains unresolved.
Future work needs conservative local triangle reduction/certificates and
revalidated plan reuse, alongside the progress repairs. No approximate rounded
pose cache or skipped safety check was introduced to improve these numbers.

## L2 and environment cost

| Public B2 operation | N | P50 ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|
| Shared forecast refresh | 416 | 8.770 | 31.920 | 34.334 | 36.198 |
| Actual belief initialize/update | 1,772 | 27.202 | 125.682 | 139.495 | 361.253 |
| Native step plus legal acquisition | 1,763 | 54.749 | 101.380 | 128.724 | 165.849 |

The held bank has 16 worlds and 1.92 s support and is shared by all candidates.
Refreshes occur for visibility changes, material observed mismatch or 0.24 s
expiry; they are not per beam candidate. The first refresh in each of 11
supported public episodes has P50/P95 8.565/16.830 ms in an already loaded
process. Actual belief updates occur on every actual packet and are not made
free by caching the forecast.

In the initial baseline smoke, model loading took 15.3 ms, initialization from
the 131-action history took 3.408 s, and the first forecast took 4.984 s. The
separate visible-human encounter's loaded forecast took 35.15 ms. These differ
in state, history and cache conditions and must not be pooled into a claimed
L2 latency improvement over the source report's representative P95 results.

Public B1 native-step/acquisition P50/P95/P99 is 56.378/99.936/125.525 ms.
Native-step timing excludes separate offline contact-label analysis, reference
preparation and result serialization. Episode wall time below includes those
costs, environment startup and per-case preparation, but excludes the runner's
one-time shared model/map loading outside each episode timer.

| Public arm | Timed wall seconds, 12 cases | Native steps/s | Attempted episodes/hour |
|---|---:|---:|---:|
| B0 | 154.472 | 7.108 | 279.7 |
| B1 | 262.015 | 6.694 | 164.9 |
| B2 | 360.840 | 4.914 | 119.7 |
| B3 | 558.581 | 4.270 | 77.3 |

These are serial measured throughput estimates. Failed early handoffs inflate
attempted episodes/hour; they are not successful training episodes/hour. Low
navigation SR and the control tails jointly preclude an RL-feasibility claim.

Reproduce with `scripts/l4_profile.py` and `scripts/l4_profile_mesh.py`; complete
timings, source receipts, raw profiler output and machine evidence are in
`artifacts/` and the retained output root.
