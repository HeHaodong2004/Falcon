# Progress-first L4: technical report

**Decision: use B1 as a provisional executor for L3 integration and initial training.** It restores progress, preserves all public12 nominal successes, and recovers one public failure. Broad navigation improvement is not established: the independent panel exchanges one success for one failure. Ordinary runtime is fast; reactive-attempt tails still miss the preferred latency targets. B2 adds no observed execution benefit.

## Implementation and integrity

The controller is **nominal path tracking → optional reactive avoidance → native action**. It uses actual poses, ordered waypoint progression and narrow local rejoin gates. Six fixed lateral/pass maneuvers provide bounded commitment; two small static nudges handle supported boundaries. Failed avoidance executes nominal. Conservative human-envelope overlap is a risk score rather than an absolute execution veto. Native center checks can support a maneuver without claiming full-body clearance. There is no online beam or mesh/FK search, query reselection, graph rerouting, BLOCKED handoff, local lifetime watchdog, or executor STOP. Native contact, termination and truncation remain authoritative.

The immutable compiler/API remains compatible; [the updated contract](EXECUTION_CONTRACT.md) specifies the v2 runtime semantics. The [historical diagnosis](DIAGNOSIS.md) ties rejection, oscillation, failed rejoining and mesh cost to retained actual traces. Old BLOCKED, budget and clock tests were revised explicitly: search failure now preserves nominal; L3 owns the schedule; actual timing/sliding/contact events are recorded rather than creating artificial episode ends.

B0 retains the original nominal schedule. All **20 B0 native traces** across public12 and frozen scenarios exactly reproduce the previous release’s actions, poses and timestamps. The protocol, episode membership, seeds, geometry, human controllers, native actions, metrics and thresholds are unchanged. All failures remain included. There are **27 passing regression tests and 108/108 passing native trace audits**. All four run receipts report unchanged controller sources and matched query semantics. The retained static-contact witness still passes its original geometry regression. Frozen L2 weights, L2.5 and L3 are untouched; the collector remains active with 28 workers.

## Native results

HC/SC count episodes with actual native human collisions / static contacts. Short-query SR, SPL and STL still use the original episode goal; QCR is separate. All action totals below include the driver’s final STOP when present.

| Panel | Arm | Native SR | QCR | HC | SC | SPL | STL | PSC | Actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Public12 | B0 | 6/12 | 6/12 | 5 | 8 | 0.5000 | 0.5000 | 0.9730 | 1098 |
| Public12 | B1 | 7/12 | 7/12 | 4 | 9 | 0.5798 | 0.5833 | 0.9784 | 1316 |
| Public12 | B2 | 7/12 | 7/12 | 4 | 9 | 0.5798 | 0.5833 | 0.9784 | 1316 |
| Frozen scenarios | B0 | 3/8 | 3/8 | 5 | 1 | 0.3739 | 0.3750 | 0.8366 | 200 |
| Frozen scenarios | B1 | 8/8 | 8/8 | 0 | 2 | 0.9437 | 1.0000 | 0.8836 | 431 |
| Frozen scenarios | B2 | 8/8 | 8/8 | 0 | 2 | 0.9437 | 1.0000 | 0.8836 | 431 |
| Short queries | B0 | 1/8 | 4/8 | 5 | 1 | 0.1250 | 0.1250 | 0.8012 | 166 |
| Short queries | B1 | 2/8 | 6/8 | 2 | 2 | 0.2272 | 0.2500 | 0.8678 | 394 |
| Short queries | B2 | 2/8 | 6/8 | 2 | 2 | 0.2272 | 0.2500 | 0.8678 | 394 |
| Additional scenes | B0 | 1/8 | 1/8 | 5 | 4 | 0.1250 | 0.1250 | 0.8675 | 344 |
| Additional scenes | B1 | 1/8 | 1/8 | 5 | 5 | 0.0997 | 0.1250 | 0.9418 | 576 |
| Additional scenes | B2 | 1/8 | 1/8 | 5 | 5 | 0.0997 | 0.1250 | 0.9418 | 576 |

The historical conservative public B1/B2 were **1/12 SR, 2 HC, 0 SC**, with eight BLOCKED queries. Historical B0 was **6/12 SR, 5 HC, 8 SC**, reproduced exactly here. Compared with historical B1, six failed public episodes now succeed and its one success is retained. Static contacts are preserved as real outcomes; improved completion does not imply collision-free execution. The historically BLOCKED native scenario now reaches its unchanged goal.

## Paired completion and failure analysis

| Panel / completion measure | Success→success | Success→failure | Failure→success | Failure→failure | Recovered − lost |
|---|---:|---:|---:|---:|---:|
| Public navigation | 6 | 0 | 1 | 5 | 1 |
| Frozen navigation | 3 | 0 | 5 | 0 | 5 |
| Short-query completion | 3 | 1 | 3 | 1 | 2 |
| Additional navigation | 0 | 1 | 1 | 6 | 0 |

Public12 gains one success with no lost nominal success. Human-collision episodes decrease **20% (5→4)**, below the desired approximately 50% reduction. The panel is a small development diagnostic, not held-out statistical evidence.

| Public episode | B0→B1 success | HC B0→B1 | Actions B0→B1 | Failed B1 outcome |
|---|---:|---:|---:|---|
| hm3d_3t8DB4Uzvkt_public_0 | 1→1 | 0→0 | 94→94 | — |
| hm3d_3t8DB4Uzvkt_public_1 | 0→0 | 1→1 | 68→68 | native human collision; static contact also observed |
| hm3d_3t8DB4Uzvkt_public_2 | 0→0 | 0→0 | 1→1 | unsupported query surface |
| hm3d_5cdEh9F2hJL_public_0 | 0→0 | 1→1 | 72→91 | native human collision; static contact also observed |
| hm3d_5cdEh9F2hJL_public_1 | 1→1 | 0→0 | 49→52 | — |
| hm3d_5cdEh9F2hJL_public_2 | 0→1 | 1→0 | 76→153 | — |
| mp3d_2azQ1b91cZZ_public_0 | 1→1 | 0→0 | 194→234 | — |
| mp3d_2azQ1b91cZZ_public_1 | 0→0 | 1→1 | 85→88 | native human collision; static contact also observed |
| mp3d_2azQ1b91cZZ_public_2 | 1→1 | 0→0 | 167→209 | — |
| mp3d_8194nk5LbLH_public_0 | 1→1 | 0→0 | 83→83 | — |
| mp3d_8194nk5LbLH_public_1 | 0→0 | 1→1 | 81→99 | native human collision; static contact also observed |
| mp3d_8194nk5LbLH_public_2 | 1→1 | 0→0 | 128→144 | — |

Actual mechanisms, with action indices and legal observations in [failure_diagnostics.json](artifacts/failure_diagnostics.json):

- **Public recovery, `hm3d_5cdEh9F2hJL_public_2`:** nominal collides at action 76. B1 starts avoidance at action 69, abandons it when the next step becomes worse than nominal, later takes a six-action static nudge, and reaches the goal in 153 actions (7.6 mm endpoint error). This is genuine extra native execution, not early termination.
- **Remaining public failures:** four native human-collision terminations, all with static contact also recorded, plus one endpoint outside the certified surface. For `hm3d_3t8DB4Uzvkt_public_1`, a human is first observed at action 54 during required heading turns; the forward-risk attempt at 63 finds no useful maneuver and nominal continues until collision at 68. These timings establish limited actionable reaction time; static-contact co-occurrence alone does not prove causation. No public failure is BLOCKED, an executor watchdog, or a native episode limit.
- **Independent loss, `hm3d_5hGmubEE6ZF_additional_0`:** B0 succeeds in 92 actions. B1 diverges at action 20 into a committed 40-action maneuver, rejoins, then collides at action 69 after 2.21 m travel with 3.61 m native goal distance remaining. The altered local path/timing creates a real failure despite a predicted improvement.
- **Independent recovery, the same scene’s episode 1:** B0 collides at action 47; B1 diverges at 40, completes a detour/rejoin and succeeds in 163 actions. Overall independent SR remains 1/8, HC remains 5/8, and SPL decreases. Five B1 failures are native human collisions and two are unsupported surfaces. These results were not used to tune the candidate.

## Short motion queries

All eight fixed scenario starts/controllers/seeds are reused with predeclared **at most 3 m** selected reference prefixes, through the canonical Stage 1 adapter. These are representative Stage 1-shaped motion queries, not sampled L3 decisions or an evaluation of the Stage 1 query distribution. No intelligent query selection or replanning is simulated.

QCR improves **4/8→6/8**: three recoveries, one lost query completion, three retained completions and one retained failure. The lost case is `scenario_multiple_short`; `scenario_stationary_short` remains a native collision. The three common completed queries have **zero additional path length and zero additional actions**. Across all eight attempts, mean actual route progress increases **2.396→2.805 m** and mean travel increases **2.396→3.346 m**. Query execution takes **162→388 native actions** (166→394 including driver STOP). Longer travel/action totals include rescued failed queries and must not be interpreted purely as detour overhead. B1 records five deviations and three completed rejoins; the two unrejoined attempts collide. Per-query distances/actions/rejoins are in [short_summary.json](artifacts/short_summary.json).

Native episode HC is **5→2**. One B0 collision occurs during the driver STOP after valid query completion (`scenario_multiple_short`): pre-STOP native mesh contacts are **4→2**, while the official native episode counts remain unchanged. Query completion is neither navigation success nor a claim about the subsequent STOP. The executor itself never emits that STOP.

## Runtime and L2 ablation

All distributions are measured on actual native trajectories on the shared 36-logical-CPU host, with one numerical thread per runner, nice 19 / idle I/O, and the 28-worker collector active. Public and additional panels briefly run concurrently. Native stepping includes legal observation acquisition; episode throughput includes setup, geometry preparation, outcome labeling and serialization, but excludes the runner’s one-time shared map/model loading.

| Public operation | N | P50 ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| B1 nominal act | 1096 | 0.357 | 0.515 | 0.650 | 6.007 |
| B1 avoidance-attempt act | 15 | 43.695 | 51.972 | 54.056 | 54.577 |
| B1 committed-maneuver act | 185 | 0.410 | 0.524 | 0.550 | 0.575 |
| B1 all act | 1308 | 0.368 | 0.539 | 35.621 | 54.577 |
| B1 act + transition feedback | 1308 | 0.842 | 1.076 | 36.148 | 55.103 |
| B2 executor + feedback + forecast-request gate | 1308 | 0.943 | 1.275 | 29.381 | 51.572 |
| B2 frozen forecast refresh | 14 | 23.286 | 33.614 | 35.307 | 35.731 |
| B2 actual belief initialize/update | 1315 | 44.414 | 111.282 | 128.325 | 334.122 |
| B1 native step + legal acquisition | 1308 | 83.077 | 153.147 | 251.035 | 1006.360 |

Ordinary nominal control meets the sub-millisecond objective and complete L4 P95 meets the 10 ms target. **Avoidance attempts do not meet the preferred 5 ms target, and P99 does not meet the preferred 20 ms target.** Fifteen public avoidance attempts account for the tail; there is no online full-mesh fallback. Public median/P95 candidate counts are one, maximum 183, versus the old P95 near 1,873. Public tracking P95 is 0.061 ms and actual ego prediction/grid/static-nudge work P95 is 0.325 ms; candidate geometry work is also included in avoidance timing, so these are not an exhaustive additive decomposition.

The additional panel has a retained **306.65 ms act outlier** (307.18 ms including feedback), at a later avoidance attempt rather than the first call. Warm deterministic replay of that actual trace exactly reproduces actions and has a 51.55 ms online maximum. The precise contribution of scheduling/cache effects to the original outlier was not isolated; it is not discarded or attributed to mesh work. [Native replay profiles](artifacts/native_profile/profile.json) and [additional replay profiles](artifacts/additional_native_profile/profile.json) retain the measurements.

| Native workload | B0 steps/s | B1 steps/s | B2 steps/s | B1 online P95/P99 ms |
|---|---:|---:|---:|---:|
| Public12 | 6.934 | 6.713 | 5.197 | 1.076/36.148 |
| Frozen scenarios | 2.915 | 5.493 | 4.629 | 0.907/25.861 |
| Short queries | 2.608 | 5.241 | 4.465 | 0.879/20.240 |
| Additional scenes | 4.220 | 4.229 | 3.369 | 1.097/39.645 |

L2 is requested only at eligible avoidance attempts: **14 public forecast refreshes**, versus 416 in the historical conservative run (different trajectory/exposure counts). Actual belief updates still process every actual packet. Across all **36 matched query/episode pairs**, B1 and B2 reproduce identical actions and poses: no SR, HC, SPL, STL or PSC improvement is measured. Keep B1 as default and the shared frozen forecast optional. This is evidence about this local ranking integration, not a negative result about using L2/L2.5 forecasts in a learned L3.

## Direct answers and freeze recommendation

| Question | Answer |
|---|---|
| Q1: Preserve nominal performance? | Yes on public12: all six successes retained, SR 6/12→7/12. Independent aggregate SR ties, but one baseline success is lost. Short-query QCR improves with one lost completion. No universal preservation claim. |
| Q2: Reduce human collisions without artificial endings? | Public 5→4, frozen 5→0, independent 5→5. Short-query native episode HC 5→2, including the separately disclosed STOP contact. No avoidance-failure STOP or early handoff exists. The approximately 50% public target is not achieved. |
| Q3: Previously failed episodes recovered? | Versus B0: one public, five frozen, one independent; short queries recover three completions. Versus historical public B1: six recoveries. Panels are separate and overlapping scenario/query trials are not pooled as independent evidence. |
| Q4: Previously successful episodes lost? | Zero public, zero frozen, one independent, and one short-query completion. |
| Q5: Oscillation/BLOCKED eliminated? | Zero BLOCKED across final runs. Public adjacent reversals fall 1,159→3; each is isolated, with no repeated alternating-turn run. Required orientation turns and finite maneuver turns remain. |
| Q6: Fast enough for large-scale L3 RL? | Routine B1 control and full online P95 are suitable for integration; rare avoidance tails exceed preferred P99/attempt targets. Simulator acquisition and frozen belief updates dominate end-to-end cost. Strict tail-latency acceptance is not established. |
| Q7: Does frozen L2 improve execution? | No measured benefit; all 36 B1/B2 action/pose traces are identical and B2 throughput is lower. |
| Q8: What must L3 learn? | Query/route choices under occlusion and moving humans, infeasible/occupied endpoints, native static sliding/contact, unsuccessful local maneuvers, and task failures from normal observations/rewards/termination/history. L4 supplies no special replanning event. |
| Q9: Suitable to freeze? | Functionally yes as a provisional observation-only research executor for L3 integration/initial training. It demonstrates a small public gain and strong frozen-scenario gain, not robust general navigation improvement. Independent performance and latency tails limit stronger acceptance. |

No L3 was implemented or trained. The illustrative **70%→80% system-level improvement remains untested**. Freeze the progress-first interface/B1 profile for research iteration, retain the nominal ablation, and evaluate the eventual trained L3 with paired system-level trials.

Complete summaries, per-episode CSVs, protocols, source receipts, audit results and full-trace SHA256 manifests are in [artifacts](artifacts/). Reproduction and retained failure locations are in [README.md](README.md).
