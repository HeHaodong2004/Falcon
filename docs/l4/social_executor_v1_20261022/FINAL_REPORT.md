# L4 delivery and acceptance decision

**Overall decision: REVISE. L4 is not sufficiently reliable or fast for L3 RL
training.** A real native implementation, canonical compiler, legal-observation
integration, tests, full-episode benchmark panel, visualizations and profiling
are delivered. Broad navigation performance remains substantially below the
references. This report does not label the prototype accepted or RL-ready.

## Provenance and runnable baseline

The independent worktree is
`/research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022`, on
`research/l4-social-executor-20261022`, created from exactly
`cf1971228c37517202cc9e5787f785a3876a0514`. Evaluated executor code is frozen at
`e19431fb61b0c525d5d5711329052d546c92c54b`. The delivery commit adds the final
reports, diagnostics, tests and artifacts; its ID is the branch HEAD. The B3
policy is unchanged from its separately retained a2155778 evaluation receipt.

Before navigator development, actual native episodes demonstrated forward and
turn transitions, 48/56 ms action timing, legal human acquisition, collision
feedback, STOP termination and real frozen-L2 inference. The visible-human
encounter produced nine actual measurements and a [16,41,1,2] short-horizon
joint future. The initial inherited prefix's static contacts remain disclosed
as protocol evidence, not successful navigation.

Interpreter: `/research-user/miniforge3/envs/eventwm-falcon/bin/python`.
Checkpoint:
`/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt`.
SHA256: `215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`.
Model: `l2-intermittent-1-fc2c500cda1b243c7526`. The source L2 branch/checkpoint
remain unchanged. No collector restart, collection reconfiguration, dataset
deletion, training, L2.5 worktree modification or production/main merge occurred.

## Implemented architecture and contract

`eventwm_graph.l4` implements immutable query compilation, ordered actual-pose
progress tracking, native-action receding-horizon beam search, retained
avoidance/rejoin maneuvers, swept static checks, time-dependent observed-human
checks, bounded mesh/FK fallback, mode management and actual execution feedback.
The search uses 32 native actions and beam width 16. Only its first legal action
executes before an actual observation/transition acknowledgment.

Forward is 25/120 m and turns are ±10/120 rad. The configured protocol has no
WAIT or BACKWARD. STOP terminates; it is never used as a wait. FOLLOW, YIELD,
AVOID and REJOIN are separate from query completion/failure. Query budgets and
reference identity persist across mode changes. Unsupported geometry or finite
search failure is reported, without fabricating graph connectivity or progress.

The shared contract is **eventwm.l25l4.contract.v1**. The canonical API is
`compile_reference(QuerySpec, legal_graph, ActionSpec) -> CompiledReference`,
compiler version `l4.reference.1`. It records immutable waypoints, nominal
native actions, timestamped poses, completion criteria, limits and a reference
hash. `l2_segment()` supplies deterministic compatible nominal segments; a test
passes one through the actual frozen L2 query-observation API.

Runtime API: `Executor.start`, `Executor.act`, `Executor.on_transition`.
Feedback contains actual action/time/pose ledgers, progress, status, reasons,
observation references and episode flags. L2.5 imports this compiler rather
than constructing a competing schedule; L4 requires no L2.5 action service.
See [EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md) for the exact handoff semantics.

The optional 1.92 s / 16-world L2 bank is shared across candidates and has
explicit actual-belief lineage, timestamps and expiry. Current exact detections
retain priority. Forecasts contribute only a soft ranking cost and never clear
an immediate observation-based collision rejection. The actual-observation
owner retains persistent belief; prediction failure falls back without resetting
it. No online planner consumes hidden human positions, goals or future tapes.

## Validation and benchmark outcome

Nineteen automated tests pass. All 72 final native traces pass independent
protocol/feedback audits. The fixed analytic suite completes 12/12; 64
speed-varied analytic trials complete 61/64, with 0/64 human collisions and
three retained slow-leader progress failures. Analytic results are not Habitat
collision statistics.

Both native L4 arms complete 7/8 frozen scenarios with 0/8 human and 0/8 static
collisions. The permanently obstructed passage returns BLOCKED. Each arm
demonstrates four detours and four rejoins. The seven positive native cases
include head-on, crossing, stationary-human, multiple-human and static-detour
successes. Controlled native reobservation/slow-leader coverage remains incomplete.

The separate 12-episode/four-scene public panel uses complete native episodes
and unchanged evaluator metrics. It is a development panel, not the complete
published corpus. All failures and unsupported references remain included.

| Variant | SR | Human collisions | Static contacts | SPL | STL | PSC |
|---|---:|---:|---:|---:|---:|---:|
| B0 nominal | 6/12 | 5/12 | 8/12 | 0.5000 | 0.5000 | 0.9730 |
| B1 observed L4 | 1/12 | 2/12 | 0/12 | 0.0821 | 0.0833 | 0.9847 |
| B2 L4 + L2 | 1/12 | 2/12 | 0/12 | 0.0799 | 0.0833 | 0.9847 |
| B3 Falcon | 5/12 | 5/12 | 11/12 | 0.3122 | 0.4167 | 0.9406 |

L4 has 8/12 blocked episodes and one unsupported reference in each public arm.
The two human collisions occur while no human tracks are legally available.
Falcon lacks L4's explicit map and structured detection privileges, so this is
an explicitly qualified local comparison. The small scene-cluster intervals
and all per-case results are in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

L2 provides no public-panel SR/collision/PSC gain and worsens time/SPL on the
single completed public query. Its small-suite PSC improvement is not evidence
of general navigation improvement.

Warm isolated core P95 is 0.220 / 3.798 / 6.266 ms for zero/two/six humans.
Public native control P95/P99 is **267.933/385.688 ms for B1** and
**350.099/453.814 ms for B2**, excluding fresh L2. Measured public throughput is
6.694 / 4.914 native steps/s, or 164.9 / 119.7 attempted episodes/hour. These
rates include failed episodes and must not be read as successful training
throughput. Mesh triangle checks dominate the long tail. L2 refresh P95 is
31.920 ms; actual belief update P95 is 125.682 ms. See
[PERFORMANCE.md](PERFORMANCE.md) for cold starts, memory and complete timing scope.

## Separate acceptance decisions

| Gate | Decision | Evidence and limitation |
|---|---|---|
| BASELINE_RUNNABLE | PASS | Real native transitions, legal packets and frozen L2 predictions demonstrated before development. |
| QUERY_COMPILATION | PASS | Immutable deterministic reference and direct L2 segment-consumption tests; unsupported surfaces explicitly rejected. |
| STATIC_NAVIGATION | REVISE | Basic native empty/detour success; broader static bottlenecks and excessive mesh cost remain. |
| BASIC_HUMAN_AVOIDANCE | REVISE | Seven positive native cases pass, but full requested controlled native coverage and randomized progress criteria are not established. |
| ROUTE_REJOINING | REVISE | Four/four scenario rejoins; one/four public detours rejoin and three slow-leader trials fail progress. |
| PROGRESS_AND_DEADLOCK_HANDLING | REVISE | Honest bounded handoff works, but eight/twelve public queries block and turning oscillation is excessive. |
| ACTUAL_FEEDBACK | PASS | 72/72 independent trace audits; no illegal actions, fictional waits or false L4 completion claims found. |
| L2_INTEGRATION | PASS | Real frozen-model inference, shared cache, current-observation priority and fallback verified. Added navigation value is unproven. |
| FULL_NAVIGATION_BENCHMARK | REVISE | Full-episode development panel fails SR; complete frozen-corpus acceptance is not established. |
| CONTROL_LATENCY | REVISE | Normal fixtures pass, but actual public grid/mesh workloads exceed the targets. |
| RL_EXECUTOR_READINESS | REVISE | Low completion, blind collisions, deadlocks and long control tails prohibit training-readiness approval. |

Research claims remain separate: `BASIC_AVOIDANCE_PASS` is supported only for
the narrower fixed analytic and seven positive native demonstrations, not the
entire requested native coverage. `NAVIGATION_BENCHMARK_PASS` and
`PREDICTION_AUGMENTATION_PASS` are **not achieved**.

## Remaining blockers and delivered evidence

The next required repairs are stable escape/rejoin progress without alternating
turns, bounded active observation for unobserved approach directions, and fast
full-body support over complicated native geometry. The current surface/grid
and mesh fallback do not constitute an unconditional collision guarantee.
Revalidation must preserve the failed roots and include independent native
slow-leader, temporary-block and controlled reobservation cases. No timeout
increase, oracle input or retraining is proposed as a substitute.

Every meaningful failed attempt is recorded in [FAILURE_LOG.md](FAILURE_LOG.md).
The 5 mm arm-contact miss was repaired with the original 25 mm margin and a
checked translation escape; its actual witness and standalone regression are
retained. Public failures and long tails are retained despite that local repair.

Absolute output root:
`/research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022/output/l4_social_executor_v1_20261022/`.
Final subdirectories are `release_scenarios`, `release_navigation`,
`final_falcon`, `release_analytic`, `release_profile`, `profile_cold`,
`mesh_profile` and `figures`. Committed artifacts contain case tables,
summaries, receipts, hashes, profile evidence, trajectory plots and a head-on
comparison animation. [README.md](README.md) contains exact reproduction and
integration commands. No claim of final acceptance is made beyond the PASS
entries above.
