# v10 evaluation protocol (declared and committed before any run on the second unseen suite)

## Suites
* **Development set** (may be inspected freely): DEV (28 episodes; 24 feasible, 4 certified infeasible) and
  the first unseen suite (`suites_unseen.json`, 40 episodes; consumed by the v9 evaluation; 31 feasible,
  7 certified infeasible, 2 unresolved).  Development runs skip the certified-infeasible episodes
  (`scripts/querygen_dev_v10.sh`).
* **Test set: second unseen suite** (`docs/querygen/suites_unseen2.json`, 45 HM3D + 15 MP3D Falcon
  train-split scenes, salted hash, disjoint from DEV / HOLDOUT / unseen; frozen at commit d9f23c01,
  sha256 3bfb1472...).  Never run with the system before the freeze.  Privileged witness searches
  (`scripts/querygen_witness_suite.sh`, 3,600 s, 1.6 M simulations, independent of the system under
  test) were started before the freeze; their results are not inspected before the freeze.  Feasibility
  certificates are computed only after the freeze.

## Frozen system
Tag `querygen-freeze-v10`.  Launcher `scripts/querygen_final_v10.sh SUITE CONFIG OUT_ROOT [WORKERS]` (refuses a
dirty package; every run executes a frozen copy of the package, `<run>/code_snapshot`).  Three configurations
share the generator code:

| id | planner | dynamics | generator (overrides of the code defaults) |
|---|---|---|---|
| C1-K10 | ORCA (full responsibility) | native | `tc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False`, `goal_relax_m=0.5`, budget 10 (quota) |
| C1-K6 | ORCA | native | `select=compact`, budget 6, `tc_variants=(goal,)`, `st_goal`, `maneuver_spawn_only=False`, `goal_relax_m=0.5` |
| C2-K10 | DWA | inertia (12 levels, accel/brake 2, turns at rest) | as C1-K10 (st_goal and close passes are unavailable for momentum dynamics; tight variants remain) |

v10 generator/executor defaults (all legal-only): exact rotation ends (`rot_exact`), exact goto arrival
(`arrive_exact`), K=6 reset-state coverage (`root_cover`), goal relaxation (`goal_relax_m=0.5`: a planning
target inside observed BLOCKED cells -- a goal on the navmesh edge next to furniture -- seeds the field from
the passable cells within 0.5 m instead of making every goal query 'unreachable'), momentum robots brake to
rest before STOP, DWA's blocked fallback brakes at the brake limit.

**Oracle** (privileged selection only; identical for all configurations): v9 options (`heading_h`,
`escape_penalty=80`, `escape_h`, `prefix_cache=48`) and a **two-member portfolio** (`OracleConfig.portfolio`)
under one deterministic budget of **12,000 simulated edges** (wall-clock cap 40,000 s, a safety net only):
* member 0: weight 3.5 (greedier), space-time radius 0.5 m with the cell-sound fallback table;
* member 1: weight 2, space-time radius 0.6 m, no fallback (the v9 heuristic: pessimistic about human blocking).
Members are independent best-first searches over the same generator queries from the same reset, run
interleaved one simulated edge at a time with equal shares; the first success wins.

Choice (development set only, single-member runs capped at 6,000 edges, `output/runs/v10dev/pf/`): the
single weight-2 search of r2 lost episodes that v9 or other orderings solve (LEFTm3JecaC, LT9Jq6dN3Ea,
Pm6F8kyY3z2) -- search cost is heavy-tailed and order-dependent.  Member 0 is the strongest single
ordering (e.g. iePHCSf119p 116 vs 1,375 edges, SByzJLxpRGn 723 vs 3,382, zUG6FL9TYeR 1,082 vs 3,739,
LEFTm3JecaC 2,091 vs failure); member 1 solves cases member 0 misses (LT9Jq6dN3Ea 253 edges,
iigzG1rtanx 3,036).  A third member (weight 2 with the 0.5 m model) added no episode on the development
data, so the budget is split two ways.

## Verdict (v9 + dynamics check)
PASS iff the Oracle found a complete branch of generator queries and the authoritative continuous replay
from reset with the unchanged `habitat.Env.step` reproduces the branch actions exactly, ends with official
success (STOP, geodesic < 0.2 m, <= 500 steps), official human collision 0, zero body-proxy overlap steps
(robot rectangle >= 0.25 m from same-floor human centres), zero non-exempt static-contact steps
(declared spawn-escape rule), bitwise shadow-pose agreement, no canonical human contact, **and zero
dynamics-model violations** (`dynamics.dynamics_violations`: speed-level change beyond accel/brake,
turning while moving, STOP while moving; always 0 for the native robot).

## Feasibility accounting
As v9: witness-proven feasible (verified witness, or a replay-verified PASS of any configuration),
certified infeasible (start escape / exact-gait progress / goal region certificates), unresolved otherwise.
C2 (inertia) feasibility is reported against the native-dynamics witnesses and certificates (an inertia
robot can only be slower), with the caveat stated.

## Reporting
Test numbers: `scripts/querygen_final_v10.sh unseen2 CONFIG output/runs/final_v10` for the three
configurations; certificates `scripts/querygen_unseen_certificates.sh output/runs/final_v10/certificates_unseen2
output/runs/witness_unseen2 3 unseen2`.  Development numbers of the frozen system (same launcher, suites dev and
unseen, certified-infeasible episodes skipped) are run alongside and reported separately from the test.
Per configuration: SR over feasible episodes (primary), strict SR, failures, certified-infeasible and
unresolved lists; SR as a function of the edge budget (from the edges-to-solution of each run); SPL, steps,
minimum human clearance, Oracle edges and wall time (P50/P95), generator latency, cached-step share,
per-member wins of the portfolio; Layer-3 proxies (decisions per episode, default-query share, distinct
query types).
