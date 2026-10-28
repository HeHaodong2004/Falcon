# Final evaluation protocol (declared before any HOLDOUT run)

Suites are frozen in `docs/querygen/suites.json` (sha256 recorded in `suites.py`): DEV = 28 HM3D
episodes (used for all iteration), HOLDOUT = 28 HM3D + 11 MP3D episodes (never run, never inspected,
no certificates computed before the generator is frozen).

## Frozen system (tag `querygen-freeze-v1`)
- Generator v6 (`GeneratorConfig()` defaults: budget 10; manoeuvre mode after a legal stall near
  the start/goal), executor with the fixed body-aware lattice, full-responsibility ORCA, native
  dynamics.
- Oracle (`OracleConfig()` defaults): lazy weighted best-first (w = 2) over generator queries,
  privileged space-time cost-to-go heuristic, privileged manoeuvre-mode gate (+120), manoeuvre
  prior 10 + pad to a full 16-step interval, fine de-duplication for manoeuvre edges,
  **budget 6000 simulated edges** (deterministic; wall-clock cap 9000 s as a safety net only).
- Both suites run with exactly these defaults (`--oracle '{}'`), 9 worker processes, frozen code
  snapshot per run.
- Generator: `eventwm_graph/querygen/generator.py` at the freeze commit (tag `querygen-freeze-*`),
  default `GeneratorConfig` (budget 10). Planner: full-responsibility ORCA (`planners.py`).
  Dynamics: benchmark-native discrete actions. Oracle: lazy weighted best-first over generator
  queries with the configuration recorded in every result file (`oracle_config`).
- Every suite run executes a frozen copy of `eventwm_graph/querygen` (`<run>/code_snapshot`, content
  hash in each result) so later edits cannot leak into a running evaluation.

## Per-episode verdict (unchanged since the harness was built)
PASS iff the Oracle found a complete branch made only of generator queries, and the authoritative
continuous replay from reset with the unchanged `habitat.Env.step` (re-executing the query sequence
through the live executor) reproduces the branch actions exactly, ends with official success
(STOP, geodesic < 0.2 m, within 500 steps), official human collision 0, zero body-proxy overlap
steps (robot rectangle >= 0.25 m from same-floor human centres), zero static-collision steps under the
declared spawn-escape rule, bitwise shadow-pose agreement at every step, no canonical human contact.

## Infeasibility (only with a complete certificate; never by judgement)
An episode may be reported as *certified infeasible* only if one of these privileged certificates
is complete (search exhausted, no budget/depth cut-off reached before exhaustion):
1. start escape (`feasibility.certify`): no native primitive sequence leaves the spawn by 1 m
   without a declared collision (BFS keyed on 2 cm position, exact turn index, gait phase, escape flag);
2. progress (`feasibility.certify_progress --exact-gait`): the full reachable set (same key incl.
   gait phase, up to the stated depth) contains no state closer to the goal than the spawn;
3. goal region (`feasibility.certify_goal`): no pose within the success region (4 cm grid, 75
   headings, 5 gait phases) is free of static contact.
Each certificate also reports measured penetration depths / reachable-state counts as evidence.
Certificates are computed for HOLDOUT only after the freeze, for every episode (not only failures).

## Reporting
- Strict: PASS count over all episodes (certified-infeasible episodes count as failures).
- Feasible set: PASS count over episodes without a complete infeasibility certificate.
- Gate verdict: **PASS** only if every HOLDOUT episode without a complete infeasibility certificate
  passes (and DEV likewise); certified-infeasible episodes are listed with their evidence and the
  strict number is always reported next to it. Otherwise **REVISE** (or **BLOCKED** if a failure is
  caused by something outside the track's control that cannot be certified).
- Also reported: SPL, time, PSC, minimum clearance, waiting/oscillation/retreat/deadlock (trace
  definitions in `report.py`), raw/kept query counts, generator and Oracle runtime P50/P95, replay
  time, and the executor-only baseline on the same suites.
