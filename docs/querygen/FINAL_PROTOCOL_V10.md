# v10 evaluation protocol (DRAFT -- to be completed and committed before any run on the second unseen suite)

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
Tag `querygen-freeze-v10` (to be set).  Configurations: TBD from the development runs (C1 ORCA x native
K=10 and K=6; C2 TBD).  Oracle budget: TBD (edge budget, deterministic; wall-clock cap only as a safety net
set high enough never to bind before the edge budget).

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

## Reporting
Per configuration: SR over feasible episodes (primary), strict SR, failures, certified-infeasible and
unresolved lists; SR as a function of the edge budget (from the edges-to-solution of each run); SPL, steps,
minimum human clearance, Oracle edges and wall time (P50/P95), generator latency, cached-step share.
