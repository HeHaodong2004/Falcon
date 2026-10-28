# v11 evaluation protocol -- OFFICIAL Falcon rules (DRAFT: declared before any run on the third unseen suite)

Status: draft written 2026-10-28 13:25 while the v11 development runs are in progress.  It becomes binding when
the freeze tag `querygen-freeze-v11` is created; any change before then is recorded here with its reason.

## Rules
`docs/querygen/OFFICIAL_RULES.md`: Falcon's own definitions, verified in the benchmark code.  Success = official
`success` (STOP with geodesic distance < 0.2 m, within 500 steps; the episode ends at the first robot-human
contact, `HumanCollision` -> `task.should_end`).  Human collision, PSC (share of steps with every human >= 1 m),
SPL and STL are reported as Falcon defines them.  No additional safety constraint, body proxy or static-contact
rule enters any verdict.  Actions: Falcon's discrete set (forward 0.25 m, turn 10 deg, stop) for every
configuration.

**Verdict:** PASS iff the selected branch's authoritative continuous replay from reset with the unchanged
`habitat.Env.step` reproduces the branch's actions exactly and ends with official success.

**Denominator:** every episode of the suite except those PROVEN natively infeasible: an exact exhaustive search
over all native action sequences (`native_cert.py`, bitwise-identical states merged) closes with no successful
sequence, and its closed sequences are verified in the official `Env.step`.  Episodes that are neither solved
nor proven infeasible (unresolved) stay in the denominator as failures.  Reported separately: proven infeasible,
confirmed feasible (a PASS of any configuration, or a native witness `witness.py --rules native` verified in the
official `Env.step`), unresolved.

## Suites
* Development set (inspected freely): DEV (28), first unseen (40), second unseen (60) = 128 episodes; proven
  infeasible 3 (unseen2 5Poh4Qz68hd, sKLMLpTHeUy, AMEM2eWycTq); unresolved 1 (unseen nMeXfQU4PMS).
* **Test set: third unseen suite** `docs/querygen/suites_unseen3.json` (45 HM3D + 15 MP3D Falcon train-split
  scenes, salted hash `querygen-unseen3-train-scenes-20261028`, disjoint from DEV / HOLDOUT / unseen /
  unseen2; frozen at commit 4a40ed00, sha256 6351fed3...).  Never run or inspected before the freeze.

## Frozen system (v11)
Launcher `scripts/querygen_final_v11.sh SUITE CONFIG OUT_ROOT [WORKERS]` (refuses a dirty package; every run
executes a frozen copy of the package, `<run>/code_snapshot`).

| id | planner | dynamics | generator |
|---|---|---|---|
| C1-K10 | ORCA (full responsibility) | native | `tc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False`, budget 10 (quota) |
| C1-K6 | ORCA | native | `select=compact`, budget 6, `tc_variants=(goal,)`, `st_goal`, `maneuver_spawn_only=False` |
| C2-K10 | DWA | native | as C1-K10 |
| Floor-C1 / Floor-C2 | ORCA / DWA | native | default policy, no selection: keep the running query, else the goal query |

v11 foundation options (legal-only, all configurations): `goal_unstick=6` (static deadlock on the goal query:
six still steps with no human within 2 m -> tight legal map + plan following; with no body-feasible plan,
rotate toward the route -- in-place rotation is always executable natively), `bump_memory_m=0.25`
(proprioception: a forward step that moves < 2 cm marks a disc around the body's front contact point BLOCKED),
`evade=True` (human-threat reflex: when the robot's own next step plus holding meets a tracked human's
constant-velocity prediction within 0.30 m in 14 steps, take the clearance-maximising short native sequence);
v10 defaults (`goal_relax_m=0.5`, exact rotation ends and arrivals, K=6 reset coverage).

Choice (development set, paired episodes, official rules): default-policy success ORCA 5 -> 9 of 21, DWA 2 -> 10
of 21 with all three options (`output/runs/official/exec` vs `exec_ube`); the evasion reflex added 2 ORCA
episodes and lost none against the breaker + bump arm.  Oracle capability with v11: `output/runs/official/v11`
(to be filled in before the freeze).

**Oracle** (privileged selection only): two-member portfolio (weight 3.5 with the 0.5 m space-time model and the
cell-sound fallback; weight 2 with the 0.6 m model), `heading_h`, `escape_penalty=80`, `prefix_cache=48`, one
deterministic budget of 12,000 simulated edges.

## Reporting
Per configuration and suite: SR over the denominator (primary), failures, proven-infeasible and unresolved
lists; official SPL, PSC, human collisions, steps on solved episodes; Oracle edges (P50/P95).  Floor SR per
planner.  Layer-3 learnability on the test suite's solved episodes (`tests/querygen/l3_readiness.py`): good
candidates per decision state, top-1 success of legal selection rules, one-decision recovery.  Development
numbers of the frozen system are reported separately from the test.
