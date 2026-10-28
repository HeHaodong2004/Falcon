# Gate test: frozen v10 system under the OFFICIAL Falcon rules (declared before any run on the third unseen suite)

Declared 2026-10-28 ~14:00, before any episode of `docs/querygen/suites_unseen3.json` was run or inspected.

## Gate
Oracle success rate >= 95% over every test episode that is not PROVEN natively infeasible (user objective,
2026-10-27/28), per configuration, under the official Falcon rules (`OFFICIAL_RULES.md`).  The Oracle is an offline
diagnostic of the capability ceiling of planner + dynamics + executor + query generator; generation and execution
are legal-only, selection is privileged.

## Frozen system
Tag `querygen-freeze-v10-official` = commit 2786115d; querygen package sha256
`bcecb8e8119faf9c20b5d1011455c166755e01efc9132c094b844f3db397da87` (the hash every result record carries as
`code_snapshot.sha256`).  This is exactly the code of the official-rules development runs
`output/runs/official/base/{dev,unseen,unseen2}_{c1k10,c1k6,c2k10}` (verified file by file).  Launcher
`scripts/querygen_test_v10_official.sh CONFIG OUT_ROOT [WORKERS]` extracts the tagged package with `git archive`
into each run's `code_snapshot` and refuses a hash mismatch, so later work in the tree (the v11 / v12 default-policy
options, all off by default) cannot leak into the test.

| id | planner | dynamics | generator | Oracle |
|---|---|---|---|---|
| C1-K10 | ORCA (full responsibility) | native | `tc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False`, budget 10 | v10 portfolio [weight 3.5 / 0.5 m space-time model + cell-sound fallback; weight 2 / 0.6 m model], `heading_h`, `escape_penalty=80`, `prefix_cache=48`, 12,000 simulated edges |
| C1-K6 | ORCA | native | `select=compact`, budget 6, `tc_variants=(goal,)`, `st_goal`, `maneuver_spawn_only=False` | same |
| C2-K10 | DWA | native | as C1-K10 | same |

Official discrete actions for every configuration; `goal_relax_m=0.5` (v10 default).

## Verdict and accounting
PASS iff the authoritative continuous replay from reset with the unchanged `habitat.Env.step` reproduces the
selected branch's actions exactly and the official `success` measure is 1.  Every episode the Oracle does not
solve gets, after the runs, (1) a native witness search (`witness.py --rules native`, independent of the system,
exact native primitives, verified in the official `Env.step`) -- found = confirmed feasible, stays a failure; (2)
if no witness is found, the exact native certificate (`native_cert.py`): only a closed tree whose closed sequences
reproduce the official human-collision measure in the official environment proves infeasibility and removes the
episode from the denominator.  Unresolved episodes stay in the denominator as failures.

## Reporting
Per configuration: SR over the denominator (gate), proven-infeasible / confirmed-feasible / unresolved lists,
failures, official SPL, PSC, human collisions and steps on solved episodes, Oracle edges P50 / P95.  Development
numbers of the same frozen system (`output/runs/official/base`) are reported separately.
