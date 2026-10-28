# Gate test: frozen v12 system under the OFFICIAL Falcon rules (declared before any run on the fourth unseen suite)

Declared 2026-10-28 ~17:45; becomes binding when the test is launched (the v12 development run must first confirm
>= 95% on the development set).  Supersedes the draft FINAL_PROTOCOL_V11.md.

## Gate (user, 2026-10-28)
At least ONE configuration reaches an Oracle success rate >= 95% over every test episode not PROVEN natively
infeasible (K=10 is acceptable; the others need reasonable performance).  Official Falcon rules
(`OFFICIAL_RULES.md`); verdict = the authoritative continuous replay with the unchanged `habitat.Env.step`
reproduces the selected actions and the official `success` measure is 1.  Also reported: SPL, PSC, human collisions,
turn reversals and backtracking (route quality), Oracle search cost (speed), decision robustness.

## Frozen system
Tag `querygen-freeze-v12` = commit b0f3c1a3, querygen package sha256
`d90e91837ba070560901ffd0ddf0636dcf050c743b12030a2e22809e78678efb` -- exactly the code of the v12 development runs
`output/runs/official/v12/` (verified file by file).  Launcher `scripts/querygen_test_v12.sh CONFIG OUT_ROOT
[WORKERS]` extracts the tagged package with `git archive` and refuses a hash mismatch.

| id | role | planner | generator (all v12 foundation flags below) |
|---|---|---|---|
| C1-K10 | primary | ORCA | `tc_variants=(goal, st_goal)`, `vc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False` |
| C1-K6 | secondary | ORCA | compact K=6, `tc_variants=(goal,)`, `st_goal`, `compact_long_turns`, `compact_human_first` |
| C2-K10 | secondary | DWA | as C1-K10 |
| Floor-C1 / Floor-C2 | default policy | ORCA / DWA | no selection |

v12 foundation (legal-only): `goal_unstick=6` + `unstick_progress` (deadlock breaker only on genuine deadlocks) +
`unstick_keep` + `static_humans`, `bump_memory_m=0.25` + `bump_front`, `evade`, `stop_strict`, `goal_relax_m=0.5`,
`stop_closest_m=1.2` (legal closest-approach STOP).  Oracle: v10 portfolio, `heading_h`, `escape_penalty=80`,
`prefix_cache=48`, 12,000 edges.

## Test set
Fourth unseen suite `docs/querygen/suites_unseen4.json` (45 HM3D + 15 MP3D Falcon train scenes, salt
`querygen-unseen4-train-scenes-20261028`, disjoint from DEV and unseen 1-3, sha256 73d307e3...), frozen before any v12
evaluation and never run or inspected.

## Accounting
As FINAL_PROTOCOL_V10_OFFICIAL.md: unsolved test episodes get a native witness search, then the exact native
certificate; only an officially verified certificate removes an episode from the denominator.
