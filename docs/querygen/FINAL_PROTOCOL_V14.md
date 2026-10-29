# Held-out test: frozen v14 system under the OFFICIAL Falcon rules (declared before any run on the sixth unseen suite)

Drafted 2026-10-28 ~19:30; binding since 2026-10-29 00:20 (tag created).

**Choice v14 vs v15** (pre-registered rule, PROGRESS.md 2026-10-28 22:58, freeze time declared 23:53): on the 246
development episodes finished by both runs, PASS v14 233 vs v15 232 (flip: jh4fc5c5qoQ v14 PASS -> v15 FAIL_SEARCH);
mean search minutes 7.85 vs 6.50; SPL on both-pass 0.917 vs 0.919.  The two v15 episodes still running at freeze
(kdFEfVoT1WE, VLzqgDo317F) are v14 PASS, so they cannot change the outcome: **v14 is frozen**.

## Purpose
v14 improves the query-level system (no privileged fallback: user, 2026-10-28 19:10 -- the fallback is the last
resort) on the user's four requirements: success rate, route quality (SPL), decision robustness (query-space
learnability) and speed.  Gate (unchanged): at least ONE configuration reaches an Oracle success rate >= 95% over
every test episode not PROVEN natively infeasible; official Falcon rules (`OFFICIAL_RULES.md`); verdict = the
authoritative continuous replay with the unchanged `habitat.Env.step` reproduces the selected actions and the
official `success` measure is 1.

## Frozen system
Tag `querygen-freeze-v14` -> commit e8fb1c77 (the querygen package every v14 development episode imported: code
snapshot sha256 21cc7deb8790e4eb81fc4797daf174de3873fba1ff084da13e27dfa2298dc61a), launcher
`scripts/querygen_test_v14.sh CONFIG OUT_ROOT [WORKERS]` (git archive of the tag, package hash check).

Development evidence (official rules, query level, no fallback; DEV + unseen 1-4 = 248 episodes, 242 not proven
infeasible): **235 / 242 = 97.1%** PASS, SPL 0.915, PSC 0.928, 0 human collisions, Oracle edges P50 77 / P90 1,493,
mean search 8.3 min (v12 on its 182: 94.0%, 18.4 min).

v14 = v12 (FINAL_PROTOCOL_V12.md) + exact fast legal sensor (`QG_FAST_SENSE=1`, identical observations) +
* generator `goal_unknown_cost=1.25`: the goal queries' legal route prices unobserved cells 1.25x instead of 2.5x
  (a goal behind the spawn was approached through the observed area in front; default policy, 36 paired development
  episodes: success 16 -> 20, SPL on both-solved 0.65 -> 0.68);
* Oracle path-aware refinement for first solutions below path efficiency 0.85 (`improve_edges=300`,
  `improve_lambda=1`, `improve_below_spl=0.85`; 9 lowest-SPL development solutions 0.70 -> 0.80);
* generator `th_variants=(goal, st_goal)` replacing `vc_variants` in the K=10 configurations: contact-tolerant goal
  copies (`executor._ThruLegal`: observed wall-band cells are soft obstacles -- body checks optimistic, routes at
  cost x6 -- except proprioceptive bump-memory discs; human clearance contact-level as `_vc`).  The legal band is
  visual and stricter than Falcon's kinematic, navmesh-bound robot, which clips through furniture (static contact is
  legal): every hard development witness crosses band cells at body height.  p32JzpQyhPk (v12: 12,000-edge failure)
  solved in 105 edges.
* `xc_variants` (committed contact-level copies) tested and NOT adopted (no gain on VqCa / p32J / Nfvx).
No `primitives`, no `prim_fallback_sims`.

| id | role | planner | generator |
|---|---|---|---|
| C1-K10 | primary | ORCA | v12 C1-K10 with `_th` instead of `_vc` + the v14 options |
| C1-K6 | secondary | ORCA | v12 C1-K6 + the v14 options (no `_vc` / `_th` copies) |
| C2-K10 | secondary | DWA | as C1-K10 |
| Floor-C1 / Floor-C2 | default policy | ORCA / DWA | no selection |

## Test set
Sixth unseen suite `docs/querygen/suites_unseen6.json` (60 HM3D Falcon train scenes; the MP3D train split is
exhausted), salt `querygen-unseen6-train-scenes-20261028`, disjoint from DEV / HOLDOUT / unseen 1-5, frozen at
commit b08c2487 before any v14 development, never run or inspected.

## Accounting
As FINAL_PROTOCOL_V10_OFFICIAL.md: unsolved test episodes get a native witness search, then the exact native
certificate; only an officially verified certificate removes an episode from the denominator.
