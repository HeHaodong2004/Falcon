# Held-out test: frozen v15 system under the OFFICIAL Falcon rules (DRAFT -- declared before any run on the sixth unseen suite)

Drafted 2026-10-28 ~23:05 as the alternative to FINAL_PROTOCOL_V14.md; the pre-registered freeze rule (PROGRESS.md,
2026-10-28 22:58) chooses one of the two.  Binding once the tag below exists.

## Purpose
v14 improves the query-level system (no privileged fallback: user, 2026-10-28 19:10 -- the fallback is the last
resort) on the user's four requirements: success rate, route quality (SPL), decision robustness (query-space
learnability) and speed.  Gate (unchanged): at least ONE configuration reaches an Oracle success rate >= 95% over
every test episode not PROVEN natively infeasible; official Falcon rules (`OFFICIAL_RULES.md`); verdict = the
authoritative continuous replay with the unchanged `habitat.Env.step` reproduces the selected actions and the
official `success` measure is 1.

## Frozen system
Tag `querygen-freeze-v15` (same querygen package as v14), launcher
`scripts/querygen_test_v15.sh CONFIG OUT_ROOT [WORKERS]` (git archive of the tag, package hash check).

v15 = v14 + short-interval goal copies near tracked humans in the K=10 configurations (generator
`short_human_m=2.5`, `short_steps=4`: when a legally tracked human is within 2.5 m, the human category offers a goal
copy whose Oracle edge lasts 4 steps -- fine timing in crowds; VqCaAuuoeWk 7,154 -> 280 edges, Pm6F8kyY3z2 solved in 749).
The K=6 configuration is exactly v14's.

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
| C1-K10 | primary | ORCA | v14 C1-K10 + short-interval copies |
| C1-K6 | secondary | ORCA | v12 C1-K6 + the v14 options (no `_vc` / `_th` copies) |
| C2-K10 | secondary | DWA | as C1-K10 (short copies untested with DWA in development) |
| Floor-C1 / Floor-C2 | default policy | ORCA / DWA | no selection |

## Test set
Sixth unseen suite `docs/querygen/suites_unseen6.json` (60 HM3D Falcon train scenes; the MP3D train split is
exhausted), salt `querygen-unseen6-train-scenes-20261028`, disjoint from DEV / HOLDOUT / unseen 1-5, frozen at
commit b08c2487 before any v14 development, never run or inspected.

## Accounting
As FINAL_PROTOCOL_V10_OFFICIAL.md: unsolved test episodes get a native witness search, then the exact native
certificate; only an officially verified certificate removes an episode from the denominator.
