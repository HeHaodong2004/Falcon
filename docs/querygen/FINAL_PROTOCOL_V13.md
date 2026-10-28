# Gate test: frozen v13 system under the OFFICIAL Falcon rules (declared before any run on the fifth unseen suite)

Declared 2026-10-28 ~18:30 while the v12 gate test runs; run if v12 misses the gate (user, 18:00: then go straight
to the next version), otherwise as the improved system's own held-out evaluation.

## Gate (user, 2026-10-28)
At least ONE configuration reaches an Oracle success rate >= 95% over every test episode not PROVEN natively
infeasible (K=10 acceptable; the others need reasonable performance).  Official Falcon rules (`OFFICIAL_RULES.md`);
verdict = the authoritative continuous replay with the unchanged `habitat.Env.step` reproduces the selected actions
and the official `success` measure is 1.  Also reported: SPL, PSC, human collisions, route quality (turn reversals,
backtracking), Oracle search cost, decision robustness -- and, for v13, SR without the fallback phase and the share
of primitive (planner-bypassing) decisions.

## Frozen system
Tag `querygen-freeze-v13` = commit 95585f5f, querygen package sha256
`d97b30b3c83e550bb2bc74652355c2a54f01a8809ef417ab7013c12522e922b9` -- exactly the code of the v13 development runs
(`output/runs/official/v13fb`; the identity runs `v13id` differ only in the takeover default, which they never
reached).  Launcher `scripts/querygen_test_v13.sh CONFIG OUT_ROOT [WORKERS]` (extracts the tagged package with
`git archive`, refuses a hash mismatch).

v13 = v12 (FINAL_PROTOCOL_V12.md, unchanged search: the generator's primitive queries are never expanded by the
ordinary search; the fast legal sensor is exact) plus
* `QG_FAST_SENSE=1` -- exact fast legal sensor (`fastsense.py`; identical rows, trackers and certified cells);
* generator `primitives=True` -- exact one-step native queries prim_f / prim_l / prim_r / prim_s (outside K);
* Oracle `prim_fallback_sims=400000` -- after a failed ordinary search, a native witness search (privileged
  selection; true human tape) from the reset, executed as primitive queries through the search runtime;
  `prim_takeover` off (the pure witness: VqCaAuuoeWk's takeover saved 30 of 94 primitive decisions for SPL 0.989 ->
  0.909).

| id | role | planner | generator |
|---|---|---|---|
| C1-K10 | primary | ORCA | v12 C1-K10 + primitives |
| C1-K6 | secondary | ORCA | v12 C1-K6 + primitives |
| C2-K10 | secondary | DWA | v12 C2-K10 + primitives |
| Floor-C1 / Floor-C2 | default policy | ORCA / DWA | no selection |

## Development evidence (before the freeze)
* v13 ordinary search == v12 search: 12 solved development episodes re-run with the v13 code (fast sensor, primitive
  queries offered) give identical decisions, edge counts, steps and official metrics
  (`tests/querygen/compare_runs.py output/runs/official/v12 output/runs/official/v13id`); the fast sensor is
  identical observation by observation (`tests/querygen/test_fastsense.py`, 5 episodes, 2,196 observations).
  Hence v13's development results are v12's, plus the fallback on v12's failures.
* Fallback (ordinary search skipped, `output/runs/official/v13fb`): VqCaAuuoeWk (v12: 12,000-edge failure) PASS
  SPL 0.989; Nfvxx8J5NCo PASS SPL 1.0; ANmWrL7Kz7h (v10: 12,000-edge failure) PASS SPL 0.959; others running.

## Test set
Fifth unseen suite `docs/querygen/suites_unseen5.json` (56 HM3D + the last 4 unused MP3D Falcon train scenes, salt
`querygen-unseen5-train-scenes-20261028`, disjoint from DEV / HOLDOUT / unseen 1-4, sha256 0f260cdf...), frozen at
commit 1bda275b before any post-v12 development and never run or inspected.

## Accounting
As FINAL_PROTOCOL_V10_OFFICIAL.md: unsolved test episodes get a native witness search, then the exact native
certificate; only an officially verified certificate removes an episode from the denominator.
