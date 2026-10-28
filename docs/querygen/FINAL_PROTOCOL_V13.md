# Gate test: frozen v13 system under the OFFICIAL Falcon rules (DRAFT -- declared before any run on the fifth unseen suite)

Drafted 2026-10-28 ~18:30 while the v12 gate test runs; binding only if v12 misses the gate (user, 18:00: then go
straight to the next version) and only once the tag below exists.

## Gate (user, 2026-10-28)
At least ONE configuration reaches an Oracle success rate >= 95% over every test episode not PROVEN natively
infeasible (K=10 acceptable; the others need reasonable performance).  Official Falcon rules (`OFFICIAL_RULES.md`);
verdict = the authoritative continuous replay with the unchanged `habitat.Env.step` reproduces the selected actions
and the official `success` measure is 1.  Also reported: SPL, PSC, human collisions, route quality (turn reversals,
backtracking), Oracle search cost, decision robustness -- and, for v13, SR without the fallback phase and the share
of primitive (planner-bypassing) decisions.

## Frozen system
Tag `querygen-freeze-v13` (to be created), launcher `scripts/querygen_test_v13.sh CONFIG OUT_ROOT [WORKERS]`
(extracts the tagged package with `git archive`, refuses a hash mismatch).

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

## Test set
Fifth unseen suite `docs/querygen/suites_unseen5.json` (56 HM3D + the last 4 unused MP3D Falcon train scenes, salt
`querygen-unseen5-train-scenes-20261028`, disjoint from DEV / HOLDOUT / unseen 1-4, sha256 0f260cdf...), frozen at
commit 1bda275b before any post-v12 development and never run or inspected.

## Accounting
As FINAL_PROTOCOL_V10_OFFICIAL.md: unsolved test episodes get a native witness search, then the exact native
certificate; only an officially verified certificate removes an episode from the denominator.
