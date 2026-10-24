# Development failures and retained evidence

All paths below are relative to `output/l4_social_executor_v1_20261022/`.
Failed runs remain available. Neither evaluator thresholds nor the fixed
scenario membership were changed to remove failures.

| Evidence | Diagnosis | Resolution or remaining limit |
|---|---|---|
| `baseline_smoke_attempt1.log`, `baseline_smoke.json` | An inherited prefix did not produce a partial-map motion query; it also contacted static geometry. | Exercised the existing native orientation compiler and a separate legal visible-human encounter. Prefix evidence establishes protocol operation, not safe navigation. |
| `analytic_dev02`–`analytic_dev04` | Beam pruning removed sustained turning/avoidance; a short lookahead and endpoint-saturated progress created loops. | Preserved fully checked turn/advance/rejoin proposals; used the next ordered waypoint; counted new best endpoint distance as rejoining progress. |
| `native_dev04`, `headon_dev06` | A short safe beam prefix could dominate a complete avoidance maneuver and lead to blocked head-on behavior. | Complete native maneuver proposals repaired the fixed native head-on case. |
| `benchmark_dev06` | Configuration-space squares and a single floor approximation falsely rejected many native routes. | Added bounded native mesh/FK proposals, with legal measured ego articulation. Multi-floor and tight-start failures remain. |
| `benchmark_static_dev07`, `benchmark_static_dev08` | Full-gait union boxes were overly restrictive at tight poses. | Used current/next acknowledged gait phases and isolated native FK; retained conservative failure when no checked action exists. |
| `falcon_probe`, `falcon_probe02` | EGL driver library mismatch, then auxiliary-head configuration incompatible with the supplied no-aux weights. | Process-local matching cached NVIDIA libraries; disabled only absent auxiliary training heads, then strict-loaded every supplied policy tensor. No system driver changes. |
| `profile_dev09`, `profile_final` | Repeated disk checks in wholly supported regions caused dense-fixture P95 over 10 ms. | Exact enclosing-region shortcut, with obstacle/height checks over every enclosing cell. Added an independent dense pose-sampling regression. |
| `final_scenarios/scenario_static_detour_B1` and B2 | A 5 mm mesh margin missed a native arm contact at action 11. The contact witness is retained. | Offline replay rejected this same action at 15, 25, 40 and 60 mm. Restored the inherited 25 mm mesh margin. This is a measured regression repair, not a global proof. |
| `static_margin025_dev10` | Margin repair alone avoided contact but turned without completing. | A bounded forward-before-turn proposal and shorter four-turn escape were tested next. |
| `static_escape_dev11` | The native constrained starting pose needed translation before turning toward its route. | Completed in 81 native actions with zero human/static contact, under the original query and budget. The final evaluation repeats the whole frozen suite. |
| `escape_public_dev12` | The same escape extension did not solve the two tested HM3D public roots. | Keep these failures in the navigation denominator. Extending arbitrary timeouts is not an accepted repair. |
| `final_analytic/random_10.json`, `random_26.json`, `random_50.json` | Three slower-leader variants failed to rejoin before the meaningful-progress watchdog. | Unresolved. Fixed-suite success does not establish randomized completion. |
| `release_navigation` | Eight/twelve L4 queries block; two collide with humans while their legal packets have no tracks; one goal is outside the requested certified surface. | Unresolved public-panel failures. No zero-occupancy safety inference, successful handoff claim or dropped denominator is allowed. |
| `mesh_profile` | At the slow retained actual state, triangle/link checks consume 85.6% of a 390 ms mesh-proposal replay. | Normal-grid optimization is insufficient. Conservative mesh acceleration and progress recovery remain release blockers. |

The full public panel is a development evaluation after inspection of earlier
failures, not a held-out confirmation set. No reserved confirmation data were
read or modified. A larger independent evaluation and robust static support
are required before an RL-readiness claim.
