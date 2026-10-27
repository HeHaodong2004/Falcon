# v9 evaluation protocol (declared before any unseen-suite run of v9)

The earlier protocol (`FINAL_PROTOCOL.md`, frozen system `querygen-freeze-v1`) is closed: its HOLDOUT
split has been used.  v9 is evaluated on the **unseen-scene suite** frozen at commit 5c60e909
(`docs/querygen/suites_unseen.json`, 30 HM3D + 10 MP3D Falcon *train-split* scenes, none of them DEV or
HOLDOUT scenes; sha256 recorded in `suites.py`).  It was never used for development.  Only aggregate
counts of the privileged witness search on it were looked at before the freeze (25 verified witnesses,
15 not found); no per-episode inspection, no certificate.

## Frozen system (tag `querygen-freeze-v9`, set before the unseen runs)
Configurations (planner x dynamics), each with the same query generator code:

| id | planner | dynamics | generator config | Oracle config |
|---|---|---|---|---|
| C1-K10 | ORCA (full responsibility, v9 executor) | native | `tc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False`, budget 10 (quota) | `heading_h`, `escape_penalty=80`, `escape_h`, `prefix_cache=48`, 6000 edges |
| C1-K6 | ORCA | native | `select=compact`, budget 6, `tc_variants=(st_goal,)`, `st_goal`, `maneuver_spawn_only=False` | as C1 |
| C2-K10 | DWA | inertia | as C1-K10 (st_goal and close passes are automatically unavailable for momentum dynamics; tight variants remain) | as C1 |

Launcher: `scripts/querygen_final_v9.sh SUITE CONFIG OUT_ROOT` (refuses a dirty package).
Everything else at the code defaults of the tag (including the v9 selection rules and the 6000-edge
deterministic budget; wall-clock cap 9000 s as a safety net only).  Every run executes a frozen copy of
the package (`<run>/code_snapshot`).

## Verdict (unchanged since the harness was built)
PASS iff the Oracle found a complete branch of generator queries and the authoritative continuous
replay from reset with the unchanged `habitat.Env.step` reproduces the branch actions exactly, ends with
official success (STOP, geodesic < 0.2 m, <= 500 steps), official human collision 0, zero body-proxy
overlap steps (robot rectangle >= 0.25 m from same-floor human centres), zero non-exempt static-contact
steps (declared spawn-escape rule), bitwise shadow-pose agreement, no canonical human contact.

## Feasibility accounting (privileged, after the freeze)
- *witness-proven feasible*: a verified witness (`witness.py`, exact primitives, Env.step-verified) exists;
- *certified infeasible*: a complete certificate (`feasibility.py` start escape / exact-gait progress /
  goal region with > 0 test points);
- *unresolved*: neither.  Witness searches for unresolved episodes may be extended (longer budgets) but
  never use the system under test.
Certificates for the unseen suite are computed only after the freeze
(`scripts/querygen_unseen_certificates.sh`).

## Reporting
SR over witness-proven feasible episodes (primary), strict SR over all episodes, certified-infeasible and
unresolved counts listed with evidence; per configuration and K: SPL, steps, minimum human clearance,
Oracle edges and wall time to solution (P50/P95), generator latency, cached-step fraction.
