# Query Generator + privileged Oracle on native Falcon: final report

Worktree `eventwm-falcon-querygen-claude-20261026`, branch `research/querygen-oracle-20261026`,
frozen system tag `querygen-freeze-v1` (commit 0d9eabb6). Protocol: `FINAL_PROTOCOL.md` (declared
before HOLDOUT). Iteration log: `PROGRESS.md`. Generator / Oracle specifications: `GENERATOR.md`,
`ORACLE.md`. All numbers below are copied from run artefacts under `output/runs/` (`report.py`,
study `summary.json` files).

## 1. Verdict
<!-- FILLED AFTER HOLDOUT -->

## 2. What was built (deliverables 1-4)
**Native harness and legality.** The unchanged Falcon benchmark (`social_nav_v2/falcon_{hm3d,mp3d}`,
Spot, discrete native actions, 500 steps). A branch simulator (human tape + exact robot setter replay
+ joint clamping + canonical, history-free contact detection) is bitwise equivalent to `Env.step`
(`tests/querygen/native_equivalence.py`). The legal state is the project's ideal geometry observer
(certified FREE/BLOCKED cells; UNKNOWN stays UNKNOWN) plus legal human tracks; no future observations,
no ground-truth future humans, no unseen map ever reach the generator or the executor.

**Planners (deliverable 2).** Full-responsibility ORCA (RVO2 port; the robot takes all avoidance
responsibility; minimum distance = combined radius in 300 random holonomic scenarios,
`tests/querygen/test_orca.py`), social-force model, DWA with exact-dynamics rollouts, and a path
follower; all share the body-aware static lattice (hybrid A* over Spot's 1.23 x 0.49 m body on the
legal map; >= 2 heading bins per turn step after the bin bug fix) and the native duty-cycle adapter.
The public Falcon `ORCAPolicy` is not canonical ORCA (it uses a privileged oracle path) and is not used.

**Dynamics families and calibration (deliverable 3).** All families keep Falcon's structure
(translate straight or turn in place each 0.048 s step) and are executed through the same
`BaseVelAction.update_base` path (`native_velocity.py`; installing the velocity action leaves native
behaviour bitwise unchanged). Calibration in the native simulator on an open 4.4 m line
(`calibrate.py`, `output/runs/calibration/calibration.json`):

| family | v_max m/s | turn deg/step | steps to full speed | steps to stop | stop distance model / measured m | max per-step error |
|---|---|---|---|---|---|---|
| native | 4.34 | 4.77 | 1 | 1 | 0 / 0 | 1.8e-7 m |
| slowturn | 4.34 | 2.39 | 1 | 1 | 0 / 0 | 4.1e-7 m |
| inertia | 4.34 | 4.77 | 6 | 6 | 0.521 / 0.521 | 3.6e-7 m |
| slowturn_inertia | 4.34 | 2.39 | 6 | 6 | 0.521 / 0.521 | 4.4e-7 m |
| heavy_brake (held out) | 4.34 | 4.77 | 6 | 12 | 1.146 / 1.146 | 4.4e-7 m |
| slowturn_heavy (held out) | 4.34 | 1.59 | 6 | 12 | 1.146 / 1.146 | 4.7e-7 m |

Turn errors are 0 (float32-exact). Near walls the navmesh step filter shortens forward motion
(native benchmark behaviour); this is simulated, not modelled.

**Query generator (deliverable 1)** - `GENERATOR.md`. **Oracle harness (deliverable 4)** -
`ORACLE.md`; privileged selection only, real execution, per-branch isolated state, authoritative
continuous replay.

## 3. Replay evidence (deliverable 5)
Every result JSON (`output/runs/{dev_final,holdout_final}/*.json`) holds the replay block: official
metrics from the unchanged `Env.step` run from reset, `actions_identical`, shadow-simulator pose
mismatch count (must be 0), static / escape / body-overlap step counts, minimum rectangle clearance,
and the per-step trace (actions, pose, geodesic, clearance).

## 4. Iteration report (deliverable 6)
All iteration used DEV only; HOLDOUT was first run after the freeze.

| run | system | DEV result | main lesson |
|---|---|---|---|
| v0 | generator v0, best-first Oracle | 8/23 PASS (stopped) | spawn contacts need a declared escape rule; narrow passages with humans dominate |
| v2 | v1 candidates, lazy Oracle, escape rule | 6/15 (stopped) | Bullet persistent manifolds leaked stale contacts across branches -> canonical, history-free contact detection |
| v3 | canonical contacts, link-set escape rule, shadow replay | **11/28** (11/24 feasible) | every Oracle success replayed bit-exactly; failures: 3 confined starts, 11 budget exhaustions (humans traverse/park in the passages the route needs) |
| v4 | + exact manoeuvres anywhere, wait_clear, block-time | regression (stopped) | manoeuvres offered broadly explode the search (HMkoS756sz6 0 progress) |
| v5 | generator v6 (manoeuvre mode), deterministic budget | partial | confined starts solved (vd3HHTEpmyA) but wait-induced manoeuvre mode regressed auFeVz9Go4m |
| final | v6 + space-time heuristic + manoeuvre gate (`querygen-freeze-v1`) | see section 7 | |

Two bugs found and fixed on the way: (i) lossy float32 transform write-back / joint clamping /
stale Bullet manifolds (harness exactness), (ii) the static lattice keyed 4.77 deg turns on 5 deg bins,
pruning rotation chains after ~50 deg (affected the executor and the generator's stuck test in all
runs up to DEV v3). Single-change ablations (HMkoS756sz6, vd3HHTEpmyA, auFeVz9Go4m, AWUFxHEyV3T) are
recorded in PROGRESS.md; the decisive changes were the manoeuvre mode with an Oracle gate (confined
starts) and the privileged space-time cost-to-go (human timing: AWUFxHEyV3T solved for the first time).
Failed/abandoned: sandbox duplicate prediction (deferred ~4,700 duplicate edges per run but found no
new solutions), block-time term (creates equal-f plateaus while waiting). Not established either way:
deeper-first tie-breaking and a greedier weight (w = 5) were only tried together with the
broad-manoeuvre generator (no success there); the clean A/B was stopped to free CPU for the final
runs, so neither is part of the frozen system.

## 5. Certificates: benchmark spawns that make the gate unattainable
Falcon places Spot with a navmesh built for a 0.25 m-radius agent; Spot's body is 1.23 x 0.49 m plus a
stowed arm at 0.8-1.0 m height, and there is no reverse action. Complete privileged searches
(`feasibility.py`; exact native primitives, canonical contacts, the declared collision rules; states
merged only on 2 cm position x exact turn index x gait phase x escape flag) prove that some episodes
admit **no** collision-free success at all:

| episode | certificate | evidence |
|---|---|---|
| DEV yr17PDCnDDW/5 | start escape (25 reachable states, complete) | spawned with the arm wrist/finger 7.6-8.3 cm inside static geometry; one forward step drives base + front legs + arm into it (8.3 cm); turns add leg contacts |
| DEV 7Ukhou1GxYi/14 | start escape (4 states) | wrist 1.4 cm inside at spawn; forward 7.9 cm with base/legs; 2-3 turn steps either way bring a leg into contact |
| DEV Dd4bFSTQ8gi/12 | start escape (2 states) | base 3 cm and wrist 9 cm inside at spawn; forward 12.5 cm with 10 new links |
| DEV 5cdEh9F2hJL/6 | progress to the 500-step episode limit (88,377 states, exact gait, exhausted) | spawned facing away from the goal in a corridor too narrow to turn in; no state closer to the goal than the spawn is reachable |
| HOLDOUT cYkrGrCg2kB/14 | start escape (286 states, complete) | clear spawn inside a pocket: at most 0.41 m of collision-free displacement (blocked by arm, base, front legs) |
| HOLDOUT dHwjuKfkRUR/18 | start escape (289 states, complete) | clear spawn, at most 0.63 m collision-free displacement |

Goal-region certificates (4 cm grid in the success region x 75 headings x 5 gait phases) found
contact-free final poses for every DEV episode (min 1.7% of poses for 7Ukhou1GxYi, 3.2% for
Nfvxx8J5NCo). An MP3D *train* episode used only as an MP3D smoke test was also certified infeasible
(49 reachable states). The strict 100%-of-suite gate is therefore unattainable on both frozen suites
for any policy, independent of the generator.

## 6. Diversity, sandbox approximation, runtime (deliverable 7)
**Planner / dynamics diversity measured by regret** (`diversity.py`, `output/runs/div_dev_v3`; 19
matched decision states from 4 DEV episodes, the same legal candidate set executed natively under all
12 configurations {ORCA, SFM, DWA} x {native, slowturn, inertia, slowturn_inertia}; equal-horizon
value at H = 48 steps with goal continuation). For configurations a, b and q*_a, q*_b their preferred
queries: *fail* = q*_b fails natively under a although a has a feasible choice; *regret* =
V_a(q*_a) - V_a(q*_b) in metres for non-failing cases; *material* = failure or >= 0.25 m.

| pairs | n | P(fail) | mean regret (m) | P(material) | P(rank flip) |
|---|---|---|---|---|---|
| same dynamics, different planner | 452 | 0.080 | 0.39 | 0.36 | 0.54 |
| same planner, different dynamics | 678 | 0.097 | 0.45 | 0.41 | 0.62 |
| both differ | 1356 | 0.102 | 0.55 | 0.44 | 0.65 |

Candidate feasibility differs per configuration (share of candidates that survive H = 48: 0.74
SFM/native ... 0.87 DWA/slowturn_inertia). Rank flips overstate diversity (54-65%), but the regret
is material in 36-44% of states and a borrowed choice fails in 8-10%: a query selector must be
conditioned on the executing planner and dynamics, or use per-executor consequence features.

**Single nominal rollout + vectorised scoring vs multi-future closed loop** (`sandbox_study.py`,
`output/runs/sb_dev_v3`; 25 decision states with tracked humans, 7 DEV episodes, 238 queries, M = 16
legal constant-velocity futures). One nominal rollout of the real executor in the kinematic sandbox,
scored against all 16 futures with `score_vec` (identical to the loop reference to 4e-16, 75x faster),
versus 16 closed-loop rollouts (planner reacting to each future):

| metric | value |
|---|---|
| overlap-probability MAE / correlation | 0.037 / 0.69 (nominal is conservative: 0.045 vs 0.012) |
| median min-clearance MAE | 0.036 m |
| top-choice agreement | 96% |
| time per query, nominal+vectorised vs closed loop | 190 ms vs 2,894 ms (15.2x) |
| sandbox vs native with the true future: mean deviation / overlap agreement | 0.16 m / 97.5% |
| sandbox vs native time per query | 220 ms vs 1,277 ms (5.8x) |

The approximation is adequate for ranking (96% agreement) and conservative for risk; the measured
speed-up is real (end-to-end, same executor code).

## 7. Untouched HOLDOUT (deliverable 8)
<!-- filled -->

## 8. Commits and reproduction (deliverable 9)
<!-- filled -->

## 9. Readiness for Layer 3 (deliverable 10)
<!-- filled -->

## Appendix A. Reproduction commands
All commands run from the worktree root with
`PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python`, `CUDA_VISIBLE_DEVICES=1`, `nice -n 10`
(Habitat runs headless on CPU; `output -> /var/tmp/falcon-querygen-claude`).

```bash
# native equivalence / unit checks
$PY scripts/querygen_run.py tests.querygen.native_equivalence
$PY scripts/querygen_run.py tests.querygen.test_orca
$PY scripts/querygen_run.py tests.querygen.test_velocity_action
$PY scripts/querygen_run.py tests.querygen.test_canonical_contacts
$PY scripts/querygen_run.py tests.querygen.test_sandbox_score
# dynamics calibration
$PY scripts/querygen_run.py eventwm_graph.querygen.calibrate --out output/runs/calibration
# Oracle + replay on a suite with the frozen defaults (code snapshot per run, 6000-edge budget)
git checkout querygen-freeze-v1 -- eventwm_graph/querygen     # (or use the run's code_snapshot)
$PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite dev --out output/runs/dev_final --workers 9 --oracle '{}'
$PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite holdout --out output/runs/holdout_final --workers 9 --oracle '{}'
# executor-only baseline (goal query only)
$PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite dev --policy executor --planner orca \
    --dynamics native --out output/runs/<run> --workers 4
# certificates (privileged diagnostics)
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --depth 60          # start escape
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --goal              # goal region
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --progress 0.3 \
    --exact-gait --max-depth 500 --budget 14000                                                   # progress
# studies on a run's PASS records
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout diversity output/runs/<run> --out DIR
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout sandbox_study output/runs/<run> --out DIR
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout l3_interface output/runs/<run> --out DIR
# metrics
$PY scripts/querygen_run.py eventwm_graph.querygen.report output/runs/<run> --cert output/runs/feasibility_v2
```
