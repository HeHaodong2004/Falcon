# Query Generator + privileged Oracle on native Falcon: final report

Worktree `eventwm-falcon-querygen-claude-20261026`, branch `research/querygen-oracle-20261026`,
frozen system tag `querygen-freeze-v1` (commit 0d9eabb6). Protocol: `FINAL_PROTOCOL.md` (declared
before HOLDOUT). Iteration log: `PROGRESS.md`. Generator / Oracle specifications: `GENERATOR.md`,
`ORACLE.md`. All numbers below are copied from run artefacts under `output/runs/` (`report.py`,
study `summary.json` files).

## 1. Verdict: **REVISE**
The primary hard gate (100% collision-free, replay-verified full-episode success with
full-responsibility ORCA, native dynamics and generator-only queries on the frozen suites) is **not
met**:

| suite (frozen system `querygen-freeze-v1`) | PASS | certified infeasible | PASS on feasible | other outcomes |
|---|---|---|---|---|
| DEV (28 HM3D) | 14/28 | 4 | **14/24 (58%)** | 10 FAIL_SEARCH |
| HOLDOUT (28 HM3D + 11 MP3D, untouched) | 19/39 | 3 | **19/36 (53%)** | 15 FAIL_SEARCH, 2 FAIL_REPLAY |

Every PASS is a continuous replay from reset with the unchanged `Env.step`: official success, 0
human collisions, 0 static collisions, 0 body-proxy overlaps, actions and shadow pose identical to the
Oracle branch. Across both suites there are **0 human-collision episodes** and **0 static-collision
steps** in any accepted run.

Why REVISE and not BLOCKED: (a) the literal gate is unattainable for *any* policy on these suites -
7 of 67 episodes are certifiably infeasible for Spot (section 5) - so the gate must be restated over
certifiably feasible episodes; (b) on the feasible episodes the remaining gap is within this track's
control: Oracle search budget for human-timing puzzles and long confined/static routes (most
FAIL_SEARCH), one generator gap (no candidate survives the reset in a tight spawn, HOLDOUT
vBMLrTe4uLA), and one harness isolation defect (shared private sensor cache made 2 HOLDOUT Oracle
branches irreproducible -> FAIL_REPLAY; confirmed and fixed). The post-freeze revision v7b fixes the
generator gap, the isolation defect and an Oracle near-goal merge defect; on DEV it reaches 15/24
feasible (+1, no regressions), but it has **not** been evaluated on an untouched split (HOLDOUT is now
used), so no gate claim is made for it.

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

## 7. Frozen-system results: DEV final and untouched HOLDOUT (deliverable 8)
Frozen system `querygen-freeze-v1`, Oracle defaults (`--oracle '{}'`), 6000 simulated edges per
episode, byte-identical code snapshots for both suites. Full per-episode tables: `output/runs/
dev_final/report.md`, `output/runs/holdout_final/report.md`.

**DEV final: 14/28 PASS; 14/24 on certifiably feasible episodes** (DEV v3: 11/24). All 14 PASS
replays are bit-identical to the Oracle branches, official success, 0 human collisions, 0 static
collisions (escape-exempt spawn contacts only in auFeVz9Go4m, hkr2MGpHD6B, vd3HHTEpmyA), 0 body
overlaps; SPL 0.72-1.00 (mean in report), minimum rectangle clearance to humans 0.43-2.81 m, no
deadlock, mean 2.6 waiting steps, 3.0 turn reversals / 100 steps, 0.09 m geodesic regress.
Failures (all FAIL_SEARCH, none FAIL_REPLAY): 4 certified infeasible; the 10 others exhaust the
6000-edge budget - human timing in narrow passages (Nfvxx8J5NCo 615 overlap-killed edges, Qpor2mEya8F
154 human collisions + 471 overlaps, 3t8DB4Uzvkt, TEEsavR23oF, SByzJLxpRGn), near-goal endings
(rsggHU7g7dh reached 0.07 m but the state was merged before trying STOP - an Oracle defect, fixed in
v7b; bCPU9suPUw9 0.44 m, LEFTm3JecaC 0.46 m, bzCsHPLDztK 0.81 m) and one long confined escape
(YRUkbU5xsYj).

**HOLDOUT final (untouched until the freeze; run once): 19/39 PASS; 19/36 on certifiably feasible
episodes.** HM3D 13/28, MP3D 6/11. PASS episodes: mean SPL 0.82, mean official PSC 0.985, minimum
rectangle clearance 0.31 m, mean 6.3 s, no deadlock, 0.9 waiting steps, 2.75 turn reversals / 100
steps, 0.35 m geodesic regress. Failures:
- certified infeasible (3): cYkrGrCg2kB/14 and dHwjuKfkRUR/18 (start-escape: the robot can move at most
  0.41 / 0.63 m without a declared collision), kJJyRFXVpx2/17 (progress certificate: no reachable
  state, of 4,419 with exact gait, is closer to the goal than the spawn);
- **FAIL_REPLAY (2): X7gTkoDHViv/2 and mp3d Z6MFQCViBuw/6** - the Oracle found a success but the
  authoritative replay diverged (111 vs 112 steps; "cannot continue a finished execution"). Root cause
  confirmed by deterministic reproduction: the legal sensor's private support/probe cache is shared by
  all branches and keyed by 0.1 m height layer while each certificate is computed at the exact height
  of the first request; on sloped floors (base height -0.02..0.18 m here) a branch's legal map
  depends on the search history. Re-running the Oracle reproduces the 112-step branch; re-executing it
  in the same process gives 112 steps, in a fresh process 111. With branch-pure caches (v7) both give
  110 (identical). Z6MFQCViBuw/6 first crashed in the driver (no result written); it was re-run once
  from the frozen snapshot with only the result-recording fix (the search is deterministic and
  re-found its 104-step branch);
- generator gap (1): vBMLrTe4uLA/21 - all 10 reset candidates collide (even waiting: it alternates
  in-place turns and the first right turn hits a hind leg), so no child exists from which manoeuvre
  mode can start; two plain forward steps (a progress witness) are collision-free (fixed in v7);
- budget (14): human timing in narrow passages (5hGmubEE6ZF 558 overlap-killed edges, TbHJrupSAjP
  1,201, y9hTuugGdiq, 8CRYizAb6yd, h1zeeAwLh9Z, QaLdnwvtxbs), long static routes with Spot-tight
  passages (BHXhpBwSMLh 17.3 m, SUHsP6z2gcJ 20.0 m, dVW2D7TDctW 21.3 m, X7HyMhZNoso 13.6 m: only
  2-5 m gained), near-goal endings (q5QZSEeHe5g 0.37 m, wcojb4TFT35 0.40 m, x8F5xyUWy9e 0.38 m);
  mp3d zsNo4HB9uLZ/42 is the only episode whose search stopped at the 9000 s safety cap (4,052 of 6,000
  edges at 2.2 s/edge in a large MP3D scene), so that one result is load-dependent.

Runtime (HOLDOUT, shared 36-core machine at load 25-37): generator 39 / 72 ms per call (P50/P95,
107,527 calls), Oracle edge 0.56 / 1.82 s (135,024 edges), Oracle per episode 2,382 / 5,548 s,
replay 56 / 97 s. Queries: 12.2 raw / 9.5 kept per decision node.

**Post-freeze revision v7b on DEV (not a held-out result):** 15/28 PASS, 15/24 feasible;
rsggHU7g7dh/12 now passes (goal test), no other change in verdicts.

## 8. Commits and reproduction (deliverable 9)
Branch `research/querygen-oracle-20261026`:

| commit | content |
|---|---|
| 32144298 | native Falcon harness, exact branch/replay equivalence, ORCA executor stack |
| 5f409979 | frozen DEV / HOLDOUT suites (before any generator iteration) |
| f97d2acd, 1e451068, efda35d3 | generator v0-v2, Oracle, evaluation driver, dynamics families, DWA |
| 4aaff8cb | canonical contacts, link-set escape rule, shadow replay, feasibility certificates (DEV v3) |
| e23c1fd4 | replay traces, behaviour/runtime report, regret summary, vectorised sandbox scoring |
| 13423b6e | lattice heading-bin fix, manoeuvres, wait_clear, certificates, calibration, L3 interface, protocol |
| 154135f3, 9fd6298f | generator v6 manoeuvre mode; space-time heuristic, manoeuvre gate |
| **0d9eabb6 (tag `querygen-freeze-v1`)** | **frozen system used for DEV final and HOLDOUT** |
| 55b60cad, b21d56d1, 1dae0fb0, e71ea4e8, 48da3d82 | post-freeze: pure caches, root probes (v7), goal test (v7b), replay-exception recording, vacuous-certificate fix |

Run artefacts (outside git): `output/runs/{dev_final,holdout_final,dev_v7b}` (result JSONs with replay
traces, `report.md`, `code_snapshot/`), `output/runs/feasibility_v2` (DEV certificates),
`output/runs/feasibility_holdout`, `output/runs/calibration`, study directories. Commands: Appendix A.

## 9. Readiness for Layer 3 (deliverable 10)
**Interface.** `l3_interface.py` defines the per-query consequence features a learned selector would
receive (all legal): query descriptors (kind, category, via distance/bearing, duration), sandbox
progress on the legal cost-to-go, completion / STOP, effort (steps, path, in-place turns), social risk
over M predicted futures (P[overlap], P5/P50 of the minimum rectangle clearance, PSC exposure) and
static risk on the legal map (max body cost and its increase, share of poses touching UNKNOWN). One
nominal sandbox rollout per query plus vectorised scoring costs ~0.2 s per query (section 6). The
feature validation against native outcomes was started but stopped to free CPU for the final runs; the
only completed sample (5 states of one episode, no humans tracked at those states) shows the obvious
limit of legal features: native failures there involved humans or obstacles that were not yet
observed, so legal risk features cannot predict them (AUC 0.5) - Layer 3 must be trained with that
partial observability in mind.

**What is ready.** (1) A legal, deterministic generator whose candidate sets contain a complete,
replay-verified successful sequence for 14/24 feasible DEV episodes and the HOLDOUT passes (section 7);
(2) an exact native harness with branch == Env.step equivalence, canonical contacts and authoritative
replay, now known to need branch-pure sensor caches (v7) for strict isolation on sloped floors;
(3) privileged Oracle labels (query sequences, per-node candidate sets with native outcomes in
`stats`) for supervised selection; (4) evidence that the selector must condition on planner and
dynamics (8-10% failure regret, 36-44% material regret); (5) a validated 15x-cheaper consequence
estimator for ranking.

**What is not ready.** The Oracle does not find solutions for ~40% of feasible DEV episodes within
6000 edges (human timing in narrow passages, long confined escapes), so Oracle labels are biased
toward easier episodes; the frozen generator has a root-collision gap (vBMLrTe4uLA) and the frozen
harness a sensor-cache isolation defect (X7gTkoDHViv) - both fixed in revision v7/v7b but not yet
validated on an untouched split; and ~10-15% of Falcon episodes are certifiably infeasible for Spot,
so any 100%-of-suite gate must first filter episodes by certificate.

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
