# Query Generator / Planner / Oracle track — progress record

Worktree `eventwm-falcon-querygen-claude-20261026`, branch `research/querygen-oracle-20261026`.
Package `eventwm_graph/querygen/`; runner `scripts/querygen_run.py <module> [args]`
(python `/home/haodong/miniforge3/envs/eventwm-falcon/bin/python`, `CUDA_VISIBLE_DEVICES=1`,
`nice -n 10`; Habitat runs headless/CPU). Outputs: `output -> /var/tmp/falcon-querygen-claude`
(`/home` is 93% full). Isolated `data/` (symlinks to Falcon data; own `navmeshes/` cache).
No Layer-3 training, no Stage 1/2 retraining, no world-model changes; other worktrees read-only.

## Declared evaluation rules (frozen before DEV iteration unless marked)
- Benchmark: original Falcon `social_nav_v2/falcon_{hm3d,mp3d}.yaml`, unchanged sources, Spot,
  discrete native actions (forward 25/120 m nominal, turns +-1/12 rad, 0.048 s/step, no pause),
  500-step limit, success = STOP with geodesic < 0.2 m.
- Hard-gate PASS per episode (`evaluate.py`): Oracle found a complete successful branch built only
  from generator queries; continuous replay from reset with the unchanged `habitat.Env.step`
  re-executes the *query sequence* through the live executor and reproduces the branch actions
  exactly; official success 1, official human collision 0; zero body-proxy overlap steps; zero
  static-collision steps; lockstep shadow simulator pose equal bitwise at every step; no
  canonical human contact.
- Body proxy (declared at harness time): the official humanoid collider is a 2 cm stick skeleton,
  so the robot's all-gait rectangle must stay >= 0.25 m (BODY_PROXY_M) from same-floor human centres.
- Static collision: non-support robot-scene contact points (same classification as
  `partial.metrics.contact_labels`: foot contacts with floor-like normals or unobstructed low
  lateral normals are support). **Canonical, history-free detection** (rev. 4aaff8cb): Bullet keeps
  persistent manifolds across kinematic teleports, so a contact query can report stale points from
  the previous branch; we move the robot far away, detect, restore the exact target, detect again.
- Spawn-contact escape rule (rev. 4aaff8cb, DEV-stage revision): let C0 = links in static contact at
  reset; during the first 24 steps and 0.3 m of travel a contact step is exempt iff all contacting
  links are in C0. Any other contact is a collision.

## Verified facts (with evidence)
- Falcon humans are exogenous (avoid only agents 1..N): human trajectories under random robot
  actions are bitwise identical to a robot-free tape.
- Branch simulation (tape + exact robot setter replay + joint-limit clamping) is bitwise
  equivalent to Env.step: `tests/querygen/native_equivalence.py` -> ALL_EQUIVALENT.
- Pitfalls: float32 transform write-back is lossy -> restore from setter inputs; physics clamps
  Spot arm joint1 (-3.1686 -> -pi); `np.asarray(mn.Matrix4)` is row-major; Magnum teardown aborts
  -> runner hard-exits after flushing; Bullet persistent manifolds -> canonical contacts
  (`tests/querygen/test_canonical_contacts.py`).
- Public Falcon `ORCAPolicy` is not canonical ORCA (uses a privileged oracle path). Our ORCA is an
  RVO2 port with full responsibility (robot takes the whole avoidance), min distance = combined
  radius in 300 random holonomic scenarios (`tests/querygen/test_orca.py`).
- Velocity action (continuous base velocity through the same `update_base` path) leaves native
  behaviour bitwise unchanged (`tests/querygen/test_velocity_action.py`); used only by non-native
  dynamics families.

## Implementation map
`native.py` harness/branch sim/contacts; `legal.py` legal map + tracks (UNKNOWN stays UNKNOWN);
`gridplan.py` Dijkstra/routes; `lattice.py` body-aware hybrid-A* static layer; `control.py` body
checks + native duty-cycle adapter; `orca.py`, `planners.py` (ORCA full responsibility, SFM, DWA
with exact-dynamics rollouts, path follower); `dynamics.py` families; `motion.py` speed realiser;
`executor.py` query semantics; `episode.py` branch runtime + authoritative replay; `generator.py`
deterministic legal query generator; `oracle.py` privileged best-first search over generator
queries; `evaluate.py` suites/verdicts; `feasibility.py` start-escape certificates; `diversity.py`
planner/dynamics regret; `sandbox.py`/`sandbox_study.py` nominal vs closed-loop futures;
`report.py` metric aggregation.

## Results log
- DEV v0 (generator v0): 8/23 PASS at 23 done (pre escape rule).
- DEV v2 (stopped at 15/28, 6 PASS): contaminated by stale Bullet contacts and the old
  first-free-step escape rule -> motivated canonical contacts + link-set escape rule.
- Executor-only (goal query only, ORCA/native, `output/runs/exec_orca_native_v2`, mixed code
  across two launches): 2/28 strict PASS, 2 more native-success-only; failures = traps revealed
  by the map, human jamming, timeouts. The executor alone is weak; selection matters.
- Start-feasibility certificates (`feasibility.py`, complete BFS over native primitives with
  canonical contacts and the escape rule): **yr17PDCnDDW/5 and 7Ukhou1GxYi/14 are infeasible**
  (spawned facing an obstacle at arm height with the arm already touching it; forward drives
  base+front legs into it, 2-6 turn steps either way bring legs into contact; search complete with
  25 / 4 reachable states, 0 m contact-free displacement). Contact-ignoring BFS escapes only by
  scraping (`tests/querygen/probe_wedge.py`). All other certified DEV episodes are feasible.
- DEV v3 (commit 4aaff8cb, Oracle 1800 s / 5000 edges, ORCA/native): **11/28 PASS** (11/25 on
  certifiably feasible episodes; 3 certified infeasible: yr17PDCnDDW/5, 7Ukhou1GxYi/14,
  Dd4bFSTQ8gi/12 - spawned 1.4-9 cm inside static geometry, one forward step adds 8-12.5 cm).
  Zero replay failures: every Oracle success replayed bit-identically with 0 human collisions,
  0 static contacts, 0 overlaps. Failures are all search failures:
  * 3 stall at the start (YRUkbU5xsYj/10, vd3HHTEpmyA/8, 5cdEh9F2hJL/6): certified escapes exist
    but need exact primitive sequences (e.g. `r12 f l28 f6`, `r10 f r f r12 f4`); the generator
    had no single-step manoeuvres and the Oracle's 0.3 m / 45 deg de-duplication merged them.
  * 11 exhaust 1800 s: dominated by humans traversing/parking in the narrow passages the route
    needs (AWUFxHEyV3T: two humans walk the 1.2 m corridor head-on, steps 48-144; TEEsavR23oF: one
    head-on, one parks at the corridor end; Nfvxx8J5NCo/Qpor2mEya8F: humans park in the passage /
    corridor exit from ~step 90-175; LEFTm3JecaC: human walks through the near-wall goal area
    steps 84-170, best branch 0.236 m from the goal). Solutions need waiting at the right place;
    with f = g + 2h every wait step costs like 1/12 m of lost progress, so waits are explored last.
  * Search waste: 60-77% of simulated edges end in an already-seen signature; the legal sensor
    is ~80% of step time (~1500 rays/step re-testing occluded cells; inherent to the accepted
    observer, so not approximated).
- Goal-region certificates (`feasibility.py --goal`, all 75 headings x 5 gait phases on 4 cm
  points within the success region, canonical contacts; placement self-check reproduces the reset
  transform and contacts exactly): rsggHU7g7dh 97% of final poses contact-free, TEEsavR23oF 41%.
- Sandbox study (1 episode, 5 states, 50 queries, M=16 CV+noise futures): nominal-rollout +
  vectorised scoring vs closed-loop per-future rollouts: overlap-probability MAE 0.015 (corr
  0.72), median d_min MAE 0.007 m, top-choice agreement 100%, 166 vs 2464 ms/query (~15x);
  sandbox vs native with true future: mean deviation 0.064 m, outcome agreement 100%.

- v4/v5 iteration (DEV only):
  * **Lattice bug** (present in all runs up to DEV v3): the static hybrid-A* keyed headings on 72
    bins of 5 deg while a native turn is 4.77 deg, so some single turns stayed in the parent's bin
    and were pruned as duplicates; rotation chains broke after ~50 deg (TEEsavR23oF reset: only 23
    reachable headings, 42,518 expansions without progress -> spurious 'stuck'). Fixed: >= 2 bins
    per turn step (`lattice.heading_bins`); the same query now finds a progressing plan in 3,813.
  * Exact manoeuvres (rotations of 1/4/10 turn steps, 1/3-step creeps; creeps like explicit
    rotations are not filtered by the dilated wall band) solve vd3HHTEpmyA/8 (PASS, 128 steps), but
    offered at every lattice-'stuck' node they exploded the search (HMkoS756sz6: 1,823 expansions,
    1,087 s vs 148 / 124 s without; ablation: block-time term irrelevant there). Now gated: only
    within 1 m of the start pose or the goal AND after the legal history shows a stall (no
    translation > 0.15 m and no net turn > 0.35 rad over the last <= 16 steps).
  * `wait_clear` query (hold until the legally predicted route conflict clears, <= 48 steps) and a
    privileged Oracle block-time term (smallest departure delay after which the true geodesic path
    is free of humans within 0.5 m for 8 m, from the human tape) so that necessary waits are not
    penalised as lost progress.
  * Progress witness search (`feasibility.py --progress`): 5cdEh9F2hJL/6 exhausted all 36,799
    reachable states (gait-merged) without any state closer to the goal than the spawn (the robot
    spawns facing away from the goal in a corridor too narrow to turn in; there is no reverse);
    an exact-gait rerun is running to make this a certificate.

- Manoeuvre gating experiments (HMkoS756sz6 = normal start needing a 180 deg turn; vd3HHTEpmyA =
  confined start): no manoeuvres 124 s / FAIL; lattice-gated near start/goal 174 s / FAIL; stall
  (translation+turn) 471 s / FAIL; stall (translation) 735 s / FAIL; anywhere (lattice or stall) +
  manoeuvre prior 10: slow / PASS 305 s. Edge-type fine de-dup (only states reached by a manoeuvre
  edge) keeps ordinary exploration coarse. Remaining cost: a 1-step manoeuvre adds only 1 step of g,
  so even with +10 it undercuts a 16-step ordinary decision -> `maneuver_pad` charges a full interval.
- Sandbox duplicate predictor (true human future, kinematic): deferred ~4,700 edges per run for
  94-133 s of prediction but did not solve AWUFxHEyV3T / TEEsavR23oF -> off.
- Block-time term alone: AWUFxHEyV3T search reaches the corridor entrance within 300 s, then spends
  1,500 s on an equal-f plateau of (position x time) states before the corridor (waiting keeps f
  constant because block time drops as g grows) -> optional bucketed-f tie-breaking toward later
  nodes (`tie_bucket`).
- MP3D smoke test on a *train* episode (no suite scene): pipeline runs; the episode itself is
  certified infeasible (49 reachable states, forward drives the arm wrist 7 cm into geometry at every
  reachable heading) -> Falcon spawn validity for Spot is a benchmark-level issue (DEV: 3 start
  certificates + 5cdEh9F2hJL progress search; MP3D train sample also infeasible).
- The Oracle's wall-clock budget makes results load-dependent (the machine reached load 36/36);
  final runs will bind on a deterministic edge budget (max_edges) with a generous wall-clock cap.

## Important failed hypotheses / lessons
- Velocity-tracking alone cannot handle Spot's 1.23 m body in tight spaces -> body-aware lattice.
- ORCA static horizon 0.4 s at native speed forbade all motion near walls (now 0.15 s).
- DWA long straight rollouts dither -> auto-stopping advances + commitment bonus.
- Executor-only ORCA cannot escape a human walking up a 1 m corridor toward it (90 deg rotation =
  19 steps); handled at query level (wait/yield before entering) + Oracle danger term.
- Over-triggered boxed-in maneuvers regressed rsggHU7g7dh/12 -> lattice-based stuck test.

## Next steps
1. Finish DEV v3; analyse failures; iterate generator (DEV only).
2. Diversity (regret) study across {orca,sfm,dwa} x dynamics; sandbox study across DEV states.
3. Freeze generator; certificates + one HOLDOUT run; final report (PASS/REVISE/BLOCKED).
