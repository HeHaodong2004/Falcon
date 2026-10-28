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

- DEV v4 (manoeuvres offered anywhere on lattice-or-stall, prior 10): regression - HMkoS756sz6 (124 s
  without manoeuvres) failed with zero progress; stopped after 5 failures.
- Generator v6 (commit 154135f3): explicit manoeuvre mode (entered after a strict legal stall near
  the start/goal; persists while the last decision was a manoeuvre). vd3HHTEpmyA PASS, XNeHsjL6nBB
  PASS, HMkoS756sz6 PASS (710 exp); YRUkbU5xsYj escape found (r8 f l24 then goal x3 to 5.5 m) but the
  budget went to start-region manoeuvres (weight 2 favours cheap early states).
  DEV v5 = v6 + block term + deterministic 6000-edge budget (running).
- **Space-time heuristic** (commit 9fd6298f, `st_heuristic`): backward DP over 0.2 m navmesh cells x
  500 steps with the true human tape (unsafe within 0.6 m), unit moves + waits -> earliest arrival.
  Built in 0.3-0.5 s. Waiting on the optimal schedule lowers h one-for-one, so the search dives
  forward in time. **AWUFxHEyV3T/0 PASS** (284 steps, 636 expansions, 3540 edges; failed in v0-v5).
- auFeVz9Go4m regressed in DEV v5 (84% of 1,685 expansions in start-region manoeuvres after a
  wait-induced stall; DEV v3 solved it in 91 expansions with an ordinary turn) -> privileged Oracle
  gate (`maneuver_gate`): entering manoeuvre mode costs +120 unless every ordinary movement query
  tried from the parent failed natively.

- **Freeze** `querygen-freeze-v1` (commit 0d9eabb6): generator v6 + Oracle defaults (space-time
  heuristic, manoeuvre gate 120, 6000-edge deterministic budget). DEV final and HOLDOUT final run
  from byte-identical snapshots. HOLDOUT certificates computed only after the freeze.
- HOLDOUT findings (no system change made after them):
  * cYkrGrCg2kB/14, dHwjuKfkRUR/18: start-escape certificates (complete, 286/289 states) -> infeasible.
  * kJJyRFXVpx2/17: exact-gait progress search exhausted (4,419 states, max depth 68) without any
    state closer to the goal than the spawn -> infeasible.
  * vBMLrTe4uLA/21: **generator gap** - all 10 root candidates (goal, turns, sidesteps, alt route,
    frontiers and both waits = alternating in-place turns) collide (first right turn hits the hind
    leg); no candidate survives, so manoeuvre mode (needs a surviving stalled child) is never offered.
    The progress witness `ff` (two forward steps) reduces the geodesic 3.51 -> 3.14 m. Fix for a
    revision: offer a 1-step creep (and 1-step rotations) when the robot has no motion history.
  * X7gTkoDHViv/2: **FAIL_REPLAY** - the Oracle branch (112 steps) and the authoritative replay (111
    steps, official success, 0 collisions/overlaps/static, shadow pose identical) differ by one STOP
    step; re-executing the same decisions in a fresh branch simulator gives 111 = replay, so the Oracle
    branch was influenced by search history. Suspect: the legal sensor's private support/probe caches
    are keyed by 0.1 m height layer but computed at the exact base height of the first request; this
    episode's floor varies by 0.2 m (-0.02..0.18 m base heights), so a cached certificate can depend on
    which branch asked first. **Confirmed** (`tests/querygen/repro_replay_divergence.py`): the
    deterministic Oracle re-finds the identical 112-step branch; re-executing its decisions in the same
    process (search-populated private cache) gives 112 steps, in a fresh process 111. Branch-pure caches
    (`LegalSensor(pure_caches=True)`: 1 cm height quantisation + per-height namespace) implemented and
    verified identical to the accepted observer on a flat floor.

- **Final results (frozen `querygen-freeze-v1`)**: DEV final 14/28 PASS (14/24 feasible), HOLDOUT
  19/39 PASS (19/36 feasible; HM3D 13/28, MP3D 6/11; 2 FAIL_REPLAY from the cache-isolation defect;
  1 generator gap; 14 budget). 0 human-collision episodes, 0 static-collision steps in all accepted
  runs. Verdict REVISE (see REPORT.md). Post-freeze revision v7b (root probes, pure caches, goal test)
  on DEV: 15/28 (15/24), rsggHU7g7dh fixed; v7 pure caches make the X7gTkoDHViv Oracle branch
  reproducible in a fresh process (110 == 110 steps).
- Certificate semantics fix: a goal-region check with zero test points (q5QZSEeHe5g/0) is
  undetermined, not a proof.

## Important failed hypotheses / lessons
- Velocity-tracking alone cannot handle Spot's 1.23 m body in tight spaces -> body-aware lattice.
- ORCA static horizon 0.4 s at native speed forbade all motion near walls (now 0.15 s).
- DWA long straight rollouts dither -> auto-stopping advances + commitment bonus.
- Executor-only ORCA cannot escape a human walking up a 1 m corridor toward it (90 deg rotation =
  19 steps); handled at query level (wait/yield before entering) + Oracle danger term.
- Over-triggered boxed-in maneuvers regressed rsggHU7g7dh/12 -> lattice-based stuck test.

## Clarified objective (user, 2026-10-27)
Evaluate and improve the Query Generator's *induced search space* via privileged Oracle tree search:
does it contain safe successful routes, how efficiently can the Oracle find them, how good are they;
compare simpler generators under identical budgets and other planners/dynamics; separate generator
limitations from search-budget failures and certified infeasibility. Not Layer 3 / RL (the
`l3_interface.py` study is a finished side artefact, not pursued further).

Search-tree metrics (report.py): DEV frozen 14/28 solved, success@edges 500:6 1000:9 2000:13
3000:13 6000:14; median 603 edges / 218 nodes / 8 decisions to a solution; effective branching
factor 1.52; 63% of non-failing edges land in already-seen states; 20% of simulated edges end in a
static contact, 3% in human contact/overlap. HOLDOUT frozen: Oracle solved 21/39 (19 replay-verified);
17 budget-limited, 1 exhausted (vBMLrTe4uLA generator gap). Executor-only (goal query, no search,
frozen code): 3/24 feasible DEV pass (7 official successes, 8 human-collision episodes, 13 timeouts).

Running: generator ablations at an identical 3000-edge budget on DEV (goal-only; goal + waits; then
spatial-only; full minus manoeuvres), then planners/dynamics (SFM, DWA, ORCA x inertia/slowturn),
then budget scaling of representative DEV failures.

## Phase 2 (user, 2026-10-27): >= 95% Oracle success on genuinely feasible episodes, K ~ 6
- **Privileged witness planner** (`witness.py`, commit 9f2aedc3): weighted A* over exactly simulated
  native primitives (no legal sensor, ~6 ms/step) with the space-time cost-to-go + in-place turn steps
  to the geodesic direction; every witness is re-executed from reset with Env.step + shadow simulator
  (all declared rules). DEV: verified witnesses for every episode that is not certified infeasible,
  including all Oracle failures of the frozen system (Nfvxx8J5NCo 142 steps, Qpor2mEya8F 109,
  TEEsavR23oF 134, LEFTm3JecaC 55, bzCsHPLDztK 239, bCPU9suPUw9 106, YRUkbU5xsYj 130), typically in
  1-4 min (max 17 min). => the ~60% Oracle success was a search limitation, not infeasibility.
- **Witness-guided Oracle** (`OracleConfig.witness_dir`): h = max(geodesic, space-time, h_w) where h_w
  is the remaining time when joining the verified witness (late joins penalised). Still selects only
  generator queries; generator and partial maps unchanged (legal).
- **Compact generator** (`select='compact'`, K = 6): goal/continue + one hold + situational slots
  (manoeuvres when stalled; pass/yield on conflict; approaches near the goal; turn; advance / alt route
  / frontier / retreat) + a root forward probe.
- **Unseen-scene suite frozen** (`docs/querygen/suites_unseen.json`, 30 HM3D + 10 MP3D train-split
  scenes, sha256 a5704e8b..., commit 5c60e909) for generalization; not used for development.

- Guided Oracle (witness_dir) interim (v7b generator): K=10 7/7 feasible solved so far; K=6 compact 7/9
  - failures LEFTm3JecaC, Nfvxx8J5NCo exhausted 6000 edges with 62-90% of nodes in manoeuvre mode
  (goal + 5 manoeuvres per node, fine de-dup). All manoeuvres in any solution (DEV final, v7b,
  HOLDOUT, guided) occur in the first 40 steps at the spawn -> v8: manoeuvre mode only before the
  robot first leaves the 1 m spawn region; witness heuristic also charges heading mismatch near
  witness poses (commit after 9f2aedc3).
- DEV witness coverage: 23/28 verified witnesses; 4 certified infeasible (yr17PDCnDDW, 7Ukhou1GxYi,
  Dd4bFSTQ8gi, 5cdEh9F2hJL); 3t8DB4Uzvkt unresolved (4 humans share the needed corridor; 30-min
  witness search reached 4.0 m of 11.1 m, not exhausted).

- **Guided K=10 (v7b + witness heuristic), DEV complete except 3**: solved only episodes that already
  pass unguided (13 PASS); every hard episode still fails (Qpor2mEya8F, Nfvxx8J5NCo, TEEsavR23oF,
  LEFTm3JecaC, SByzJLxpRGn, YRUkbU5xsYj, bzCsHPLDztK, 3t8DB4Uzvkt) and AWUFxHEyV3T **regressed** (PASS
  unguided in 3,540 edges, FAIL guided): the join-the-witness term is inadmissible for the generator's
  reachable set and pulls the search toward witness states its queries cannot reach.  Dropped as the
  main configuration.  v8 (spawn-only manoeuvres) K=6: LEFTm3JecaC, Nfvxx8J5NCo still fail.
- **Why the hard episodes fail (diagnostics, 2026-10-27):**
  * Not needles: witnesses with a 0.5 m human margin (2x the body proxy; `witness.py --margin`) exist for
    Qpor2mEya8F (115 steps), Nfvxx8J5NCo (165), AWUFxHEyV3T (245), LEFTm3JecaC (56), bCPU9suPUw9 (111);
    TEEsavR23oF none in 1200 s.
  * Witness tracking (`tests/querygen/witness_track.py`): generator queries follow the witness until the
    critical human interaction, then every child of the near-witness states is doomed.
  * Witnesses of passing *and* failing episodes put the body axis on the legal map's inflated wall band
    (`witness_legal.py`: 13-93 steps per witness), so legal-map conservatism alone does not separate them;
    but waiting/squeezing next to humans and tight corners (LEFTm3JecaC, YRUkbU5xsYj spawn 3-point
    turn) need it.
  * Executor (`trace_exec.py`): ORCA's disc proxy (0.55 + 0.30 + 0.25 m, 1.5 s horizon = ~6.5 m of
    native travel) forbids approaching any human ahead and makes plan-following dither left/right (the
    static plan turns one way, ORCA's velocity the other): Qpor2mEya8F's goal query wastes ~25 of its
    first 58 steps; near-goal failures (bCPU9suPUw9 0.44 m) have a human standing ~0.7 m from the goal.
  * Timing: at Qpor2mEya8F's corridor exit every forward query steps into a crossing human's path and is
    trapped (no reverse, humans do not react); the witness crosses ahead of the human diagonally ~10
    steps earlier.  No query reasons about time.
- **v9 candidates** (commits e8dcb244, 1e5897a2): ORCA turn commit + non-holonomic speed governor
  (executor default); query options `tight` (outer band ring as UNKNOWN), `close` (body-width ORCA
  proxies, 0.5 s horizon, 0.27 m rectangle check), `commit` (plan + hard check only); **`st_goal`**:
  legal space-time lattice plan (stplan.py: exact native primitives incl. holds, legal map, legal CV
  human predictions with growing margin) re-planned every 4 steps.  Ablations on the 9 DEV failures
  running (`dev_tight1`, `dev_v9tc`, `dev_v9st`).

- **Objective update (user, 2026-10-27 afternoon):** 95-100% Oracle SR on feasible episodes (incl. unseen),
  planner + dynamics + generator + executor + Oracle jointly optimisable, legality/safety/dynamics rules
  unchanged, configurations kept genuinely different (a shared Layer-3 policy must transfer across them).
- **v9 ablations on the 9 hard DEV episodes** (ORCA/native, K=10, 6000 edges):
  | config | Qpor2mEya8F | bzCsHPLDztK | bCPU9suPUw9 | LEFTm3JecaC | Nfvxx8J5NCo | TEEsavR23oF | YRUkbU5xsYj | SByzJLxpRGn | 3t8DB4Uzvkt |
  |---|---|---|---|---|---|---|---|---|---|
  | v9 exec + tight + close | PASS | PASS | PASS | fail | fail | fail | fail | fail | fail |
  | v9 exec + st_goal | PASS | PASS | PASS | fail | fail | fail | fail | | |
  | v9 exec + tight2 + close + st | | | | fail | | | fail | | |
  | v9 exec + tight + close + commit + st | | PASS | | | fail | fail | | fail | |
  The time-aware query (st_goal) and the close/tight executor each independently solve the same three.
  Failure analysis: LEFTm3JecaC = 64 steps lost at the spawn (legal route through an unobserved wall) +
  corner turn inside the legal band; Nfvxx8J5NCo = overshoots the still-UNKNOWN passage (route avoids
  UNKNOWN), alt/frontier candidates were crowded out of K=10; YRUkbU5xsYj/3t8DB4Uzvkt = 72-79% of edges
  spent on spawn manoeuvres (search budget); TEEsavR23oF = witness passes humans at 0.26 m (no 0.5 m
  witness); SByzJLxpRGn = ~70-step wait in a side pocket next to a human (no 0.5 m witness).
- **Generic fixes from this analysis (committed):** approach gated by the legal route cost-to-go (UNKNOWN
  2.5x); K=10 quotas from solution usage (wait_long / retreat / look_lost_human never appear in ~60
  solutions -> leftovers; alt/frontier get slots); Oracle `heading_h` (turn steps to the true geodesic
  path), `escape_penalty` (de-prioritise spawn manoeuvres once any branch escaped), exact duplicate skip
  before generation, exact sibling prefix sharing (`prefix_cache`; ~45% of sibling steps are shared;
  equivalence test identical search).  Close/commit only for robots that stop within one step.
- **Other configurations (4 easy DEV episodes, v9 generator):** DWA x native 4/4, SFM x native 4/4,
  ORCA x inertia 2/3 so far (tQ5s4ShP627 fails: close mode with momentum -> 2,873 overlap edges; fixed by
  restricting close to stateless dynamics).

- **Mixed mode (v9, `tc_variants`)**: conservative queries plus explicit tight+close copies of the
  progress queries (`goal_tc`, `st_goal_tc`) instead of tight/close everywhere (all-tight planning cut
  the corner at Nfvxx8J5NCo's passage top).  With the Oracle options heading_h + escape_penalty +
  escape_h + prefix_cache it solves **LEFTm3JecaC** (86 steps, 1,289 edges), **TEEsavR23oF** (216 steps;
  also solved by the all-tight/close config with escape_penalty, min clearance 0.275 m) and
  Qpor2mEya8F (634 edges, min clearance 0.405 m).  Note: no witness with a 0.30/0.35 m human margin was
  found for TEEsavR23oF in 2,700 s, yet the Oracle found a 0.275 m solution -- margin searches are
  incomplete evidence.
- **3t8DB4Uzvkt is feasible**: 3-hour witness searches found verified 400/409-step solutions (two long
  waits of ~80 and ~130 steps).  All 24 non-certified DEV episodes now have verified witnesses.
- Full v9 (tight+close everywhere) K=10 DEV: all 15 previously passing + Qpor2mEya8F, bzCsHPLDztK,
  bCPU9suPUw9 pass; YRUkbU5xsYj, Nfvxx8J5NCo, TEEsavR23oF, LEFTm3JecaC fail (3t8DB4Uzvkt, SByzJLxpRGn
  pending at the time of writing).
- Escape map v2 (Oracle `escape_h`): confinement test (static contact within 3 primitive layers) then a
  weighted A* escape path (BFS never reached YRUkbU5xsYj's 49-primitive escape within 12k sims).
- Second configuration: DWA x inertia passes 4/4 easy DEV episodes; ORCA x inertia fails 2/4 (one-step
  safety check cannot protect a robot that brakes over 6 steps) -> ORCA kept for instant-stop dynamics.
  Full-DEV DWA x inertia run started.

- **Mixed-mode DEV runs (pre-freeze, ORCA x native)**: K=10 with `tc_variants=(goal, st_goal)` solved
  22/24 feasible episodes (incl. 3t8DB4Uzvkt in 432 steps, SByzJLxpRGn, LEFTm3JecaC, TEEsavR23oF); failures
  Nfvxx8J5NCo (also with 15,000 edges) and YRUkbU5xsYj.  Defects found and fixed on the way:
  * manoeuvre starvation: extra fixed progress variants left K=10 too few slots for exact manoeuvres
    (vd3HHTEpmyA / yZME6UR9dUN regressed) -> manoeuvre mode keeps only continue/goal/stop;
  * persistent manoeuvre mode: rotate/creep chains continued anywhere while the last decision was a
    manoeuvre (YRUkbU5xsYj 88% of expansions) -> only near the start or the goal;
  * `Query.signature` ignored execution meta, so compact K=6 de-duplicated `st_goal_tc` away;
  * escape penalty extended to every edge from spawn nodes once a branch escaped.
  `tc_variants=(st_goal,)` alone loses LEFTm3JecaC and 3t8DB4Uzvkt (both solutions use `goal_tc`).
- **Nfvxx8J5NCo (open)**: the passage opening lies at the end of the inflated south-wall band and is
  unobserved while the robot drives past; neither the normal nor an optimistic (UNKNOWN = FREE) legal route
  enters it; every branch reaches the junction 10-20 steps after the witness and meets a returning human.
- **DWA x inertia (pre-freeze code)**: 14/16 feasible DEV episodes completed so far pass (Nfvxx8J5NCo,
  TEEsavR23oF fail).

- **Final pre-freeze fixes (2026-10-27 evening)**: manoeuvre mode keeps `goal_tc` (3t8DB4Uzvkt leaves its
  spawn manoeuvres with it) and orders creep1 before the rotations; escape map = exact breadth-first map
  first, A* escape path only when BFS finds no escape (A*-only mis-guided 3t8DB4Uzvkt); escape penalty
  back to manoeuvre edges only; DWA exact per-call memo (2.8x faster planner, identical search).
  K=6 uses `goal_tc` (solves SByzJLxpRGn, 3t8DB4Uzvkt, LEFTm3JecaC, TEEsavR23oF in checks), K=10 both
  variants.  Hard episodes near the budget edge flip with small search changes (3t8DB4Uzvkt: PASS in
  earlier K=10 snapshots, 342/432 steps and FAIL in the last check, PASS at K=6).
- **Freeze `querygen-freeze-v9`** (commit aea47081); final DEV + unseen runs launched with
  `scripts/querygen_final_v9.sh` (C1-K10, C1-K6, C2-K10) and unseen certificates
  (`scripts/querygen_unseen_certificates.sh`) -> `output/runs/final_v9/`.

- **v9 results (frozen runs, `output/runs/final_v9/`)**: DEV feasible SR C1-K10 21/24, C1-K6 21/24,
  C2-K10 (DWA x inertia) 18/24; unseen suite C1-K10 28/30, C1-K6 25/30 (7 certified infeasible, 3
  unresolved; C2 still running) -- details in `REPORT_V9.md`.

## Post-freeze (v10 development, 2026-10-27 night)
- **Second unseen suite frozen** (`docs/querygen/suites_unseen2.json`, 45 HM3D + 15 MP3D train scenes, new
  salt, disjoint from DEV/HOLDOUT/unseen; sha256 3bfb1472..., commit d9f23c01) before any post-v9 work.
  The first unseen suite is consumed by the v9 evaluation and is now part of the development set
  (DEV 24 + unseen 33 non-certified episodes; `scripts/querygen_dev_v10.sh`).  unseen2 is never run
  before the v10 freeze.
- **Defect: in-place rotations ended one step late.**  The executor noticed "heading reached" at the
  start of the next step and spent that step on a native hold, which is itself an in-place turn that
  always starts right: `rot_left1` = left+right (a no-op), `rot_left<k>` turned k-1 steps, `rot_right<k>`
  k+1.  At pUneSGJDrvY's wall-side spawn no left turn was expressible, so every branch hit the wall and
  the tree was exhausted after 587 edges.  Fix (`rot_exact`): the decision ends on the primitive that
  reaches the target heading.  pUneSGJDrvY: PASS in 39 s.
- **K=6 reset-state coverage (`root_cover`)**: at TQSiMZJawkS's spawn only a left turn is free; all six
  K=6 root candidates started with 'r' (goal, st_goal, goal_tc, wait = native hold starting right,
  turn_right) or 'f' (creep1).  Nothing behind the robot is observed at reset, so no legal test can tell
  the sides apart; K=6 now offers both route turns instead of the goal_tc copy at the reset state.
  TQSiMZJawkS K=6: PASS (1,368 edges).
- **Momentum STOP**: v9's Oracle goal test issued STOP at any speed level -- an instantaneous halt the
  inertia model forbids.  8 v9 DWA x inertia solutions stopped while moving (levels 2-6 of 12); all 8
  still pass when re-verified with braking to rest first (`tests/querygen/rebrake_check.py`), so the v9
  numbers stand.  v10: STOP of a stateful robot brakes at the brake limit and declares at rest.
- Budget use: 50-75% of all edges end in already-seen (coarse-key) states; long equal-f plateaus at
  human timing points (SByzJLxpRGn: 800 expansions at t=100-150 before the breakthrough edge).  The
  sandbox duplicate predictor (off since the wall-clock era) is being re-tested under the edge budget.
- **Search-order fragility (2026-10-28 morning, r2 = all v10 fixes, 12,000 edges).**  Small, unrelated
  changes move single episodes between tens and >12,000 edges: v9 -> r2 HMkoS756sz6 K6 2,119 -> 29,
  hkr2MGpHD6B K6 44 -> 1,645, BJovXQkqbC3 K10 5,778 -> 228, LEFTm3JecaC K10 1,269 -> FAIL (the v9 branch
  relied on the stray rotation step), LT9Jq6dN3Ea K10 233 (r1, space-time radius 0.6) -> >7,000 (r2,
  radius 0.5 + cell-sound fallback); iePHCSf119p K6 FAIL at 12,000 with weight 2 but PASS in 645 edges
  with weight 3.5.  The K6/K10 search spaces contain these solutions; the single best-first order misses
  them.  -> **Oracle portfolio** (`OracleConfig.portfolio`, commit 89d2fade): independent member searches
  (override dicts) interleaved under one edge budget, first success wins; identical to the single search
  for one member (checked).  Members only share exact caches, so an equal-share portfolio of n members
  succeeds at ~ n * min_i(edges_i): single-member runs score any combination offline
  (`tests/querygen/virtual_portfolio.py`).  Two v9/r2-style variants at a fixed budget gain little
  (budget halving offsets the complementary solves), so members must be chosen by data.
- **Where branches lose time against witnesses** (`tests/querygen/branch_lag.py`, r2 best branches):
  * LEFTm3JecaC (witness 54 steps, a continuous right arc through the inflated band): after the first turn
    the robot sees the furniture next to the goal and the legal map's 0.25 m inflated band covers the goal
    cell; gridplan's Dijkstra skipped blocked sources, so every goal-directed query was 'unreachable' (the
    v9 solution spun 12 steps, then tight queries found the way).  13 of 68 DEV/unseen goals lie within
    0.1 m of the navmesh edge (LEFT, iePH, Vvot 0.0 m; 3t8D, wz9F, mJXq, Nfvx < 0.075 m).  -> v10 option
    `goal_relax_m` (field seeded from passable cells near a blocked target; commit 81d01a84), under test.
  * Nfvxx8J5NCo: the branch tracks the witness within 3-8 steps for 80 steps, then loses ~15 at the side
    passage entrance (stop-turn-crawl where the witness turns while moving through the band); a human
    then blocks the junction.  YRUkbU5xsYj: 11 steps lost in the confined spawn turn-around (the witness:
    12 right, 1 forward, 29 left), then a passage the witness crosses 0.7 m from humans.
  * p4ZPcGtk6Ex / W16Bm4ysK8v: the witness reaches a waiting spot early, waits (50 / 15 steps) and passes
    a bottleneck in a short window (0.7 m from a human); the branches arrive 18-45 steps late after
    non-progress decisions (a goal query that alternates left/right 'jammed' for 16 steps; advances that
    lose ground).
  * 3t8DB4Uzvkt (r2): 2,119 of 2,624 expansions at t = 25-50 near the spawn; waiting far from humans is
    pruned (time enters the duplicate key only within 4 m of a human) -- `time_key_slack` under test.

- **Portfolio members on the development set** (single-member runs capped at 6,000 edges, `output/runs/v10dev/pf/`):
  weight 3.5 (0.5 m space-time model + fallback) is the strongest single ordering (iePHCSf119p 116 vs r2 1,375
  edges, SByzJLxpRGn 723 vs 3,382, zUG6FL9TYeR 1,082 vs 3,739, TEEsavR23oF 3,147 vs 6,923, LEFTm3JecaC 2,091 vs
  r2 failure) but misses iigzG1rtanx; the v9 heuristic (radius 0.6, no fallback, weight 2) solves
  LT9Jq6dN3Ea in 253 edges (r2: failure at 12,000) and iigzG1rtanx (3,036).  `time_key_slack=20` did not help
  3t8DB4Uzvkt (expansions still end before t=100; the witness waits 80 + 130 steps).
- **Goal relaxation** (`goal_relax_m=0.5`, v10 default, commit f6d08023): on the 14 near-edge-goal episodes
  LEFTm3JecaC 2,091 -> 1,782 edges and 205 -> 144 steps with no 'unreachable' goal decisions, eF36g7L6Z9M
  401 -> 306, the rest identical.
- **v10 FROZEN** (tag `querygen-freeze-v10`, commit aa09bf88; `FINAL_PROTOCOL_V10.md`): Oracle portfolio
  [weight 3.5 | v9 heuristic] with equal shares under 12,000 edges; generator with goal relaxation.  Test runs on
  the second unseen suite (`output/runs/final_v10/unseen2_*`), its certificates and the frozen system's
  development numbers (`output/runs/final_v10_devset/`) launched 2026-10-28 07:17.

- **Test-set observations while the v10 test runs** (recorded for v11; v10 is unchanged):
  * K=6 reset state without single-step rotations: 5737gQA9p2T and tJ7XVoEN82a spawn in static contact; K=10
    solves both starting `rot_left1 -> rot_left4/rot_left10 -> creep` (113 / 177 edges), K=6's root set
    (goal, st_goal, wait, creep1, both long route turns) has every movement child colliding and the tree is
    exhausted after 24 edges.  Candidate v11 fix: the K=6 reset set offers rot_left1 / rot_right1 (a manoeuvre
    puts the next state into manoeuvre mode, which then offers the finer rotations).
  * Narrow-corridor spawns facing away (BCWtGkh8CHv, PFLHHbjscNN, no humans): witnesses drive 1-1.6 m away to a
    wider spot and turn there; manoeuvre mode only exists within 1 m of the start and the legal map at reset holds
    none of the walls that stop an in-place turn -> trees exhausted (C1) / budget (C2).  Candidate v11 fix:
    manoeuvre mode within a larger spawn radius (needs a regression check against the v8 search explosion).
- **Third unseen suite frozen** (`suites_unseen3.json`, commit 4a40ed00) for the post-v10 version.

## Official Falcon rules (user, 2026-10-28 ~10:45) -- the binding evaluation from here on
- Declared project rules (static contact, body proxy) are dropped; verdict = official success in the authoritative
  replay (`OFFICIAL_RULES.md`).  Denominator = every episode not PROVEN natively infeasible: exact native-rule
  certificates (`native_cert.py`, officially verified) prove 5Poh4Qz68hd, sKLMLpTHeUy, AMEM2eWycTq; native witnesses
  (`witness.py --rules native`) proved every other formerly 'certified infeasible' episode solvable (incl. the 4 DEV
  ones, sT4fr6TAbpF, Wo6kuutE9i7); nMeXfQU4PMS unresolved (in the denominator).  C2 = DWA x native (official actions).
- Oracle capability (v10 portfolio Oracle, official rules; `output/runs/official/base`, partial at 13:10): only
  PFLHHbjscNN fails among finished non-infeasible episodes (C1 K10 and K6; narrow corridor section that the
  conservative legal wall band blocks); SPL on solved 0.87-0.97, PSC 0.92-0.97, H-Coll 0 among solved.
- User refocus (2026-10-28 ~11:50): the Oracle is only a diagnostic; priorities are the foundation (planner,
  dynamics, executor, generator) and Layer-3 learnability.  No more Oracle-search work (anytime refinement dropped).
- No-selection floor (default policy) failure modes: human collisions (13/16 analysed hit a robot holding/turning in
  place while a tracked human walked into it) and static deadlocks (inflated wall band treated as hard; obstacles the
  camera cannot see).  Foundation options (legal): `goal_unstick` (tight map + plan following; rotate toward the
  route when no body-feasible plan exists), `bump_memory_m` (proprioceptive bump marks), `evade` (human-threat
  reflex).  Floor A/B: baseline ~19%, breaker+bump ~39%, +evade ~42% (partial).
- Learnability: minimum-intervention Oracle (C1 K10) -- solved episodes need 0-2 non-default decisions (median 1;
  most common critical choice: the initial turn direction); `l3_env.QueryEnv` (decision-level environment);
  `l3_readiness.py` (recovery / identifiability study); exact human-tape cache halves episode setup.
- Visualization: `viz_episode.py` (figures), `render_topdown.py` (navigation videos; RGB chase-camera recording is
  blocked by the GPU driver/library mismatch: EGL cannot create a context), `build_page.py` (results page).

## 2026-10-28 afternoon: v11 foundation, default-policy diagnosis, v12 candidate fixes
- v11 default (paired floor A/B on the same episodes, official rules): `goal_unstick=6 + bump_memory_m=0.25 +
  evade` >= breaker + bump >= v10 baseline (ORCA 5 -> 9 of 21, DWA 2 -> 10 of 21 at 13:10).  v11 Oracle capability
  runs `output/runs/official/v11/`, minimum intervention `minint_v11/`, readiness `readiness_v11/` (watch mode).
- Remaining v11 default-policy failures (`exec_ube`, first 60 episodes): human collisions -- in-FOV head-on
  approaches where the constant-velocity reflex mispredicts stop-and-go humans (Qpor2mEya8F: the human's estimated
  velocity went 1.6 -> 0.2 -> 0 -> 1.9 m/s in the last 8 steps) and humans beside the robot outside its 90-degree
  camera (yr17PDCnDDW, FgswoxWb3uN, PUNuHY5M7MS: within 1 m for 20 steps at bearings 50-120 deg); these are
  decision problems for Layer 3 (yield / sidestep earlier), not executor bugs.  Static failures were executor
  bugs: (1) an escalated unstick goal query re-planned at every decision boundary; from the intermediate pose of
  an in-place turn the lattice found no plan and the route fallback turned back (HMkoS756sz6, bXu6SSWkJY8,
  LEFTm3JecaC: a 32-step left/right cycle for 400 steps); (2) the evasion reflex and the unstick fallback checked
  forward steps at the base centre on the tight map, so a bump-memory disc 0.684 m ahead never stopped them
  (iePHCSf119p pushed 'forward' for 100+ steps).  v12 candidate flags `unstick_keep` (meta 'ukeep') and
  `bump_front` (meta 'bumpf'); keep-plan experiment: 6 of 6 cycles broken, 3 of 6 then succeed.
- gTV8FGcVJC9 (unseen2): C1-K6 search tree exhausted after 2,478 edges (every branch met a human by step 22);
  feasible (declared-rule witness, 127 steps: 11 right turns first).  K=6 coverage gap at a spawn threat.

## 2026-10-28 ~14:00: gate test launched; v11 superseded
- User: has the gate been met, or is this polishing?  Answer: met on the finished development episodes (C1-K10
  86/87, C1-K6 83/85, C2-K10 69/70) but not yet formally -- the hardest development episodes are still running and
  the held-out suite had not been run.  The default-policy work is Layer-3 foundation work, not gate work.
- Gate test (`FINAL_PROTOCOL_V10_OFFICIAL.md`): the exact code of the official development runs, tag
  `querygen-freeze-v10-official` (2786115d, package sha256 bcecb8e8...), on the untouched third unseen suite, three
  configurations, `output/runs/official/test_v10/`; feasibility of unsolved test episodes afterwards with
  `scripts/querygen_test_feasibility.sh` (native witness, then exact certificate).
- v11 default policy superseded: its unstick trigger counted in-place turns as "still", so the breaker took over in
  88 of 100 runs (mostly at the spawn turn) and ORCA and DWA produced identical action sequences in 24 of 48
  episodes -- the planners were bypassed.  v12 candidate flags (all default off): `unstick_progress` (genuine
  deadlocks only), `unstick_keep`, `bump_front`, `static_humans` (Falcon's humans stop for good after 100-400
  steps: by step 300 most stand still), `stop_strict` (0.19 m stuck-STOP).  To be evaluated after the gate test.
- Interim results page published (top-down recordings: default vs Oracle on collision and deadlock cases, the
  PFLH failure; charts; metrics table).

Technical report of the current implementation and results: `docs/querygen/TECHNICAL_REPORT.md` (2026-10-28 15:15).

## Next steps (REVISE)
1. Evaluate revision v7b (root probes, branch-pure caches, goal test) on a *new* untouched split,
   with the gate restated over certificate-filtered episodes.
2. Oracle search for human timing in narrow passages and long Spot-tight routes (largest failure
   class): larger budgets or a cheaper sensor-free search layer (space-time plan + native
   verification), and better de-duplication of time-indexed states.
3. Layer 3: train selectors only on certificate-feasible episodes; condition on planner/dynamics.
