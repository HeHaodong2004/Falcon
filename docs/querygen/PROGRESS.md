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
- DEV v3 (commit 4aaff8cb, running): early results auFeVz9Go4m/16, HMkoS756sz6/10,
  hkr2MGpHD6B/8 PASS; both certified-infeasible episodes FAIL_SEARCH.
- Sandbox study (1 episode, 5 states, 50 queries, M=16 CV+noise futures): nominal-rollout +
  vectorised scoring vs closed-loop per-future rollouts: overlap-probability MAE 0.015 (corr
  0.72), median d_min MAE 0.007 m, top-choice agreement 100%, 166 vs 2464 ms/query (~15x);
  sandbox vs native with true future: mean deviation 0.064 m, outcome agreement 100%.

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
