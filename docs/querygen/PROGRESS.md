# Query Generator / Planner / Oracle track — progress record

Worktree `eventwm-falcon-querygen-claude-20261026`, branch `research/querygen-oracle-20261026`.
Package `eventwm_graph/querygen/`; runner `scripts/querygen_run.py <module> [args]`
(python `/home/haodong/miniforge3/envs/eventwm-falcon/bin/python`, `CUDA_VISIBLE_DEVICES=1`,
`nice -n 10`; Habitat runs headless/CPU). Outputs: `output -> /var/tmp/falcon-querygen-claude`
(`/home` is 93% full). Isolated `data/` (symlinks to Falcon data; own `navmeshes/` cache).

## Verified facts (with evidence)
- Native benchmark = original `social_nav_v2/falcon_{hm3d,mp3d}.yaml`, sources identical to
  baseline Falcon. 0.048 s world time per Env.step; forward 25/120 m nominal (actual varies
  near walls, mean 0.173 in random probe), turns exactly +-1/12 rad; no pause/wait action;
  STOP ends episode; success = STOP & geodesic < 0.2 m; human contact terminates; static
  contact does not terminate (500-step limit).
- Falcon humans are exogenous (avoid only agents 1..N). Verified: human trajectories under
  random robot actions are bitwise identical to a robot-free tape.
- Branch simulation (tape + exact robot setter replay + joint-limit clamping) is bitwise
  equivalent to Env.step: `tests/querygen/native_equivalence.py` -> ALL_EQUIVALENT
  (3 episodes x 4 trials incl. forced human collisions; poses, humans, collision flags,
  static contacts, legal observations, mid-episode restores).
- Pitfalls found: float32 transform write-back is lossy (Bullet double quaternion) -> restore
  from setter inputs; physics clamps Spot arm joint1 (-3.1686 -> -pi) on step 1;
  `np.asarray(mn.Matrix4)` is row-major (yaw = atan2(-m[2][0], m[0][0])); process exit
  aborts in Magnum teardown -> runner hard-exits after flushing.
- Official human collision uses the humanoid **stick skeleton (2 cm boxes)**: a human centre
  can be inside the robot footprint without a "collision". Declared extra criterion:
  robot all-gait rectangle >= 0.25 m from active same-floor human centres (BODY_PROXY_M).
- Static contact definition: non-support robot-scene contact points (project definition,
  same detection as the official human measure).
- ORCA port (RVO2 LPs, full responsibility) keeps min distance = combined radius in 300
  random holonomic scenarios (`tests/querygen/test_orca.py`).

## Current implementation (commit 32144298)
native harness, legal map/tracks, numba grid Dijkstra, body-aware hybrid-A* lattice
(static layer), ORCA/SFM/DWA/follow planners with native duty-cycle adapter, query
executor (goto/goal/wait/rotate/stop), episode runtime + authoritative replay, viz.

- Spawn contact: 10/28 DEV episodes start with Spot touching static geometry (arm wrist/finger
  0.5-0.6 m ahead at 0.8-1.0 m height, body front, or hind legs). Declared rule (native.ESCAPE_STEPS):
  if spawned in contact, contacts are exempt only until the first contact-free step (<= 24 steps);
  afterwards (and in clear-spawn episodes) any non-support static contact is a collision.
- Static contacts are often millimetre grazes; whether a graze registers can depend on Bullet
  manifold history (e.g. far-away human motion). Branch == real still holds exactly per episode.
- Velocity action (continuous base velocity, same update_base path) verified: installing it leaves
  native behaviour bitwise unchanged; (25,0)/(0,+-10) reproduce native actions; branch == real for
  random arc commands (`tests/querygen/test_velocity_action.py`).
- Dynamics families (dynamics.FAMILIES): native; slowturn (turn/2); inertia (12 speed levels,
  +-2 levels/step, 6 steps to stop, 0.52 m stop distance, rotate only at rest);
  slowturn_inertia; held-out heavy_brake (brake 1 level/step, 1.15 m) and slowturn_heavy.

## Results log
- DEV v0 (generator v0, oracle w=2.5, 300 exp): 8/23 PASS at 23 done; failures: 7 spawn-contact
  (pre escape rule), 6 search-exhausted dominated by body-proxy overlaps (robot inside human's
  path in narrow passages), 2 static at start.

## Important failed hypotheses / lessons
- Velocity-tracking adapter alone cannot handle Spot's 1.23 m body in tight spaces
  (oscillation, cornering); fixed by body-aware lattice static layer + relative body cost.
- ORCA static horizon 0.4 s at 4.34 m/s native speed forbade all motion near walls.

- DWA with long fixed-length straight rollouts dithers (turn-1 flips); fixed by advances that stop
  before walls + commitment bonus.
- Executor-only ORCA cannot escape a human walking up a 1 m corridor toward it (90 deg rotation =
  19 steps); this must be avoided at query level (wait before entering) -> Oracle danger term.

## Next steps
1. Generator v0 + Oracle v0 (best-first over decision nodes), replay verification.
2. Declare frozen dev/acceptance and untouched test suites.
3. Iterate on failures; dynamics families; SFM/DWA evaluation; runtime.
