# EventWM v3 root-sampling source trace (before implementation)

Baseline: clean `research/eventwm-social-outcome-v3`, commit `82097ac974cba4f8aa604d3319238f7a2fe75a50`; all 77 collector tests pass. Labels, checker, human dynamics, sensor/tracker/controller behavior are frozen for this task.

## Prefix and selection pipeline

`scripts/eventwm_v3.py::run` chooses the four smoke fixtures explicitly: Castle h0/wait/8301, h1/seek_visible/8401, h2/seek_lost/8502, HM3D h0/explicit static-open-space fixture/13301. It writes a recipe, invokes `eventwm_collect.py collect`, then seals successful roots. `collector.runtime.worker` seeds Python/NumPy/Torch and Habitat, constructs the existing environment, records observations, and consumes `collector.root_sampling.prefix_commands(recipe, history)`. Each command is executed by the same native BaseVelAction and human actions used in branches. The root is captured immediately after the prefix. Episode termination before then raises `Prefix terminated before root`. No alternate time is substituted.

For the first root process, generated controls are frozen into `roots/replay.json`. Every branch starts a fresh process, resets the same episode/seed, replays these explicit controls, and checks the exact physical/controller/RNG/history/image fingerprint. The explicit `prefix_controls` path has priority over every named profile.

| Existing profile | Exact prefix / stopping rule | Conditioning |
|---|---|---|
| `wait` | Exactly `sampling.wait_steps` copies of `[0,0]`; smoke uses 20 (~0.97 native seconds). | None on humans. Fixed-time stationary baseline, not a sampled ordinary-navigation distribution. |
| `seek_visible` | Before each of up to 126 `[0,6]` turns, inspect `history[-1].human_tracks`; stop turning if **any** track has `visible_now=true`. Then **12 WAIT commands**. | Legal current visibility conditions stopping. No guarantee a track is still visible after the tail. |
| `seek_lost` | Same visibility-conditioned search, then **24 additional `[0,6]` turns**. | Visibility conditions initial stopping. Lost status is **not** directly checked; the tail is intended to create lost tracks, not a guarantee. |
| `seek_move` | Same search, then **12 `[1.2,0]` commands** (0.12m total nominal translation). | Also visibility-conditioned; not ordinary navigation. |
| Explicit controls | Replay supplied list exactly; names do not override it. | Depends on the provenance of that list. The existing non-Castle fixture is static-map/robot-pose selected and human-independent. |
| Ordinary/default | **No ordinary/natural sampled-time profile exists.** The no-recipe correctness collector supplies a fixed 40-step WAIT fixture; the v3 smoke runner explicitly picks coverage profiles. | Fixed fixture only. |

`[0,6]` uses 6rad/s integrated at 1/120s per action: about 0.05rad each. The action clock is ~0.048667s and is distinct from controller integration time; native timestamps, not nominal multiplication alone, remain authoritative.

No existing named prefix function reads human distance, PSC, future labels, hidden GT positions or human future trajectories. Seek profiles nonetheless **condition root timing and orientation on legal visibility**. The words `seek_lost` do not mean “wait until a lost track exists”. These are conditional coverage profiles and cannot stand in for an ordinary distribution.

## Human initialization and dynamics (unchanged)

- `scripts/controlled_matrix.py::prepare` copies the existing reference episode/navmesh bundle; case manifests retain the original episode paths and hashes. Changing collector episode seed does **not** regenerate spawn points or routes.
- `scripts/controlled/generate.py::generate` documents the original static sampling: robot start/goal on a connected 4–10m geodesic pair, similar floor height; human points at least 1.2m horizontally from start, separated by 0.8m from supplied avoid points, connected to start; six slots, three persisted waypoints each. This generation code is not run by the collector and will not be changed or rerun here.
- `socialnav_v2/socialnav_task.py::reset` uses `_use_episode_start_goal=True`: robot from episode start; active humans from waypoint 0 and saved rotation. `scripts/controlled/runtime.py::ControlledPointNavTask.reset` verifies counts, positions and goals, clearing/parking inactive slots at remote sentinel positions. No sampler changes any of this.
- `falcon/additional_action.py::OracleNavRandCoordAction_Obstacle.reset` sets active-human goals from persisted waypoints 1 and 2, resets goal index and navigation state. Native pathfinding/gait follows those goals; the goal index advances through the finite saved goal list, without regenerating goals to seek the robot. Avoidance may affect realized paths, but the stored goal coordinates remain fixed.
- `OracleNavObstacleAction.__init__` draws `rand_human_speed_scale ~ Uniform(0.8,1.2)` from the existing seeded NumPy RNG. The collector seeds before constructing the configuration/environment. Sampler randomness must use a **separate local RNG**, never consume this stream.
- Native avoidance in `OracleNavObstacleAction` enumerates agent indices **1..num_articulated_agents−1**, explicitly excluding robot index 0. Humans can respond to other humans; there is no configured reciprocal response to the robot. Native physical contact can terminate the task. The optional stop-and-walk-to-robot gate is disabled (`human_stop_and_walk_to_robot_distance_threshold=-1.0` in every resolved active-human action). No robot-conditioned human policy/replanning is added. The inherited `human_reactivity` diagnostic may remain FAIL.

## Existing smoke root-time evidence

Read-only `roots/privileged_root.json` gives:

| Root | Time | Nearest canonical PSC distance | Already <1m? |
|---|---:|---:|---|
| Castle h0/wait | 0.968s | N/A | No humans |
| Castle h1/seek_visible | 4.032s | 2.572014m | **No** |
| Castle h2/seek_lost | 3.744s | 2.740424m | **No** |
| HM3D h0/explicit prefix | 12.160s | N/A | No humans |

Thus the smoke's 18 social-success / 8 collision outcomes were not caused by initial PSC violation. Conditional timing/orientation, fixed routes, later interaction and query horizons remain plausible contributors; the pilot must measure them rather than assert causation.

## Implementation decision

Add `natural`: draw a discrete root step from a declared inclusive window using only episode seed + independent sampler seed, **before** observing outcomes. For meaningful position variation reuse the canonical static-feasible query compiler as a human-independent robot prefix: select one of three accepted immediate normal-speed movement paths using the local sampler RNG, execute until the selected time, then WAIT if its controls end earlier. No learned or human-seeking policy. The existing non-Castle static-open-space prelude may precede this motion; its length is included in the declared absolute step window. Save exact resolved controls for unchanged replay.

This is a declared static-navigation-prefix distribution over existing episode fixtures, not a claim to sample every possible navigation state or regenerate human routes. Sampling is independent of social signals. If the native episode terminates before the preselected step, retain the attempt as a censored prefix failure; never move the selected time, rescue it at collision, or resample based on outcome. Report this survivorship limitation separately from root rejection and future branch labels.
