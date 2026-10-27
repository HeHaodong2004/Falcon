# Query Generator v6 (frozen as `querygen-freeze-v1`; deterministic, non-learning, legal inputs only)

Code: `eventwm_graph/querygen/generator.py` (`QueryGenerator.propose(legal, execution)`), query
semantics: `executor.py`. Parameters: `GeneratorConfig` (recorded in every result file).

## Inputs (legal only)
- `legal.map`: the robot's own certified occupancy grid (FREE / BLOCKED / UNKNOWN, 0.1017 m cells;
  BLOCKED = observed wall hits dilated by 0.25 m). UNKNOWN is never treated as free: routes may cross
  it optimistically at 2.5x cost, via-points must be on FREE (or, for sidesteps, non-BLOCKED) cells.
- `legal.tracks`: legally observed human tracks (position, estimated velocity, age); predictions are
  constant-velocity extrapolations of these tracks only.
- `legal.pose`, `legal.poses` (own pose history), the public PointGoal, the dynamics family, and the
  currently executing query (for `continue`).
Never used: simulator state, true human futures, unseen map, navmesh, geodesic distance.

## Output
At most `budget` (10) executable, de-duplicated `Query` objects + diagnostics (raw candidates per
category, kept count, conflicts, stuck flag).

## Candidate categories and legal triggers
| category | trigger | queries |
|---|---|---|
| fixed | always | `continue` (if a non-wait query is running), `goal`, `stop` (legal goal distance < 0.40 m) |
| human | a tracked human's CV path meets the next 3 m of the route within 1.3 m | `pass_left/right` (via 1.15 m beside the human at the conflict time, then rejoin the route), `pass_behind` |
| yield | conflict | `yield` (nearby reachable body-clear point maximising predicted clearance), `sidestep_left/right` (0.9 m lateral, tracked human within 3 m) |
| wait | always; `wait_clear` on conflict | `wait_clear` (hold until the legal route conflict clears, <= 48 steps), `wait_short` (6), `wait_long` (16), `goal_slow` (following a same-direction human ahead) |
| approach | legal goal distance < 3 m | goal via a pre-goal point 0.9 m before the goal on headings whose body pose is clear of observed walls |
| turn | route heading differs > 30 deg | rotate toward the route heading, both directions (sweep side matters for a 1.23 m body) |
| maneuver | manoeuvre mode: entered after a legal stall within 1 m of the start pose or the goal (over the last 6-16 steps no translation > 0.15 m and no net turn > 0.35 rad); kept while the previous decision was itself a manoeuvre | exact rotations of 1/4/10 turn steps each way, 1/3-step forward creeps (8 candidates) |
| retreat / advance | always (if available) | back 1.6 m along the own legal trajectory / 0.7 m straight ahead |
| alt / frontier | alternative homotopy with a diverging first 2.5 m / frontier clusters | via-point routes |
| observe | goal or a recently lost track out of view | rotate toward it |

Selection: situation-aware quotas in priority order (maneuver 8, human 3, approach 2, yield 2, turn 2,
wait 2, retreat 1, advance 1, alt 1, frontier 1, observe 1), then round-robin leftovers until the
budget; de-duplication by signature and by first via-point distance (< 0.45 m).

## Execution contract (executor.py)
`goto`/`goal` follow the legal-map route through via-points with the body-aware static lattice
(hybrid A*, heading bins >= 2 per turn step) and the chosen local planner (ORCA / SFM / DWA) under
the dynamics family; `goal` issues STOP within 0.15 m (legal); `wait`/`wait_clear` hold (native robot:
alternating in-place turns); `rotate` turns in place (explicit direction); `creep` steps forward;
human one-step clearance (0.30 m) is enforced for all kinds. Queries run until done or the Oracle's
decision interval (16 steps; waits run their duration) and may be continued.
