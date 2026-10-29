# Falcon query-space foundation: technical report

Snapshot: 2026-10-28, ~15:15 GMT. Branch `research/querygen-oracle-20261026`.
Scope: the planner / dynamics / executor / query-generator / Oracle stack in `eventwm_graph/querygen/`, its
evaluation tools, and its results under the **official Falcon benchmark rules**.

Every result below is marked either **[verified]** (finished, replay-checked, reproducible from the frozen
code and the recorded run files) or **[in progress]** (runs still executing at the time of writing; numbers
are partial and will change).

---

## 0. Status in one page

| item | status |
|---|---|
| Evaluation rules | Official Falcon definitions only (Section 1), adopted 2026-10-28 by user decision. |
| Gate (>= 95 % Oracle success over every episode not proven natively infeasible, per configuration) | **Not yet formally established.** The held-out test (third unseen suite, frozen code) is [in progress]. Every finished test episode passes so far: ORCA K=10 40/40, ORCA K=6 36/36, DWA K=10 28/28, out of 58 episodes not proven infeasible. |
| Development set (DEV + unseen 1 + unseen 2, 125 episodes not proven infeasible) | [in progress], same frozen code. Finished episodes: ORCA K=10 98/99 (99.0 %), ORCA K=6 100/103 (97.1 %), DWA K=10 82/83 (98.8 %). |
| Proven natively infeasible (exact certificates, officially verified) | 5: development 5Poh4Qz68hd, sKLMLpTHeUy, AMEM2eWycTq; test sWAdxSLVPQC, q28T9C3q2dv. [verified] |
| Default policy alone (no selection, "floor") | v10 ~18 %; v12 candidate ~56 % with the planners kept distinct ([in progress]). |
| Layer-3 learnability | Pilot results only (Section 13.4). The scaled readiness study waits for the v12 Oracle runs. |
| Layer 3 (learned policy) | Not trained (out of scope by instruction). |

Frozen systems: tag `querygen-freeze-v10-official` = commit `2786115d`, querygen package sha256 `bcecb8e8...`
(the gate system); tag `querygen-freeze-v10` (declared-rule era, superseded); earlier tags `querygen-freeze-v1`,
`querygen-freeze-v9`.

---

## 1. Rules and accounting

**Official Falcon rules**, verified in the benchmark code (`docs/querygen/OFFICIAL_RULES.md`):

| item | definition | source |
|---|---|---|
| actions | discrete forward (lin_speed 25 -> 25/120 m = 0.208 m per step), turn left/right (ang_speed 10 -> 10/120 rad = 4.77 deg per step), stop; `allow_dyn_slide` | `social_nav_v2/falcon_hm3d.yaml` |
| episode | at most 500 steps | `falcon_hm3d_task.yaml` |
| success | STOP with geodesic `distance_to_goal` < 0.2 m (`end_on_success`) | habitat `nav.py` |
| collision | `DidMultiAgentsCollide`: physics contact between robot and a human; `HumanCollision` ends the episode at the first contact | `falcon/additional_metric.py` |
| PSC | share of steps with every human >= 1.0 m (metric only) | `falcon/additional_metric.py` |
| static contact | not a rule | -- |

**Verdict** (`evaluate.py`, `--rules official`): PASS iff the authoritative continuous replay from reset, with
the unchanged `habitat.Env.step`, reproduces the selected branch's action sequence exactly and the official
`success` measure is 1. SPL, PSC, human collisions and static-contact counts are reported but never enter the
verdict. The earlier stricter project rules ("declared": zero static contact outside a spawn-escape window,
robot rectangle >= 0.25 m from every human centre) remain available as `--rules declared` only for
reproducing old runs.

**Denominator.** Every episode, except those **proven** natively infeasible. Proof requires an exact native
certificate (`native_cert.py`): exhaustive search over all native action sequences, merging only bitwise-identical
robot states, a branch closed only by an official human contact. Every closed sequence must then reproduce the
official human-collision measure when replayed in the official environment. Search failures, declared-rule
certificates and relaxations do not count. Unresolved episodes stay in the denominator as failures.

**Legal vs privileged.** Query generation and execution use legal information only: the robot's own observations,
pose/action history and the public PointGoal. Only the Oracle's *selection* among generated queries may use
privileged information (the recorded future of the humans, the navmesh, geodesic distances). The Oracle is an
offline diagnostic of the capability ceiling of planner + dynamics + executor + generator; it is not deployed.

---

## 2. Architecture

```
                 +----------------------------------------------------------------------+
  privileged     |  Oracle (oracle.py): best-first search over generator queries,       |
  selection      |  privileged heuristics (true human tape, navmesh, geodesic)          |
                 +-------------------------------+--------------------------------------+
                                                 | chooses 1 of K queries per decision
                 +-------------------------------v--------------------------------------+
  legal          |  Query Generator (generator.py): <= K queries from the legal state   |
  decision       |  (categories, triggers, K=10 quota / K=6 compact selection)          |
  layer          |  -- Layer 3 (l3_env.QueryEnv) plugs in here instead of the Oracle -- |
                 +-------------------------------+--------------------------------------+
                                                 | Query(kind, via, heading, duration, meta)
                 +-------------------------------v--------------------------------------+
  legal          |  Executor (executor.py): query semantics, routes, static plans,      |
  execution      |  STOP logic, decision boundaries, reflexes (evade / unstick)         |
                 |    Planning substrate: gridplan (Dijkstra fields), lattice (body-     |
                 |    aware hybrid A*), stplan (legal space-time lattice)               |
                 |    Planner (planners.py, orca.py): ORCA / DWA / SFM -> primitive     |
                 |    Dynamics (dynamics.py, motion.py): primitive -> native command    |
                 +-------------------------------+--------------------------------------+
                                                 | native action (forward/left/right/stop)
                 +-------------------------------v--------------------------------------+
  simulator      |  native.py: NativeTask (unchanged habitat.Env.step, replay) and      |
                 |  BranchSimulator (exact robot-state restoration + recorded HumanTape)|
                 +-------------------------------+--------------------------------------+
                                                 | observation row (geometry sensor)
                 +-------------------------------v--------------------------------------+
  legal state    |  legal.py: LegalSensor (ideal geometry observer) -> LegalState:      |
                 |  certified grid map, human tracks, pose/action history, bump memory  |
                 +----------------------------------------------------------------------+
  runtime        |  episode.py: EpisodeRuntime (branch states for search) and replay()  |
  evaluation     |  evaluate.py, witness.py, native_cert.py, report tools               |
  Layer 2.5      |  sandbox.py, l3_interface.py (consequence features over predicted     |
                 |  human futures; Stage-1 samples plug into the same tensor)            |
```

**One decision cycle.** At a decision state:
1. The generator builds at most K queries from the legal state.
2. The selector (Oracle, default policy or a future Layer 3) picks one.
3. The executor runs it for at most one decision interval (16 native steps; waits run their duration), or until
   it completes or ends the episode.

Each native step goes through: route on the current legal map -> static plan -> planner -> dynamics realiser ->
native command. Then the simulator steps (branch simulator or `Env.step`), the legal sensor observes, and the
legal state updates. Branch states own all of this (simulator state, sensor state, legal map, tracks, executor
and planner state), so nothing observed in one branch can enter another.

**Source files** (all under `eventwm_graph/querygen/`; lines of code as of this snapshot):

| file | LOC | role |
|---|---|---|
| `native.py` | 596 | benchmark episodes, `NativeTask` (Env.step), `BranchSimulator`, `HumanTape`, contact classification |
| `episode.py` | 351 | `EpisodeRuntime` (branch states, run a query for a horizon), prefix cache, `replay()` |
| `legal.py` | 325 | `GridMap`, `LegalState` (incl. bump memory), `LegalSensor` |
| `control.py` | 371 | planar frame, body rectangle checks, legal human estimates, adapters |
| `gridplan.py` | 269 | legal-map planning view, Dijkstra fields, routes (numba) |
| `lattice.py` | 176 | body-aware hybrid A* over native primitives |
| `stplan.py` | 347 | legal space-time lattice planner (native and momentum variants) |
| `orca.py` | 189 | RVO2 port, full robot responsibility |
| `planners.py` | 558 | ORCA / SFM / DWA / follow planners, one-step hard check |
| `dynamics.py`, `motion.py`, `native_velocity.py` | 138 / 91 / 83 | dynamics families, primitive realiser, velocity action registration |
| `executor.py` | 853 | `Query`, `QueryExecution` (query semantics and execution) |
| `generator.py` | 879 | `GeneratorConfig`, `QueryGenerator` |
| `oracle.py` | 999 | `OracleConfig`, `Oracle` (search, heuristics, portfolio) |
| `evaluate.py` | 302 | search -> replay -> verdict, default-policy runner, suite driver |
| `witness.py`, `native_cert.py`, `feasibility.py`, `stcert.py`, `bodyh.py` | 262 / 147 / 300 / 119 / 165 | feasibility witnesses and certificates, privileged body cost-to-go |
| `sandbox.py`, `l3_interface.py`, `l3_env.py` | 214 / 274 / 164 | Layer 2.5 / 3 interfaces |
| `suites.py` | ~240 | frozen episode suites |
| `report.py`, `viz.py`, `diversity.py`, `sandbox_study.py`, `fanout.py`, `calibrate.py` | | reporting and studies |

---

## 3. Simulator interface (`native.py`)

* **Episodes.** `EpisodeSpec(dataset, split, scene, episode_id, seed)` addresses an original Falcon episode
  (`social_nav_v2/falcon_{hm3d,mp3d}.yaml`, `RearrangeSim-v2`, its own recomputed navmesh).
* **`NativeTask.step`** is the unchanged `habitat.Env.step`. It is used for authoritative replay and its
  official measures.
* **`BranchSimulator`** (Oracle only).
  * Falcon's humans never consult the robot (agent 0). `OracleNavObstacleAction` avoids only agents 1..N, and the
    robot-distance coupling is disabled. So one recorded `HumanTape` (human positions and setter inputs per step,
    recorded from the reset) is a valid exogenous future for every robot branch.
  * Robot states are restored exactly. Branch states keep the *setter inputs* (the last base target computed by
    an exact replica of `BaseVelAction.update_base`, joint positions, gait phase), because writing back a float32
    transform is lossy by about 1 ulp.
  * Equivalence with `Env.step` is a tested property: every authoritative replay runs a lockstep shadow simulator
    whose robot pose must match bitwise at every step.
* **Exact tape cache** (`HumanTape.cached`, `QG_TAPE_CACHE`): the tape is a deterministic function of the reset,
  so caching it is exact. It cuts episode setup from ~24 s to ~11 s.
* **Constants.** `MAX_EPISODE_STEPS = 500`; success distance 0.2 m. The robot body is the measured all-gait
  rectangle: x in [-0.646, 0.584] m, lateral [-0.243, 0.238] m, i.e. 1.23 x 0.49 m. The navmesh only keeps the
  body *centre* 0.25 m from obstacles.

## 4. Legal state and sensor (`legal.py`)

* **`LegalSensor`** wraps the project's ideal geometry sensor (`partial/fast_observer.py`, exact batched
  implementation; shared with Stage 1/2).
  * Jaw depth camera frustum: 90 deg horizontal field of view, 10 m range (Falcon task config).
  * FREE certificates come from visible configuration-space probes.
  * BLOCKED cells are observed wall hits dilated by 0.25 m (about the 0.243 m half width).
  * Human detections: exact torso with static and human occlusion. The tracker is constant-velocity, with tracks
    kept for 2 s after the last sighting.
  * `pure_caches=True` makes the sensor's private caches independent of branch history (required for exact
    branch isolation on sloped floors).
* **`GridMap`**: growable dense grid of 0.1017 m cells (robot collision reference / 6, as in Stage 1/2 data)
  holding UNKNOWN / FREE / BLOCKED. `apply()` reproduces `partial.map.PartialGraph.update` exactly (unit test
  `test_legal_map.py`), so a Stage-2 `PartialGraph` can always be exported. UNKNOWN is never treated as free.
* **`LegalState`**: map, tracks (CV estimates with age), pose / time / action history, the public goal.
  * v11 bump memory (`bump_memory_m`): a commanded forward step that moves the base < 2 cm marks a disc around the
    body's front contact point (0.684 m ahead) BLOCKED, and re-applies it after every observation. This is
    proprioceptive evidence of obstacles the camera cannot see.
* **Cost.** The sensor dominates the per-step runtime (Section 13.6).

## 5. Dynamics (`dynamics.py`, `motion.py`, `native_velocity.py`)

`DynamicsModel` describes motion capability in per-step units. Per step the robot either translates straight
ahead or turns in place:

| family | turn per step | forward levels | accel / brake | notes |
|---|---|---|---|---|
| `native` | 1/12 rad | 0 or 25/120 m | instant | the official Falcon action set (`native_actions`) |
| `slowturn` | 1/24 rad | native | instant | velocity action |
| `inertia` | 1/12 rad | 12 levels | 2 / 2 per step, turns at rest | velocity action |
| `slowturn_inertia` | 1/24 rad | 12 levels | 2 / 2 | velocity action |
| `heavy_brake`, `slowturn_heavy` | -- | 12 levels | 2 / 1 | held-out families, never used for development |

* **Official benchmark.** Only `native` is admissible, because the official action space is discrete. Both
  current configurations (C1 ORCA, C2 DWA) use native dynamics. The momentum families command a continuous base
  velocity through `QGBaseVelocityAction`: the same `update_base` integration, navmesh step filter and leg
  animation as the discrete actions. They were used for C2 in the declared-rule era (v9/v10), are outside the
  official benchmark, and are kept for research.
* **`MotionRealizer`** turns kinematic primitives into commands. For momentum families it picks the highest
  forward level that can still brake before the next planned rotation, keeps the full braking trajectory clear of
  legally predicted humans, and keeps the body off observed walls; otherwise it brakes. Rotation requires rest.
* **Calibration.** `calibrate.py` measures model vs simulator per family; `dynamics_violations` counts executed
  commands that break the model (always 0 for native).

## 6. Planners (`planners.py`, `orca.py`, `control.py`)

**Interface.** All planners receive a `PlanContext` built by the executor: planar pose, measured velocity, legal
human estimates, nearby observed-BLOCKED samples, preferred velocity from the static plan / legal route, the
static plan's primitives and poses, and a progress function (legal cost-to-go). They return a primitive
(`forward` / `left` / `right`), or a realised command for momentum families, via adapters (`DiscreteAdapter` /
`NativeAdapter`: a discrete robot without a stop/pause action tracks a velocity by duty cycling; holding = an
alternating in-place turn pair).

| planner | algorithm | key parameters (defaults) |
|---|---|---|
| **ORCA** (P1, required) | Direct port of RVO2 (`computeNewVelocity`, `linearProgram1/2/3`) with **full robot responsibility**: the robot takes the whole velocity correction because Falcon humans never react. Observed walls are hard constraints that the 3-D fallback never relaxes. Governed speed along the preferred direction. | robot disc 0.55 m, human 0.30 m, margin 0.25 m, human time horizon 1.5 s, wall horizon 0.15 s, static radius 0.18 m, neighbour range 8 m (max 8), radius growth 0.6 m/s for stale tracks |
| **DWA** (P3) | Dynamic window over the robot's admissible command *sequences* (turn options x speed caps), forward-simulated 16 steps; score = route progress + capped human clearance + heading + static-plan bonus + commitment; inadmissible if the rectangle comes within 0.30 m of a predicted human centre. | turn options (0, 1, 2, 4, 7, 11, 16), speed caps (1.0, 0.5), w_progress 1.0, w_clearance 0.6 (cap 1.2 m), w_heading 0.25, w_plan 0.15, w_commit 0.3 |
| **SFM** (P2) | Helbing social force with velocity-dependent elliptical repulsion (Johansson, Helbing & Shukla 2007) and anisotropy. | relaxation 0.25 s, contact radius 0.95 m, B 0.45 m, lambda 0.3 |
| Follow | static-plan / route follower without avoidance (diagnostic only) | -- |

SFM is implemented and was evaluated in earlier versions, but is not part of the official configurations.
Every query kind also enforces a **one-step hard check**: the next robot rectangle must stay >= 0.30 m from every
legally predicted human centre (0.27 m for `close` queries).

## 7. Planning substrate (`gridplan.py`, `lattice.py`, `stplan.py`)

* **`PlanningView`** (numba): Dijkstra cost-to-go fields on a cropped copy of the legal map.
  * BLOCKED cells are impassable; diagonals may not cut BLOCKED corners.
  * UNKNOWN is traversable optimistically at extra cost (2.5x), which supports frontier exploration.
  * Routes descend the field.
  * v10 goal relaxation (`BLOCKED_TARGET_RELAX_M = 0.5`): a target inside observed BLOCKED cells (a goal on the
    navmesh edge next to furniture) seeds the field from passable cells within 0.5 m.
* **`lattice.plan`**: body-aware hybrid A* over the robot's exact nominal primitives (forward one step, turn one
  step). Duplicates are keyed by (cell, heading bin), with at least 2 bins per turn step. Every pose's body-axis
  samples must avoid observed BLOCKED cells; forward poses also keep a nose margin. UNKNOWN costs a small penalty.
  The heuristic is the 2-D cost-to-go field.
* **`stplan.plan`** (`st_goal` queries): legal space-time lattice over (x, z, psi, t) with forward, turn and a
  two-step hold. The robot rectangle must keep 0.35 m + 0.004 m/step from every *legally predicted*
  (constant-velocity) human centre (0.28 m for `close`). It is weighted A*: time cost, static cost-to-go
  heuristic, horizon 40 steps, re-planned every 4 steps. `plan_inertia` is the momentum variant with a
  braking-tail safety check.

## 8. Executor (`executor.py`)

**Query contract.** `Query(kind, via, heading, duration, speed, label, meta)`. `signature()` (kind, rounded
via-points, heading, duration, speed and the execution-changing meta flags) is used for de-duplication.

| kind | semantics |
|---|---|
| `goal` | follow the legal-map route to the public PointGoal (optionally through via-points). STOP when the legal Euclidean goal distance < 0.15 m, or < 0.22 m with no progress over 4 steps. |
| `goto` | follow the route through ordered via-points; done within 0.30 m of the last |
| `stgoal` | follow a legal space-time lattice plan to the goal (re-planned every 4 steps) |
| `wait`, `wait_clear` | hold (native: alternating in-place turns) for `duration` steps, or until no tracked human is predicted to conflict with the route (two consecutive steps; <= 48 steps) |
| `rotate` | turn in place toward a heading (exact k-step rotations for manoeuvres) |
| `creep`, `kturn` | exact forward steps (each only if the one-step human check passes) / multi-point turns; manoeuvre mode only |
| `stop` | STOP now (offered only near the goal) |
| `continue` | keep executing the running query (via-point progress persists) |

**Per-step execution** for movement queries:
1. Compute the route on the current legal map.
2. Compute or follow the body-aware static plan (lattice), re-validated on new map evidence.
3. Derive the preferred velocity from the plan's next poses (route pursuit as fallback).
4. Call `planner.act`.
5. Realise the primitive under the dynamics model and apply the one-step hard check.

**Decision boundaries.** Every 16 steps, or when a query ends, the derived caches (planning view, fields, static
plan) are dropped identically in branches and in replay, so a continued query re-derives them from the current
legal map.

**Execution-changing meta flags** (attached by the generator):

* `tight=N`: the outer N rings of the inflated BLOCKED band become UNKNOWN.
* `close`: smaller human proxies, 0.27 m hard check.
* `commit`: follow the static plan under the hard check only.
* `exact_end` / `arrive_exact`: v10 exact rotation ends and arrivals.
* `blind`: creeps skip the body gate.
* v11 / v12 foundation flags, all default off in the frozen gate system (Section 10).

## 9. Query Generator (`generator.py`)

Deterministic and non-learning. Inputs are legal only: legal map, legal tracks, pose and action history, the
public PointGoal, the dynamics model, and the running query (for `continue`). Output: at most K executable,
de-duplicated queries plus diagnostics (raw candidates per category, conflicts, stuck flag, reset flag).

**Candidate categories and legal triggers:**

| category | trigger | queries |
|---|---|---|
| fixed | always | `continue` (running non-wait query), `goal`, `stop` (legal goal distance < 0.40 m) |
| space-time / mixed (v9) | `st_goal`, `tc_variants` | `st_goal` (legal space-time plan to the goal); `goal_tc`, `st_goal_tc` (tight(1) + close copies of the fixed goal queries) |
| human | a tracked human's CV path meets the next 3 m of the route within 1.3 m | `pass_left/right` (via 1.15 m beside the human at the conflict time, rejoin after 1.6 m), `pass_behind` |
| yield | route conflict | `yield` (nearby reachable body-clear point maximising predicted clearance), `sidestep_left/right` (0.9 m, tracked human within 3 m) |
| wait | always; `wait_clear` on conflict | `wait_short` (6), `wait_long` (16), `wait_clear` (<= 48), `goal_slow` (follow a same-direction human) |
| approach | legal route to the goal < 3 m | goal via a pre-goal point 0.9 m before the goal on headings whose body pose is clear |
| turn | route heading differs > 30 deg | rotate toward the route heading, both directions |
| maneuver | manoeuvre mode: legal stall within 1 m of the start or the goal (window 16, <= 0.15 m translation and <= 0.35 rad net turn), kept while the previous decision was a manoeuvre | exact rotations 1/4/10 steps each way, creeps 1/3 steps |
| retreat / advance | always | back 1.6 m along the own legal trajectory / 0.7 m straight ahead |
| alt / frontier | alternative homotopy whose first 2.5 m diverge / frontier clusters | via-point routes |
| observe | goal or a recently lost track out of view | rotate toward it |

**Selection.**
* `select='quota'` (K=10): fixed queries first, then priority quotas. Quotas: maneuver 8, human 3, approach 2,
  yield 2, turn 2, wait 2, alt 1, frontier 1, advance 1, observe 1; retreat, wait_long and look_lost_human only
  as leftovers (v9 quotas from solution usage). Then round-robin leftovers up to K.
* `select='compact'` (K=6): slots by situation.
  * Progress: STOP, CONTINUE (only for a non-goal query), GOAL, ST_GOAL.
  * Timing: one hold.
  * Then, by situation: manoeuvres when stuck; yield/pass on a conflict; approach near the goal; the shorter
    route turn; then advance, alt, frontier, retreat.
  * At the reset state, both route turns replace the `_tc` copies (`root_cover`).
* De-duplication in both modes: by signature and by first via-point distance < 0.45 m.

## 10. Oracle (`oracle.py`)

**Role.** Offline privileged *selection* only. Each search edge = one generator query, proposed from the branch's
own legal state and executed by the fixed executor/planner/dynamics in the branch simulator. A branch ends at an
official human contact, STOP, or step 500.

**Search: lazy weighted best-first over edges.** An edge (node, query) is simulated only when popped.

`f(child) = steps + w * h + penalties`, with:

* **h**: max over
  * geodesic / 0.17 m per step;
  * the privileged **space-time arrival time** (`SpaceTimeHeuristic`: backward dynamic programming over 0.2 m
    navmesh cells x episode steps against the true human tape; member-specific human radius; a cell-sound fallback
    table where the main table finds no arrival; `st_fallback` 300 steps).
  * Plus the in-place turn steps needed to face the geodesic path (`heading_h`).
* **penalties**:
  * rectangle clearance < 0.6 m to humans along the branch: 30 per metre;
  * a "doomed held pose" (true human future within 0.8 m over the next 60 steps): 60 per metre;
  * small per-category priors;
  * exact manoeuvres +10, padded to a full decision interval;
  * entering manoeuvre mode +120 unless every ordinary movement query from the parent failed natively;
  * once a branch has left the spawn region (1.5 m), manoeuvre edges from spawn nodes +80 (`escape_penalty`).

Further search mechanics:

* **De-duplication.** 0.3 m cells x 8 heading bins (x 10-step time bins within 4 m of a human). States reached by
  a manoeuvre edge, or within 1 m of the goal, use 5 cm x 75 heading bins. Duplicate children are dropped before
  the generator runs.
* **Goal test.** Any new state inside the success radius immediately tries the generator's STOP.
* **Exact prefix sharing** (`prefix_cache=48`): sibling edges that issue the same command prefix reuse its
  simulated steps. Results are identical with or without it.
* **Open list.** Trimmed above 4 x 3,000 entries.
* **Portfolio (v10).** Two independent best-first members interleaved one edge at a time under one shared budget:
  member 0 is weight 3.5 with the 0.5 m space-time model plus fallback; member 1 is weight 2.0 with the 0.6 m
  model and no fallback. The first success wins. Search cost is heavy-tailed and order-dependent, and the
  portfolio cuts the tail.
* **Budget.** 12,000 simulated edges (deterministic, independent of load); a wall-clock cap of 40,000 s is a
  safety net only.

**Diagnostic variants:**
* `intervention_cost` (minimum-intervention Oracle: an f penalty per decision that deviates from the default
  policy) gives the fewest "critical" decisions.
* `improve_edges` (anytime refinement) was tested with no gain and dropped at the user's request.
* `noise` (randomised tie-breaks) and `witness_dir` guidance are unused in the frozen configuration.

## 11. Configurations

Frozen gate system (`querygen-freeze-v10-official`, launcher `scripts/querygen_test_v10_official.sh`):

| id | planner | dynamics | generator (overrides) | Oracle |
|---|---|---|---|---|
| C1-K10 | ORCA | native | `tc_variants=(goal, st_goal)`, `st_goal`, `maneuver_spawn_only=False`, K=10 quota | portfolio [{weight 3.5}, {st_radius 0.6, st_radius_opt 0}], `heading_h`, `escape_penalty 80`, `prefix_cache 48`, 12,000 edges |
| C1-K6 | ORCA | native | `select=compact`, K=6, `tc_variants=(goal,)`, `st_goal`, `maneuver_spawn_only=False` | same |
| C2-K10 | DWA | native | as C1-K10 | same |

**Default-policy (foundation) options in the current tree.** These are flags, all off in the frozen gate system,
attached by the generator as query meta.

* **v11** (superseded; Section 13.3):
  * `goal_unstick=6`: deadlock breaker. A goal query whose robot has not translated for 6 steps with no tracked
    human within 2 m switches to the tight map and static-plan following, or rotates toward the route when no
    body-feasible plan exists.
  * `bump_memory_m=0.25`.
  * `evade`: human-threat reflex. If the query's own next primitive plus holding comes within 0.30 m of a tracked
    human's CV prediction within 14 steps, it takes the clearance-maximising short native sequence instead.
* **v12 candidate** (adds):
  * `unstick_progress`: escalate only on a genuine deadlock, i.e. no translation for N steps with <= 1.5 turn
    steps of net rotation, or no translation for 40 steps.
  * `unstick_keep`: the escalated plan survives decision boundaries.
  * `bump_front`: the reflex and the breaker respect bump memory at the body's front contact point.
  * `static_humans`: tracked humans with a near-zero velocity estimate do not block escalation, and escalated
    plans route around a 0.35 m disc per standing human.
  * `stop_strict`: a stuck STOP only within 0.19 m.
  * Launcher draft: `scripts/querygen_final_v12.sh`.

**Episode suites** (`docs/querygen/suites*.json`, `suites.py`; salted-hash construction, pairwise disjoint scenes):

| suite | episodes | use |
|---|---|---|
| DEV | 28 (HM3D val) | development |
| unseen 1 / unseen 2 | 40 / 60 (HM3D + MP3D train) | development (consumed by earlier version tests) |
| **unseen 3** | 60 | **gate test of the frozen v10 system** (first run 2026-10-28 13:51) |
| **unseen 4** | 60 | frozen 2026-10-28 (sha256 `73d307e3...`) for the v12 held-out test; never run |

## 12. Evaluation and feasibility tooling

* **`evaluate.py`**
  * `one` / `suite`: Oracle search, then `episode.replay`: the recorded query sequence is re-executed through the
    live executor from reset with the unchanged `Env.step`, plus a lockstep shadow `BranchSimulator` whose pose
    must match bitwise. Then the verdict.
  * `exec_one`: the default policy (no selection).
  * The suite driver copies the querygen package into `<run>/code_snapshot` and every episode subprocess imports
    from it; each record carries the snapshot's sha256.
* **Feasibility.**
  * `witness.py --rules native`: weighted A* over exactly simulated native primitives with the privileged
    space-time heuristic. A found witness is replayed and verified in `Env.step` and proves feasibility.
  * `native_cert.py`: proves infeasibility (Section 1).
  * `scripts/querygen_test_feasibility.sh` applies both to unsolved test episodes.
  * `feasibility.py` (declared-rule start certificates) and `stcert.py` (a sound but loose space-time relaxation)
    are not used for official accounting.
* **Reporting and visualisation.** Under `tests/querygen/`:
  * `report_official.py` (aggregate JSON), `native_accounting.py`;
  * `viz_episode.py` (top-down path figures, incl. failed branches), `render_topdown.py` (top-down videos: default
    policy vs Oracle side by side, shared human motion), `curate_media.py`, `build_page.py` (results page);
  * `render_video.py` (RGB chase camera; blocked, Section 14).
* **Unit tests** (`tests/querygen/test_*.py`, run as modules via `scripts/querygen_run.py`):
  * legal map == PartialGraph update;
  * ORCA LP correctness (300 scenarios);
  * yaw conventions;
  * vectorised sandbox scoring == reference;
  * velocity action;
  * pure caches;
  * canonical contacts;
  * space-time heuristic.

  The legal-map, ORCA, yaw and sandbox tests were re-run on the current tree on 2026-10-28 and pass; the others
  were last run at earlier milestones.

### Layer 2.5 / 3 interfaces

* **`sandbox.py`**: runs the same `QueryExecution` on a frozen legal map against predicted human futures,
  `F [M, T+1, N, 3]`.
  * `rollout()` gives an open-loop nominal trajectory; `score_vec()` gives vectorised clearance statistics over
    all M futures.
  * `cv_futures()` (constant velocity + noise) stands in for Stage-1 samples; the interface takes any array of
    that shape.
* **`l3_interface.consequence_features`**: per candidate, legal features. Descriptors, legal progress, effort,
  social risk over the futures (P(overlap), P5/P50 minimum distance, PSC steps) and static risk.
* **`l3_env.QueryEnv`**: the decision-level environment.
  * `reset()` / `step(i)` over the K candidates; observations are legal only.
  * The reward is a training signal (progress, time, success, collision).
  * `official_replay()` re-executes the decision sequence with `Env.step` and is the evaluation of record.
* **`tests/querygen/l3_readiness.py`**: the readiness study. At decision states along Oracle solutions it labels
  every candidate as "candidate, then default policy, succeeds", computes top-1 success of legal selection rules
  (including a leave-one-episode-out logistic regression on the legal features), and one-decision recovery.

## 13. Results

### 13.1 Feasibility accounting [verified]

| suite | episodes | proven natively infeasible | evidence |
|---|---|---|---|
| DEV | 28 | 0 | all 28 confirmed feasible (witness or PASS) |
| unseen 1 | 40 | 0 | 39 confirmed feasible; nMeXfQU4PMS **unresolved** (2-h native witness searches found no solution and no proof) |
| unseen 2 | 60 | 3 | 5Poh4Qz68hd, sKLMLpTHeUy (every first action -> official contact at step 1), AMEM2eWycTq (all sequences by step 6) |
| unseen 3 (test) | 60 | 2 so far | sWAdxSLVPQC, q28T9C3q2dv: every first action -> official human contact at step 1. Certified in 34 s / 56 s; all 3 closed sequences reproduce the official collision. |

### 13.2 Oracle capability, frozen system, official rules

**Development set** [in progress]. Counts are solved / finished episodes not proven infeasible. Unfinished
episodes are the slowest searches (some take more than 3 h under load).

| configuration | DEV | unseen 1 | unseen 2 | all finished | SPL (solved) | PSC (solved) | human collisions (solved) |
|---|---|---|---|---|---|---|---|
| C1 ORCA K=10 | 25/25 | 24/24 | 49/50 | **98/99 (99.0 %)** of 125 | 0.91 | 0.92-0.95 | 0 |
| C1 ORCA K=6 | 25/25 | 24/25 | 51/53 | **100/103 (97.1 %)** of 125 | 0.89-0.91 | 0.89-0.94 | 0 |
| C2 DWA K=10 | 18/18 | 21/21 | 43/44 | **82/83 (98.8 %)** of 125 | 0.96-0.97 | 0.92-0.94 | 0 |

Failures so far (all episodes are feasible):
* **PFLHHbjscNN** fails in all three configurations. The Oracle's best branch stalls 4.2 m from the goal before a
  narrow corridor after 12,000 edges; a 127-step native witness exists.
* **gTV8FGcVJC9** and **s19Uyn7AWwv** fail with K=6 only. The search tree is exhausted (2,478 / 4,956 edges)
  because every K=6 branch meets a human early; witnesses of 127 / 111 steps exist. This is a K=6 coverage gap
  at a spawn-area human threat.

Other statistics:
* Oracle edges per solved episode, P50 / P95: C1-K10 103-157 / 1,435-2,019; C1-K6 83-107 / 1,491-3,029;
  C2-K10 57-83 / 761-1,713.
* Steps on solved episodes: P50 95-118.

**Held-out gate test, unseen 3** [in progress; first run 13:51]. 60 episodes; 2 proven infeasible -> denominator 58.

| configuration | solved / finished | failures among finished | SPL | PSC |
|---|---|---|---|---|
| C1 ORCA K=10 | 40/40 | none | 0.935 | 0.950 |
| C1 ORCA K=6 | 36/36 | none | 0.914 | 0.963 |
| C2 DWA K=10 | 28/28 | none | 0.966 | 0.982 |

The gate is decided when all 58 are finished. Unfinished or unresolved episodes count as failures, so each
configuration may lose at most 2 of the 58 (>= 56/58 = 96.6 %).

**Superseded results** (declared rules, see `REPORT_V9.md` / `REPORT_V10.md`). They were judged under stricter,
non-official rules and are reported only for history: v9 re-scored as headline DEV C1-K10 21/28 (75 %), unseen 1
28/40 (70 %).

### 13.3 Default policy alone ("floor"; no selection) [partial]

The default policy keeps the running query, else issues the goal query. Episodes proven infeasible are excluded.

| foundation | ORCA | DWA | status |
|---|---|---|---|
| v10 (frozen gate system) | 18/94 (19 %) | 15/88 (17 %) | [in progress] |
| v11 (unstick + bump memory + evade) | 28/49 (57 %) | 28/49 (57 %) | stopped; superseded |
| v12 candidate (+ progress-gated unstick, plan keeping, bump-aware reflex, standing humans, strict stuck-STOP) | 48/81 (59 %) | 44/83 (53 %) | [in progress] |

* Paired on the same episodes, v10 -> v12: ORCA 16 -> 48 of 81, DWA 15 -> 44 of 83.
* **v11 was superseded for a design reason, not its success rate.** Its breaker counted in-place turns as "not
  moving", so it took over in 88 of 100 runs (mostly the first turn toward the route). ORCA and DWA then executed
  identical action sequences in 24 of 48 paired episodes: the planners were bypassed. Under v12, identical
  sequences fell to 8 of 82.
* **Remaining v12 failures.**
  * Human collisions (ORCA 18, DWA 30). Humans approaching head-on with stop-and-go motion that the CV reflex
    mispredicts (e.g. Qpor2mEya8F: estimated speed 1.6 -> 0.2 -> 0 -> 1.9 m/s in the last 8 steps), and humans
    beside the robot outside its 90 deg camera. These are decision problems for Layer 3 (yield or side-step
    earlier).
  * Timeouts at legal dead ends, e.g. iePHCSf119p, whose bump memory walls off the passage it chose.
* **Behavioural fact (Falcon).** Humans stop for good after 100-400 steps (by step 300 most stand still). Standing
  humans in corridors were the main late-episode deadlock of v11's planners (Nfvxx8J5NCo held 300 steps behind two
  standing humans at 1.0 / 1.3 m). v12's `static_humans` addresses this.

### 13.4 Learnability for Layer 3 [pilot]

* **Critical decisions** (minimum-intervention Oracle, v10, C1-K10, intervention cost 300 steps, 4,000 edges; 11
  solved episodes before it was switched to v11 and then stopped): 0 non-default decisions in 3 episodes, 1 in 7,
  2 in 1 (median 1). Most common critical choice: the initial turn direction (`turn_right` 4), then `goal_tc` 2,
  `st_goal` 2. An earlier probe found 8B43pG641ff needs 11 of 17.
* **Structure of Oracle solutions** (frozen system, solved development episodes):
  * Median 8 (C1-K10), 9 (C1-K6) and 7 (C2-K10) decisions per episode.
  * 58 % / 50 % / 64 % of decisions are the default (continue / goal / stop).
  * The non-default vocabulary is compact: `st_goal`, `advance`, `goal_tc`, `st_goal_tc`, route turns,
    `wait_short`.
* **Readiness study pilot** (3 episodes on the superseded v11 foundation plus one v10 smoke test; the numbers only
  illustrate the instrument):
  * Easy episodes have 6-9 of 10 good candidates per state.
  * 13 of 14 wrong candidates were recoverable with one later decision.
  * Smoke test 3Ao63EY7J83: 3 of 10 candidates good at the critical first decision.
* **Not done yet:** the readiness study at scale on v12 Oracle solutions (identifiability AUCs, top-1 success of
  legal rules, recovery rates, K=6 vs K=10 good-candidate counts).

### 13.5 Visualisations [verified, interim]

Interim results page (private artifact): https://claude.ai/artifact/4ukZJJDjoYWJeQS2H2Q3CN. It contains capability
and floor charts, the metrics table, and top-down recordings: default vs Oracle on two collision and two deadlock
cases (ORCA and DWA), the PFLHHbjscNN failure, and a whole-episode path figure. Media and page sources are
generated under `output/runs/official/` (not in git).

### 13.6 Speed (loaded 36-core machine, load ~57-67; per process)

* Generator: P50 49-60 ms per decision (P90 112-140 ms).
* One 16-step decision edge: P50 0.58 s (ORCA) to 0.80 s (DWA), i.e. 62-76 ms per simulated step including the
  legal sensor. The sensor is about two thirds of a step (~0.10 s/step in default-policy runs).
* Episode setup: ~43 s P50 without the tape cache, ~11 s with it.
* Sandbox consequence features: ~0.7 s per decision (K rollouts x 16 futures).
* Implication for Layer 3: a decision-level environment runs about 1-2 decisions/s per process. Decision-level RL
  or imitation over ~10 decisions per episode needs process parallelism (tens of processes) or offline datasets
  (Oracle search trees).

## 14. Unresolved issues and limitations

1. **Gate not yet formally established.** The held-out test is running; the development runs have unfinished
   long searches. nMeXfQU4PMS is unresolved and counts as a failure.
2. **PFLHHbjscNN** (narrow corridor) fails in every configuration. The Oracle's queries cannot get the 1.23 m body
   through within 12,000 edges, though a primitive-level witness exists.
3. **K=6 coverage gaps** at spawn-area human threats (gTV8FGcVJC9, s19Uyn7AWwv): every K=6 branch is caught by a
   human early, while K=10 or witnesses succeed.
4. **v12 is not frozen.**
   * Its Oracle capability (all three configurations) has not been measured.
   * Its default-policy floor run is partial.
   * The fourth unseen suite is reserved for its held-out test.
5. **Partial observability bounds the floor.** 90 deg camera, 2 s track memory, CV predictions. Many remaining
   default-policy collisions involve humans the robot cannot see or predict; only earlier, better query choices
   (Layer 3, Layer 2.5 predictions) can address them.
6. **Runtime.** The legal sensor dominates simulation cost. Speeding it up without changing its output is open
   work (the observer is an exact, already batched implementation shared with Stage 1/2).
7. **RGB recordings blocked.** EGL cannot create a context because the NVIDIA kernel driver (595.91) and
   user-space library (595.99) mismatch. Top-down videos are used instead.
8. **Out of scope by instruction:** training Layer 3, retraining Stage 1/2. The sandbox uses the
   constant-velocity stand-in until Stage-1 samples are plugged in.

## 15. Reproduction

* Runner: `python scripts/querygen_run.py <module> [args]`. It installs `eventwm_graph` as a namespace, pins
  single-threaded math libraries and the verified native ray-batch build, and imports a frozen snapshot when
  `QG_SNAPSHOT` is set.
* Gate test: `scripts/querygen_test_v10_official.sh {c1k10|c1k6|c2k10} output/runs/official/test_v10 7`. Then
  `scripts/querygen_test_feasibility.sh output/runs/official/test_v10 unseen3`.
* Development runs (official rules): `scripts/querygen_dev_official.sh CONFIG OUT_ROOT [GEN_JSON] [ORACLE_JSON]`.
* Default policy: `python scripts/querygen_run.py eventwm_graph.querygen.evaluate suite SUITE --policy executor
  --out DIR --planner {orca|dwa} --rules official --gen JSON`.
* Aggregate: `python3 tests/querygen/report_official.py --out report.json`. Page:
  `python3 tests/querygen/build_page.py --report report.json --out DIR`.
* Protocols: `docs/querygen/FINAL_PROTOCOL_V10_OFFICIAL.md` (gate), `OFFICIAL_RULES.md`, `LAYER3_READINESS.md`;
  development log `PROGRESS.md`.
## 16. Versions v12 and v13 (2026-10-28 evening)

### 16.1 v12: legal foundation and capability fixes (frozen `querygen-freeze-v12` = b0f3c1a3)
Generator / executor flags, all legal (generation uses only the legal state):

| flag | what it does | evidence on development data |
|---|---|---|
| `vc_variants=(goal, st_goal)` | contact-level copies `goal_vc`, `st_goal_vc` (tight=2 map, 0.20 m hard clearance to predicted human centres, small planner proxies; instant-stop robots only) | half of v12 solutions use a `_vc` query; s19Uyn7AWwv solved (crowded spawn) |
| `stop_closest_m=1.2` | STOP also offered at the legal closest approach when the goal point is not legally reachable | s8pcmisQ38h (goal 0.72 m off the navmesh): 12,000-edge failure -> 607 edges |
| `compact_long_turns`, `compact_human_first` (K=6) | 10-step rotations and both sidesteps get slots | gTV8FGcVJC9 K=6: exhausted -> 153 edges |
| `maneuver_vclose` | manoeuvres use the contact-level one-step check | s19Uyn7AWwv DWA |
| `goal_unstick=6` + `unstick_progress` + `unstick_keep` + `static_humans`, `bump_memory_m=0.25` + `bump_front`, `evade`, `stop_strict` | default-policy foundation (deadlock breaker only on genuine deadlocks, bump memory at the body front, human-threat reflex, strict stuck-STOP) | default policy alone: ORCA 72/128 (56%), DWA 64/128 (50%) vs v10 ~18% |

Development run (ORCA K=10, DEV + unseen 1-3, 188 episodes): mean Oracle edges 232 vs v10 397 on common episodes,
SPL unchanged (0.92), fewer turn reversals (1.5-2.3 vs 2.0-3.2 per 100 steps).

### 16.2 v13: exact speed-up and the privileged witness fallback (frozen `querygen-freeze-v13` = 95585f5f)
* **Exact fast legal sensor** (`fastsense.py`, `QG_FAST_SENSE=1`).  Same rows, trackers and certified cells as the
  accepted observer (checked observation by observation: 5 episodes, 2,196 observations, 0 differences; search
  digests identical).  (1) Static-occlusion memo: support probes are cast through the same native castRay with all
  hits and classified; a probe blocked by the stage stays blocked from that exact camera origin whatever moves, so
  the cell is skipped at later observations from there (a waiting robot, branches through the same pose).  (2) The
  inflated wall band is computed in one vectorised pass with the reference's float64 arithmetic and boundary rule
  and published as arrays (`UpdateBlock`), read directly by the legal map and the prefix cache.  Load-normalised
  sensor speed-up 1.0-2.8x (median ~1.8x); remaining search cost: native castRay ~35%, planning ~25%.
* **Witness fallback** (generator `primitives`, Oracle `prim_fallback_sims=400000`).  The generator also offers the
  exact one-step native queries `prim_f/l/r/s` (outside K, no legal input).  The ordinary search never expands them,
  so it is the v12 search: 12 development episodes re-run with the v13 code give identical decisions, edge counts,
  steps and official metrics.  When the ordinary search fails, the Oracle runs the native witness search
  (`witness.py`: weighted A* over the native primitives, true human tape -- privileged selection only) from the
  reset inside the search runtime and executes the witness as primitive queries; every step is simulated and
  legally observed again and the authoritative replay verifies it like any branch.  Primitive decisions bypass the
  planner, so SR with and without the fallback and the primitive-decision share are always reported separately.
  A takeover option (hand control back to the default policy as early as possible) exists but is off: on
  VqCaAuuoeWk it saved 30 of 94 primitive decisions for SPL 0.989 -> 0.909.

### 16.3 Gate tests of v12 and v13 [verified, final]
Untouched suites, frozen code snapshots, official rules, query level (the v13 witness fallback never fired):

| test | configuration | PASS | certified infeasible | PASS rate over the denominator | SPL (passes) | PSC | human collisions | Oracle edges P50 / P95 |
|---|---|---|---|---|---|---|---|---|
| v12, fourth unseen suite (60) | ORCA K=10 (primary) | 58 | 0 | **58 / 60 = 96.7%** | 0.906 | 0.941 | 0 | 91 / 5,000 |
| v13, fifth unseen suite (60) | ORCA K=10 | 58 | 2 | **58 / 58 = 100%** | 0.904 | 0.934 | 0 | 94 / 2,526 |
| v13, fifth unseen suite | ORCA K=6 | 58 | 2 | **58 / 58 = 100%** | 0.894 | 0.933 | 0 | 80 / 2,659 |
| v13, fifth unseen suite | DWA K=10 | 58 | 2 | **58 / 58 = 100%** | 0.950 | 0.921 | 0 | 81 / 2,831 |

* v12 failures, both FAIL_SEARCH, both kept in the denominator. yqNxxJnA3iL (closest 0.48 m) is natively feasible:
  the witness search found and officially verified a 135-step solution (SPL 0.956). For oKFJo8jpzRW the native
  witness search was inconclusive (400,000 simulations; best geodesic 6.23 m).
* v13's infeasible episodes are wrq3kiEU4VR and Uxmj2M2itWa. Both carry exact native certificates with official
  verification.
* Earlier: frozen v10 on the third unseen suite reached 53/57 (93.0%) with ORCA K=10 and DWA K=10, and 51-52/57 with
  ORCA K=6. The gate was missed.
* v13 fallback on v12's development failures (`output/runs/official/v13fb`, ordinary search skipped -- it is the
  v12 search): VqCaAuuoeWk SPL 0.989, p32JzpQyhPk 0.975, ANmWrL7Kz7h 0.959, Nfvxx8J5NCo 1.0, FXuXGH9YQTW 0.98 -- all
  official PASS; 94-159 primitive decisions each.  Not used for any gate claim (user, 19:10: last resort).

### 16.4 Decision robustness of the v12 solutions [in progress, 28 of 35 episodes]
`querygen/decision_robustness.py` (`output/runs/official/decision_v12`): at every decision state of an Oracle
solution all K candidates are executed for one interval; near-duplicates (same outcome, end time +-3 steps, end
pose 0.25 m / 20 deg, path 0.25 m) are merged into choices; each choice is continued by the default policy ("easy")
or a bounded minimum-intervention Oracle ("clever", 200 edges) or found dead.  110 states: 5.3 distinct choices of
10 candidates, 4.9 viable, 3.5 recoverable by the default policy alone, 3.8 with good SPL (>= 90% of the best
choice); a single viable choice in 0.9% of states, a single good-SPL choice in 13.6%.  Of 451 suboptimal choices
(not the Oracle's), 64% are recovered by the default policy, 27% need a clever continuation, 9% are dead; mean SPL
loss 0.108.  The good-SPL shortfall mostly reflects the default policy's own route efficiency (success SPL ~0.81).

## 17. Versions v14 and v15; the frozen v14 system (2026-10-28 night)

### 17.1 What changed (all legal; query level -- the privileged witness fallback is not used, user 19:10)
* **v14 = v12 plus four changes:**
  * the exact fast legal sensor;
  * generator `goal_unknown_cost=1.25` (goal queries price unobserved cells 1.25x instead of 2.5x);
  * Oracle path-aware SPL refinement (`improve_edges=300`, `improve_lambda=1`, below path efficiency 0.85);
  * contact-tolerant `_th` goal copies replacing the `_vc` copies in the K=10 configurations.

  The `_th` copies (`executor._ThruLegal`) treat observed wall-band cells as soft obstacles: body checks are
  optimistic and routes cost x6, except at proprioceptive bump-memory discs. Long-tail analysis showed why: every
  hard development witness crosses wall-band cells at body height. The legal band is visual (wall hits 0.2-1.4 m
  above the base, dilated 0.25 m), while Falcon's kinematic, navmesh-bound Spot clips through furniture (static
  contact is legal).
* **v15 = v14 + short-interval goal copies near tracked humans** (`short_human_m=2.5`, `short_steps=4`; Oracle edges
  of 4 steps). VqCaAuuoeWk was solved in 280 edges (v14: 7,154).
* Tested and not adopted: `xc_variants` (committed contact-level copies) and `maneuver_thru` (thru semantics for
  manoeuvres).

### 17.2 Development results (DEV + unseen 1-4; official rules; query level)

| version | episodes not proven infeasible | PASS | SPL (passes) | PSC | human collisions | Oracle edges P50 / P90 | mean search |
|---|---|---|---|---|---|---|---|
| v12 | 182 (DEV + unseen 1-3) | 171 (94.0%) | 0.900 | 0.932 | 0 | 87 / 2,255 | 18.4 min |
| v14 | 242 (DEV + unseen 1-4) | 235 (97.1%) | 0.915 | 0.928 | 0 | 77 / 1,493 | 8.3 min |
| v15 | 240 of 242 finished at freeze | 232 of 240 paired | 0.919 | 0.930 | 0 | 71 / 1,155 | 6.1 min |

* **v14 vs v12, 205 paired episodes:**
  * PASS 188 -> 194;
  * gained VqCaAuuoeWk, p32JzpQyhPk, Nfvxx8J5NCo, YRUkbU5xsYj, Pm6F8kyY3z2, jh4fc5c5qoQ and uNb9QFRL6hY;
  * lost FXuXGH9YQTW: a 7,008-edge v12 solution that uses no `_vc` copy, so this is search-order sensitivity;
  * SPL on both-pass 0.896 -> 0.914; median edges 82 -> 77; mean minutes 16.8 -> 7.5.
* **Remaining v14 development failures:**
  * 3t8DB4Uzvkt, ANmWrL7Kz7h, FXuXGH9YQTW, tEafuWwhhwr: human timing / band.
  * nMeXfQU4PMS: native witness search inconclusive.
  * oKFJo8jpzRW: witness inconclusive.
  * yqNxxJnA3iL: natively feasible, verified witness; the query-level search stops 0.48 m short.
* Expert benchmark (1,500-edge cap, query level): 64 / 68 feasible solved (v12-era fast sensor: 63 / 68). P50 46 s,
  mean 152 s, P95 676 s per episode, under a heavier machine load.

### 17.3 Freeze
Pre-registered rule (PROGRESS.md, 22:58; freeze time declared 23:53, before the last v15 results):
* freeze v15 iff its paired PASS >= v14's and its mean search time is lower; otherwise v14.

Result on 246 paired development episodes:
* PASS 233 (v14) / 232 (v15; jh4fc5c5qoQ PASS -> FAIL); mean minutes 7.85 / 6.50; SPL 0.917 / 0.919.
* The two v15 episodes still running were v14 PASS and could not change the outcome.

**v14 frozen.** Tag `querygen-freeze-v14` points to e8fb1c77, the code snapshot every v14 development episode
imported (package sha256 21cc7deb...). Protocol: `FINAL_PROTOCOL_V14.md`; launcher: `scripts/querygen_test_v14.sh`.

### 17.4 Held-out test of the frozen v14 (sixth unseen suite, 60 HM3D episodes, never run or inspected before)
<!-- V14TEST -->
