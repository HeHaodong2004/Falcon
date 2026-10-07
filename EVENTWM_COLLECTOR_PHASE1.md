# EventWM collector Phase 1

One-root correctness experiment on the existing `eventwm-falcon` environment,
branch `research/eventwm-data-collector`, checkpoint/tag
`a706993` / `eventwm-pre-collector`. No environment installation, baseline edits,
branch switch, commit, or scene sweep. This is a counterfactual intervention
collector, not a FALCON policy evaluation or a set of successful trajectories.

## Files and interfaces

Added:

- `scripts/eventwm_collect.py`: bounded one-root orchestration, grouped storage,
  label CLI, machine-readable validation and diagnostic plot/video.
- `scripts/collector/core.py`: immutable query serialization, legal root/observation
  whitelist, local target and temporal variants, motion compiler, offline labels,
  explicit `load_wm_input(root_dir, query_id)` reader.
- `scripts/collector/runtime.py`: standalone static planner, seeded process replay,
  native environment runner, contact/state recorder, controller fingerprints.
- `tests/collector/test_collector.py`: semantic regression tests.
- `EVENTWM_COLLECTOR_PHASE1.md`: this document.

No existing upstream or controlled-matrix source was modified.

Reused `scripts/controlled/run_case.py::build_config`,
`controlled/runtime.py::{ControlledRearrangeSim,ControlledPointNavTask}`,
`evaluation/controlled50` episode `002_castle_h1` (episode 3002, seed 2301),
Skokloster Castle, and its existing experiment-owned static navmesh. Native imports
load `habitat_sim` first, as required by the validated installation.

Relevant discovered interfaces:

- `habitat.Env.reset/step`, `env.sim`, `env.task`; tuple task actions run the robot
  and active native human controller, then advance physics and update sensors.
- `falcon/rearrange_sim_v2.py::capture_state/set_state`: transforms/joints and some
  rigid-body velocities, but no complete human-controller, clock, task or RNG
  snapshot. `set_state` explicitly zeros articulated joint velocities.
- `get_agent_data(i).articulated_agent`, `sim_obj.transformation`, joint positions /
  velocities, root linear/angular velocities; `base_pos`, base transform.
- `OracleNavRandCoordAction_Obstacle.step`: live waypoint progression and humanoid
  gait/controller update on **every branch**, never a copied future trajectory.
- `habitat_sim.PathFinder.load_nav_mesh/snap_point/find_path`, `ShortestPath`.
  Query planning uses a **separate standalone PathFinder**, not the live world.
- Native `BaseVelAction` integrates a local (+x forward, +y yaw) velocity over
  `1 / sim.ctrl_freq`; `step_filter` constrains translation to the static NavMesh.
- `sim.step_world`, `get_world_time`, `get_physics_time_step`,
  `perform_discrete_collision_detection`, `get_physics_contact_points`.
  Recorder wraps `step_world` passively; it does not change the stepping schedule.

The continuous robot action uses existing `BaseVelAction`, maximum native linear
scale 25 and angular scale 10, and `allow_dyn_slide=True`. It does not call a human
avoidance planner or inspect human states. Robot gait animation is not driven by
this action (unlike discrete policy walking). Human action configurations are
unchanged. Only collision and step-count task measures are retained; navigation
success/reward/goal sensors and oracle sensors are unnecessary for fixed queries.
Native robot-human termination is retained; collector obstacle-contact and
execution-failure termination are additional research collection rules.

## Root chronology and restoration

Initialize scene, robot and all six allocated human slots (one active, five parked)
using the validated controlled reset. Then execute **40 real WAIT actions with the
human controller running** and acquire real robot observations. The root occurs
at simulator time **1.944 seconds**, not an assumed `40 / fps`.

Each query uses a brand-new Python process and environment, identical Python,
NumPy, Torch, environment seed, episode and exact prefix controls. This resets
module globals, controller objects, Bullet world/clock, sensor setup and task state.
It avoids relying on incomplete snapshots or position-only restoration.

Before every branch, require an exact canonical root fingerprint containing active
robot/human transforms, joints, native velocities, contacts, controller goals,
progress, old human positions, random speed scale and numeric gait fields, task
step, Python/NumPy/Torch RNG hash, history hash and first-person image hash. The
fingerprint is an audit, not a claim that every private simulator field can be
serialized. Repeated full consequences provide the stronger replay check.

## Query representation and clock semantics

Eight local directions, 1.5 m target radius, static target projection (at most
1.5 m), static geodesic length in (0.15, 4) m, duplicate projected target removal.
Each path gets immediate normal, wait 0.5 s normal, wait 1 s normal, immediate slow
(0.5 speed scale). Explicit WAIT lasts as long as the longest compiled query.
No future human state, human goal, collision label or hidden human position is an
input to generation. Local target coordinates are forward/left metres.

Each `Query` owns an immutable canonical JSON string. It includes `query_id`,
`root_id`, local target, projected world path, nominal `[t,x,y,z,yaw]` samples,
**all executable `[linear, angular]` controls**, control/integration dt, requested
and effective wait, speed scale, duration, path length, robot/research radius,
NavMesh specification, generation config and timing policy. SHA256 covers the
entire representation, including ID. Each ID is a hash of the body before adding
its ID. Controls are frozen before any branch. JSON access returns a fresh copy.
Both pre/post hashes are asserted and saved; execution never replaces nominal data.

Compilation uses rotate-then-translate segments (no online tracker), effective
normal speed 0.7 m/s and turn speed at most 1.5 rad/s. Native command magnitudes
account for the distinct control integration dt. WAIT issues zero velocity and
holds the kinematic base; it does not invoke STOP, which ends the native task.

**Measured clocks:** integration dt = 1/120 s; native physics quantum = 0.008 s;
the task requests 1/60 s physics and the simulator adds four substeps per action.
Nominal control dt = 0.0486666666667 s. Actual action intervals alternate between
0.048 and 0.056 s due to native accumulated time quantization. Fixed commands
advance once per environment action; nominal schedule error is audited at <=
0.008001 s. Sensor observations arrive once per action. Actual recorded simulator
timestamps determine observed durations, velocities, event times and exposure.
Physics-call recording gaps are at most 0.024 s (the task request can advance
three native quanta); intermediate kinematic poses do not change within that call.
This is sufficiently fine recording of the current kinematic setup, not a claim
to a callback after every Bullet internal substep. Revisit for dynamic bodies.

## Dataset schema and legal/privileged boundary

Dataset directory: `datasets/raw/phase1_castle_smoke_v3/`.
The existing `datasets` symlink is preserved. The sandbox requires host permission
for GPU execution and writes through this symlink.

```
roots/replay.json                 episode/seed/exact prefix recipe (privileged)
roots/legal_root.json             whitelisted robot state/clock for generator
roots/privileged_root.json        root state + exact fingerprint
roots/prefix_truth.json           real prefix simulator truth
legal_observations/history.json   one shared immutable legal history
legal_observations/history_images.npz  grouped timestamp/RGB/depth arrays
legal_observations/branch_images.zip   grouped branch sensor archives
queries/queries.json              [{sha256, query}, ...], frozen before rollouts
branch_truth/branches.jsonl       one record per executed branch
labels/labels.json                offline-derived labels
audits/                          provenance, machine-readable audits, worker logs
diagnostics/comparisons.json      active vs WAIT human trajectory/speed/travel
diagnostics/comparison.png       same-path timing / speed / reference plot
diagnostics/comparison.mp4       first-person RGB comparison
diagnostics/render.json         chosen branches and frame-hold convention
```

`audits/` contains `provenance.json`, `generation.json`, `audits.json` and
`workers/<branch_id>/`: resolved native config, actual episode, reset crowd audit,
root fingerprint, prefix clock trace, execution command/return code/runtime/log,
raw branch record and grouped images. Worker duplicates are intentionally retained
for Phase-1 auditability. There are no per-timestep files. Logical raw truth is one
JSONL per root; query/history/labels are each grouped.

Each raw branch contains IDs, pre/post query hash, root fingerprint, planned and
actual duration, completion, termination reason, endpoint error, `controls`,
`truth`, and future `legal_observations` metadata. Each control records index,
actual start/end timestamps, nominal and issued command. Each truth row records
actual timestamp, phase, requested physics dt, control index, issued command,
robot/human states (positions, yaw, transforms, joints, native velocities,
object IDs, finite-difference velocities when time advances), full native contacts,
and separately classified native human/static contacts. Contact fields include
object/link IDs, both world contact positions, normal, distance, force, active flag.
Termination reasons: query_completed, robot_human_collision, static_collision,
execution_failure (endpoint error >0.3 m), timeout, environment_termination;
worker process failures have separate `collection_error.json` and execution log.

The legal stream contains only timestamp, robot pose/odometry, past command,
first-person jaw RGB/depth, `human_tracks=[]`, `tracking_available=False`.
**There is no detector/tracker yet**; hidden people cannot appear as GT tracks.
Pixels have real renderer occlusion. Third-person cameras, human count/state,
controller goals/progress, contacts and all oracle task sensors are excluded.
`load_wm_input` explicitly enumerates legal history fields and reads only the
shared history/images and chosen frozen query. Future branch observations stay
under branch records and cannot be appended to the current root history.

## Offline labels

`build_labels` takes only stored branch truth, independently of rollout:

- Native human contact: any native contact pair linking robot to an active human.
- Research disc contact: horizontal centre distance <= 0.25 + 0.30 m; distinct
  from native Spot/humanoid collision geometry. Clearance is distance minus 0.55 m.
- Human collision event: union of the above, with separate first-native and
  first-disc event times retained. All times are relative to actual root time.
- Static obstacle contact: native robot/nonhuman contact excluding near-vertical
  ground support (absolute normal y >0.8 and both contact points below base y+0.12).
  Full unfiltered contacts are retained so this heuristic can be revised offline.
- Minimum human clearance over recorded samples. Zero-human value is JSON null.
- Near exposure: sum actual timestamp differences using right-endpoint clearance
  <0.5 m. This sample convention and radii are explicit research assumptions.
- Completion, censoring, event-observed mask, termination, planned/observed duration.
  `full_query_safe` is **null for every incomplete branch**, even without contact.
  Completed branches are safe only if no native/disc human or static event occurred.
  Disc events do not themselves stop collection; native contact does.

## Reproduction and validation

Use the existing environment and external GPU selection; do not reinstall:

```bash
cd /home/haodong/workspace/projects/eventwm-falcon
source ~/miniforge3/etc/profile.d/conda.sh
conda activate eventwm-falcon
# CUDA_VISIBLE_DEVICES=1 is already configured externally.
python tests/collector/test_collector.py
python tests/controlled_matrix/test_controlled.py
python -m compileall -q scripts/eventwm_collect.py scripts/collector tests/collector
python scripts/eventwm_collect.py collect --output datasets/raw/phase1_castle_manual
python scripts/eventwm_collect.py labels --output datasets/raw/phase1_castle_manual
python scripts/eventwm_collect.py validate --output datasets/raw/phase1_castle_manual
python scripts/eventwm_collect.py render --output datasets/raw/phase1_castle_manual
```

`collect` refuses nonempty outputs, runs exactly one root, all its candidate
queries, two repeated queries in reverse order, and a ten-control truncated WAIT.
No multi-root/multi-scene option exists. Its final validation raises on any
collector integrity failure; human-reactivity FAIL is retained as a scientific
result rather than hidden by a successful process return code.

## Results and limitations

Actual successful collection command (2026-10-05):

```bash
python scripts/eventwm_collect.py collect --output datasets/raw/phase1_castle_smoke_v3
python scripts/eventwm_collect.py labels --output datasets/raw/phase1_castle_smoke_v3
python scripts/eventwm_collect.py validate --output datasets/raw/phase1_castle_smoke_v3
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames,r_frame_rate -of json \
  datasets/raw/phase1_castle_smoke_v3/diagnostics/comparison.mp4
```

One real root `root_castle_h1_6939818f90`, 41 shared legal history observations,
33 distinct frozen queries (32 motion + one WAIT), 36 branches including the two
reverse-order repeats and truncated WAIT. Saved **13,616 future truth records**
(13,048 in the 33 primary branches), 2,716 executed controls, and 36 offline label
records. Example first query: `q_5041e2dd556a`; reference WAIT:
`q_fdf74421f956` / `b032`.

Primary outcomes: 23 query completions, 10 native human-collision terminations,
zero static-contact terminations. Native human contact appears in 10 branches,
research disc contact in 9, and their union in 12. Of the 23 completed branches,
21 meet the declared full-query safety definition; the other two have disc events.
Including repeats/audit: 25 completions, 10 collision terminations, one timeout.
The truncated WAIT observes 0.488 s of a nominal 6.472667 s query and has
`completed=false`, `censored=true`, `full_query_safe=null`.

| Audit | Result | Evidence / scope |
|---|---|---|
| Query immutability | PASS | All saved canonical hashes match before/after execution |
| Same actual root | PASS | All 36 branches match exact replay fingerprint |
| Same root + query reproducibility | PASS | Two full native/command/legal traces match exactly; RGB/depth arrays also checked bit-identical |
| Branch-order independence | PASS | Queries 0,1 repeated in order 1,0 after all candidates |
| Generator privilege boundary | PASS | Typed root whitelist rejects injected hidden/future/goal fields |
| Hidden GT excluded from legal observations | PASS | Sensor poison test; only first-person pixels and robot proprioception; no GT tracks |
| Offline labels recomputable | PASS | All 36 rebuilt labels equal saved labels; native contact pairs independently reclassified for 13,048 primary records |
| Truncation censoring | PASS | Real ten-control WAIT truncation remains unknown, never fully safe |
| Timestamps | PASS | Actual 8–24 ms recorder gaps, fixed commands, <=8 ms nominal schedule quantization |
| Static planner human-blind | PASS | Identical query set regenerated from standalone NavMesh and poisoned-extra-field legal root |
| Human reactivity | **FAIL** | Maximum matched human trajectory deviation from WAIT is exactly 0 m, including close/contact branches |

Human-reactivity FAIL is a measured property of this configuration, **not** an
unperformed test. Comparisons use the common observed horizon of each active/WAIT
pair, not extrapolation after termination. The minimum active disc clearance was
-0.24149 m. Human speed change and travelled-distance difference are also zero.
The controller continued running separately in every process; trajectories were
not precomputed or copied.

Same spatial path, different timing (selected offline only for visualization):

| Branch | Variant | Observed seconds | Minimum clearance (m) | Native / disc contact |
|---|---|---:|---:|---|
| b008 | Immediate normal | 3.256 | 1.54825 | No / No |
| b009 | Wait 0.5 s, normal | 3.792 | 0.74340 | No / No |
| b010 | Wait 1 s, normal | 4.136, terminated | 0.15965 | Yes / No |
| b011 | Immediate slow | 4.328, terminated | -0.24149 | Yes / Yes |
| b032 | Reference WAIT | 6.472 | 0.19059 | No / No |

The PNG shows all four variants and WAIT; the MP4 shows b008, b009, b011, b032
in that order. Video is 1024×256, 134 decoded frames, approximately 20.55 fps;
ended panels freeze and are marked `ended`. It is a nominal-cadence diagnostic;
JSON timestamps remain authoritative. The PNG was visually inspected and the
video decoded with ffprobe.

Validation: six new semantic tests PASS; six existing controlled-matrix tests
finish OK with one documented sandbox multiprocessing-socket skip. Compile checks
and `git diff --check` pass. The source stayed on the original branch/checkpoint;
all implementation files are uncommitted additions. No zero-human simulator root
or additional scene was needed for this milestone.

Known limitations before scale-up: deterministic replay is validated for this
machine/build/root, not across versions or hardware; no complete portable snapshot;
no human detector or track/last-seen estimates; one human/one lightweight scene;
kinematic robot fixed leg pose; approximate research discs and floor-contact filter;
no continuous swept-collision proof between samples; nominal clock quantization;
no scale-format/resume/concurrency optimizations. Scene/asset contents and the
validated environment must remain fixed when reproducing.

The native human obstacle loop in `falcon/additional_action.py` iterates slots
`range(1, num_articulated_agents)`, excluding robot slot 0, and the optional
robot-distance stop/walk threshold defaults to -1. We do not change either to
manufacture reactivity or call a modified controller official FALCON.

Failed preflight attempts are preserved: `phase1_castle_smoke` failed before
prefix execution because the copied forward action had zero angular scale;
`phase1_castle_smoke_v2` rejected nonuniform native clock intervals. The final
adapter explicitly sets the existing turn scale and represents native timing
quantization. These failed attempts are not training samples.
