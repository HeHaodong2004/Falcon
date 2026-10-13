# Structured legal human observations

Implemented on `research/structured-human-observation`, starting at `b041762`
(`eventwm-collector-phase1`). No controller, query-generator, planner, compiler,
replay, label definition, model or training change. Nothing committed automatically.

> This is an ideal structured perception protocol: visible human position is measured perfectly after legal range/FOV/occlusion gating. It is not a claim of realistic computer-vision performance.

## Authoritative sensor model

`ObservationConfig.from_depth_sensor` reads the **resolved Phase-1 jaw depth
sensor**, not the sensor-setup template defaults or the narrower RGB camera.
The existing resolved configuration is 256×256, HFOV **90°**, max depth **10 m**.
The structured sensor uses that 10 m value as a conservative *radial* range limit,
and derives VFOV **90°** from HFOV/aspect ratio. It does not change either camera.
The suggested 120°/4 m assumptions were not substituted for the validated config.

Every worker writes `audits/workers/<id>/observation_config.json`; the root also
writes `legal_observations/observation_config.json`. These resolved values are the
observation configuration/provenance. Configurable protocol fields are a frozen
`ObservationConfig` dataclass; all values are serialized, including:

| Field | Value |
|---|---:|
| horizontal_fov_deg / vertical_fov_deg | 90 / 90 |
| max_range_m | 10 |
| torso_height_m | 1.0 |
| los_tolerance_m | 0.02 |
| track_max_age_s | 2.0 |
| association_gate_m | 0.8 |
| prediction_horizon_s | 0.5 |
| velocity_max_gap_s | 0.5 |
| min_velocity_dt_s | 0.000001 |

No noise, learned detector, false-positive/negative model, or artificial ID switch
was introduced. These are fixed protocol choices, not performance-tuned values.

At each actual observation timestamp, the adapter obtains the **current absolute
camera transform** from
`sim._sensors['agent_0_articulated_agent_jaw_depth']._sensor_object.node.absolute_transformation()`.
This includes Spot's camera-link pose and mounting transform; it does not assume
robot yaw alone equals the camera direction. Camera local forward is -Z. A human
must be active, finite, within radial range, inside horizontal and vertical FOV,
and have an unobstructed torso ray before a detection is created.

The target point is current human `base_pos + [0, 1.0, 0]` in world coordinates.
This is a declared standing-body proxy, not a hidden joint/controller feature.
The ray origin is the live jaw depth camera origin. Ray direction is normalized;
`habitat_sim.geo.Ray` and `sim.cast_ray(ray, max_distance=target_distance)` therefore
report distances in metres. Sorted hits block visibility when they occur before
`target_distance - 0.02 m`. Stage collision geometry and static rigid geometry
block the ray. Agent root/link IDs and non-static rigid objects are ignored;
non-agent articulated geometry, if present, is not explicitly ignored. Castle
has no such dynamic scene articulations. The protocol models **static** occlusion,
not human-on-human occlusion. No NavMesh visibility shortcut is used.

Inspected/reused interfaces: Habitat's
`datasets/rearrange/navmesh_utils.py::snap_point_is_occluded` demonstrates native
raycasting, ignored object IDs and target-distance tests. The focused adapter uses
the same API with an actual camera origin, one torso point, explicit metric
tolerance, bounded ray length and diagnostic witness. It does not invoke the
helper's search across hypothetical observer heights.

## Legal detection and track semantics

Only `HabitatHumanObserver` accesses live human positions/active slots. Its world
records contain current position, active flag and a diagnostic-only ID; they do
not read GT velocity, goals, future positions or controller state.

`IdealHumanDetector` emits `HumanDetection(timestamp, position, visible=True,
measurement_valid=True)` **only after all gates pass**. Detected positions are
perfect measurements of the visible person's base position. Detections are sorted
by measured position; simulator ordering/identity is not used for association.

`LegalHumanTracker` consumes only these detections and actual timestamps:

- Deterministic greedy minimum-distance matching, globally sorted by distance,
  track ID and detection index, with a 0.8 m XYZ gate.
- Association may predict from the last legal position/velocity for at most 0.5 s.
  The predicted position is **not** emitted as a measurement.
- A new track gets a monotonic `track_000000` bookkeeping ID. Simulator IDs never
  reach it. The ID groups a sequence; it is not a numerical/semantic entity feature
  for a neural network. This task adds no tensorizer or model architecture.
- Velocity is `(new_visible_position - last_visible_position) / actual_dt`.
  It is null/invalid at first sight, at dt <=1e-6 s, or on reacquisition after a
  gap >0.5 s. No hidden trajectory is reconstructed. No GT velocity is consulted.
- While hidden, retain exactly the last measured position and last estimated
  velocity, count and last-seen time. Set `visible_now=False` and increase age.
- `valid=True` means a retained legal memory within its lifetime, not a current
  hidden-position measurement. `velocity_valid` refers to the last legal estimate.
- Expire when age **exceeds** 2 s. Never-seen hidden people create no tracks.
  Reappearance outside the legal association gate creates a new ID even if the
  simulator says it is the same person. There is no oracle ReID.
- Duplicate timestamps return a deep-copied cached result without incrementing
  counts. This handles the existing root observation shared by prefix/branch.
  Decreasing/nonfinite timestamps are rejected; invalid detections are ignored.

The legal track whitelist is exactly:

```
track_id, position, estimated_velocity, velocity_valid,
visible_now, valid, first_seen_time, last_seen_time,
observation_age, num_observations
```

Positions/velocities use world XYZ (y up); estimated velocity is either three
numbers or JSON null. No redundant relative position/clearance features are added.
The existing legal observation fields remain unchanged, with `tracking_available=True`
and the structured list replacing the previous empty list in real workers.

`legal_observation` and `load_wm_input` deep-copy and whitelist nested track fields.
The loader also whitelists the frozen query's fields and nested robot/generation
config. It reads only legal history and the chosen query. It never opens privileged
root/branch files. Unit tests inject `future_position`, `gt_velocity`, `goal`,
`hidden_human_position`, and `controller_state` at observation, track, query and
nested query-config levels; returned WM input remains unchanged.

## Online timing, replay and causality

Humans are initialized by the unchanged controlled reset before the 40 real WAIT
prefix actions. The same observer processes initial observation and every prefix
observation **online**, then continues through each independently replayed branch.
No retrospective GT trajectory conversion constructs the history.

Actual simulator timestamps drive association, velocity, age and expiration.
The existing 0.048/0.056 s action intervals are preserved; no 0.1 s assumption is
introduced. The observer does not step physics, consume randomness, change poses,
or modify controllers. Only isolated diagnostic fixtures deliberately set up poses.

The root fingerprint continues to hash legal history as before. Enriching that
history changes root/query identifiers; all query fields other than those IDs
remain exactly identical to Phase 1. Replay mechanics and hashing algorithms are
unchanged. Each branch replays the tracker from the same prefix. Shared root
history is a deep snapshot, and branch observations never update it backward.

Privileged visibility diagnostics are separate:
`audits/workers/<id>/observation_diagnostics.json` contains camera pose, GT position,
GT diagnostic ID, rejection reason, ray witness, detection count and copied legal
tracks. These files are **not WM inputs**. Hidden population, GT rejection reason,
controller state and GT IDs never appear in legal observations.

## Files added/modified

Added:

- `scripts/collector/observation.py`: visibility, detector, legal tracker, Habitat adapter.
- `scripts/collector/observation_audit.py`: native wall fixtures, controlled real
  visible-prefix audit, saved-data physical/causality checks, timeline and plot.
- `tests/collector/test_observation.py`: 11 regression tests, including protected
  Phase-1 component AST equality against the checkpoint tag.
- `EVENTWM_STRUCTURED_HUMAN_OBSERVATION.md`: this document.

Modified:

- `scripts/collector/core.py`: optional structured tracks and nested legal loader
  whitelists only. Query generation/compilation and label construction unchanged.
- `scripts/collector/runtime.py`: initialize observer, call it at the existing
  observation point, save resolved config and privileged diagnostics.
- `scripts/eventwm_collect.py`: update the former “all tracks must be empty” audit
  to enforce the nested track whitelist. Dedicated observer audits test gating and
  causal tracking; all other collector checks remain intact.

## Commands and outputs

Run in the existing `eventwm-falcon` environment with external
`CUDA_VISIBLE_DEVICES=1`. No reinstall or environment changes are required.

Executed:

```bash
python tests/collector/test_collector.py
python tests/collector/test_observation.py
python tests/controlled_matrix/test_controlled.py
python -m compileall -q scripts/collector scripts/eventwm_collect.py tests/collector
python scripts/eventwm_collect.py collect --output datasets/raw/structured_human_castle_v1
python scripts/collector/observation_audit.py scenarios --output datasets/raw/structured_human_castle_v1
python scripts/collector/observation_audit.py prefix --output datasets/raw/structured_human_castle_v1
python scripts/eventwm_collect.py worker --output datasets/raw/structured_human_castle_v1 \
  --worker-id observed_repeat --query-id q_6d84ea7d1706
python scripts/collector/observation_audit.py validate \
  --output datasets/raw/structured_human_castle_v1 \
  --baseline datasets/raw/phase1_castle_smoke_v3
```

`collect` automatically runs the previous label builder, integrity audits and
comparison renderer. To reproduce collection, choose a **new output directory**;
existing samples are never overwritten by `collect`. Query IDs for an extra replay
can be read from that run's `queries/queries.json`.

Reports/artifacts within `datasets/raw/structured_human_castle_v1/`:

- `audits/audits.json`: previous collector integrity and human-reactivity checks.
- `audits/structured_observation_audits.json`: observation checks, counts and
  exact comparisons against Phase 1.
- `audits/observation_scenarios.json`: A–G fixture inputs, native ray witnesses,
  tracking results and PASS/FAIL assertions.
- `audits/visible_prefix.json`: two real 40-WAIT Castle prefixes with identical
  initialization, observer off/on, complete physical traces and online history.
- `legal_observations/visible_prefix_audit_history.json`: separate legal visible
  prefix for inspection; it is not appended to the primary root's history.
- `diagnostics/human_observation_timeline.csv`: real branch b023, with a clearly
  marked **privileged** GT visibility-reason annotation alongside legal tracks.
- `diagnostics/observation_occlusion.png`: **PRIVILEGED DIAGNOSTIC OVERLAY** of the
  static wall fixture, FOV, hidden human GT, blocked ray and remembered legal track.

## Results

Primary root: `root_castle_h1_28a6811906`. **33 queries, 36 grouped branches**,
13,616 truth samples, unchanged from Phase 1. An additional audit-only replay of
b023 / `q_6d84ea7d1706` verifies populated tracking, giving 37 branch executions.
Primary outcomes remain 23 completions and 10 native human-collision terminations;
repeats and truncated WAIT retain their original outcomes.

The original WAIT-prefix human is outside the real camera FOV throughout: **zero
root detections** is the correct result, not a missing tracker. The 33 primary
branches produce **160 legal detections and 227 lost-track observations**.
There are 2,257 samples with no detection and no legal track (root counted once).
No natural static-occlusion or reappearance event occurred in this short collection;
those cases are exercised explicitly by the native wall fixtures, not claimed as
natural Castle events.

For a nonempty *root history*, the separate controlled-prefix audit initializes
robot yaw toward the human **before** history generation, then executes the same
40 WAIT actions with the native human controller. It produces **41 real legal
human detections**, with history-derived velocity after the first observation.
Observer-off/on traces for that identical initialization match exactly. This is a
bounded observation audit, not a change to the primary collection's root recipe,
controller or query semantics. The extra b023 replay adds 41 repeated detections;
these are excluded from the primary count of 160.

| Check | Result |
|---|---|
| Visible / range / FOV gating (A, B, C) | PASS |
| Native static-wall occlusion and never-seen hidden (D, F) | PASS |
| Visible → hidden legal memory (E) | PASS |
| Legal association on reappearance (G) | PASS |
| Expiration / first observation / tiny dt / invalid data | PASS |
| Legal-history velocity and no hidden refresh | PASS |
| Native GT metadata poison + nested WM loader poison | PASS |
| Real online visible prefix, observer off/on physical equality | PASS |
| Populated visible/lost branch replay | PASS |
| Root/future causal separation | PASS |
| Every grouped physical trace/control versus Phase 1 | PASS, exact |
| Every label versus Phase 1 | PASS, exact except query ID |
| Frozen query semantics versus Phase 1 | PASS, exact except IDs |
| Previous ten collector integrity checks | PASS |
| Existing human reactivity diagnostic | **FAIL, unchanged** |

Tests: 11 observation tests and six original collector tests PASS. Existing
controlled-matrix suite: OK (six tests, one documented sandbox socket skip).
Protected query/compiler/planner/recorder/label components are AST-identical to
`eventwm-collector-phase1`. All 36 branches' full physical states, contacts,
timestamps and controls compare exactly with the saved Phase-1 run.

Example real timeline (world x/z; seconds are actual simulator time):

| Time | Visible | x | z | vx estimate | vz estimate | Age |
|---:|---|---:|---:|---:|---:|---:|
| 2.968 | yes | -1.58771 | 9.62570 | null | null | 0 |
| 3.016 | yes | -1.61910 | 9.69222 | -0.65403 | 1.38587 | 0 |
| 4.960 | no | -2.81552 | 12.22738 | -0.70179 | 1.48708 | 0.048 |
| 5.008 | no | -2.81552 | 12.22738 | -0.70179 | 1.48708 | 0.096 |
| 5.056 | no | -2.81552 | 12.22738 | -0.70179 | 1.48708 | 0.144 |

The hidden rows retain last-seen time 4.912 and the same position/velocity even
while privileged human truth continues moving.

The first native fixture audit genuinely failed: the test attempted to move a
STATIC Bullet wall, whose transform setters were ignored, leaving it at the
origin. The corrected fixture repositions while KINEMATIC, asserts the position,
then sets STATIC **before** testing LOS. Observer code was not weakened.
`audits/observation_scenarios_initial_failure.json` preserves that failure.
The native fixture uses declared ideal sensor-world human samples and an actual
Bullet wall; it is not a navigation rollout or a learned-detector result.

## Limitations

One real human/one scene. Multi-person crossing/association ambiguity is not
validated in simulation. A single standing-torso ray can miss partial visibility;
collision geometry may differ from rendered visual/transparent geometry. Dynamic
human occlusion and realistic vision errors are intentionally absent. Short-gap
velocity is a finite difference between visible measurements, not instantaneous
GT velocity; long-gap velocity remains unknown until fresh evidence arrives.
Root IDs change with the legal schema, so physical comparisons normalize only
root/query IDs after checking all remaining query fields. No training, sweep,
controller reactivity change, or scale-format redesign was performed.
