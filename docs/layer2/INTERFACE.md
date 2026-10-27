# Unified Layer 2 world model: architecture and interface (for Layer 2.5)

Package `eventwm_graph/layer2/` · release manifest `release/layer2_v1/RELEASE.json` · tests `tests/layer2/` ·
results `docs/layer2/RESULTS.md`.

## 1. What it is

One decision state, K candidate robot trajectories → one shared prediction of the humans and K candidate-specific
hypothetical map increments, plus per-candidate human-interaction reductions against the shared human bank.

```
                    DecisionState (legal, at decision time t)
          frames[0..n] ── partial_graph G_t ── range_packet[0..n] ── stream id
                │                      │                      │
     ┌──────────┴──────────┐   ┌───────┴──────────────────────┴──────────────┐
     │ Stage 1 (CPU, 1 thr) │   │ Stage 2 (GPU)                               │
     │ hwm_lite_e12         │   │ v17_a LegalVoxelPredictor                   │
     │ ONE joint human bank │   │ per candidate tau_k: Delta G_k (hypothetical)│
     │ 40 worlds x 3 s      │   │ + complete hypothetical future graph        │
     └──────────┬──────────┘   └───────┬──────────────────────────────────────┘
                │   runs on a worker   │   (overlapped; results identical to sequential)
                └─────────┬────────────┘
                          ▼
       per-candidate joint reduction (robot envelope vs human disk, 1 m social, per-world minima)
                          ▼
              Layer2Prediction: humans (shared) + candidates[k] (map, humans, summary)
```

* **Stage 1** (`research/stage1-claude-20261026` @ `cd7f3761`, merged): `HumanWorldModel`, checkpoint
  `release/stage1_hwm_lite_e12/hwm_lite_e12.pt` (byte-identical to `docs/stage1_hwm/SELECTED.json`, SHA-256 `53d44d85…`).
  Humans are exogenous to the robot in this simulator, so one bank serves every candidate.
* **Stage 2** (`research/stage2-claude-exploration-20261026` @ `70fae3cf`, the base of this branch):
  `LegalVoxelPredictor`, checkpoint `release/stage2_claude_v17a/stage2_claude_v17a.pt` (SHA-256 `bef7f5fd…`), `lanes=3`,
  256 px range frames.
* Neither stage was retrained or modified. Both are reached only through the two adapters in `layer2/backends.py`
  (`Stage1HumanBackend`, `Stage2MapBackend`). Either can be replaced by any object with the same four methods
  (`test_backends_are_replaceable` swaps Stage 1 for a constant-velocity stand-in, and the Stage 2 outputs stay
  bit-identical).
* Stage 2 keeps its own legal human channel: the constant-velocity sweep of the currently tracked humans, used as
  dynamic occluders. It does **not** consume Stage 1 worlds. Coupling the two would change Stage 2's validated
  function and would require retraining.

## 2. Calling it from Layer 2.5

```python
from eventwm_graph.layer2.model import Layer2WorldModel
from eventwm_graph.layer2.state import DecisionState, CandidateTrajectory, trajectory_from_poses

l2 = Layer2WorldModel.from_release(map_device='cuda:0')   # verifies both SHA-256s; sets torch threads = 1
                                                          # (~20 s: Stage 2 NVRTC kernels + CUDA graphs)

# once per decision (real time t of the episode / environment `env_id`)
state = DecisionState(frames=arrived_frames,              # list of legal observation dicts, oldest first; [-1] is now
                      partial_graph=legal_graph_t,        # causal_partial_cspace_graph_1 at frames[-1]['timestamp']
                      range_packet=dict(ranges=R,         # [n, 256, 256] float32 Euclidean m, NaN = no return <= 10 m
                                        origins=O,        # [n, 3]   == frames[i]['camera_origin']
                                        rotations=Rot,    # [n, 3, 3] == frames[i]['camera_rotation']
                                        timestamps=T,     # [n]      == frames[i]['timestamp']
                                        sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1'),
                      stream=env_id)                      # one stream per robot/episode (incremental Stage 1 state)

# candidates: one per Query, rolled out by the L2.5 planner + dynamics model from the current pose
cands = [CandidateTrajectory(trajectory_from_poses(rollout_poses_q, state.timestamp), query_id=q.id) for q in queries]
out = l2.predict(state, cands)                            # K = len(cands) >= 0

out.humans                     # shared Stage 1 HumanWorlds: positions [40, 64, N, 2] world X/Z on times [64] (relative s)
out.human_futures(steps=48)    # the same bank on the planner clock: positions [40, 49, N, 2] at t + i*0.048 s (absent = +inf),
                               #   present [40, N], weights [40], absolute times, track keys
out.humans_at(abs_times)       # [40, T, N, 2] at arbitrary absolute times in [t, t + 3 s]
for c in out.candidates:       # same order as `cands`
    c.map                      # Stage 2 delta (predict_segments contract, below)
    c.future_graph             # complete HYPOTHETICAL graph at the trajectory end (hypothetical=True)
    c.humans                   # p_collision, p_social, expected_social_duration/severity, minimum [40], minimum_quantiles,
                               #   first_collision_mass [T], social_endpoint_probability [T], collision_world_event [40],
                               #   p_no_humans, steps, horizon_seconds, horizon_truncated
    c.summary()                # flat dict: predicted FREE/BLOCKED counts, uncalibrated expected gains, p_collision, ...
```

`predict` accepts a plain `N x 5` array in place of a `CandidateTrajectory`. `l2.predict_humans(state)` and
`l2.predict_maps(state, cands)` run a single stage. `l2.reset(stream)` drops incremental state. `l2.close()` stops the
worker thread.

### What Layer 2.5 must supply

| input | requirement (validated in `DecisionState` / `predict`) |
|---|---|
| `frames` | Every arrived legal observation of the episode so far, oldest first, in the dataset format (`timestamp`, `robot_pose` [x, y, z, yaw], `camera_origin`, `camera_rotation`, `tracks`, `static_updates`, `observation_valid`, `sensor_protocol`). Strictly increasing clock. **No** simulator state, identities, future frames or predicted frames. Stage 1 reads only the frames it has not yet seen on `stream`, so passing the full list each time costs only the new frames. |
| `partial_graph` | The legal graph at `frames[-1]['timestamp']`, built from those frames' certificates. A graph with `hypothetical=True`, `prediction_checkpoint_sha256`, or any `predicted*` cell source is rejected (`LegalityError`). |
| `range_packet` | Exactly one measured range frame per observation frame: same timestamps (1e-7 s), same camera origin and rotation (1e-5). 256 px, 90° frustum, Euclidean metres, `NaN` beyond 10 m. This is the legal sensor extension Stage 2 was trained on; a real front end must provide equivalent calibrated ranges. |
| `stream` | A stable id per robot/episode. Concurrent environments must use different ids. Passing an older or unrelated history on the same id simply rebuilds the state (correct, slower). |
| candidates | `N x 5 [t, x, y, z, yaw]`, absolute time on the frame clock. Row 0 must equal `[t_now, *robot_pose]` exactly (1e-7). Times must increase. Yaw follows the frame convention: forward = (cos yaw, −sin yaw) in world (x, z), checked on 1,328 recorded steps and identical to the querygen rollouts. Use the native 0.048 s step (`trajectory_from_poses`); other spacings are accepted. |

### Clocks, frames and units

* World metres, y up; the floor plane is (x, z). The partial-graph cell `(i, j)` covers `[i·res, (i+1)·res) × [j·res, (j+1)·res)`
  in (x, z) with `res = graph['resolution']` (0.1017 m). Human positions are world (x, z).
* `out.timestamp == state.timestamp == out.humans.timestamp == c.map['base_timestamp']` (asserted on every call);
  `c.map['timestamp']` is the trajectory end. Human times are relative to t (`humans.times`, 0…3 s).
* Human-interaction reductions sample each candidate at `t + k·0.048 s` (k = 1…T; linear interpolation, yaw unwrapped).
  This is the L2.5 nominal action clock. Coverage is capped by the 3 s human horizon: at most 62 steps (2.976 s). A
  longer candidate still gets its full map prediction; `c.humans['horizon_truncated']` is then `True`. About 1 % of
  the recorded Query executions are longer than 3 s.

### Outputs that must stay hypothetical

* `c.map`, `c.future_graph` and `c.map['static_updates']` are **predictions** (`predicted=True`,
  `provenance.hypothetical=True`, cell source `predicted_stage2_research`). Never merge them into the legal graph.
  Never count predicted FREE cells as observed FREE space. Never build a later `DecisionState` from them;
  `DecisionState` rejects them.
* `c.map['probabilities']` are **uncalibrated**: expected gains run about 17 % high (Stage 2 query benchmark). Use
  `states` (frozen operating point) or recalibrate on TRAIN validation before using them as probabilities.
* The human bank is a 40-world Monte Carlo sample, not a safety certificate. `p_collision` uses an enclosing body
  envelope.

## 3. Cost model (details and numbers in RESULTS.md)

* Construction: ~20 s (Stage 2 kernel compilation and CUDA-graph capture), 1.2–1.5 GB of GPU memory per process.
* Steady per-decision latency (one new frame, new state), K=6: 37 ms P50 / 45 ms P95. One RTX 3090 sustains
  65 K=6 decisions/s with 6 processes (RESULTS.md §4–5).
* A CUDA error poisons the process's CUDA context: treat it as fatal and restart the worker (RESULTS.md §3).
* Per decision: one Stage 1 call (CPU, overlapped with Stage 2) + one Stage 2 call for all K candidates + reductions.
  Further `predict` calls with the **same `DecisionState` object** reuse the human bank and the Stage 2 legal map,
  so only the new candidates are paid for.
* Throughput scales with processes that share one GPU (each process: 1 Stage 1 CPU thread + Stage 2 GPU work).

## 4. Guarantees (tested)

* Bit-identical to the standalone stages. Banks equal standalone `HumanWorldModel.predict`. Each increment equals a
  cold single-candidate v17_a `predict_segments` call, whatever the batch or cache. In-extent candidates are batched;
  each candidate whose crop leaves Stage 2's default ±19 m legal map runs alone (RESULTS.md §3.1). Checked on all
  recorded Query executions of the 2,110 development Roots.
* Shared decision state: batched K equals K single calls (maps and banks bit-identical; reductions agree up to
  ~1e-17 summation order). Overlapped equals sequential. A rebuilt state object gives an identical bank.
  An incremental stream gives the same bank as a fresh stream.
* No input mutation; strict clock, pose and alignment checks; hypothetical inputs rejected.

## 5. Not part of Layer 2 (open interfaces)

* Planner/dynamics/Query generation (in development in the querygen worktree). Layer 2 only consumes its rollouts.
* Chained look-ahead (a second Query from a predicted state) is not supported legally: a hypothetical future graph is
  not an observation, and no future range frames or tracks exist for it.
* Layer 3 and any RL training.
