# World Model Sandbox Core (Layer 2.5): interface

Package `eventwm_graph/sandbox/` · tests `tests/sandbox/test_sandbox.py`, `tests/sandbox/test_service.py` · results [REPORT.md](REPORT.md).
Built on the unified Layer 2 (`eventwm_graph/layer2/`, [docs/layer2/INTERFACE.md](../layer2/INTERFACE.md)) and the
querygen planners / executor / dynamics / Query generator (`eventwm_graph/querygen/`), all unchanged.

## 1. What it does

One legal decision state, K candidate Queries (each with its own low-level planner and dynamics family) →

```
 legal decision state at t ──┬── Stage 1 (hwm_lite_e12): ONE joint human bank, 40 worlds x 3 s   (shared)
 (frames, partial graph,     │        │
  range packet, executor     │        └─ nominal future of the tracked people ──┐
  legal state)               │                                                   ▼
                             ├── imagined execution of every candidate: querygen QueryExecution with ITS planner +
                             │   dynamics family, frozen legal map, nominal family kinematics   -> K trajectories
                             ▼
                 Layer 2: Stage 2 (v17_a) map increment + complete hypothetical future graph per trajectory
                          (one batched call) + joint collision / social risk of each trajectory vs all 40 worlds
                             ▼
                 SandboxResult: per candidate trajectory, execution outcome, map increment, future graph, risk
```

Nothing in it learns or selects: it imagines the consequences of the candidates it is given. Selection (Layer 3)
and graph compilation are out of scope.

## 2. Calling it

```python
from eventwm_graph.sandbox.core import WorldModelSandbox, Candidate, SandboxConfig
from eventwm_graph.sandbox.episode import BranchEpisode, RealEpisode, generator, default_query

sb = WorldModelSandbox.from_release(SandboxConfig(), map_device='cuda:0')   # Layer 2 release (~20 s, ~1.5 GB GPU)
gen = generator('k10', 'native')               # frozen querygen candidate generator ('k10' or 'k6')

ep = BranchEpisode(spec, 'orca', 'native', range_sensor='fast', gc_freeze=True)   # or RealEpisode (habitat.Env)
while not ep.over:
    state = ep.decision_state()                 # layer2 DecisionState at the current observation
    queries, _ = gen.propose(ep.legal, ep.execution)
    cands = [Candidate(None, 'orca', 'native', execution=ep.execution, level=ep.level) if q.kind == 'continue'
             else Candidate(q, 'orca', 'native', level=ep.level) for q in queries]
    res = sb.evaluate(state, ep.legal, cands)   # K imagined consequences (hypothetical)
    choice = ...                                # Layer 3 (not part of the sandbox)
    ep.run(None if choice is continue else cands[choice].query, horizon=16)
```

Any environment can drive it. It must feed `context.LegalStream.observe(row, range_frame)` after every step, keep a
querygen `LegalState` from the same rows, and pass both at a decision. `RealEpisode` (unchanged habitat.Env) and
`BranchEpisode` (querygen's exact branch simulator) are the two reference drivers. For long-running workers,
`episode.freeze_gc()` after setup (and `thaw_gc()` at release) avoids 150-215 ms cyclic-GC pauses.

### Inputs (validated)

| input | contract |
|---|---|
| `state` | `layer2.state.DecisionState` from `LegalStream.decision_state()`. Every arrived legal observation (dataset frame format; static updates as `CompactUpdates`, a read-only sequence of the reference dicts), the causal partial graph built with `PartialGraph.update` from those frames, one range frame per frame. Hypothetical graphs and rows are rejected. |
| `legal` | The querygen executor's `LegalState` at the same instant (time 1e-7, pose 1e-7, map clock). It is read only. Imagination uses a frozen alias of its map. |
| candidates | `Candidate(query, planner='orca', dynamics='native', horizon=None, execution=None, level=0, planner_kw=None, query_id=None)`. `query`: any querygen `Query`; `query=None` + `execution=` continues the running execution (a deep copy is imagined). Planner ∈ {orca, sfm, dwa, follow}; dynamics ∈ querygen `FAMILIES`. Mixed planners/dynamics in one call are allowed. |
| range frames | `ideal_current_jaw_range_dynamic_occlusion_1`, 256 × 256, from the environment's sensor: `ranges.NativeRangeSensor` (reference, ~1.5 s per frame) or `fastrange.FastRangeSensor` (numerically equivalent, ~10 ms GPU + native recast of rays near humans). |

### Outputs (all hypothetical)

`SandboxResult`: `timestamp`, `robot_pose`, `stream`, `humans` (the shared Stage 1 `HumanWorlds`), `nominal` (the
`NominalHumans` the imagined planners perceived), `candidates` (tuple of `CandidateOutcome`, same order as the
input), `layer2` (the `Layer2Prediction`), `profile` (timings), `provenance`. Methods: `summaries()`,
`human_futures(steps)`, and `record(maps=False)`, a JSON-ready dict (schema `wm_sandbox_record_1`) for logging and the
Graph Compiler.

`CandidateOutcome`:

* `candidate`: descriptor (kind, label, via, meta, planner, dynamics, continued).
* `rollout` (`imagine.ImaginedRollout`): `poses [T+1,4]` (x, y, z, yaw; row 0 = the real pose at t) and `times`
  (t + k·0.048 s). Also `commands` (as the environment would receive them), `levels`; `status` / `reason`
  (executor) and `ended_by` (horizon | done | stop | failed | error). Then `stop`, `goal_distance`, `progress`,
  `path_length`, `blocked_steps` (imagined poses on legally BLOCKED cells), `nominal_clearance`, `planner_ms`,
  `error`. `trajectory()` gives the N × 5 Layer 2 candidate.
* `map`: the Stage 2 increment (`predict_segments` contract: `candidate_cells`, `probabilities` (uncalibrated),
  `states`, `heights`, `static_updates`, `future_partial_graph`, provenance). `future_graph`: the complete
  hypothetical graph at the trajectory end.
* `humans`: the Layer 2 joint reductions against the shared bank. These are `p_collision`, `p_social`,
  `expected_social_duration` / `severity`, `minimum [40]`, `minimum_quantiles`, `first_collision_mass`,
  `social_endpoint_probability`, `collision_world_event`, `steps` and `horizon_truncated` (3 s / 62 steps).
* `summary()`: a flat dict of the above.

### Configuration (`SandboxConfig`)

| field | default | meaning |
|---|---|---|
| `horizon` | `'interval'` | imagined steps per candidate: querygen `horizon_for` (16 per movement decision, durations for wait / creep, 8 for STOP) or an int |
| `lookahead` | 0 | extra steps continuing the query after its interval |
| `max_steps` | 120 | hard cap |
| `nominal` | `'mean'` | human future the imagined planner perceives: `mean` / `medoid` / `world:i` of the Stage 1 bank, or `cv` |
| `worlds`, `seed` | release | Stage 1 sampling |
| `maps` | True | run Stage 2 (False: trajectories + human risk only) |
| `materialize` | True | build complete future graphs |
| `memoize` | True | exact per-decision caches (`accel.py`) |
| `prefetch_maps` | False | warm Stage 2's candidate-independent context on a worker thread (exact; no measured gain) |

### Split-process mode (`service.py`): many environment workers per GPU

In-process, every worker holds Layer 2 (~2.5 GB of GPU memory), which caps an RL loop at 6-7 workers per GPU while
the loop itself is CPU-bound. In split mode Layer 2 runs in a few server processes and each worker keeps only the
environment, its sensors, the generator and the imagined rollouts. A worker's only GPU use is the fast range sensor:
0.30 GB with its torch-free driver backend (`range_sensor='fast-driver'`, bit-identical frames), 0.73 GB with the
PyTorch one (a full CUDA context; torch 1.12 has no lazy module loading). The simulator itself uses no GPU memory.

```python
from eventwm_graph.sandbox.service import RemoteSandbox, start_server, wait_ready

srv = start_server('wm0', 'wm0.ready')         # or: scripts/sandbox_run.py eventwm_graph.sandbox.service serve --address wm0
wait_ready(['wm0.ready'], [srv])               # ~20 s (Layer 2 build)
sb = RemoteSandbox('wm0', SandboxConfig())     # in each worker; several workers share a server
ep = BranchEpisode(spec, 'orca', 'native', range_sensor='fast-driver', stream=sb.open_stream(spec.key))
res = sb.evaluate(ep.decision_state(), ep.legal, cands)   # the same SandboxResult as in process
```

* **Exactness.** The server rebuilds each worker stream's decision state with `context.LegalStream` from the same
  frames and makes the same Layer 2 calls on it. The worker runs the same imagination code: `RemoteSandbox` subclasses
  `WorldModelSandbox` and only answers its Layer 2 calls remotely. Results are identical bit for bit, also with
  several workers interleaved on one server (`tests/sandbox/test_service.py`).
* **Protocol.** `RemoteStream.observe` checks every observation against the `LegalStream` contract and buffers it.
  The frames travel with the decision's first request (at most 64 are buffered). A decision makes two round trips:
  `stage1` (frames -> decision state -> Stage 1 bank, which the worker needs for its nominal humans) and `predict`
  (the K imagined trajectories -> Stage 2 + joint risk; the bank is not sent back).
* **Server.** One model thread serves requests in arrival order. It is the thread that built the model, because Stage
  2's driver-API CUDA modules are bound to it. Each worker stream gets its own `LegalStream` and Stage 1 stream state.
  Stream names are suffixed with the worker's pid by default and cannot be taken over by another connection. A
  closed or vanished worker's streams are released.
* **Contract.** A decision handle (`RemoteDecision`) is valid until its stream observes again; stale handles are
  rejected. `prefetch_maps` is in-process only. Transport: `multiprocessing.connection` over Unix sockets (Linux
  abstract namespace) with pickled messages. Stage 2's read-only outputs are rebuilt through their constructors.

## 3. Legality

* Inputs are the legal decision state and the legal executor state, both built from the arrived observations only.
  No simulator state, navmesh, scene mesh, Oracle information, identities or future observations enter the sandbox.
  The range frame is produced by the environment's sensor and only the frame is passed.
* Imagined executions run on a frozen alias of the legal map at t. They perceive only the currently tracked people:
  every track `control.human_estimates` keeps (valid, at most 2 s old), mapped to Stage 1's `track:<id>` slot, never
  the unseen residual slots. Each person is placed at the bank's present-weighted mean, or at constant velocity if
  Stage 1 does not model the track. The robot moves by the dynamics family's nominal kinematics, without the navmesh.
* Every output is flagged hypothetical. Predicted cells and future graphs are never merged into either legal state.
  `DecisionState` rejects them as inputs of a later decision (tested).
* The real episode's legal state, map arrays and running execution are unchanged by `evaluate` (tested bit for bit).

## 4. Exactness of the accelerations

* `accel.py`: per-decision memoisation of the executor's pure functions of the frozen map (tight / thru map
  variants, planning views, fields, routes, lattice plans). It also provides exact numba forms of the body-axis map
  lookup and the blocked-sector search. Rollouts are bit-identical with and without them (all planners × dynamics
  families, `test_memo_is_exact_across_planners_and_dynamics`).
* `fastrange.py`: static rays are cast on the GPU with an exact float32 emulation of Bullet's triangle callback.
  Rays near a human's collision capsules, rays near a float32-ambiguous edge or 10 m decision, and grazing rays are
  recast natively. Hit / no-hit masks are identical and range values agree within 8.8e-5 m (100 frames, 4 episodes; Bullet's own ray
  transform is not reproduced bit for bit, so about half the pixels are bit-equal and the rest a few ulps apart).

## 5. Not in scope / known limits

* Layer 3, the Graph Compiler, training of any kind. Stage 1 / Stage 2 are used as released (Stage 2's full-dev
  accuracy 0.4831 < 0.50 is a known Layer 2 limit).
* Imagination does not observe new map evidence during the rollout. It has no navmesh sliding, and the people it
  perceives follow ONE nominal future. Risk is still scored against all 40 worlds (open loop).
* Chained imagination (deciding again from an imagined state) is not legal input for Layer 2 and is not offered.
