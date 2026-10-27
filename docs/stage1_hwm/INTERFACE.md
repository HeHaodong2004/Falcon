# Stage 1 human world model: integration interface

## Construction

```python
from eventwm_graph.stage1_hwm.api import HumanWorldModel, load_selected

model = load_selected()                      # docs/stage1_hwm/SELECTED.json, checkpoint SHA-256 verified
# or: model = HumanWorldModel('/path/best.pt', device='cpu')   # CPU deployment path (1 thread)
#     model = HumanWorldModel('/path/best.pt', device='cuda:0') # batched GPU serving
```

Construction compiles and warms up the CPU engine (~1 s); per-call latency is then stable.
Set `torch.set_num_threads(1)` in the inference process for the reported CPU latency.

## Prediction (legacy-compatible signature)

```python
bank = model.predict(history, obs, partial_graph=None, horizon=3., worlds=40, seed=None,
                     actions=(), executed_action=None, times=None, stream='default')
```

* `history`: list of arrived observation dicts (oldest first) **or** a legacy
  `OnlineContext`; `obs` is the newest frame (appended if newer than the history).
* `partial_graph` (optional) supplies the static-certificate grid resolution and is checked
  against the observation clock; the map itself is rebuilt from the frames' certificates.
* `stream` separates concurrent robots/episodes. A call whose history extends the previous
  call of the same stream ingests only the new frames (incremental, O(new frames)).
* `times`: optional relative output times (default 64 uniform points on [0, horizon]).

Returns `HumanWorlds`, a subclass of the legacy `JointHumanWorlds`:

| field | meaning |
|---|---|
| `positions` [W,T,N,2] | world X/Z (m) of every human slot in every world |
| `present` [W,N] | slot exists in that world (constant over time) |
| `times` [T], `timestamp` | relative seconds; absolute anchor time |
| `weights` [W] | uniform (IID worlds) |
| `track_keys`, `unseen` | `track:<id>` slots first (tracker keys, not identities), then `residual:i` |
| `count_log_probability` | exact log P(total present = c), legally truncated |
| `existence_probability` | per tracker key (1 for keys visible now) |
| `minimum_count` | legal lower bound enforced exactly in every world |
| `knots` [W,N,13,2] | 0.25 s knots; `positions_at(t)` is exact for any t ∈ [0, 3.028] |

Guarantees: every world has ≥ `minimum_count` humans; every path respects the 3 m/s bound;
worlds are IID samples of one joint law (human-human and time correlations preserved within
a world). Humans are exogenous in this simulator, so one bank serves every robot candidate.

## Batched serving (many environments)

```python
banks = model.predict_batch([dict(history=h, obs=o, partial_graph=g, stream=sid), ...], worlds=40)
```

One network call for all streams (GPU or CPU). Per-stream features are identical to
single-stream calls (tested).

## Scoring K candidate robot trajectories with one shared bank

```python
out = bank.evaluate_candidates(robot_paths)   # robot_paths [K,T,4] = (x, y, z, yaw) at (1..T)*0.048 s
out['p_collision'], out['p_social'], out['minimum'], out['first_collision_mass'], ...
```

This calls the L2.5 `aggregate_joint` reduction (robot body envelope vs human disk, 1 m
social events, per-world minima) on the shared bank: no re-prediction per candidate
(≈1 ms for 3–24 candidates). `bank.sandbox_positions(t)` returns positions with absent
slots set to +inf for custom reductions.

## Legal information boundary

Only arrived frames are consumed: robot/camera pose, tracker positives (with tracker keys)
and static C-space certificates. No simulator state, future observation, identity label or
query is accepted. Privileged positions are used only by offline extraction/evaluation.
