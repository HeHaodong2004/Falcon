# Stage 2 (Claude redesign): legal voxel simulation + masked BEV U-Net

Trajectory-conditioned prediction of newly observable static certificates:
`(G_t, tau_k, legal range history, legal tracks) -> hypothetical Delta G_k`.

## Public interface (unchanged contract)

```python
from eventwm_graph.stage2_claude.api import LegalVoxelPredictor

predictor = LegalVoxelPredictor(checkpoint, device='cuda:0')      # CUDA graphs captured here (startup)
deltas = predictor.predict_segments(
    partial_graph=G_t,                  # causal_partial_cspace_graph_1 dict (never mutated)
    trajectories=[tau_1, ..., tau_K],   # each N_k x 5 [time, x, y, z, yaw]; shared starting pose = current pose
    range_observations=packet,          # dict(ranges FxSxS Euclidean m (NaN = no return <=10 m), origins Fx3,
                                        #      rotations Fx3x3, timestamps F, sensor_protocol=
                                        #      'ideal_current_jaw_range_dynamic_occlusion_1'); last frame time == G_t time
    track_history=tracks,               # F lists of legal track rows aligned with the range frames
    materialize=True)                   # complete native hypothetical future graph per query
predictor.predict(partial_graph=..., trajectory=..., range_observations=..., track_history=...)  # K=1 convenience
predictor.clear_cache()                 # drops graph-preparation and legal-map caches
```

Each `delta` (one per trajectory, same order) contains:

| key | meaning |
| --- | --- |
| `candidate_cells` (n,2) int32 | ORIGINAL universe: initially UNKNOWN cells whose f32 centers lie within 11 m of a supplied f32 position, original row-major order (bit-identical to `stage2_quality.features.candidate_mask`) |
| `probabilities` (n,3) f32 | [UNKNOWN, FREE, BLOCKED] = [1-r, r(1-b), rb] |
| `states` (n,) uint8 | hard decision (frozen operating point stored in the checkpoint) |
| `heights` (n,) f32 | start base height (FREE/BLOCKED certificates carry base/navmesh height) |
| `static_updates` | concrete `SparseIncrement` of the predicted FREE/BLOCKED cells, timestamp = path end |
| `future_partial_graph` | complete `NativePartialGraph` (known cells, heights and metadata retained exactly; `hypothetical=True`) |
| `unknown_cells`, `base_graph_identity`, `timestamp`, `robot_pose`, `probability_columns`, `observation_scope`, `prediction_task`, `provenance` | as in the previous range API |

Guarantees (tested in `test_api.py`): exact original candidate domain; a starting-pose-only path reveals
nothing; inputs are never mutated; **single-query and batched calls are bit-identical** (each query runs
alone in the batch-1 CUDA graph of its own frame-size bucket; wall-ray traversal uses fixed-point atomics);
results do not depend on cache state.

## Method

1. **Legal voxel map** (`geometry.py`, `kernels.py`): 0.1 m (x,z cell grid) x 0.1 m (height) tri-state
   volume (UNKNOWN / FREE / OCCUPIED) from all legal past/current range frames: measured hits mark
   OCCUPIED (optionally skipping hits on legally tracked visible humans); projective carving marks FREE.
2. **Certificate simulation along each supplied trajectory**: future camera poses from the current rigid
   camera/base extrinsic; per pose a 96x96 visibility render (first OCCUPIED / first UNKNOWN distance);
   the exact 63-probe FREE certificate geometry (5 square + 16 halo probes x 3 heights) is tested
   against the renders; the exact 35x3 wall rays are marched (DDA) with the 0.25 m inflation rule.
3. **BEV canvas per query** (`features.py`): 20 map channels (graph FREE/BLOCKED, per-height-band
   occupied/free/unknown fractions) + query channels (probe statistics, unknown depth, wall-ray hits,
   unknown traversal/entry, path raster, frame validity) on the query's original candidate crop.
4. **Masked U-Net** (`model.py`): full cell resolution; every activation outside the query frame is
   zeroed and global context is a masked mean, so outputs do not depend on padding or batch composition.
   Outputs revelation r and BLOCKED-given-revelation b.

## Files

`api.py` predictor; `features.py`/`geometry.py`/`kernels.py` legal features; `model.py` network and CUDA
graph runner; `train.py`/`data.py` training on TRAIN roots; `select.py` operating-point selection on
scene-disjoint TRAIN validation; `evaluate.py` original development evaluation (identical assertions and
`summarize()` to `stage2_repair/range_full_evaluate.py`); `benchmark.py` original 32-workload complete
latency; `test_api.py` contract tests. Diagnostics: `sim_eval.py`, `sim_check*.py`, `cam_check.py`,
`blocked_debug.py`, `ray_debug.py`, `entry_debug.py`, `fp_debug.py`, `strata.py`, `aux_eval.py`,
`fitcheck.py`, `viz*.py`.
