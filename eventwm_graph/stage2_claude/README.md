# Stage 2 (Claude redesign): legal voxel simulation + masked BEV U-Net

Trajectory-conditioned prediction of newly observable static certificates:
`(G_t, tau_k, legal range history, legal tracks) -> hypothetical Delta G_k`.

## Public interface (unchanged contract)

```python
from eventwm_graph.stage2_claude.api import LegalVoxelPredictor

predictor = LegalVoxelPredictor(checkpoint, device='cuda:0',      # CUDA graphs captured here (startup)
                                lanes=None)                     # None: checkpoint setting (default 1); >1 overlaps
                                                                # the per-query graphs on that many streams
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
results do not depend on cache state or on the number of lanes (`lanes=3` is checked bit-identical to `lanes=1`).

## Method

1. **Legal voxel map** (`geometry.py`, `kernels.py`): tri-state volume (UNKNOWN / FREE / OCCUPIED)
   over the cell grid with `sub` voxels per cell edge (production: 2, i.e. 5 cm) and 5 cm height bins
   from base-0.4 m to base+2.0 m, built from all legal past/current range frames: measured hits mark
   OCCUPIED (hits on humans legally tracked and visible in that frame are skipped), projective carving
   marks FREE. Currently tracked humans (latest legal frame, velocity-extrapolated over their
   observation age) are stamped as dynamic occluders. Unknown voxels within `near` m of a camera are
   treated as the robot's own volume.
2. **Certificate simulation along each supplied trajectory**: future camera poses from the current rigid
   camera/base extrinsic; per unique pose a `render`x`render` visibility image (first OCCUPIED/human and
   first UNKNOWN distance, production 192); the exact 63-probe FREE certificate geometry is tested against
   the renders; the exact 35x3 wall rays are DDA-marched with the 0.25 m inflation rule and the
   (base+0.2, base+1.4) band; human voxels consume rays without certificates.
3. **Inputs per query** (`features.py`, exact original candidate crop frame): map channels (graph
   FREE/BLOCKED, per-height-band occupied/free/unknown fractions, current humans, and probe/wall-ray
   statistics from up to 16 legal PAST camera poses: what was already fully visible but is still
   uncertified is unlikely to be certified later) + query channels (probe statistics incl. unknown depth
   and pose fractions, known / after-unknown wall hits, distance to known hits, unknown traversal and
   entry, path raster, window poses/duration/length, the constant-velocity sweep of the currently tracked
   humans over the window's duration (`human_future`, from legal track positions and estimated
   velocities), frame validity). Channel names are recorded in each checkpoint and selected by name, so
   older checkpoints keep working; `train.py --extend` warm-starts a checkpoint with the current channel
   set (new channels get zero stem weights, so the warm model's function is unchanged at step 0).
4. **Masked U-Net** (`model.py`): full cell resolution; every activation outside the query frame is
   zeroed and global context is a masked mean, so outputs do not depend on padding or batch composition.
   Outputs revelation r and BLOCKED-given-revelation b. Optional cascade: a second masked U-Net refines
   the first network's logits.
5. **Decision**: frozen in the checkpoint by `select.py` on scene-disjoint TRAIN validation, either
   `reveal` (r > t, then b > t_b) or `class` (FREE if r(1-b) > t_F, BLOCKED if rb > t_B, larger margin wins).

## Files

`api.py` predictor; `features.py`/`geometry.py`/`kernels.py` legal features; `model.py` network and CUDA
graph runner; `train.py`/`data.py` training on TRAIN roots; `select.py` operating-point selection on
scene-disjoint TRAIN validation; `evaluate.py` original development evaluation (identical assertions and
`summarize()` to `stage2_repair/range_full_evaluate.py`); `benchmark.py` original 32-workload complete
latency; `test_api.py` contract tests. Diagnostics: `sim_eval.py`, `sim_check*.py`, `cam_check.py`,
`blocked_debug.py`, `ray_debug.py`, `entry_debug.py`, `fp_debug.py`, `strata.py`, `aux_eval.py`,
`fitcheck.py`, `check_past.py`, `fidelity_debug.py`, `near_ab.py`, `compare_pilot.py`, `viz*.py`.
`quiet.py` runs a benchmark with this session's own background jobs paused (run as a module).
