# Legal forecasting interface

Use the checkpoint roles in the final release manifest. The retained primary is v1.
The same interface supports the completed alternatives, whose separate names
preserve their different probability and forecasting tradeoffs.

`eventforecastwm.inference` accepts a checkpoint, arrived-history JSON, optional
sensor JSON, and optional permitted static-map NPZ. Its output contains coherent
joint tracked-human paths and center-occupancy probabilities. It does not read a
training target cache. The release's standalone receipt tests this boundary and
repeatability using the exact checkpoint and source hashes.

```bash
PYTHONPATH=eventwm-source/scripts "$EVENTWM_PY" -B -m eventforecastwm.inference \
  --checkpoint "$EVENTWM_RELEASE/primary.pt" \
  --history "$EVENTWM_RELEASE/examples/history.json" \
  --sensor "$EVENTWM_RELEASE/examples/sensor.json" \
  --static-map "$EVENTWM_RELEASE/examples/static_map.npz" \
  --horizon 3 --samples 6 --seed 7 --cpu-preparation \
  --device cuda:0 --output /tmp/eventwm_forecast_NEW.npz
```

The output path must be new. CPU inference uses `--device cpu`. The static map
contains finite `features[2,32,32]` and boolean `support[32,32]`, aligned to the
last legal robot pose in the supplied history. An unavailable map is allowed for
the geometry-occupancy adapter and uses its learned unknown-map category. It is
not replaced by privileged dynamic scene state. `--navmesh` instead derives the
same features from a permitted static navmesh and requires the simulator library
for that geometry operation; supplying the NPZ avoids that dependency at inference.
The CLI ignores maps for historical checkpoints whose geometry weights were
not trained, preserving their benchmark input contract. It exports
`geometry_available` (provided) separately from `geometry_conditioning_used`.
Trained physical-map models and the geometry-occupancy adapter still use maps.

For recurrent use, keep the belief and sampling state separately:

```python
import torch
from eventforecastwm.benchmark import load_model
from eventforecastwm.composable_sampling import sample_transition
from eventforecastwm.inference import select_static_geometry

model = load_model(checkpoint).to(device).eval()
static_map = select_static_geometry(model, static_map)
with torch.no_grad():
    belief = model.encode_history(
        legal_batch, static_map, sensor_config, {"radius": 0.25},
        history_timestamps=absolute_history_times,
        track_acquisition_times=absolute_acquisition_times,
    )
    generator = torch.Generator(device=device).manual_seed(7)
    first = model.imagine(belief, horizon_s=1.0)
    paths1, sample_state = sample_transition(first, 6, generator)
    second = model.imagine(first.next_belief, horizon_s=2.0)
    paths2, sample_state = sample_transition(second, 6, generator, sample_state)
    paths = torch.cat([paths1, paths2], dim=2)
```

The legal batch comes from `eventquerywm.inputs.tensorize` and `collate`, as in
the CLI. Both clock sidecars must preserve native float64 absolute timestamps.
The encoder converts them to a fixed root-relative clock. Canonical imagination
segments are positive multiples of 0.25 seconds. The optional candidate-query
argument is accepted for compatibility and does not condition human dynamics.
Do not independently call a fresh joint sampler on each segment when one
continuous scenario is required.

An arrived correction uses the same coordinate frame, existing slot order and
root-relative clock as the current belief:

```python
observation = {
    "source": "actual_arrived",
    "positions": arrived_local_positions,        # [B,N,2]
    "observed": arrived_mask,                    # [B,N], bool
    "acquisition_times": acquired_since_origin,  # [B,N], float64
    "timestamp_s": current_belief.core.clock,    # [B]
    "track_ids": current_belief.core.track_ids,  # unchanged existing slots
}
with torch.no_grad():
    corrected = model.observe(current_belief, observation)
    future = model.imagine(corrected, horizon_s=1.0)
    paths, new_sample_state = sample_transition(future, 6, generator)
```

Only actual arrivals may set `observed=True`. Acquisition time must be no later
than the belief clock and no more than one canonical tick earlier. A stale
acquisition supplies no additional evidence. Do not substitute predicted points
for observations or retain the old sampling state after a correction. New slots
require re-encoding legally acquired history; this campaign does not perform
cross-ID association. Re-encoding also establishes a new map/frame origin.

`samples` has shape `[B,K,T,N,2]`; `valid` identifies forecast slots. Scene-mode
`positions` are conditional means and are not themselves sampled scenarios.
For a stopped-motion law, the export additionally supplies regime means,
probabilities and regime-conditional variance-category probabilities. Use the
distribution-aware readouts when evaluating density or occupancy. A single
Gaussian at the exported mean loses the persistent mixture structure.

The occupancy outputs are **human-center cell marginals**, not body collision
probabilities or a free-space certificate. The anonymous component supplies no
identity trajectories or joint anonymous sample paths. Its support ends five
seconds after the encoding origin. Tracked predictions beyond three seconds
are explicitly marked as extrapolation beyond physical-training support. These
limitations remain part of the interface contract for downstream use.

The prepared six-track latency example is in
`example_legal_dense_inputs/` under the v2 output directory. Its legal history
is byte-identical to the v1 six-track example. The static grid was independently
derived from the permitted mesh and its last legal ego pose and matched the
cached root-origin grid exactly. The provenance records both hashes; no dynamic
target was needed to construct the map. This preparation is not a latency result.

## Completed matched six-track timing

The candidate and released v1 were measured sequentially on GPU 1 with the same
129-frame, six-track legal history and independently verified static map, CPU
legal preprocessing, one CPU thread, five warmups and 50 repetitions. The
production collector remained active and bounded CPU analysis could run; this
is not an idle-host latency claim. Three-second full forecast/readout p95 is
153.748 ms for the age/geometry candidate and 154.056 ms for v1. Median is
115.099 / 132.594 ms and mean 121.816 / 125.409 ms. This single-scene sample
does not establish a universal speedup. Peak allocated tensors are 18,183,168 /
17,797,120 bytes; each reserves 20,971,520 bytes. Disk IO, sensors, simulator
and initial input transfer are excluded. Legal-feature recomputation and its
transfer are included on every encode; trajectory sampling and total/tracked
occupancy readouts are included in the full timing.

CPU/GPU preprocessing parity has maximum position difference 1.431e-6 m,
mode-weight difference 2.981e-8, and candidate variance difference 2.384e-7.
Receipts: candidate_review_primary_latency/result.json and
fallback_primary_latency/result.json. The earlier 149 ms p95 reference used
a different run condition and is not treated as a strict latency target.
