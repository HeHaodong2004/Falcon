# Reproduction and research-only operations

Use the existing environment; no package installation or driver change is
required. All commands run from the isolated worktree:

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_checkout
export EVENTWM_PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
export EVENTWM_RUN=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_20261014
export PYTHONPATH="$PWD/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Completed owner and operational commands

The final owner was workflow v10, containing evaluation queue v8. Both are
complete and inactive. Earlier workflow/queue generations are completed or
stopped historical configurations. The commands below document operations;
completed runs reject resume, and a reproduction needs a new output directory.
Never resume two owners of the same child runs.

```bash
bash scripts/run_eventwm_forecast.sh status configs/eventwm_forecast_workflow_v10_launch.json
bash scripts/run_eventwm_forecast.sh status configs/eventwm_forecast_evaluation_queue_v8_launch.json
bash scripts/run_eventwm_forecast.sh stop configs/eventwm_forecast_workflow_v10_launch.json
bash scripts/run_eventwm_forecast.sh resume configs/eventwm_forecast_workflow_v10_launch.json
```

Stopping the owner requests cooperative stops of its research children. Confirm
`active: false` and current `launcher_exit.json` receipts before launching a new
owner. The stop script never signals the collector. Training checkpoints retain
parameters, AdamW state, scheduler step, sampled root order/cursor, and NumPy,
PyTorch and CUDA RNG states. A 32-step versus interrupted 8+24-step GPU test
matched these states bit for bit (`resume_audit/result.json`).

One trainer per GPU and one CPU evaluation run concurrently. Each worker uses
one CPU thread and low CPU/IO priority. Launch checks retain 48 GiB available
RAM, 50 GiB disk, and at least 8 GiB available GPU memory. Throughput below 65%
of the original 3,615.97 queries/hour baseline for three rolling windows pauses
research; recovery requires three windows above 85%, spanning at least 120 s.
Do not delete guard files or change thresholds to force progress.

The monitor may run while guarded children are paused. Its explicit 16-hour
lifetime bounds unattended dispatch; check status before resuming. Completed
result directories reject relaunch. A resource-paused benchmark repeats its
bounded evaluation; training resumes exact state. Confirmation resumes completed
model/bank panels only if all source and artifact hashes still match.

## Training and evaluation

The selected initial-belief family has exact training configurations
`configs/eventwm_forecast_posterior_full_b64_s20261014.json` and
`...s20261015.json`, with corresponding `_launch.json` wrappers. Original full
baselines are `...deterministic_gp_full_cached.json` and
`...correlated_full_cached.json`. Read their `result.json` for completion;
checkpoint existence alone does not imply a completed budget.

For a new reproduction, first choose a free research GPU and a new output
directory. Copy the configuration and change only `run` and, if needed,
`device`; preserve its data/protocol/model/seed/optimization fields. Do not point
a new reproduction at a completed run. The launch wrapper is a JSON object:

```json
{
  "run": "/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_reproduction_s20261014",
  "module": "eventforecastwm.train_posterior",
  "args": ["--config", "/absolute/path/to/new_training_config.json"]
}
```

```bash
$EVENTWM_PY -B scripts/belief_resource_probe.py --cuda --output /tmp/eventwm_reproduction_resources.json
bash scripts/run_eventwm_forecast.sh launch /absolute/path/to/new_launch_config.json
bash scripts/run_eventwm_forecast.sh status /absolute/path/to/new_launch_config.json
bash scripts/run_eventwm_forecast.sh stop /absolute/path/to/new_launch_config.json
bash scripts/run_eventwm_forecast.sh resume /absolute/path/to/new_launch_config.json
```

Evaluation configurations explicitly name the frozen manifest, protocol,
checkpoint and fitting population. For example,
`eventwm_forecast_benchmark_full_posterior_s20261014_ou_reuse.json` fits only
uncertainty on fitting roots, then scores development. The corresponding
`..._scene_scale.json` and `..._agent_scale.json` are separate probability-law
experiments. Their output checkpoints include calibration memberships and the
common training-only anonymous occupancy prior. For a repeat, copy the JSON
and change `run` and `--output` to new paths, then use the same launcher.

```bash
bash scripts/run_eventwm_forecast.sh launch /absolute/path/to/new_evaluation_launch_config.json
bash scripts/run_eventwm_forecast.sh status /absolute/path/to/new_evaluation_launch_config.json
```

```bash
$EVENTWM_PY -B -m unittest discover -s tests -p 'test_forecast*.py' -v
$EVENTWM_PY -B -m eventforecastwm.benchmark_reuse --help
$EVENTWM_PY -B -m eventforecastwm.window_evaluate --help
$EVENTWM_PY -B -m eventforecastwm.confirmation_evaluate --help
```

The clean confirmation evaluator additionally requires a committed decision
file containing exact checkpoint, membership and cache SHA256 hashes. It
performs no calibration or model selection. Do not inspect a partial panel and
then change the selected recipe.

## Legal-only inference

The release includes the selected initial-belief model with persistent per-agent
variance categories, its independently trained second seed, and the Gaussian OU
alternative. All were frozen before reserved confirmation; the release manifest
links that fixed decision and the final evaluation location.

```bash
export EVENTWM_CHECKPOINT="$EVENTWM_RUN/release_forecast_v1_20261015/primary.pt"
$EVENTWM_PY -B -m eventforecastwm.inference \
  --checkpoint "$EVENTWM_CHECKPOINT" \
  --history "$EVENTWM_RUN/release_forecast_v1_20261015/example_history.json" \
  --sensor "$EVENTWM_RUN/release_forecast_v1_20261015/example_sensor.json" \
  --horizon 3 --samples 6 \
  --output /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_inference_reproduction/forecast.npz
```

Primary SHA256:
`d092b5d05d38be4619251e009d2aa2dafc691f3f57440473d72e87acf7887970`.
The identical physical means with Gaussian OU residuals are in
`release_forecast_v1_20261015/gaussian_alternative.pt`, SHA256
`ec4e3b0790531f7b688116ccf5be00b73aca0c26e6ed36cb8c4d1d94b231f1e6`.
The legal example contains arrived observations and provenance, not future
targets or candidate outcomes. Use a fresh output filename for every run.

The output includes component mean trajectories, coherent sampled scenarios,
weights, covariance, occupancy and validity. See [MODEL.md](MODEL.md) for the
`encode_history` / `imagine` / `observe` API and persistent sampling example.
Only map-trained recipes should receive a static-map sidecar: supplying geometry
to a model trained with unavailable-map inputs changes its physical model.
Standalone inference does not enter the training or confirmation target cache.

For GPU use, timestamp sanitization/regression may be computed on the same legal
CPU history, then passed through `prepared_history=prepared.to(device)` to
`encode_history`. The learned GRU is still recomputed; no target cache or learned
embedding is reused. `latency.py --cpu-preparation` measures this complete path,
including feature recomputation and transfer. On six tracks and 129 history
frames, its 50-repeat full-forecast p95 was 149 / 126 / 196 ms for batches
1 / 8 / 32, versus 312 / 313 / 358 ms with GPU preprocessing. Forecast position
differences were at most 1.44e-6 m. These are shared-workstation measurements on
repeated copies of one scene, not an idle-device service-level guarantee.

## Separate 500k scaling

The following are commands for a future dataset version, not evidence that a
500k audited snapshot exists. Use fresh paths and check collection, resources
and audit status first. The parent must be the current 350k snapshot directory.

```bash
$EVENTWM_PY -B scripts/eventwm_belief_freeze.py \
  --parent /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_350k_belief_snapshot_20261014 \
  --output /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_500k_forecast_snapshot_NEW

$EVENTWM_PY -B -m eventforecastwm.scaling \
  --parent-protocol "$EVENTWM_RUN/protocol/protocol.json" \
  --snapshot /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_500k_forecast_snapshot_NEW/snapshot.json \
  --snapshot-sha256 VERIFIED_NEW_SNAPSHOT_SHA256 \
  --parent-cache "$EVENTWM_RUN/cache_full" \
  --new-cache /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_500k_NEW/cache \
  --output /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_500k_NEW/protocol
```

Check `--help` and the new audit receipt before extraction/training. The scaling
tool rejects fewer than 495,000 validated queries by default, preserves exact
old subset IDs, verifies inherited joins and reservations, and hard-links
verified compact payloads where possible. It never changes the production
target or restarts a training experiment. New roots require incremental target
extraction and an independently versioned training run with unchanged evaluation
membership. Report both matched optimizer budgets and matched root exposure.
