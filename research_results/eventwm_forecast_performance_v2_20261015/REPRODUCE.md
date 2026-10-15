# Reproduce the completed local campaign

These commands target the validated cv-w06 workspace. The publication README
separately documents portable source/tests/inference; raw Habitat scene assets,
trajectory archives and compact training caches are not public source artifacts.
No command here changes production collection. Completed outputs are immutable;
choose a new output name when repeating an experiment.

## Environment

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_performance_v2_checkout
export EVENTWM_PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
export EVENTWM_V2=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_performance_v2_20261015
export EVENTWM_V1=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_20261014
export EVENTWM_RELEASE="$EVENTWM_V2/release_forecast_v2_20261015"
export PYTHONPATH="$PWD/scripts:$PWD/tests"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/eventwm_matplotlib
export LD_LIBRARY_PATH="$EVENTWM_V2/runtime/nvidia-595.91.07/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Validated versions include Python 3.9.19, PyTorch 1.12.1, NumPy 1.23.5, SciPy
1.13.1, Matplotlib 3.3.2, PyYAML 6.0.3, Pillow 8.4.0 and Habitat-Sim/Lab 0.3.1.
`environment_v2.json` and `environment_reproduction.yml` record the complete
working environment. The YAML is an exported inventory, not a claim that a fresh
cross-machine environment installation was tested. Habitat extraction additionally
requires the licensed scene assets and the exact simulator build. Inference and
unit tests use compact legal input and do not need those simulator assets.

On this host only, cached matching userspace libraries resolve the existing
595.91.07-kernel / 595.99.02-userspace mismatch:

```bash
dpkg-deb -x /var/cache/apt/archives/libnvidia-compute-595_595.91.07-0ubuntu0.22.04.1_amd64.deb \
  "$EVENTWM_V2/runtime/nvidia-595.91.07"
```

This extracts a research-local package; it does not install drivers. Launch
manifests pin that library path because tmux does not inherit a new client's
environment. Other hosts should use their own working driver environment.

## Tests, integrity and standalone inference

```bash
$EVENTWM_PY -B -m unittest discover -s tests -p 'test_forecast*.py' -v
$EVENTWM_PY -B -m eventforecastwm.performance_v2.release verify --release "$EVENTWM_RELEASE"
$EVENTWM_PY -B -m eventforecastwm.inference \
  --checkpoint "$EVENTWM_RELEASE/primary.pt" \
  --history "$EVENTWM_RELEASE/examples/history.json" \
  --sensor "$EVENTWM_RELEASE/examples/sensor.json" \
  --static-map "$EVENTWM_RELEASE/examples/static_map.npz" \
  --device cpu --cpu-preparation --horizon 3 --samples 6 --seed 7 \
  --output /tmp/eventwm_primary_NEW.npz
```

Use `age_calibrated_alternative.pt` with the same arguments for the probability
alternative. `replicate.pt` and `age_calibrated_replicate.pt` are independent seed
15 roles. `stopped_alternative.pt` and `horizon_alternative.pt` preserve the other
predeclared tradeoffs. The primary is byte-identical to v1. `primary.pt` is not a
newly improved physical architecture. The interface accepts legal history,
optional sensor context and permitted map data, never a target cache. See
INFERENCE.md for NPZ fields, units, clocks, support and stochastic semantics.

Archive-only verification extracts `source.tar.gz` to a new directory, sets
`PYTHONPATH` to its `scripts/`, and runs the same inference command from outside
the research worktree. The external `release_validation_receipt.json` records
array parity, exact source/checkpoint hashes, finite outputs and the final
manifest verification. Sealed release files are not edited to add this receipt.

## Research ownership, stop and exact resume

```bash
read -r EVENTWM_OWNER < "$EVENTWM_V2/CURRENT_OWNER"
bash scripts/run_eventwm_forecast.sh status "$EVENTWM_OWNER"
$EVENTWM_PY -B -m eventforecastwm.performance_v2.status --output /tmp/eventwm_status_NEW.json
# During an active run, request a cooperative research-only stop:
bash scripts/run_eventwm_forecast.sh stop "$EVENTWM_OWNER"
# Only after that owner and every child have exited:
bash scripts/run_eventwm_forecast.sh resume "$EVENTWM_OWNER"
```

The final finite queue is `workflow_training_priority_v18_launch.json`. A completed
queue intentionally refuses resume. Earlier owners must not be launched alongside
it. A fresh reproduction below creates its own finite owner. Never directly start
a child managed by an owner. Resource guards retain health/RAM/disk/OOM stops;
low collector throughput alone produces bounded throttling. The physical trainers
resume model, Adam state, deterministic scheduler counter, CPU/CUDA/sampler RNG,
root order/cursor, seen-root state and presentation counters. A 32-step continuous
versus interrupted CUDA audit is bitwise equal. The two real full runs resumed
from 2,677/2,723 and completed 24,576 updates each.

Incremental probability fits verify 64-root density chunks and preserve their
optimizer state. Short legacy calibration stages deterministically restart their
small calibration fit if interrupted; they do not reset physical training.
Lifecycle JSONL records distinguish attempted wall time from successful-fit time.

## Fresh fitting and development evaluation

The helper below copies explicit completed configurations to a new output root,
rewrites producer dependencies and preserves the scientific recipe. It references
verified data/prepared caches instead of duplicating them. Preparation launches no
workers. The 8-job primary, 15-job probability alternative and 16-job new physical
control recipes have passed configuration/dependency preparation checks. Only
fresh assembly of the selected probability law was actually repeated: its tensors
and all development metrics exactly match the original assembly.

```bash
# Retained primary family, two seeds and its original calibration chain:
$EVENTWM_PY -B -m eventforecastwm.performance_v2.prepare_reproduction \
  --spec configs/eventwm_performance_v2/reproduction_primary.json \
  --output "$EVENTWM_V2/reproduce_primary_NEW"
# After stopping any prior research owner and verifying all children exited:
bash scripts/run_eventwm_forecast.sh launch "$EVENTWM_V2/reproduce_primary_NEW/configs/workflow_launch.json"
bash scripts/run_eventwm_forecast.sh status "$EVENTWM_V2/reproduce_primary_NEW/configs/workflow_launch.json"
bash scripts/run_eventwm_forecast.sh stop "$EVENTWM_V2/reproduce_primary_NEW/configs/workflow_launch.json"
bash scripts/run_eventwm_forecast.sh resume "$EVENTWM_V2/reproduce_primary_NEW/configs/workflow_launch.json"
```

Use `reproduction_selected_family.json` and a new destination for the age-aware
alternative, or `reproduction_full_physical_controls.json` for both completed
self-only full-data experiments and their calibrations. These recipes train all
8,965 fitting roots for 24,576 updates per seed. An early best checkpoint is
selected only within the fixed development protocol; final exposure and selected
checkpoint exposure remain separate. Earlier pilots and intermediate runs have
exact launch/config files and completed receipts; see TRAINING.md and ABLATIONS.md.

For development evaluation alone, prepare
`configs/eventwm_performance_v2/reproduction_evaluation.json` into a new destination
using the same helper and launch that destination's workflow. It evaluates both
v1 roles and both age-aware roles with fixed support, ES sample counts, K=6 and
all-window diagnostics. It performs no fitting. All source configurations are
under `configs/eventwm_performance_v2/` and copied in each run's launch receipt.

The immutable new confirmation is already completed, not a tuning loop. Its exact
command is recorded in `configs/eventwm_performance_v2/confirmation_panel_launch.json`.
The decision, membership, cache and checkpoint identities are all required by
`eventforecastwm.confirmation_evaluate`; repeating it needs a new output directory
and cannot change the panel or make these opened banks untouched again. The
completed paired-summary command is:

```bash
$EVENTWM_PY -B research_results/eventwm_forecast_performance_v2_20261015/summarize_confirmation.py \
  --campaign "$EVENTWM_V2"
```

It intentionally refuses to overwrite `confirmation_analysis/`; use its existing
verified receipt unless reproducing in an isolated copied artifact layout.

## Audited 500k continuation, not executed

```bash
$EVENTWM_PY -B -m eventforecastwm.performance_v2.scaling_freeze check
```

Proceed only when `eligible_to_attempt_freeze` is true. The default gate is
495,000 validated queries. It never gates the completed 350k LDM experiments.
The live collector count alone is not a sealed/audited dataset. After the prior
research owner and all children exit, the finite freeze workflow keeps the shared
adaptive guard fresh while using the original audited freezer:

```bash
bash scripts/run_eventwm_forecast.sh launch configs/eventwm_performance_v2/workflow_scaling500k_freeze_launch.json
bash scripts/run_eventwm_forecast.sh status configs/eventwm_performance_v2/workflow_scaling500k_freeze_launch.json
# If interrupted, use the same stop/resume operations only after workers exit.
```

Wait for `scaling500k_snapshot/result.json` with `scaling_snapshot_complete`.
Then preserve exact 350k memberships and hardlink reusable compact cache payloads:

```bash
export EVENTWM_500K_SNAPSHOT="$EVENTWM_V2/scaling500k_snapshot/frozen/snapshot.json"
EVENTWM_500K_SHA=$(sha256sum "$EVENTWM_500K_SNAPSHOT" | cut -d ' ' -f 1)
$EVENTWM_PY -B -m eventforecastwm.scaling \
  --parent-protocol "$EVENTWM_V1/protocol/protocol.json" \
  --snapshot "$EVENTWM_500K_SNAPSHOT" --snapshot-sha256 "$EVENTWM_500K_SHA" \
  --parent-cache "$EVENTWM_V1/cache_full" \
  --new-cache "$EVENTWM_V2/scaling500k_NEW/cache" \
  --output "$EVENTWM_V2/scaling500k_NEW/protocol"
$EVENTWM_PY -B research_results/eventwm_forecast_performance_v2_20261015/prepare_500k_continuation.py \
  --protocol "$EVENTWM_V2/scaling500k_NEW/protocol/protocol.json" \
  --cache "$EVENTWM_V2/scaling500k_NEW/cache" \
  --geometry "$EVENTWM_V2/scaling500k_NEW/geometry" \
  --output "$EVENTWM_V2/scaling500k_primary_updates_NEW" --family primary --budget updates
```

The generated preparation receipt contains exact launch/status/stop/resume commands.
It validates the >=495k sealed audit, parent hashes, inherited fitting roots,
unchanged development and exclusion of all new confirmation IDs. Extraction uses
one CPU worker and only missing targets; no raw archives are copied. Physical
training has priority, one run per available GPU. `--budget updates` uses 24,576
updates per seed. After that run completes, use a fresh training destination and
`--budget exposure` to match at least 174.2998 presentations per fitting root,
accounting for epoch-tail batches. Reuse only the verified completed expanded
cache/geometry. `--family probability_alternative` selects the separately named
age-aware recipe; it does not change the final primary selection.

The helper's epoch-tail arithmetic and real 350k gate rejection passed a bounded
self-test. A positive 500k extraction/training run has not occurred, so no end-to-
end 500k success is claimed. If a deliberate stop interrupts old data/geometry
builders, preserve failure/partial-cache receipts and inspect the known explicit-
stop failure before resuming; automatic exact recovery of that expanded-data path
has not been demonstrated. The completed 350k optimizer recovery is separately
verified. Already-opened confirmation cannot select new scaling hyperparameters;
reserve another untouched population for any newly tuned generalization claim.

## Seal the verified release

```bash
$EVENTWM_PY -B -m eventforecastwm.performance_v2.release build \
  --config configs/eventwm_performance_v2/final_release.json
$EVENTWM_PY -B -m eventforecastwm.performance_v2.release verify --release "$EVENTWM_RELEASE"
```

Build requires committed clean source/reports, current source-bound tests and
standalone proofs, recurrent checks, two completed full training seeds, matching
dataset hashes, the frozen complete confirmation panel and its hash-bound family
review. It refuses an existing release destination. Primary/replicate cannot be
reassigned based on test ranking. `release_manifest.json` provides file sizes and
SHA256 values; this detects changed bytes but is not a cryptographic signature.
The original v1 release is never overwritten. Large data, caches and intermediate
models remain outside Git.
