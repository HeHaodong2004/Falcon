#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON=${PYTHON:-/home/haodong/miniforge3/envs/eventwm-falcon/bin/python}
OUTPUT=${OUTPUT:-datasets/raw/eventwm_social_outcome_v3_train_scale_01}
TARGET_PRIMARY_QUERIES=${TARGET_PRIMARY_QUERIES:-300000}
MIN_FREE_GB=${MIN_FREE_GB:-150}
STORAGE_PROFILE=${STORAGE_PROFILE:-structured_production}
CONCURRENCY=${CONCURRENCY:-$("$PYTHON" -c 'import json; print(json.load(open("reports/eventwm_mass_collection/release.json"))["selected_concurrency"])')}
GPU_DEVICES=${GPU_DEVICES:-$("$PYTHON" -c 'import json; print(",".join(map(str,json.load(open("reports/eventwm_mass_collection/release.json"))["gpu_assignment"])))')}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
args=(run --output "$OUTPUT" --target "$TARGET_PRIMARY_QUERIES" --concurrency "$CONCURRENCY" --gpus "$GPU_DEVICES" --min-free-gb "$MIN_FREE_GB" --storage-profile "$STORAGE_PROFILE")
if [[ -n ${MAX_ROOTS:-} ]]; then args+=(--max-roots "$MAX_ROOTS"); fi
exec "$PYTHON" -u scripts/eventwm_mass_collection.py "${args[@]}" "$@"
