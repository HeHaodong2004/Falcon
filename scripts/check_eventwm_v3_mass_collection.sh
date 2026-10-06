#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON=${PYTHON:-/home/haodong/miniforge3/envs/eventwm-falcon/bin/python}
exec "$PYTHON" scripts/eventwm_mass_monitor.py --output "${OUTPUT:-datasets/raw/eventwm_social_outcome_v3_train_scale_01}"
