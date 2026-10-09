#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export EVENTWM_PRODUCTION_CONFIG="${EVENTWM_PRODUCTION_CONFIG:-$REPO/configs/eventwm_300k_production.json}"
# Select the configured Conda interpreter, never whichever Python is on PATH.
PYTHON="$(/usr/bin/python3 -c 'import json,os; print(json.load(open(os.environ["EVENTWM_PRODUCTION_CONFIG"]))["python"])')"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON" "$REPO/scripts/eventwm_production/control.py" "${@:-status}"
