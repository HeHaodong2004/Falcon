#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
command=${1:-status}
shift || true
case "$command" in
  start) command=suite ;;
  freeze|episode|suite|resume|status|stop|summarize|timing-smoke|verify-isolation) ;;
  *) echo "Usage: $0 {freeze|episode|start|suite|resume|status|stop|summarize|timing-smoke|verify-isolation} [options]" >&2; exit 2 ;;
esac
exec nice -n 15 ionice -c 3 "$PY" -B scripts/eventwm_closed_loop.py "$command" "$@"
