#!/usr/bin/env bash
set -euo pipefail
echo 'V1 production retired. Use the explicit v2 small-run entry point.' >&2
exit 2
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PATH="/home/haodong/miniforge3/envs/eventwm-falcon/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
collector_python=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
if [[ "${1:-}" == --dry-run || "${1:-}" == --preflight-only ]]; then
    exec "$collector_python" scripts/eventwm_overnight.py preflight
fi
if [[ $# -ne 0 ]]; then
    echo 'Usage: scripts/run_eventwm_overnight.sh [--dry-run|--preflight-only]' >&2
    exit 2
fi
production_output="$repo_dir/datasets/raw/eventwm_prod_v1"
mkdir -p "$production_output/logs"
launcher_log="$production_output/logs/launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$launcher_log") 2>&1
printf 'Launcher log: %s\n' "$launcher_log"
exec "$collector_python" -u scripts/eventwm_overnight.py launch
