#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec /home/haodong/miniforge3/envs/eventwm-falcon/bin/python -B scripts/eventforecastwm/control.py "$@"
