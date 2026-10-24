#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONDONTWRITEBYTECODE=1
exec /research-user/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_overnight.py status
