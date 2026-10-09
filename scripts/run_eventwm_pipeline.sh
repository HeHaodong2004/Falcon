#!/usr/bin/env bash
# Compatibility entry point: all production actions use the immutable 300k release.
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "$REPO/scripts/run_eventwm_300k.sh" "$@"
