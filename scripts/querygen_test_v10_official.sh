#!/usr/bin/env bash
# Held-out gate test of the frozen v10 system under the OFFICIAL Falcon rules
# (docs/querygen/FINAL_PROTOCOL_V10_OFFICIAL.md).
#   scripts/querygen_test_v10_official.sh CONFIG OUT_ROOT [WORKERS] [SUITE]
#   CONFIG = c1k10 | c1k6 | c2k10;  SUITE = unseen3 (default: the test suite)
# Every episode subprocess imports the querygen package of the freeze tag querygen-freeze-v10-official (commit
# 2786115d, package sha256 bcecb8e8...) from <run>/code_snapshot, extracted with git archive -- the exact code of
# the development runs output/runs/official/base -- whatever the working tree contains.
set -eu
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_test_v10_official.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
CFG=${1:?config}; ROOT=${2:?out root}; W=${3:-7}; SUITE=${4:-unseen3}
TAG=querygen-freeze-v10-official
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
ORACLE='{"heading_h": true, "escape_penalty": 80, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}]}'
case "$CFG" in
  c1k10) PL=orca; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c1k6)  PL=orca; GEN='{"select": "compact", "budget": 6, "tc_variants": ["goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c2k10) PL=dwa;  GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
OUT="$ROOT/${SUITE}_${CFG}"
mkdir -p "$OUT/code_snapshot"
if [ ! -d "$OUT/code_snapshot/eventwm_graph/querygen" ]; then
  git archive "$TAG" eventwm_graph/querygen | tar -x -C "$OUT/code_snapshot"
fi
HASH=$($PY - "$OUT/code_snapshot/eventwm_graph/querygen" <<'EOF'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
for p in sorted(Path(sys.argv[1]).glob('*.py')):
    h.update(p.name.encode()); h.update(p.read_bytes())
print(h.hexdigest())
EOF
)
if [ "$HASH" != "bcecb8e8119faf9c20b5d1011455c166755e01efc9132c094b844f3db397da87" ]; then
  echo "frozen package hash mismatch: $HASH" >&2; exit 1
fi
echo "$TAG $(git rev-parse "$TAG^{commit}") $HASH" > "$OUT/FROZEN_SYSTEM"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
  --out "$OUT" --workers "$W" --planner $PL --dynamics native --gen "$GEN" --oracle "$ORACLE" --rules official \
  > "$OUT.log" 2>&1
