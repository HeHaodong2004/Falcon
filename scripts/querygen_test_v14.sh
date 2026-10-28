#!/usr/bin/env bash
# Held-out test of the frozen v14 system under the OFFICIAL Falcon rules (docs/querygen/FINAL_PROTOCOL_V14.md)
# on the untouched sixth unseen suite.
#   scripts/querygen_test_v14.sh CONFIG OUT_ROOT [WORKERS] [SUITE]
#   CONFIG = c1k10 (primary) | c1k6 | c2k10 | floor_c1 | floor_c2;  SUITE = unseen6 (default)
# Every episode subprocess imports the querygen package of tag querygen-freeze-v14 (package sha256 checked below) from
# <run>/code_snapshot (git archive) -- exactly the code of the v14 development runs.
set -eu
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_test_v14.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
CFG=${1:?config}; ROOT=${2:?out root}; W=${3:-8}; SUITE=${4:-unseen6}
TAG=querygen-freeze-v14
HASH_EXPECTED=__V14_HASH__
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
export QG_TAPE_CACHE="$QG_REPO/output/runs/official/tapes"     # exact human-tape cache
export QG_FAST_SENSE=1                                          # exact fast legal sensor
ORACLE='{"heading_h": true, "escape_penalty": 80, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}], "improve_edges": 300, "improve_lambda": 1.0, "improve_below_spl": 0.85}'
F='"goal_unstick": 6, "bump_memory_m": 0.25, "evade": true, "unstick_keep": true, "bump_front": true, "static_humans": true, "stop_strict": true, "unstick_progress": true, "goal_relax_m": 0.5, "stop_closest_m": 1.2, "goal_unknown_cost": 1.25'
POLICY=oracle
case "$CFG" in
  c1k10) PL=orca; GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"th_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $F}" ;;
  c1k6)  PL=orca; GEN="{\"select\": \"compact\", \"budget\": 6, \"tc_variants\": [\"goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, \"compact_long_turns\": true, \"compact_human_first\": true, $F}" ;;
  c2k10) PL=dwa;  GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"th_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $F}" ;;
  floor_c1) PL=orca; POLICY=executor; GEN="{$F}" ;;
  floor_c2) PL=dwa;  POLICY=executor; GEN="{$F}" ;;
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
if [ "$HASH" != "$HASH_EXPECTED" ]; then
  echo "frozen package hash mismatch: $HASH" >&2; exit 1
fi
echo "$TAG $(git rev-parse "$TAG^{commit}") $HASH" > "$OUT/FROZEN_SYSTEM"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
  --policy $POLICY --out "$OUT" --workers "$W" --planner $PL --dynamics native --gen "$GEN" --oracle "$ORACLE" \
  --rules official > "$OUT.log" 2>&1
