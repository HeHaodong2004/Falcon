#!/usr/bin/env bash
# Final v9 evaluation (docs/querygen/FINAL_PROTOCOL_V9.md).  Run from a clean checkout of the freeze tag.
#   scripts/querygen_final_v9.sh SUITE CONFIG OUT_ROOT [WORKERS]
#   SUITE  = dev | unseen ; CONFIG = c1k10 | c1k6 | c2k10
set -eu
SUITE=${1:?suite}; CFG=${2:?config}; ROOT=${3:?out root}; W=${4:-6}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain -- eventwm_graph/querygen)" ]; then
  echo "querygen package is dirty; refusing to run a final evaluation" >&2; exit 1
fi
ORACLE='{"heading_h": true, "escape_penalty": 80, "escape_h": true, "prefix_cache": 48}'
case "$CFG" in
  c1k10) PL=orca; DY=native; GEN='{"tc_variants": ["st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c1k6)  PL=orca; DY=native; GEN='{"select": "compact", "budget": 6, "tc_variants": ["st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c2k10) PL=dwa;  DY=inertia; GEN='{"tc_variants": ["st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
OUT="$ROOT/${SUITE}_${CFG}"
mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
  --out "$OUT" --workers "$W" --planner $PL --dynamics $DY --gen "$GEN" --oracle "$ORACLE" > "$OUT.log" 2>&1
