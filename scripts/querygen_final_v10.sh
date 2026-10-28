#!/usr/bin/env bash
# Final v10 evaluation (docs/querygen/FINAL_PROTOCOL_V10.md).  Run from a clean checkout of the freeze tag.
#   scripts/querygen_final_v10.sh SUITE CONFIG OUT_ROOT [WORKERS]
#   SUITE  = unseen2 (test) | dev | unseen (development set, for the frozen system's development numbers)
#   CONFIG = c1k10 | c1k6 | c2k10
set -eu
# run from a private copy: bash reads scripts incrementally (see querygen_dev_v10.sh)
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_final_v10.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
SUITE=${1:?suite}; CFG=${2:?config}; ROOT=${3:?out root}; W=${4:-6}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
if [ -n "$(git status --porcelain -- eventwm_graph/querygen)" ]; then
  echo "querygen package is dirty; refusing to run a final evaluation" >&2; exit 1
fi
# Oracle: v9 options + the v10 portfolio under one deterministic edge budget (wall-clock cap = safety net only)
ORACLE='{"heading_h": true, "escape_penalty": 80, "escape_h": true, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}]}'
case "$CFG" in
  c1k10) PL=orca; DY=native; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false, "goal_relax_m": 0.5}' ;;
  c1k6)  PL=orca; DY=native; GEN='{"select": "compact", "budget": 6, "tc_variants": ["goal"], "st_goal": true, "maneuver_spawn_only": false, "goal_relax_m": 0.5}' ;;
  c2k10) PL=dwa;  DY=inertia; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false, "goal_relax_m": 0.5}' ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
ONLY=()
if [ "$SUITE" = dev ] || [ "$SUITE" = unseen ]; then
  # development numbers of the frozen system: certified-infeasible episodes are skipped (as in development)
  ONLY=(--only "$($PY - "$SUITE" <<'EOF'
import json, sys
s = sys.argv[1]
inf = {'dev': {'yr17PDCnDDW', '7Ukhou1GxYi', 'Dd4bFSTQ8gi', '5cdEh9F2hJL'},
       'unseen': {'sT4fr6TAbpF', 'D8bT1ambLFc', 'FgswoxWb3uN', 's19Uyn7AWwv', 'JeFG25nYj2p', 'VLzqgDo317F', 'gDDiZeyaVc2'}}[s]
f = 'docs/querygen/suites.json' if s == 'dev' else 'docs/querygen/suites_unseen.json'
print(','.join(r['scene'] for r in json.load(open(f))[s] if r['scene'] not in inf))
EOF
)")
fi
OUT="$ROOT/${SUITE}_${CFG}"
mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
  --out "$OUT" --workers "$W" --planner $PL --dynamics $DY --gen "$GEN" --oracle "$ORACLE" "${ONLY[@]}" > "$OUT.log" 2>&1
