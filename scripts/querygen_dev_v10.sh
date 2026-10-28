#!/usr/bin/env bash
# v10 development runs: DEV (24 feasible) + the first unseen suite (30 feasible + 3 unresolved), which is a
# development set after the v9 evaluation.  Certified-infeasible episodes are skipped.  The second unseen
# suite (suites_unseen2.json) is never run with this script.
#   scripts/querygen_dev_v10.sh CONFIG OUT_ROOT [WORKERS] [GEN_OVERRIDES_JSON] [ORACLE_OVERRIDES_JSON]
#   CONFIG = c1k10 | c1k6 | c2k10 (the v9 configurations; overrides are merged on top)
#   ONLY_DEV / ONLY_UNSEEN (env, comma-separated scene names) restrict the episodes, e.g. to a hard set
set -eu
# run from a private copy: bash reads scripts incrementally, so editing this file under a running instance
# made it resume at a stale offset and relaunch a suite (2026-10-27)
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_dev_v10.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
CFG=${1:?config}; ROOT=${2:?out root}; W=${3:-6}; GX=${4:-'{}'}; OX=${5:-'{}'}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
ORACLE='{"heading_h": true, "escape_penalty": 80, "escape_h": true, "prefix_cache": 48}'
case "$CFG" in
  c1k10) PL=orca; DY=native; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c1k6)  PL=orca; DY=native; GEN='{"select": "compact", "budget": 6, "tc_variants": ["goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c2k10) PL=dwa;  DY=inertia; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
merge() { $PY -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' "$1" "$2"; }
GEN=$(merge "$GEN" "$GX"); ORACLE=$(merge "$ORACLE" "$OX")
ONLY=$($PY - <<'EOF'
import json
dev_inf = {'yr17PDCnDDW', '7Ukhou1GxYi', 'Dd4bFSTQ8gi', '5cdEh9F2hJL'}
un_inf = {'sT4fr6TAbpF', 'D8bT1ambLFc', 'FgswoxWb3uN', 's19Uyn7AWwv', 'JeFG25nYj2p', 'VLzqgDo317F', 'gDDiZeyaVc2'}
d = [r['scene'] for r in json.load(open('docs/querygen/suites.json'))['dev'] if r['scene'] not in dev_inf]
u = [r['scene'] for r in json.load(open('docs/querygen/suites_unseen.json'))['unseen'] if r['scene'] not in un_inf]
print(','.join(d)+' '+','.join(u))
EOF
)
if [ -n "${ONLY_DEV:-}${ONLY_UNSEEN:-}" ]; then ONLY="${ONLY_DEV:-none} ${ONLY_UNSEEN:-none}"; fi
mkdir -p "$ROOT"
# both suites start together so they snapshot the same code; WORKERS is split between them
for SUITE in dev unseen; do
  if [ $SUITE = dev ]; then O=${ONLY% *}; WS=$(( (W*24/57) > 0 ? (W*24/57) : 1 )); else O=${ONLY#* }; WS=$(( W - (W*24/57) > 0 ? W - (W*24/57) : 1 )); fi
  [ "$O" = none ] && continue
  OUT="$ROOT/${SUITE}_${CFG}"
  CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
    --out "$OUT" --workers "$WS" --planner $PL --dynamics $DY --gen "$GEN" --oracle "$ORACLE" --only "$O" > "$OUT.log" 2>&1 &
done
wait
