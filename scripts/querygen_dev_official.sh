#!/usr/bin/env bash
# Development runs under the OFFICIAL Falcon rules (2026-10-28): every episode of DEV, the first and the second
# unseen suites (all development data now; nothing is excluded -- the denominator is every episode not proven
# natively infeasible).  The third unseen suite (suites_unseen3.json) is never run with this script.
#   scripts/querygen_dev_official.sh CONFIG OUT_ROOT [GEN_OVERRIDES_JSON] [ORACLE_OVERRIDES_JSON]
#   CONFIG = c1k10 | c1k6 | c2k10 (c2k10 = DWA x native: official discrete actions)
#   ONLY_DEV / ONLY_UNSEEN / ONLY_UNSEEN2 (env, comma-separated scene names, 'none' = skip) restrict the episodes
#   W_DEV / W_UNSEEN / W_UNSEEN2 (env) worker counts (default 2 / 3 / 5)
set -eu
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_dev_official.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
CFG=${1:?config}; ROOT=${2:?out root}; GX=${3:-'{}'}; OX=${4:-'{}'}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
ORACLE='{"heading_h": true, "escape_penalty": 80, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}]}'
case "$CFG" in
  c1k10) PL=orca; DY=native; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c1k6)  PL=orca; DY=native; GEN='{"select": "compact", "budget": 6, "tc_variants": ["goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  c2k10) PL=dwa;  DY=native; GEN='{"tc_variants": ["goal", "st_goal"], "st_goal": true, "maneuver_spawn_only": false}' ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
merge() { $PY -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' "$1" "$2"; }
GEN=$(merge "$GEN" "$GX"); ORACLE=$(merge "$ORACLE" "$OX")
mkdir -p "$ROOT"
for SUITE in dev unseen unseen2; do
  case $SUITE in
    dev) O=${ONLY_DEV:-}; WS=${W_DEV:-2} ;;
    unseen) O=${ONLY_UNSEEN:-}; WS=${W_UNSEEN:-3} ;;
    unseen2) O=${ONLY_UNSEEN2:-}; WS=${W_UNSEEN2:-5} ;;
  esac
  [ "$O" = none ] && continue
  OUT="$ROOT/${SUITE}_${CFG}"
  ONLY=(); [ -n "$O" ] && ONLY=(--only "$O")
  CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
    --out "$OUT" --workers "$WS" --planner $PL --dynamics $DY --gen "$GEN" --oracle "$ORACLE" --rules official \
    "${ONLY[@]}" > "$OUT.log" 2>&1 &
done
wait
