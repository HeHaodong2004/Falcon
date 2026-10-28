#!/usr/bin/env bash
# v14 development runs under the OFFICIAL Falcon rules: DEV + unseen 1-5 (unseen 5 is consumed by the v13 gate test;
# run it here only after that test has finished).  v14 = the v12 query-level system (fast exact sensor; NO privileged
# witness fallback -- user, 2026-10-28 19:10: the fallback is the last resort, used only if every other approach fails)
# + goal-route UNKNOWN cost 1.25 (generator goal_unknown_cost: the default policy no longer drives away from a goal
# behind the spawn) + the path-aware SPL refinement (Oracle improve_edges 300, improve_lambda 1, improve_below_spl 0.85).
#   scripts/querygen_dev_v14.sh CONFIG OUT_ROOT [GEN_OVERRIDES_JSON] [ORACLE_OVERRIDES_JSON]
#   CONFIG = c1k10 | c1k6 | c2k10
#   W_DEV / W_UNSEEN / W_UNSEEN2 / W_UNSEEN3 / W_UNSEEN4 / W_UNSEEN5 (env) worker counts (default 4 / 5 / 5 / 6 / 6 / 6)
#   ONLY_<SUITE> (env, comma-separated scene names, 'none' = skip) restrict the episodes
set -eu
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_dev_v14.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
CFG=${1:?config}; ROOT=${2:?out root}; GX=${3:-'{}'}; OX=${4:-'{}'}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
export QG_TAPE_CACHE="$QG_REPO/output/runs/official/tapes"     # exact human-tape cache (setup 28 s -> ~0 s)
export QG_FAST_SENSE=1                                          # exact fast legal sensor (tests/querygen/test_fastsense.py)
ORACLE='{"heading_h": true, "escape_penalty": 80, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}], "improve_edges": 300, "improve_lambda": 1.0, "improve_below_spl": 0.85}'
# v12 foundation (legal-only): default-policy options + capability fixes
F='"goal_unstick": 6, "bump_memory_m": 0.25, "evade": true, "unstick_keep": true, "bump_front": true, "static_humans": true, "stop_strict": true, "unstick_progress": true, "goal_relax_m": 0.5, "stop_closest_m": 1.2, "goal_unknown_cost": 1.25'
case "$CFG" in
  c1k10) PL=orca; GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"vc_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $F}" ;;
  c1k6)  PL=orca; GEN="{\"select\": \"compact\", \"budget\": 6, \"tc_variants\": [\"goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, \"compact_long_turns\": true, \"compact_human_first\": true, $F}" ;;
  c2k10) PL=dwa;  GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"vc_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $F}" ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
merge() { $PY -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' "$1" "$2"; }
GEN=$(merge "$GEN" "$GX"); ORACLE=$(merge "$ORACLE" "$OX")
mkdir -p "$ROOT"
for SUITE in dev unseen unseen2 unseen3 unseen4 unseen5; do
  case $SUITE in
    dev) O=${ONLY_DEV:-}; WS=${W_DEV:-4} ;;
    unseen) O=${ONLY_UNSEEN:-}; WS=${W_UNSEEN:-5} ;;
    unseen2) O=${ONLY_UNSEEN2:-}; WS=${W_UNSEEN2:-5} ;;
    unseen3) O=${ONLY_UNSEEN3:-}; WS=${W_UNSEEN3:-6} ;;
    unseen4) O=${ONLY_UNSEEN4:-}; WS=${W_UNSEEN4:-6} ;;
    unseen5) O=${ONLY_UNSEEN5:-none}; WS=${W_UNSEEN5:-6} ;;
  esac
  [ "$O" = none ] && continue
  OUT="$ROOT/${SUITE}_${CFG}"
  ONLY=(); [ -n "$O" ] && ONLY=(--only "$O")
  CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
    --out "$OUT" --workers "$WS" --planner $PL --dynamics native --gen "$GEN" --oracle "$ORACLE" --rules official \
    "${ONLY[@]}" > "$OUT.log" 2>&1 &
done
wait
