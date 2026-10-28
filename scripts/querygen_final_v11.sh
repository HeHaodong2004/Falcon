#!/usr/bin/env bash
# Final v11 evaluation under the OFFICIAL Falcon rules (docs/querygen/FINAL_PROTOCOL_V11.md).  Run from a clean
# checkout of the freeze tag.
#   scripts/querygen_final_v11.sh SUITE CONFIG OUT_ROOT [WORKERS]
#   SUITE  = unseen3 (test) | dev | unseen | unseen2 (development set: the frozen system's development numbers)
#   CONFIG = c1k10 | c1k6 | c2k10            (Oracle selection over the generator's queries)
#          | floor_c1 | floor_c2             (default policy, no selection: ORCA / DWA)
# Nothing is excluded: the denominator is every episode not proven natively infeasible (native_cert.py).
set -eu
# run from a private copy: bash reads scripts incrementally (see querygen_dev_v10.sh)
if [ -z "${QG_LAUNCH_COPY:-}" ]; then
  export QG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  tmp=$(mktemp "${TMPDIR:-/tmp}/querygen_final_v11.XXXXXX.sh"); cp "$0" "$tmp"
  QG_LAUNCH_COPY=1 exec bash "$tmp" "$@"
fi
SUITE=${1:?suite}; CFG=${2:?config}; ROOT=${3:?out root}; W=${4:-6}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$QG_REPO"
if [ -n "$(git status --porcelain -- eventwm_graph/querygen)" ]; then
  echo "querygen package is dirty; refusing to run a final evaluation" >&2; exit 1
fi
# Oracle (privileged selection only; identical for every configuration): the v10 two-member portfolio under one
# deterministic budget of 12,000 simulated edges (wall-clock cap = safety net only).  No escape heuristic: the
# official rules have no static-contact rule.
ORACLE='{"heading_h": true, "escape_penalty": 80, "prefix_cache": 48, "max_edges": 12000, "max_seconds": 40000, "portfolio": [{"weight": 3.5}, {"st_radius": 0.6, "st_radius_opt": 0}]}'
# v11 foundation (legal-only): static-deadlock breaker on the goal query, proprioceptive bump memory, human-threat
# evasion reflex; goal relaxation as v10
V11='"goal_unstick": 6, "bump_memory_m": 0.25, "evade": true, "goal_relax_m": 0.5'
POLICY=oracle
case "$CFG" in
  c1k10) PL=orca; GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $V11}" ;;
  c1k6)  PL=orca; GEN="{\"select\": \"compact\", \"budget\": 6, \"tc_variants\": [\"goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $V11}" ;;
  c2k10) PL=dwa;  GEN="{\"tc_variants\": [\"goal\", \"st_goal\"], \"st_goal\": true, \"maneuver_spawn_only\": false, $V11}" ;;
  floor_c1) PL=orca; POLICY=executor; GEN="{$V11}" ;;
  floor_c2) PL=dwa;  POLICY=executor; GEN="{$V11}" ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
OUT="$ROOT/${SUITE}_${CFG}"
mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite "$SUITE" \
  --policy $POLICY --out "$OUT" --workers "$W" --planner $PL --dynamics native --gen "$GEN" --oracle "$ORACLE" \
  --rules official > "$OUT.log" 2>&1
