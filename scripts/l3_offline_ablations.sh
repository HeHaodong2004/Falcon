#!/bin/bash
# Representation ablations with the data held fixed (E3): critics trained OFFLINE on the main run's replay (same
# learner, same warm-up data, same number of updates), differing only in the input feature groups, then evaluated
# greedily on dev.  Usage: scripts/l3_offline_ablations.sh SEED_SPOOL OUT_ROOT UPDATES [CONFIGS]
#   e.g. scripts/l3_offline_ablations.sh output/runs/R4/spool output/ablations/A1 40000
set -e
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
SEED=$1; OUT=$2; UPD=${3:-40000}
CONFIGS=${4:-orca-native-k10,dwa-native-k10,orca-native-k6,orca-inertia-k10,dwa-slowturn-k10,sfm-native-k10,dwa-inertia-k10}
mkdir -p $OUT
git rev-parse HEAD > $OUT/code_commit.txt
declare -A MODEL=(
  [full]='{}'
  [no_map]='{"drop": ["map"]}'
  [no_human]='{"drop": ["human"]}'
  [flat]='{"flat": true}'
  [no_imagination]='{"drop": ["motion", "progress", "static", "human", "map"]}'
)
for v in full no_map no_human flat no_imagination; do
  if [ ! -f $OUT/$v/policy.pt ]; then
    mkdir -p $OUT/$v
    $PY scripts/l3_run.py eventwm_graph.l3.learner --run $OUT/$v --device cuda:1 --seed-spool $SEED \
      --warm output/warm/replay/spool --cf output/warm/cf --pretrain 1000 --offline-updates $UPD \
      --mc-mix 0.5,0.1,300000 --model "${MODEL[$v]}" > $OUT/$v/train.log 2>&1
  fi
done
for v in full no_map no_human flat no_imagination; do
  if [ ! -f $OUT/eval_$v/summary.json ]; then
    $PY scripts/l3_run.py eventwm_graph.l3.evaluate run --out $OUT/eval_$v --suite dev --configs $CONFIGS \
      --selector policy --policy $OUT/$v/policy.pt --workers ${WORKERS:-16} --servers 2 --server-devices 0,1 \
      --tag abl_$v > $OUT/eval_$v.log 2>&1
  fi
done
$PY scripts/l3_run.py eventwm_graph.l3.evaluate matrix --out $(for v in full no_map no_human flat no_imagination; do printf "$OUT/eval_$v,"; done | sed 's/,$//') \
  --names full,no_map,no_human,flat,no_imagination > $OUT/matrix.md
cat $OUT/matrix.md
