#!/bin/bash
# Layer 3 training runs (exact launch commands).  Usage: scripts/l3_runs.sh RUN_NAME
# Every run records its code commit in output/runs/<name>/code_commit.txt and its arguments in fleet_args.json.
set -e
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
SEEN="orca-native-k10:2,dwa-native-k10:2,orca-native-k6:1,orca-inertia-k10:1.5,dwa-slowturn-k10:1.5,dwa-slowturn_inertia-k10:1,orca-slowturn_inertia-k6:1"
EVAL="orca-native-k10:1,dwa-native-k10:1,orca-inertia-k10:1,dwa-inertia-k10:1,sfm-native-k10:1"
LEARN="--device cuda:1 --cf output/warm/cf --pretrain 1000 --min-size 2000 --replay-ratio 32 --tau 0.5,0.15,400000 --eps 0.05,0.02,400000"
name=$1
mkdir -p output/runs/$name
git rev-parse HEAD > output/runs/$name/code_commit.txt
case $name in
  R1)  # main: shared policy, full world model (Sandbox + Graph Compiler), Oracle-guided warm-up, 7 seen configurations
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R1 --servers 3 --server-devices 0,0,1 --workers 20 \
      --eval-workers 2 --tag R1 --actor-args "--configs $SEEN --suite train_all --selector policy" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "$LEARN --warm output/warm/replay/spool" ;;
  B1)  # ablation: no imagination (current legal observation + query descriptors only; no Sandbox, no range frames)
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/B1 --servers 0 --workers 6 --eval-workers 1 --tag B1 \
      --actor-args "--blind --configs $SEEN --suite train_all --selector policy" \
      --eval-args "--blind --suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "$LEARN --warm output/warm/replay_blind/spool" ;;
  S1)  # ablation: specialist (ORCA x native x K=10 only), otherwise as R1
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/S1 --servers 2 --server-devices 0,1 --workers 10 \
      --eval-workers 1 --tag S1 --actor-args "--configs orca-native-k10:1 --suite train_all --selector policy" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "$LEARN --warm output/warm/replay/spool" ;;
  N1)  # ablation: no Oracle warm-up (no demonstrations, no counterfactual labels, no pre-training), otherwise as R1
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/N1 --servers 2 --server-devices 0,1 --workers 10 \
      --eval-workers 1 --tag N1 --actor-args "--configs $SEEN --suite train_all --selector policy" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 32 --tau 0.5,0.15,400000 --eps 0.05,0.02,400000" ;;
  *) echo "unknown run $name"; exit 1 ;;
esac
