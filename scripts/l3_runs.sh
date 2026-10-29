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
  R2)  # main, second attempt: R1 + Monte Carlo target mixing (0.5 -> 0.1 over 300k decisions, fixes the warm-up
       # over-optimism measured in R1) + replay ratio 64; R1's episodes seed the buffer (off-policy reuse)
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R2 --servers 3 --server-devices 0,0,1 --workers 20 \
      --eval-workers 2 --tag R2 --actor-args "--configs $SEEN --suite train_all --selector policy" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --warm output/warm/replay/spool --mc-mix 0.5,0.1,300000 --seed-spool output/runs/R1/spool" ;;
  R3)  # main, third attempt: R2 settings with the thread-free learner (R2's learner aborted on a native heap
       # corruption after 5 min); seeded with every episode R1 and R2 collected
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R3 --servers 3 --server-devices 0,0,1 --workers 20 \
      --eval-workers 2 --tag R3 --actor-args "--configs $SEEN --suite train_all --selector policy" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --warm output/warm/replay/spool --mc-mix 0.5,0.1,300000 --seed-spool output/runs/R1/spool,output/runs/R2/spool" ;;
  R4)  # main, fourth attempt: R3 + Graph Compiler v1b (imagined braking tail in the 3 s risk extension) + default-
       # anchored exploration (the actor takes the default candidate with probability 0.3 -> 0.05 over 300k decisions;
       # R3 thrashed among manoeuvres on momentum robots).  Seeded with every earlier main-run episode of the
       # configurations whose graphs v1b leaves unchanged (no speed state).
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R4 --servers 3 --server-devices 0,0,1 --workers 20 \
      --eval-workers 2 --tag R4 --actor-args "--configs $SEEN --suite train_all --selector policy --compiler '{\"brake_tail\":true}'" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --warm output/warm/replay/spool --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --seed-spool output/runs/R1/spool,output/runs/R2/spool,output/runs/R3/spool --seed-configs orca-native-k10,dwa-native-k10,orca-native-k6,dwa-slowturn-k10" ;;
  R5)  # main, continued from R4's learner state (network, optimiser, counters) and data, plus 10 on-policy
       # counterfactual labellers (every candidate at the critical states of the policy's own failed episodes,
       # followed by the default policy; exact) and 25% of each batch from labelled states.  16 training actors.
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R5 --servers 3 --server-devices 0,0,1 --workers 16 \
      --eval-workers 2 --labelers 10 --labeler-args "--k 4 --cap 160 --p-success 0.25" --tag R5 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --compiler '{\"brake_tail\":true}'" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --pretrain 0 --init output/runs/R4/learner_state.pt --warm output/warm/replay/spool --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.25 --cf-floor 0.5 --v1b --seed-spool output/runs/R4/spool,output/runs/R1/spool,output/runs/R2/spool,output/runs/R3/spool" ;;
  B3)  # no-imagination ablation matching R4's learner and exploration, seeded with B1 + B2 episodes
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/B3 --servers 0 --workers 6 --eval-workers 1 --tag B3 \
      --actor-args "--blind --configs $SEEN --suite train_all --selector policy" \
      --eval-args "--blind --suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --warm output/warm/replay_blind/spool --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --seed-spool output/runs/B1/spool,output/runs/B2/spool" ;;
  B1)  # ablation: no imagination (current legal observation + query descriptors only; no Sandbox, no range frames)
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/B1 --servers 0 --workers 6 --eval-workers 1 --tag B1 \
      --actor-args "--blind --configs $SEEN --suite train_all --selector policy" \
      --eval-args "--blind --suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "$LEARN --warm output/warm/replay_blind/spool" ;;
  B2)  # no-imagination ablation with the R2 learner (MC mixing, replay ratio 64), seeded with B1's episodes
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/B2 --servers 0 --workers 6 --eval-workers 1 --tag B2 \
      --actor-args "--blind --configs $SEEN --suite train_all --selector policy" \
      --eval-args "--blind --suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --warm output/warm/replay_blind/spool --mc-mix 0.5,0.1,300000 --seed-spool output/runs/B1/spool" ;;
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
