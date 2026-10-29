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
  R5b) # R5 restarted IN PLACE (same run directory: its spool, labels and learner state carry over) with dense
       # counterfactual supervision (--cf-reg 0.5: the value heads regress on every labelled candidate's exact outcome
       # next to TD), labels over the last 6 decisions of failures, 14 labellers / 14 training actors.
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R5 --servers 3 --server-devices 0,0,1 --workers 14 \
      --eval-workers 2 --labelers 14 --labeler-args "--k 6 --cap 160 --p-success 0.25" --tag R5b \
      --actor-args "--configs $SEEN --suite train_all --selector policy --compiler '{\"brake_tail\":true}'" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "${LEARN/--replay-ratio 32/--replay-ratio 64} --pretrain 0 --init output/runs/R5/learner_state.pt --warm output/warm/replay/spool --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.25 --cf-floor 0.5 --cf-reg 0.5 --v1b --seed-spool output/runs/R4/spool,output/runs/R1/spool,output/runs/R2/spool,output/runs/R3/spool" ;;
  R6)  # EXPERIMENTAL (perception): 360-degree people detection (QG_HUMAN_360=1; map sensing unchanged) + Graph Compiler v2
       # (continuation imagination, 24 steps) + v1b braking tail; warm start from R5's network (partial transfer for the
       # new columns); no Oracle demonstrations (they were recorded with the 90-degree sensor); weaker counterfactual
       # regression (labels at collision moments proved largely unpredictable).
    export QG_HUMAN_360=1 L3_LAYOUT=v2
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R6 --servers 3 --server-devices 0,0,1 --workers 18 \
      --eval-workers 2 --labelers 8 --labeler-args "--k 6 --cap 160 --p-success 0.25" --tag R6 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --compiler '{\"brake_tail\":true}' --sandbox '{\"continuation\":24}'" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R5/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.25" ;;
  R7)  # main (pending the user's sensing decision): 360-degree people detection, Graph Compiler v1b (no continuation
       # imagination: it cost 2.7x Sandbox time in R6 and true-future scoring of imagined continuations did not predict
       # success), warm start from R5's full learner state, 20 actors, 4 labellers (labels at collision moments are
       # mostly timing noise), weaker counterfactual regression.  NEAR_INTERVAL (default 0) enables adaptive frequency.
    export QG_HUMAN_360=1
    unset L3_LAYOUT
    NI=${NEAR_INTERVAL:-0}
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R7 --servers 3 --server-devices 0,0,1 --workers 20 \
      --eval-workers 2 --labelers 4 --labeler-args "--k 6 --cap 160 --p-success 0.25" --tag R7 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --compiler '{\"brake_tail\":true}' --near-interval $NI" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R5/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.1 --cf-floor 0.5 --cf-reg 0.25" ;;
  R8)  # main (legal 90-degree sensing; the user's primary benchmark): Graph Compiler v3 (L3_LAYOUT=v3: query semantics
       # -- kind, family, execution-style flags -- and the robot's information state: camera view age by sector, Stage 1
       # unseen-person mass by direction, lost tracks, and per candidate what its imagined camera would see) + v1b
       # braking tail; Layer 3 scan queries (look left / right, look at a lost person: the Oracle never needs to look,
       # the 90-degree robot must learn when to); warm start from R5's network with the new input columns at zero (starts
       # as R5); no Oracle demonstrations (v1b graphs; they never look around); 6 on-policy labellers (scans included).
    unset QG_HUMAN_360
    export L3_LAYOUT=v3
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R8 --servers 3 --server-devices 0,0,1 --workers 18 \
      --eval-workers 2 --labelers 6 --labeler-args "--k 6 --cap 160 --p-success 0.25" --tag R8 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}'" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R5/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.25" ;;
  R8b) # R8 restarted IN PLACE (spool, labels and learner state carry over) with the training changes: prioritised
       # replay of online decisions (alpha 0.6, IS beta 0.4 -> 1), within-state ranking loss on counterfactual labels,
       # risk-directed exploration (the default's predicted failure risk sets temperature and anchor), and an episode
       # curriculum (uncertain episodes weighted up, always/never-solved down, spawn collisions ~excluded; prior from
       # R1-R5 actor logs, updated from R8's own outcomes).
    unset QG_HUMAN_360
    export L3_LAYOUT=v3
    INIT=output/runs/R8/learner_state.pt
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R8 --servers 3 --server-devices 0,0,1 --workers 18 \
      --eval-workers 2 --labelers 6 --labeler-args "--k 6 --cap 160 --p-success 0.25" --tag R8b \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init $INIT --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.25 --per 0.6,0.4,1.0,300000 --cf-rank 0.5 --explore risk" ;;
  R9)  # main (legal 90-degree sensing): R8b's method + Graph Compiler v4 (L3_LAYOUT=v4: each candidate's INTENTION after
       # its interval -- the running query's remaining executor route (SandboxConfig.route), else the default
       # continuation's legal goal route -- followed for the rest of Stage 1's horizon and scored against every world,
       # hidden people separately, plus route exposure); warm start from R8b's network (new columns at zero); fresh spool;
       # labellers label failures over the whole episode (last 2 + 4 earlier decisions, continuations to the end).
    unset QG_HUMAN_360
    export L3_LAYOUT=v4
    INIT=${INIT:-output/runs/R8/learner_state.pt}
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R9 --servers 3 --server-devices 0,0,1 --workers 18 \
      --eval-workers 2 --labelers 6 --labeler-args "--k 6 --cap 500 --p-success 0.25 --spread" --tag R9 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --sandbox '{\"route\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init $INIT --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.25 --per 0.6,0.4,1.0,300000 --cf-rank 0.5 --explore risk" ;;
  R9b) # R9 restarted IN PLACE (spool, labels, learner state and learner_config.json carry over): 22 training actors and
       # 2 labellers (default-continuation labels proved weak for multi-step behaviour; on-policy data is worth more),
       # fleet with server CUDA-fault recovery and a larger learner restart budget.
    unset QG_HUMAN_360
    export L3_LAYOUT=v4
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R9 --servers 3 --server-devices 0,0,1 --workers 22 \
      --eval-workers 4 --labelers 2 --labeler-args "--k 6 --cap 500 --p-success 0.25 --spread" --tag R9b \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --sandbox '{\"route\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R9/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.25 --per 0.6,0.4,1.0,300000 --cf-rank 0.5 --explore risk" ;;
  R10) # main: R9's method + the execution-loop fix (L3_INHERIT=1: a goal query that interrupts a running goal query
       # inherits its stall / unstick / stop history -- re-issuing goal queries cost 5-8 SR points before), fresh spool,
       # warm start from R9's network; learner settings from output/runs/R10/learner_config.json (copied from R9);
       # 22 actors, 4 greedy dev evaluators, 2 labellers.
    unset QG_HUMAN_360
    export L3_LAYOUT=v4 L3_INHERIT=1
    INIT=${INIT:-output/runs/R10/init_from_R9.pt}
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R10 --servers 3 --server-devices 0,0,1 --workers 22 \
      --eval-workers 4 --labelers 2 --labeler-args "--k 6 --cap 500 --p-success 0.25 --spread" --tag R10 \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --sandbox '{\"route\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init $INIT --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.15 --cf-floor 0.5 --cf-reg 0.05 --per 0.6,0.4,1.0,300000 --cf-rank 0.2 --explore risk" ;;
  R10b) # R10 restarted IN PLACE with more counterfactual supervision: with the v4 features a critic trained on the
       # labels picks better than a random alternative at failure-bound states (35 vs 30%) and beats keeping the default
       # at mixed states (47 vs 31%); the exact one-step rollout policy over the default looks strong (4/4 vs floor 2/4
       # so far).  8 labellers (successes labelled with p 0.5), 16 training actors; learner_config.json raises cf_reg /
       # cf_frac / cf_rank.
    unset QG_HUMAN_360
    export L3_LAYOUT=v4 L3_INHERIT=1
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R10 --servers 3 --server-devices 0,0,1 --workers 16 \
      --eval-workers 4 --labelers 8 --labeler-args "--k 6 --cap 500 --p-success 0.5 --spread" --tag R10b \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --sandbox '{\"route\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R10/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.25 --cf-floor 0.5 --cf-reg 0.25 --per 0.6,0.4,1.0,300000 --cf-rank 0.3 --explore risk" ;;
  R10L) # R10 restarted IN PLACE as a LABEL FACTORY for approximate policy iteration: the exact one-step rollout policy
       # over the default reaches 13/15 vs the floor's 9/15, while the RL learner stays at / below the floor; exact
       # per-candidate labels are the clean signal.  6 training actors (on-policy states), 18 labellers (cheap families
       # skipped via output/runs/R10/cf_config.json), 2 greedy evaluators; the learner keeps training on what arrives.
    unset QG_HUMAN_360
    export L3_LAYOUT=v4 L3_INHERIT=1
    exec $PY scripts/l3_run.py eventwm_graph.l3.fleet --run output/runs/R10 --servers 2 --server-devices 0,1 --workers 6 \
      --eval-workers 2 --labelers 18 --labeler-args "--k 6 --cap 500 --p-success 0.5 --spread" --tag R10L \
      --actor-args "--configs $SEEN --suite train_all --selector policy --scan --compiler '{\"brake_tail\":true}' --sandbox '{\"route\":true}' --curriculum output/curriculum_90.json" \
      --eval-args "--suite dev --selector policy --tau 0 --eps 0 --spool eval_spool --configs $EVAL" \
      --learner-args "--device cuda:1 --pretrain 0 --min-size 2000 --replay-ratio 64 --tau 0.4,0.15,300000 --eps 0.05,0.02,300000 --init output/runs/R10/learner_state.pt --mc-mix 0.5,0.1,300000 --anchor 0.3,0.05,300000 --cf-frac 0.25 --cf-floor 0.5 --cf-reg 0.25 --per 0.6,0.4,1.0,300000 --cf-rank 0.3 --explore risk" ;;
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
