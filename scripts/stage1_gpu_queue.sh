#!/usr/bin/env bash
# GPU queue (one Stage 1 job at a time on the session's GPU): score hwm_a, then train/score hwm_fast_t.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN="$PY scripts/stage1_hwm_run.py"
wait_for() { while pgrep -f "stage1_hwm.train $1 " >/dev/null; do sleep 30; done; }
score() {
  $RUN eventwm_graph.stage1_hwm.bank_hwm output/stage1_hwm/runs/$1/best.pt /var/tmp/falcon-stage1-claude/banks/$1 --device cuda:0 >> output/logs/bank_$1.log 2>&1
  $RUN eventwm_graph.stage1_hwm.metrics /var/tmp/falcon-stage1-claude/banks/$1 $1 >> output/logs/metrics_$1.log 2>&1
}
wait_for hwm_a
score hwm_a &
$RUN eventwm_graph.stage1_hwm.train hwm_fast_t --arch fast --truncate --epochs 6 --batch 128 --device cuda:0 --workers 8 --val-every 3000 > output/logs/train_hwm_fast_t.log 2>&1
wait
score hwm_fast_t
echo GPU_QUEUE_DONE >> output/logs/gpu_queue.log
