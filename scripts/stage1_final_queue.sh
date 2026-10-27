#!/usr/bin/env bash
# Final-candidate queue: score lite2_e12 when it ends and start lite_e12 in its GPU slot;
# score fast_e12 and lite_e12 when they end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN="$PY scripts/stage1_hwm_run.py"
wait_for() { while pgrep -f "stage1_hwm.train $1 " >/dev/null; do sleep 30; done; }
score() {
  $RUN eventwm_graph.stage1_hwm.bank_hwm output/stage1_hwm/runs/$1/best.pt /var/tmp/falcon-stage1-claude/banks/$1 --device cuda:0 >> output/logs/bank_$1.log 2>&1
  $RUN eventwm_graph.stage1_hwm.metrics /var/tmp/falcon-stage1-claude/banks/$1 $1 >> output/logs/metrics_$1.log 2>&1
}
wait_for hwm_lite2_e12
score hwm_lite2_e12 &
$RUN eventwm_graph.stage1_hwm.train hwm_lite_e12 --arch lite --truncate --epochs 12 --batch 128 --device cuda:0 --workers 6 --val-every 4000 > output/logs/train_hwm_lite_e12.log 2>&1 &
wait_for hwm_fast_e12
score hwm_fast_e12
wait_for hwm_lite_e12
score hwm_lite_e12
wait
echo FINAL_QUEUE_DONE >> output/logs/final_queue.log
