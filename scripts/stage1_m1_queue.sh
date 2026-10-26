#!/usr/bin/env bash
# Matched full-scale single-mode control for the lite2 12-epoch candidate.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN="$PY scripts/stage1_hwm_run.py"
while pgrep -f "stage1_hwm.train hwm_fast_e12 " >/dev/null; do sleep 30; done
$RUN eventwm_graph.stage1_hwm.train hwm_lite2_m1_e12 --arch lite2 --modes 1 --truncate --epochs 12 --batch 128 --device cuda:0 --workers 6 --val-every 4000 > output/logs/train_hwm_lite2_m1_e12.log 2>&1
$RUN eventwm_graph.stage1_hwm.bank_hwm output/stage1_hwm/runs/hwm_lite2_m1_e12/best.pt /var/tmp/falcon-stage1-claude/banks/hwm_lite2_m1_e12 --device cuda:0 > output/logs/bank_hwm_lite2_m1_e12.log 2>&1
$RUN eventwm_graph.stage1_hwm.metrics /var/tmp/falcon-stage1-claude/banks/hwm_lite2_m1_e12 hwm_lite2_m1_e12 > output/logs/metrics_hwm_lite2_m1_e12.log 2>&1
echo M1_QUEUE_DONE >> output/logs/m1_queue.log
