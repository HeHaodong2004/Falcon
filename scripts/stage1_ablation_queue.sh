#!/usr/bin/env bash
# Sequential small-protocol ablations on CPU (fast arch, truncation, 30 shards, 2 epochs).
# Each run is trained, then its development banks are generated and scored.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN="$PY scripts/stage1_hwm_run.py"
export CUDA_VISIBLE_DEVICES=
COMMON="--arch fast --truncate --limit-shards 30 --epochs 2 --batch 128 --device cpu --threads 12 --workers 3 --val-every 1500"
wait_for() { while pgrep -f "stage1_hwm.train $1 " >/dev/null; do sleep 30; done; }
score() {
  $RUN eventwm_graph.stage1_hwm.bank_hwm output/stage1_hwm/runs/$1/best.pt /var/tmp/falcon-stage1-claude/banks/$1 --device cpu --chunk 128 >> output/logs/bank_$1.log 2>&1
  $RUN eventwm_graph.stage1_hwm.metrics /var/tmp/falcon-stage1-claude/banks/$1 $1 >> output/logs/metrics_$1.log 2>&1
}
wait_for abl_s0; score abl_s0
for spec in "abl_nomap --ablate map" "abl_novis --ablate vis" "abl_hard --hard-labels" "abl_m1 --modes 1"; do
  set -- $spec; name=$1; shift
  $RUN eventwm_graph.stage1_hwm.train $name $COMMON "$@" > output/logs/train_$name.log 2>&1
  score $name
done
echo ABLATIONS_DONE >> output/logs/ablation_queue.log
