#!/bin/bash
# Full-development Layer 2 validation: N shards on one GPU; a shard that hits a CUDA fault exits with 3 and is
# restarted (resume skips finished Roots; the faulting Root is retried once and logged in *.cuda_faults.jsonl).
# usage: scripts/layer2_validate_shards.sh NAME [SHARDS]
cd "$(dirname "$0")/.."
NAME=$1; N=${2:-4}
mkdir -p output/layer2/logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
for ((i=0; i<N; i++)); do
  (
    while true; do
      python scripts/layer2_run.py eventwm_graph.layer2.validate run "$NAME" --shard $i --shards $N >> output/layer2/logs/${NAME}_shard$i.log 2>&1
      code=$?; [ $code -eq 3 ] || { echo "shard $i exit $code" >> output/layer2/logs/${NAME}_shard$i.log; break; }
    done
  ) &
done
wait
python scripts/layer2_run.py eventwm_graph.layer2.validate summarize "$NAME" > output/layer2/logs/${NAME}_summary.log 2>&1
