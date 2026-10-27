#!/usr/bin/env bash
# Back-to-back latency of legacy vs the selected Stage 1 model under identical machine load.
# Usage: scripts/stage1_final_benchmarks.sh CHECKPOINT TAG
set -euo pipefail
cd "$(dirname "$0")/.."
CK="$1";TAG="$2"
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
W=$PWD;SB=$W/output/s1base
O=/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023
S=$O/frozen_sources/1ca38d61ddeab23a9022f045685ec5115f3a5a71463305855d10c6063bc188e0
LEG="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR=$SB/output/l2_two_stage_20261023/numba_cache EVENTWM_TWO_STAGE_BASE=$SB PYTHONPATH=$S:$O/deps CUDA_VISIBLE_DEVICES="
for round in 1 2; do
  uptime | tee -a output/logs/final_benchmarks_$TAG.log
  (cd /tmp && $LEG $PY $W/eventwm_graph/stage1_hwm/benchmark_legacy.py $O/experiments/final_stage1_research_20261024/last.pt \
     $W/output/stage1_hwm/benchmark/legacy_${TAG}_r$round > /dev/null 2>&1)
  CUDA_VISIBLE_DEVICES= $PY scripts/stage1_hwm_run.py eventwm_graph.stage1_hwm.benchmark "$CK" ${TAG}_cpu_r$round > /dev/null 2>&1
done
uptime | tee -a output/logs/final_benchmarks_$TAG.log
$PY scripts/stage1_hwm_run.py eventwm_graph.stage1_hwm.benchmark_batch "$CK" ${TAG}_batch_gpu --device cuda:0 > /dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES= $PY scripts/stage1_hwm_run.py eventwm_graph.stage1_hwm.benchmark_batch "$CK" ${TAG}_batch_cpu --device cpu --batches 1 8 32 > /dev/null 2>&1 || true
echo FINAL_BENCHMARKS_DONE | tee -a output/logs/final_benchmarks_$TAG.log
