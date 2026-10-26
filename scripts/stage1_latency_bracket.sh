#!/usr/bin/env bash
# Bracketed back-to-back CPU latency: legacy, candidates..., legacy (same machine state).
# Usage: scripts/stage1_latency_bracket.sh TAG CK1 [CK2 ...]
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1;shift
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
W=$PWD;SB=$W/output/s1base
O=/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023
S=$O/frozen_sources/1ca38d61ddeab23a9022f045685ec5115f3a5a71463305855d10c6063bc188e0
legacy() {
  (cd /tmp && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=$SB/output/l2_two_stage_20261023/numba_cache EVENTWM_TWO_STAGE_BASE=$SB PYTHONPATH=$S:$O/deps CUDA_VISIBLE_DEVICES= \
    $PY $W/eventwm_graph/stage1_hwm/benchmark_legacy.py $O/experiments/final_stage1_research_20261024/last.pt \
    $W/output/stage1_hwm/benchmark/${TAG}_legacy_$1 > /dev/null 2>&1)
}
echo "start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" >> output/logs/latency_${TAG}.log
legacy a
for ck in "$@"; do
  n=$(basename $(dirname $ck))
  CUDA_VISIBLE_DEVICES= $PY scripts/stage1_hwm_run.py eventwm_graph.stage1_hwm.benchmark $ck ${TAG}_${n} > /dev/null 2>&1
  echo "$n done $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" >> output/logs/latency_${TAG}.log
done
legacy b
echo "end $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" >> output/logs/latency_${TAG}.log
