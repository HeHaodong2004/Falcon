#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
PROJECT=/home/haodong/workspace/projects/eventwm-falcon
OUTPUT="$PROJECT/output/eventwm_v3_30k_pipeline_v1"
SESSION=eventwm_pipeline_v1
RUNNER="$REPO/scripts/eventwm_pipeline_runner.py"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
cd "$REPO"
ACTION="${1:-check}"
case "$ACTION" in
  check) "$PYTHON" "$RUNNER" check --output "$OUTPUT" ;;
  status) "$PYTHON" "$RUNNER" status --output "$OUTPUT" ;;
  stop) "$PYTHON" "$RUNNER" stop --output "$OUTPUT" ;;
  attach) exec tmux attach-session -t "=$SESSION" ;;
  benchmark|overnight|start|resume)
    "$PYTHON" "$RUNNER" check --output "$OUTPUT"
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "Session already exists: $SESSION" >&2; exit 2
    fi
    if [[ "$ACTION" == start && -e "$OUTPUT/manifest.json" ]]; then
      echo 'Use resume for an existing candidate dataset.' >&2; exit 2
    fi
    if [[ "$ACTION" == resume && ! -f "$OUTPUT/pipeline.sqlite3" ]]; then
      echo 'No candidate state to resume.' >&2; exit 2
    fi
    tmux new-session -d -s "$SESSION" -c "$REPO" "$REPO/scripts/run_eventwm_pipeline.sh" "_$ACTION"
    echo "Started $SESSION; attach: ./scripts/run_eventwm_pipeline.sh attach"
    ;;
  _benchmark|_overnight|_start|_resume)
    mkdir -p "$PROJECT/output/eventwm_pipeline_logs"
    LOG="$PROJECT/output/eventwm_pipeline_logs/$(date -u +%Y%m%d_%H%M%S)_$$.log"
    exec > >(tee -a "$LOG") 2>&1
    echo "Persistent log: $LOG"
    if [[ "$ACTION" == _benchmark || "$ACTION" == _overnight ]]; then
      BENCH="$PROJECT/output/eventwm_pipeline_ab_$(date -u +%Y%m%d_%H%M%S)_$$"
      "$PYTHON" -u "$REPO/scripts/eventwm_pipeline_benchmark.py" --output "$BENCH" --proposals 256 --minimum-speedup 1.5
      echo "Native A/B verdict: $BENCH/VERDICT.json"
      [[ "$ACTION" == _benchmark ]] && exit 0
    fi
    if [[ "$ACTION" == _start || "$ACTION" == _resume ]]; then
      # Explicit start/resume is available after the operator has reviewed the
      # native verdict. overnight enforces that verdict automatically.
      echo 'Starting the candidate dataset; native validation is recorded separately.'
    fi
    "$PYTHON" -u "$RUNNER" run --output "$OUTPUT"
    ;;
  *) echo 'Usage: run_eventwm_pipeline.sh {check|benchmark|overnight|start|resume|status|stop|attach}' >&2; exit 2 ;;
esac
