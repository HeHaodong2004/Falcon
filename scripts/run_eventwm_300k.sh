#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
PROJECT=/home/haodong/workspace/projects/eventwm-falcon
OUTPUT=/tmp/eventwm_v4_300k
SESSION=eventwm_300k
RUNNER="$REPO/scripts/eventwm_pipeline_runner.py"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$REPO"
ACTION="${1:-status}"
case "$ACTION" in
 check) "$PYTHON" "$RUNNER" check --output "$OUTPUT" ;;
 status) "$PYTHON" "$REPO/scripts/eventwm_300k_monitor.py" --once ;;
 stop|drain) "$PYTHON" "$RUNNER" stop --output "$OUTPUT" ;;
 attach) exec tmux attach-session -t "=$SESSION" ;;
 logs) tail -n 60 "$PROJECT/output/eventwm_300k_logs/collector.log" ;;
 start|resume)
  "$PYTHON" "$RUNNER" check --output "$OUTPUT"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo 'Collector session already exists';exit 2;fi
  if [[ "$ACTION" == start && -e "$OUTPUT/pipeline.sqlite3" ]];then echo 'Use resume for the saved dataset';exit 2;fi
  if [[ "$ACTION" == resume && ! -e "$OUTPUT/pipeline.sqlite3" ]];then echo 'No saved dataset';exit 2;fi
  "$PYTHON" -c 'import json,sys; v=json.load(open(sys.argv[1])); assert v["scientific_matches"] and v["retained"]>=100, "Native scientific gate has not passed"' "$PROJECT/output/eventwm_72h_native_ab/VERDICT.json"
  "$PYTHON" -c 'import json,sys;v=json.load(open(sys.argv[1]));assert v["passed"] and v["retained_native_primary_queries"]>=100;assert v["source_commit"]==sys.argv[2]' "$PROJECT/output/eventwm_72h_v4_branch_gate/VERDICT.json" "$(git rev-parse HEAD)"
  tmux new-session -d -s "$SESSION" -c "$REPO" "$REPO/scripts/run_eventwm_300k.sh" _run
  echo "Started $SESSION; dataset $OUTPUT"
 ;;
 _run)
  mkdir -p "$PROJECT/output/eventwm_300k_logs"
  nice -n 5 ionice -c 2 -n 6 "$PYTHON" -u "$RUNNER" run --output "$OUTPUT" --target 300000 --concurrency 36 --preparers 4 --validators 2 --prefetch 8 --backlog 72 2>&1 | tee -a "$PROJECT/output/eventwm_300k_logs/collector.log"
 ;;
 *) echo 'Usage: run_eventwm_300k.sh {check|start|status|logs|stop|drain|resume|attach}';exit 2 ;;
esac
