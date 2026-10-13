#!/usr/bin/env bash
# Run this from the frozen root-admission worktree. No configurable science flags.
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
SESSION=eventwm_30k
OUTPUT=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_v3_30k_c36_combined_exact_gate
RUNNER="$REPO/scripts/eventwm_combined_runner.py"
ACTION="${1:-start}"
if [[ $# -gt 1 ]]; then
    echo 'Usage: scripts/run_eventwm_30k.sh {check|start|status|attach|stop|resume}' >&2
    exit 2
fi
if [[ "$(pwd -P)" != "$REPO" ]]; then
    echo "ERROR: first cd to $REPO; do not run from the parent checkout." >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$PYTHON" "$RUNNER" check --output "$OUTPUT"
case "$ACTION" in
    check) ;;
    start|resume)
        command -v tmux >/dev/null || { echo 'ERROR: tmux is required' >&2; exit 2; }
        if tmux has-session -t "=$SESSION" 2>/dev/null; then
            echo "ERROR: $SESSION already exists; use status/attach, do not launch twice." >&2
            exit 2
        fi
        if [[ "$ACTION" == start && -e "$OUTPUT/manifest.json" ]]; then
            echo 'ERROR: existing manifest; use resume instead of start.' >&2
            exit 2
        fi
        if [[ "$ACTION" == resume && ! -f "$OUTPUT/manifest.json" ]]; then
            echo 'ERROR: no saved manifest; use start for the first launch.' >&2
            exit 2
        fi
        tmux new-session -d -s "$SESSION" -c "$REPO" "$REPO/scripts/run_eventwm_30k.sh" _run
        echo "Started tmux $SESSION. Output: $OUTPUT"
        echo 'Check status; attach with scripts/run_eventwm_30k.sh attach.'
        ;;
    _run)
        # tmux invokes this internal action. The Python runner takes both locks,
        # validates source/manifest identity, and resumes verified completions.
        mkdir -p "$REPO/output/eventwm_30k_logs"
        LOG="$REPO/output/eventwm_30k_logs/session_$(date -u +%Y%m%d_%H%M%S)_$$.log"
        printf 'C36 COMBINED exact-gate collection. Persistent log: %s\n' "$LOG"
        set +e
        "$PYTHON" -u "$RUNNER" run --output "$OUTPUT" 2>&1 | tee -a "$LOG"
        result=$?
        set -e
        printf 'Collector exited: %s. Log: %s\n' "$result" "$LOG"
        # The session exits after a clean drain; resume can use the same name.
        exit "$result"
        ;;
    status) "$PYTHON" "$RUNNER" status --output "$OUTPUT" ;;
    attach) exec tmux attach-session -t "=$SESSION" ;;
    stop) "$PYTHON" "$RUNNER" stop --output "$OUTPUT" ;;
    *) echo 'Usage: scripts/run_eventwm_30k.sh {check|start|status|attach|stop|resume}' >&2; exit 2 ;;
esac
