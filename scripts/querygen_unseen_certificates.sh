#!/usr/bin/env bash
# Feasibility certificates for the frozen unseen suite (run only AFTER the system freeze).
#   start escape + goal region for every episode; exact-gait progress search for episodes without a
#   verified witness.  Usage: scripts/querygen_unseen_certificates.sh OUT_DIR WITNESS_DIR [PARALLEL]
set -u
OUT=${1:?out dir}; WIT=${2:?witness dir}; PAR=${3:-3}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
KEYS=$($PY - <<'EOF'
import json
for s in json.load(open('docs/querygen/suites_unseen.json'))['unseen']:
    print(f"{s['dataset']}/{s['split']}/{s['scene']}/{s['episode_id']}")
EOF
)
jobs_file=$(mktemp)
for k in $KEYS; do
  n=$(echo "$k" | tr / _)
  echo "CUDA_VISIBLE_DEVICES=1 timeout 2400 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.feasibility $k --suite unseen --out $OUT > $OUT/start_$n.log 2>&1" >> "$jobs_file"
  echo "CUDA_VISIBLE_DEVICES=1 timeout 2400 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.feasibility $k --suite unseen --goal --out $OUT > $OUT/goal_$n.log 2>&1" >> "$jobs_file"
  w="$WIT/$n.json"
  if ! [ -f "$w" ] || ! grep -q '"ok": true' "$w"; then
    echo "CUDA_VISIBLE_DEVICES=1 timeout 4000 nice -n 10 $PY scripts/querygen_run.py eventwm_graph.querygen.feasibility $k --suite unseen --progress 1.5 --exact-gait --budget 3600 --out $OUT > $OUT/progress_$n.log 2>&1" >> "$jobs_file"
  fi
done
xargs -P "$PAR" -I{} bash -c "{}" < "$jobs_file"
rm -f "$jobs_file"
