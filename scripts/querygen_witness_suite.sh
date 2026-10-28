#!/usr/bin/env bash
# Privileged witness searches for every episode of a suite (independent of the system under test).
#   scripts/querygen_witness_suite.sh SUITE SUITE_JSON KEY OUT [PARALLEL] [BUDGET_S] [MAX_SIMS] [WEIGHT]
# e.g. scripts/querygen_witness_suite.sh unseen2 docs/querygen/suites_unseen2.json unseen2 output/runs/witness_unseen2 4 3600 1600000
# For a held-out suite, results are not inspected before the system freeze.
set -u
SUITE=${1:?suite}; JSON=${2:?suite json}; FIELD=${3:?json field}; OUT=${4:?out}; PAR=${5:-4}; B=${6:-3600}; MS=${7:-1600000}; WT=${8:-3}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
jobs_file=$(mktemp)
$PY - "$JSON" "$FIELD" <<'PYEOF' | while read k; do n=$(echo "$k" | tr / _); [ -f "$OUT/$n.json" ] || echo "CUDA_VISIBLE_DEVICES=1 nice -n 15 $PY scripts/querygen_run.py eventwm_graph.querygen.witness $k --suite $SUITE --out $OUT --budget $B --max-sims $MS --weight $WT > $OUT/$n.log 2>&1"; done > "$jobs_file"
import json, sys
for s in json.load(open(sys.argv[1]))[sys.argv[2]]:
    print(f"{s['dataset']}/{s['split']}/{s['scene']}/{s['episode_id']}")
PYEOF
xargs -P "$PAR" -I{} bash -c "{}" < "$jobs_file"
rm -f "$jobs_file"
