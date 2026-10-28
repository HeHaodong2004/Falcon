#!/usr/bin/env bash
# Summaries for REPORT_V10 (test = second unseen suite; development numbers of the frozen system).
#   scripts/querygen_report_v10.sh [OUT_JSON_DIR]
set -u
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$(dirname "$0")/.."
R=output/runs
J=${1:-$R/final_v10/summaries}
mkdir -p "$J"
cd tests/querygen
R=../../$R; J=../../$J
echo "=== TEST: second unseen suite"
for C in c1k10 c1k6 c2k10; do
  $PY summarize_v10.py $R/final_v10/unseen2_$C --witness $R/witness_unseen2 --cert $R/final_v10/certificates_unseen2 \
    --proof-runs $R/final_v10/unseen2_c1k10 $R/final_v10/unseen2_c1k6 $R/final_v10/unseen2_c2k10 --json $J/unseen2_$C.json
done
echo "=== DEV (frozen system; certified-infeasible episodes skipped)"
for C in c1k10 c1k6 c2k10; do
  $PY summarize_v10.py $R/final_v10_devset/dev_$C --witness $R/witness_dev $R/witness_dev2 $R/witness_final_dev \
    $R/witness_3t8_w3 $R/witness_3t8_w6 $R/witness_m50 --json $J/dev_$C.json
done
echo "=== first unseen suite (development data since v9; certified-infeasible episodes skipped)"
for C in c1k10 c1k6 c2k10; do
  $PY summarize_v10.py $R/final_v10_devset/unseen_$C --witness $R/witness_unseen $R/witness_unseen_long_w3 \
    $R/witness_unseen_long_w6 --cert $R/final_v9/certificates_unseen \
    --proof-runs $R/final_v9/unseen_c1k10 $R/final_v9/unseen_c1k6 $R/final_v9/unseen_c2k10 \
    $R/final_v10_devset/unseen_c1k10 $R/final_v10_devset/unseen_c1k6 $R/final_v10_devset/unseen_c2k10 --json $J/unseen_$C.json
done
