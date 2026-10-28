#!/usr/bin/env bash
# Feasibility accounting for a finished test suite (FINAL_PROTOCOL_V10_OFFICIAL.md): every episode that some
# configuration did not PASS gets a native witness search (independent of the system under test); episodes without
# a witness and without a PASS of any configuration get the exact native certificate.
#   scripts/querygen_test_feasibility.sh TEST_ROOT SUITE [PARALLEL] [WITNESS_S] [CERT_S]
# e.g. scripts/querygen_test_feasibility.sh output/runs/official/test_v10 unseen3 6 3600 3600
set -u
ROOT=${1:?test root}; SUITE=${2:?suite}; PAR=${3:-6}; WB=${4:-3600}; CB=${5:-3600}
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
cd "$(dirname "$0")/.."
WD=output/runs/witness_native/$SUITE; CD=output/runs/native_cert/$SUITE
mkdir -p "$WD" "$CD"
# keys not PASSed by every configuration, and whether any configuration PASSed them
$PY - "$ROOT" "$SUITE" > "$WD/todo.txt" <<'EOF'
import glob, json, sys
root, suite = sys.argv[1], sys.argv[2]
res = {}
for f in glob.glob(f'{root}/{suite}_*/*.json'):
    if f.endswith(('run.json', 'summary.json')):
        continue
    r = json.load(open(f))
    if 'verdict' in r:
        res.setdefault(r['key'], []).append(r['verdict'] == 'PASS')
for k, v in sorted(res.items()):
    if not all(v):
        print(k, int(any(v)))
EOF
echo "$(wc -l < "$WD/todo.txt") episodes not solved by every configuration"
# 1) native witness searches (skip keys with a PASS somewhere: already confirmed feasible)
awk '$2 == 0 {print $1}' "$WD/todo.txt" | while read k; do n=$(echo "$k" | tr / _); [ -f "$WD/$n.json" ] || \
  echo "CUDA_VISIBLE_DEVICES=1 nice -n 15 $PY scripts/querygen_run.py eventwm_graph.querygen.witness $k --suite $SUITE --out $WD --rules native --budget $WB --max-sims 1600000 > $WD/$n.log 2>&1"; done > "$WD/jobs.txt"
xargs -P "$PAR" -I{} bash -c "{}" < "$WD/jobs.txt"
# 2) exact certificates for keys without a PASS and without a witness
$PY - "$WD" <<'EOF' > "$CD/todo.txt"
import json, sys
from pathlib import Path
wd = Path(sys.argv[1])
for line in (wd/'todo.txt').read_text().split('\n'):
    if not line.strip():
        continue
    k, solved = line.split()
    if solved == '1':
        continue
    f = wd/f"{k.replace('/', '_')}.json"
    w = json.loads(f.read_text()) if f.exists() else {}
    found = bool(w.get('found')) and bool((w.get('verified') or {}).get('ok'))     # verified in the official Env.step
    if not found:
        print(k)
EOF
while read k; do n=$(echo "$k" | tr / _); [ -f "$CD/native_$n.json" ] || \
  echo "CUDA_VISIBLE_DEVICES=1 nice -n 15 $PY scripts/querygen_run.py eventwm_graph.querygen.native_cert $k --suite $SUITE --out $CD --budget $CB > $CD/$n.log 2>&1"; done < "$CD/todo.txt" > "$CD/jobs.txt"
xargs -P "$PAR" -I{} bash -c "{}" < "$CD/jobs.txt"
echo done
