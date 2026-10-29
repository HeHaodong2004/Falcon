#!/bin/bash
# Milestone dev evaluation of a Layer 3 checkpoint (pre-registered selection: mean dev SR over the 7 SEEN
# configurations; the held-out configurations are reported, never used for selection).
#   scripts/l3_milestone.sh RUN VERSION [LAYOUT] [WORKERS] [EXTRA evaluate.py args...]
# e.g. scripts/l3_milestone.sh R9 200 v4 8 --margin 0.5
set -e
cd "$(dirname "$0")/.."
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN=$1; VER=$2; LAYOUT=${3:-v4}; WORKERS=${4:-8}; shift 4 || true
SEEN7="orca-native-k10,dwa-native-k10,orca-native-k6,orca-inertia-k10,dwa-slowturn-k10,dwa-slowturn_inertia-k10,orca-slowturn_inertia-k6"
HELD7="dwa-native-k6,sfm-native-k10,dwa-inertia-k10,orca-slowturn-k10,orca-heavy_brake-k10,dwa-slowturn_heavy-k10,sfm-inertia-k10"
CK=$(printf "output/runs/%s/checkpoints/policy_v%05d.pt" "$RUN" "$VER")
TAG=$(echo "$@" | tr -c 'a-zA-Z0-9.' '_' | sed 's/_*$//')
OUT=output/eval/${RUN}_v${VER}${TAG:+_$TAG}
SANDBOX='{}'
[ "$LAYOUT" = "v4" ] && SANDBOX='{"route": true}'
export L3_LAYOUT=$LAYOUT
unset QG_HUMAN_360
exec $PY scripts/l3_run.py eventwm_graph.l3.evaluate run --suite dev --configs "$SEEN7,$HELD7" --policy "$CK" \
  --out "$OUT" --workers "$WORKERS" --servers 2 --server-devices 0,1 --nice 5 --tag "m${RUN}v${VER}" --scan \
  --compiler '{"brake_tail": true}' --sandbox "$SANDBOX" "$@"
