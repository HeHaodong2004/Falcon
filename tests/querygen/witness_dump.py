import sys, json, math, numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.native import NativeTask, HumanTape, same_floor_rect_distance
from eventwm_graph.querygen.witness import load_witness_poses
key = sys.argv[1]; t0, t1, step = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
wdir = sys.argv[5] if len(sys.argv) > 5 else 'output/runs/witness_dev2'
spec = [s for s in load('dev') if s.key == key][0]
w = json.load(open(f"{wdir}/{key.replace('/', '_')}.json"))
P = load_witness_poses(spec, w['actions'])
task = NativeTask(spec)
tape = HumanTape.record(task, 500)
acts = w['actions']
for k in range(t0, min(t1, len(P)-1)+1, step):
    p = P[k]
    hp = tape.positions[k]
    rd = same_floor_rect_distance(p, hp) if len(hp) else []
    print(f"{k:3d} {acts[k] if k < len(acts) else '-'} R({p[0]:5.2f},{p[2]:5.2f},{math.degrees(p[3]):6.1f}) ",
          ' '.join(f'H{i}({q[0]:5.2f},{q[2]:5.2f}|{d:4.2f})' for i, (q, d) in enumerate(zip(hp, rd))))
task.close()
