import sys, numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.native import NativeTask, HumanTape
key = sys.argv[1]; t0, t1, step = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spec = [s for s in load('dev') if s.key == key][0]
task = NativeTask(spec)
tape = HumanTape.record(task, t1+1)
for k in range(t0, t1+1, step):
    print(k, ' '.join(f'({p[0]:5.1f},{p[2]:5.1f})' for p in tape.positions[k]))
task.close()
