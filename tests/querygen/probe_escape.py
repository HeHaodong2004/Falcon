"""Privileged diagnostic: is there ANY static-contact-free primitive escape?

Breadth-first search over native primitives (forward/left/right) from the
reset pose using the exact branch simulator and real contact detection
(ignoring humans, which are far at the start).  States are de-duplicated on
(2 cm position, exact turn index).  Reports the farthest contact-free
displacement reached, the depth needed to get >= 1 m from the start, and the
best geodesic distance-to-goal reached.  Diagnostic only; never used online.
"""
import math
import sys
import time
from collections import deque
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, BranchSimulator
from eventwm_graph.querygen.suites import load

key = sys.argv[1]
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 60
spec = [s for s in load('dev') if s.key == key]
spec = spec[0] if spec else EpisodeSpec(*key.split('/'), 1)
task = NativeTask(spec)
sim = BranchSimulator(task, None)
root = sim.root()
p0 = task.robot_pose()
g0 = task.geodesic_to_goal()
sim.tape = HumanTape.record(task, max_depth+2)
t0 = time.perf_counter()
q = deque([(root, 0, 0, p0)])
seen = {(round(p0[0]/.02), round(p0[2]/.02), 0)}
best_disp, best_geo, first_escape, n = 0., g0, None, 0
while q and time.perf_counter()-t0 < 900:
    st, depth, turn_idx, pose = q.popleft()
    if depth >= max_depth:
        continue
    for a in ('forward', 'left', 'right'):
        sim.load(st, depth)
        r = sim.step(a, compute_geodesic=True)
        n += 1
        if r.static_contact or r.human_collision:
            continue
        ti = turn_idx+(1 if a == 'left' else -1 if a == 'right' else 0)
        key2 = (round(r.pose[0]/.02), round(r.pose[2]/.02), ti % 75)
        if key2 in seen:
            continue
        seen.add(key2)
        disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2])
        best_disp = max(best_disp, disp)
        best_geo = min(best_geo, r.geodesic)
        if disp >= 1.0 and first_escape is None:
            first_escape = depth+1
        q.append((sim.state, depth+1, ti, r.pose))
print(f'{spec.key} start_geo {g0:.2f} states {len(seen)} sims {n} best_disp {best_disp:.2f} '
      f'first_1m_escape_depth {first_escape} best_geo {best_geo:.2f} secs {time.perf_counter()-t0:.0f}')
task.close()
