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
budget_s = float(sys.argv[3]) if len(sys.argv) > 3 else 900
spec = [s for s in load('dev') if s.key == key]
spec = spec[0] if spec else EpisodeSpec(*key.split('/'), 1)
task = NativeTask(spec)
sim = BranchSimulator(task, None)
root = sim.root()
p0 = task.robot_pose()
g0 = task.geodesic_to_goal()
sim.tape = HumanTape.record(task, max_depth+2)
t0 = time.perf_counter()
from eventwm_graph.querygen.native import classify_contacts, contact_links, escape_exempt
c0 = classify_contacts(task, gate_humans=False); spawn = bool(c0['static']); C0 = contact_links(c0)
print('spawn contact', spawn)
q = deque([(root, 0, 0, p0, spawn, '', 0.)])
seen = {(round(p0[0]/.02), round(p0[2]/.02), 0, spawn)}
best_disp, best_geo, first_escape, n = 0., g0, None, 0
while q and time.perf_counter()-t0 < budget_s and first_escape is None:
    st, depth, turn_idx, pose, esc, seq, trav = q.popleft()
    if depth >= max_depth:
        continue
    for a in ('forward', 'left', 'right'):
        sim.load(st, depth)
        r = sim.step(a, compute_geodesic=True)
        n += 1
        if r.human_collision:
            continue
        trav2 = trav+math.hypot(r.pose[0]-pose[0], r.pose[2]-pose[2])
        if r.static_contact and not escape_exempt(r.static_links, C0, depth+1, trav2):
            if depth < 3: print('prune', seq+a[0], sorted(r.static_links), round(trav2, 3))
            continue
        esc2 = esc and bool(r.static_contact)
        ti = turn_idx+(1 if a == 'left' else -1 if a == 'right' else 0)
        key2 = (round(r.pose[0]/.02), round(r.pose[2]/.02), ti % 75, esc2)
        if key2 in seen:

            continue
        seen.add(key2)
        disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2])
        seq2 = seq+a[0]
        if esc2:
            q.append((sim.state, depth+1, ti, r.pose, esc2, seq2, trav2))
            continue
        best_disp = max(best_disp, disp)
        best_geo = min(best_geo, r.geodesic)
        if disp >= 1.0 and first_escape is None:
            first_escape = depth+1
            print('first 1 m escape sequence', seq2, 'geo', round(r.geodesic, 2), flush=True)
        q.append((sim.state, depth+1, ti, r.pose, esc2, seq2, trav2))
print(f'{spec.key} start_geo {g0:.2f} states {len(seen)} sims {n} best_disp {best_disp:.2f} '
      f'first_1m_escape_depth {first_escape} best_geo {best_geo:.2f} secs {time.perf_counter()-t0:.0f}')
task.close()
