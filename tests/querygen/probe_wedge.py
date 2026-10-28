"""Diagnose a spawn wedge: per primitive from reset, which links touch static geometry,
penetration depths, and translation achieved; also free (contact-ignoring) BFS escape."""
import math, sys, json
import numpy as np
from collections import deque
from eventwm_graph.querygen.native import NativeTask, HumanTape, BranchSimulator, classify_contacts, contact_links
from eventwm_graph.querygen.suites import load
key = sys.argv[1]
spec = [s for s in load('dev') if s.key == key][0]
task = NativeTask(spec)
sim = BranchSimulator(task, None)
root = sim.root()
p0 = task.robot_pose(); g0 = task.geodesic_to_goal()
names = {lid: task.robot.sim_obj.get_link_name(lid) for lid in task.robot.sim_obj.get_link_ids()}
names[-1] = 'base'
c0 = classify_contacts(task, gate_humans=False)
print('spawn pose', np.round(p0, 3), 'geo', round(g0, 2))
print('spawn contacts', {k: (v if not isinstance(v, list) else len(v)) for k, v in c0.items()} if isinstance(c0, dict) else c0)
sim.tape = HumanTape.record(task, 80)
# depth-1 / depth-2 outcomes
for seq in ['f', 'l', 'r', 'ff', 'll', 'rr', 'lf', 'rf', 'lll', 'rrr', 'llllll', 'rrrrrr']:
    sim.load(root, 0)
    out = []
    for i, a in enumerate(seq):
        r = sim.step({'f': 'forward', 'l': 'left', 'r': 'right'}[a])
        out.append((a, sorted(names.get(l, str(l)) for l in r.static_links), round(math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2]), 3)))
    print(seq, out[-1])
# contact-ignoring BFS: can the robot get 1 m away at all (navmesh motion)?
q = deque([(root, 0, 0)]); seen = set(); best = 0; sims = 0; first = None
while q and sims < 20000:
    st, d, ti = q.popleft()
    if d >= 40: continue
    for a in ('forward', 'left', 'right'):
        sim.load(st, d); r = sim.step(a); sims += 1
        if r.human_collision: continue
        ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
        k = (round(r.pose[0]/.02), round(r.pose[2]/.02), ti2 % 75)
        if k in seen: continue
        seen.add(k)
        disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2]); best = max(best, disp)
        if disp >= 1.0 and first is None:
            first = d+1; print('contact-ignoring 1 m escape at depth', first, 'static links', sorted(names.get(l, str(l)) for l in r.static_links))
        q.append((sim.state, d+1, ti2))
    if first is not None: break
print('contact-ignoring best disp', round(best, 3), 'states', len(seen), 'sims', sims)
task.close()
