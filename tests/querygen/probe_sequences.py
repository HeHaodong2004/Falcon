"""Check specific primitive sequences against the declared escape/static rule."""
import sys
from eventwm_graph.querygen.native import NativeTask, HumanTape, BranchSimulator, classify_contacts, contact_links, escape_exempt
import math
from eventwm_graph.querygen.suites import load

key = sys.argv[1]
seqs = sys.argv[2:]
spec = [s for s in load('dev') if s.key == key][0]
task = NativeTask(spec)
sim = BranchSimulator(task, None); root = sim.root()
c0 = classify_contacts(task, gate_humans=False); spawn = bool(c0['static']); C0 = contact_links(c0)
print('C0 links', sorted(C0))
sim.tape = HumanTape.record(task, 80)
m = {'f': 'forward', 'l': 'left', 'r': 'right'}
for seq in seqs:
    sim.load(root, 0); esc = spawn; marks = ''; ok = True; trav = 0.; last = task.robot_pose()
    for k, c in enumerate(seq, 1):
        r = sim.step(m[c])
        trav += math.hypot(r.pose[0]-last[0], r.pose[2]-last[2]); last = r.pose
        if r.static_contact:
            if escape_exempt(r.static_links, C0, k, trav):
                marks += 'e'
            else:
                marks += 'X'; ok = False
                if not hasattr(task, '_shown'):
                    task._shown = 0
                if task._shown < 12:
                    task._shown += 1
                    names = {lid: task.robot.sim_obj.get_link_name(lid) for lid in task.robot.sim_obj.get_link_ids()}
                    print('   ', seq, 'step', k, 'links', [names.get(l, l) for l in sorted(r.static_links)])
        else:
            esc = False; marks += '.'
    print(f'{seq:30s} {marks:30s} {"OK" if ok else "FAIL"}')
task.close()
