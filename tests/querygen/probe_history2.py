import sys
from eventwm_graph.querygen.native import NativeTask, HumanTape, BranchSimulator
from eventwm_graph.querygen.suites import load
spec = [s for s in load('dev') if s.key == 'hm3d/val/yr17PDCnDDW/5'][0]
task = NativeTask(spec)
sim = BranchSimulator(task, None); root = sim.root()
sim.tape = HumanTape.record(task, 60)
sim.load(root, 0); a = sim.step('left'); s1 = sim.state
b = sim.step('left'); print('consecutive l,l  :', sorted(a.static_links), sorted(b.static_links))
sim.load(s1, 1); c = sim.step('left'); print('reload s1, l     :', sorted(c.static_links))
sim.load(s1, 1); f = sim.step('forward'); print('reload s1, f     :', sorted(f.static_links))
sim.load(s1, 1); d = sim.step('left'); print('after f: s1, l   :', sorted(d.static_links))
sim.load(s1, 1); e = sim.step('left'); print('again: s1, l     :', sorted(e.static_links))
task.sim.perform_discrete_collision_detection()
sim.load(s1, 1); task.robot.sim_obj.transformation = task.robot.sim_obj.transformation
g = sim.step('left'); print('again2: s1, l    :', sorted(g.static_links))
task.close()

import magnum as mn
task2 = NativeTask(spec)
sim2 = BranchSimulator(task2, None); root2 = sim2.root()
sim2.tape = HumanTape.record(task2, 60)
sim2.load(root2, 0); sim2.step('left'); s1b = sim2.state
sim2.load(s1b, 1); sim2.step('forward')
# clear robot manifolds: move far away, detect, then reload
obj = task2.robot.sim_obj
obj.transformation = mn.Matrix4.translation(mn.Vector3(500., 500., 500.))
task2.sim.perform_discrete_collision_detection()
sim2.load(s1b, 1); h = sim2.step('left'); print('cleared, s1, l    :', sorted(h.static_links))
task2.close()
