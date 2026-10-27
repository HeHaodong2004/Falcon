"""Continuous base-velocity action: native parity and branch/real equivalence.

1. With the velocity action installed, native discrete actions are unchanged
   and the velocity commands (25,0)/(0,+-10) reproduce forward/left/right
   bitwise (poses and joints).
2. Random unicycle commands (arcs included) give bitwise-equal poses, joints
   and contacts in BranchSimulator vs Env.step.
"""
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, BranchSimulator, classify_contacts

spec = EpisodeSpec('hm3d', 'val', '5cdEh9F2hJL', '3', 7)
rng = np.random.default_rng(0)
native_seq = list(rng.choice(['forward', 'left', 'right'], size=120, p=[.6, .2, .2]))
vel = {'forward': (25., 0.), 'left': (0., 10.), 'right': (0., -10.)}


def run(task, seq):
    out = []
    for a in seq:
        if task.episode_over:
            break
        task.step(a)
        out.append((task.robot_pose(), np.asarray(task.robot.sim_obj.joint_positions)))
    return out


with NativeTask(spec) as t0:
    ref = run(t0, native_seq)
with NativeTask(spec, velocity_action=True) as t1:
    nat = run(t1, native_seq)
with NativeTask(spec, velocity_action=True) as t2:
    vv = run(t2, [np.asarray(vel[a]) for a in native_seq])
n = min(len(ref), len(nat), len(vv))
print('native unchanged with velocity action installed:',
      all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(ref[:n], nat[:n])))
print('velocity commands reproduce native actions:',
      all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(ref[:n], vv[:n])))

cmds = [np.asarray((float(rng.uniform(0, 25)), float(rng.uniform(-10, 10)))) for _ in range(150)]
real = []
with NativeTask(spec, velocity_action=True) as t3:
    for c in cmds:
        if t3.episode_over:
            break
        t3.step(c)
        cc = classify_contacts(t3, gate_humans=False)
        real.append((t3.robot_pose(), np.asarray(t3.robot.sim_obj.joint_positions), cc['static'], bool(t3.metrics()['human_collision'])))
task = NativeTask(spec, velocity_action=True)
sim = BranchSimulator(task, None)
root = sim.root()
sim.tape = HumanTape.record(task, 200)
sim.load(root, 0)
br = []
for c in cmds[:len(real)]:
    r = sim.step(c)
    br.append((r.pose, np.asarray(task.robot.sim_obj.joint_positions), r.static_contact, r.human_collision))
eq = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2] and a[3] == b[3] for a, b in zip(real, br))
print('branch == real for random unicycle commands:', eq, 'steps', len(real))
d = np.linalg.norm(np.diff(np.asarray([p[0] for p in real])[:, [0, 2]], axis=0), axis=1)
print('per-step displacement for random commands: mean %.4f max %.4f' % (d.mean(), d.max()))
task.close()
