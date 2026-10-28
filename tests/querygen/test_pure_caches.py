"""Branch-pure legal sensor caches (legal.LegalSensor(pure_caches=True)).

(1) flat floor: rows identical to the accepted observer along a random trajectory;
(2) sloped floor: a branch's rows do not depend on which other branches populated the shared
    private cache first (accepted observer vs pure observer, same probe)."""
import sys, json
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, BranchSimulator
from eventwm_graph.querygen.legal import LegalSensor
from eventwm_graph.querygen.suites import load
from tests.querygen.native_equivalence import random_actions


def rows_along(task, sim, sensor, root, s0, actions, start=0):
    sim.load(root, 0); sensor.restore(s0)
    out = []
    for a in actions:
        r = sim.step(a)
        row = sensor.observe(r.time)
        out.append(json.dumps(dict(u=row['static_updates'], t=row['tracks']), sort_keys=True, default=float))
    return out


def check(key, suite, pure, n=100, seed=3):
    spec = [s for s in load(suite) if s.key == key][0]
    task = NativeTask(spec)
    sensor = LegalSensor(task, pure_caches=pure)
    sim = BranchSimulator(task, None); root = sim.root(); s0 = sensor.state()
    sim.tape = HumanTape.record(task, n+5)
    rng = np.random.default_rng(seed)
    A = random_actions(rng, n); B = random_actions(np.random.default_rng(seed+1), n)
    b_fresh = rows_along(task, sim, sensor, root, s0, B)
    task.close()
    task = NativeTask(spec)
    sensor = LegalSensor(task, pure_caches=pure)
    sim = BranchSimulator(task, None); root = sim.root(); s0 = sensor.state()
    sim.tape = HumanTape.record(task, n+5)
    rows_along(task, sim, sensor, root, s0, A)          # another branch fills the private cache first
    b_after = rows_along(task, sim, sensor, root, s0, B)
    ys = sorted({round(float(task.robot.base_pos[1]), 3)})
    task.close()
    k = next((i for i, (x, y) in enumerate(zip(b_fresh, b_after)) if x != y), None)
    return b_fresh, k


flat = 'hm3d/val/HMkoS756sz6/10'
acc, _ = check(flat, 'dev', False)
pure, _ = check(flat, 'dev', True)
print('flat floor: pure == accepted rows:', acc == pure)
for pure_flag in (False, True):
    _, k = check(sys.argv[1] if len(sys.argv) > 1 else 'hm3d/val/X7gTkoDHViv/2', 'holdout', pure_flag)
    print(f'sloped floor, pure_caches={pure_flag}: branch rows history-independent: {k is None}' + ('' if k is None else f' (first difference at step {k})'))
