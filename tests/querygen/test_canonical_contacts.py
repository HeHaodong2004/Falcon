"""Canonical contacts: order independence and agreement with Env.step history."""
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, BranchSimulator, classify_contacts
from eventwm_graph.querygen.suites import load
from tests.querygen.native_equivalence import random_actions

# 1. order independence on the measured stale-manifold case
spec = [s for s in load('dev') if s.key == 'hm3d/val/yr17PDCnDDW/5'][0]
task = NativeTask(spec)
sim = BranchSimulator(task, None); root = sim.root(); sim.tape = HumanTape.record(task, 60)
sim.load(root, 0); a = sim.step('left'); s1 = sim.state
sim.load(s1, 1); ref = sorted(sim.step('left').static_links)
sim.load(s1, 1); sim.step('forward')
sim.load(s1, 1); after = sorted(sim.step('left').static_links)
print('order independent:', ref == after, ref, after)
task.close()

# 2. agreement with the real Env.step (history-dependent) classification
tot = dis_s = dis_h = 0
for key, seed in [('hm3d/val/3t8DB4Uzvkt/1', 11), ('hm3d/val/5cdEh9F2hJL/3', 12), ('hm3d/val/yr17PDCnDDW/5', 5)]:
    ds, sp, sc, ep = key.split('/')
    sp_ = EpisodeSpec(ds, sp, sc, ep, seed)
    acts = random_actions(np.random.default_rng(seed), 200)
    real = []
    with NativeTask(sp_) as t:
        for x in acts:
            if t.episode_over:
                break
            t.step(x)
            c = classify_contacts(t, gate_humans=False)
            real.append((c['static'], bool(t.metrics()['human_collision'])))
    task = NativeTask(sp_)
    sim = BranchSimulator(task, None); root = sim.root(); sim.tape = HumanTape.record(task, 220); sim.load(root, 0)
    for k, x in enumerate(acts[:len(real)]):
        r = sim.step(x)
        tot += 1; dis_s += int(r.static_contact != real[k][0]); dis_h += int(r.human_collision != real[k][1])
    task.close()
print(f'steps {tot} static disagreements {dis_s} human disagreements {dis_h}')
