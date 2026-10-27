import cProfile, pstats
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, BranchSimulator
from eventwm_graph.querygen.legal import LegalSensor
from tests.querygen.native_equivalence import random_actions

task = NativeTask(EpisodeSpec('hm3d', 'val', '5cdEh9F2hJL', '3', 7))
sens = LegalSensor(task)
sim = BranchSimulator(task, None); root = sim.root(); s0 = sens.state()
sim.tape = HumanTape.record(task, 130)
rng = np.random.default_rng(1)
# warm the private support cache with a few branches, then profile more branches
def branches(n):
    for b in range(n):
        sim.load(root, 0); sens.restore(s0)
        for a in random_actions(rng, 120):
            r = sim.step(a)
            sens.observe(r.time)
branches(3)
pr = cProfile.Profile(); pr.enable()
import time; t = time.perf_counter(); branches(6); dt = time.perf_counter()-t
pr.disable()
print('per step ms', round(1000*dt/(6*120), 2))
pstats.Stats(pr).sort_stats('tottime').print_stats(14)
task.close()
