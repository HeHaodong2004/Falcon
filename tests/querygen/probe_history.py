"""Does static-contact classification of a branch depend on prior branches?"""
import sys
from eventwm_graph.querygen.native import NativeTask, HumanTape, BranchSimulator
from eventwm_graph.querygen.suites import load

key = sys.argv[1]
spec = [s for s in load('dev') if s.key == key][0]
task = NativeTask(spec)
sim = BranchSimulator(task, None); root = sim.root()
sim.tape = HumanTape.record(task, 60)
m = {'f': 'forward', 'l': 'left', 'r': 'right'}


def run(seq):
    sim.load(root, 0)
    return [tuple(sorted(sim.step(m[c]).static_links)) for c in seq]


target = 'l'*12
base = run(target)
for pre in ['f', 'r', 'rrrr', 'ffff', 'lllllll', 'f'+'l'*12, '']:
    if pre:
        run(pre)          # unrelated branch evaluated first
    out = run(target)
    print(f'after {pre or "(repeat)":16s} equal={out == base}', '' if out == base else
          [(k, a, b) for k, (a, b) in enumerate(zip(base, out)) if a != b][:4])
task.close()
