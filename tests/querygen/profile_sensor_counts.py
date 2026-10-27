"""Ray counts per observe() call during Oracle-like branch execution."""
import sys, time
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.executor import Query
key = sys.argv[1]
spec = [s for s in load('dev') if s.key == key][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=False)
gen = QueryGenerator(NATIVE)
st = rt.root
for _ in range(5):
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else Query('goal', label='goal'), horizon=16, cont=cont)
obs = rt.sensor.observer
p0 = dict(obs.performance)
queries, _ = gen.propose(st.legal, st.execution)
t = time.perf_counter(); n = 0
for q in queries:
    ch = rt.run(st, q, horizon=16); n += ch.index-st.index
dt = time.perf_counter()-t
p1 = obs.performance
d = {k: p1[k]-p0.get(k, 0) for k in p1 if isinstance(p1[k], (int, float))}
print('steps', n, 'ms/step', round(1000*dt/n, 1))
for k in sorted(d):
    print(f'  {k:28s} {d[k]:12.1f}  per step {d[k]/n:9.2f}')
rt.close()
