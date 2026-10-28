"""Profile Oracle-like branch execution (EpisodeRuntime.run) on a DEV episode."""
import cProfile, pstats, sys, time
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
# advance with the goal query for a while (map grows), then profile edges from there
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    if st.terminal:
        break
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else Query('goal', label='goal'), horizon=16, cont=cont)
print('base state index', st.index, 'terminal', st.terminal)
queries, _ = gen.propose(st.legal, st.execution)
pr = cProfile.Profile(); pr.enable()
t = time.perf_counter(); n = 0
for rep in range(3):
    for q in queries:
        ch = rt.run(st, q, horizon=16)
        n += ch.index-st.index
dt = time.perf_counter()-t
pr.disable()
print('steps', n, 'ms/step', round(1000*dt/max(n, 1), 2), {k: round(v, 2) for k, v in rt.perf.items()})
pstats.Stats(pr).sort_stats('tottime').print_stats(22)
rt.close()
