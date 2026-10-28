"""Sensor cost breakdown along a goal-query execution (diagnostic)."""
import sys, time, json
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
key = sys.argv[1]; n = int(sys.argv[2])
spec = [s for s in load('dev') if s.key == key][0]
rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'])
obs = rt.sensor.observer
for k in list(obs.performance):
    if isinstance(obs.performance[k], (int, float)):
        obs.performance[k] = 0
st = rt.root
t0 = time.perf_counter()
while not st.terminal and st.index < n:
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else Query('goal', label='goal'), horizon=16, cont=cont)
dt = time.perf_counter()-t0
perf = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in obs.performance.items() if isinstance(v, (int, float))}
print('steps', st.index, 'wall s %.2f' % dt, 'runtime perf', {k: round(v, 2) for k, v in rt.perf.items()})
print('observer', json.dumps(perf))
rt.close()
