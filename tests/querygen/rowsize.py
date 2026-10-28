import sys, pickle, numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
spec = [s for s in load('dev') if s.key == sys.argv[1]][0]
rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'])
sizes = []; nups = []; sstate = []
orig = rt.sensor.observe
def obs(t):
    row = orig(t)
    sizes.append(len(pickle.dumps(row))); nups.append(len(row['static_updates']))
    st = rt.sensor.state(); sstate.append((len(pickle.dumps(st[0])), len(st[1])))
    return row
rt.sensor.observe = obs
st = rt.root
while not st.terminal and st.index < 80:
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else Query('goal', label='goal'), horizon=16, cont=cont)
print('row bytes median %d max %d; updates median %d max %d; tracker bytes median %d; seen size last %d' % (
    np.median(sizes), max(sizes), np.median(nups), max(nups), np.median([a for a, b in sstate]), sstate[-1][1]))
rt.close()
