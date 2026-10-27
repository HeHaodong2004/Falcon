"""Space-time heuristic sanity check on AWUFxHEyV3T/0 (corridor blocked by humans, steps ~48-144)."""
import time
import numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.oracle import SpaceTimeHeuristic
spec = [s for s in load('dev') if s.key == 'hm3d/val/AWUFxHEyV3T/0'][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE)
t = time.perf_counter(); st = SpaceTimeHeuristic(rt); dt = time.perf_counter()-t
print('build s %.2f' % dt, 'grid', st.nav.shape, 'table MB %.1f' % (st.table.nbytes/1e6))
p0 = rt.root.legal.pose
print('start geo', round(rt.root.geodesic, 2), 'steps @0 from start', st.value(p0[0], p0[2], 0))
for t in (60, 88, 100, 120, 140, 160):
    print(f'corridor entrance (6.4, 3.0) at t={t}:', st.value(6.4, 3.0, t), ' pocket (7.5, 2.9):', st.value(7.5, 2.9, t))
rt.close()
