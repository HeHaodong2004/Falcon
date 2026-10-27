import math, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.gridplan import PlanningView, INF
from eventwm_graph.querygen.control import to_plane, body_cost
from eventwm_graph.querygen import lattice
spec = [s for s in load('dev') if s.key == sys.argv[1]][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=True)
legal = rt.root.legal; pose = legal.pose
goal_xz = (float(legal.goal[0]), float(legal.goal[2]))
view = PlanningView(legal.map, (float(pose[0]), float(pose[2])), goal_xz)
gfield = view.field([goal_xz])
blocked, unknown, cost, fld, kx0, kz0 = lattice._crop(view, gfield, float(pose[0]), float(pose[2]), 48)
print('crop shape', blocked.shape, 'blocked', blocked.sum(), 'unreachable field cells', (fld > 1e17).sum(), 'field min/max finite', fld[fld < 1e17].min(), fld[fld < 1e17].max())
i0 = int(math.floor(pose[0]/view.res))-kx0; j0 = int(math.floor(pose[2]/view.res))-kz0
print('robot local', i0, j0, 'field at robot', fld[i0, j0], 'blocked at robot', blocked[i0, j0])
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
F = np.where(fld > 1e17, np.nan, fld)
ax[0].imshow(F.T, origin='lower'); ax[0].plot(i0, j0, 'r*'); ax[0].set_title('field (crop)')
ax[1].imshow(blocked.T, origin='lower'); ax[1].plot(i0, j0, 'r*'); ax[1].set_title('blocked (crop)')
fig.savefig('output/runs/debug_stuck.png', dpi=80)
rt.close()
