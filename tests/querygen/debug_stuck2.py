import math, sys
import numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.gridplan import PlanningView
from eventwm_graph.querygen.control import BODY_X, FRONT_MARGIN
from eventwm_graph.querygen import lattice
spec = [s for s in load('dev') if s.key == sys.argv[1]][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=True)
legal = rt.root.legal; pose = legal.pose
goal_xz = (float(legal.goal[0]), float(legal.goal[2]))
view = PlanningView(legal.map, (float(pose[0]), float(pose[2])), goal_xz)
gfield = view.field([goal_xz])
blocked, unknown, cost, fld, kx0, kz0 = lattice._crop(view, gfield, float(pose[0]), float(pose[2]), 48)
i, j = view.to_local(float(pose[0]), float(pose[2]))
here = gfield[i, j]
res = lattice._search(blocked, unknown, cost, fld, kx0, kz0, view.res, float(pose[0]), float(pose[2]), float(pose[3]),
                      goal_xz[0], goal_xz[1], 0.12, float(max(here-1.0/view.res, 0.)), NATIVE.step_m, NATIVE.turn_rad,
                      60000, 0.15, 0, view.res/NATIVE.step_m, BODY_X[0], BODY_X[1], BODY_X[1]+FRONT_MARGIN, lattice.heading_bins(NATIVE.turn_rad), 0.6)
node, nn, X, Z, PSI, PAR, PRIM, expanded, reached = res
X, Z, PSI = X[:nn], Z[:nn], PSI[:nn]
print('nn', nn, 'expanded', expanded, 'reached', reached)
print('x range', X.min()-pose[0], X.max()-pose[0], 'z range', Z.min()-pose[2], Z.max()-pose[2])
ci = np.floor(X/view.res).astype(int)-kx0; cj = np.floor(Z/view.res).astype(int)-kz0
ok = (ci >= 0) & (cj >= 0) & (ci < fld.shape[0]) & (cj < fld.shape[1])
print('min field reached', fld[ci[ok], cj[ok]].min(), 'goal_field', max(here-1.0/view.res, 0.), 'here', here)
print('unique headings', len(np.unique(np.round(PSI % (2*math.pi), 3))))
print('prims count', np.bincount(PRIM[1:nn], minlength=3))
rt.close()
