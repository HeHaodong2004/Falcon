"""Why does the generator's legal stuck test fire at the reset state?"""
import math, sys
import numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.gridplan import PlanningView, INF
from eventwm_graph.querygen.control import to_plane, body_cost, turn_ok
from eventwm_graph.querygen import lattice
from eventwm_graph.querygen.legal import BLOCKED, UNKNOWN, FREE
spec = [s for s in load('dev') if s.key == sys.argv[1]][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=True)
legal = rt.root.legal
pose = legal.pose
goal_xz = (float(legal.goal[0]), float(legal.goal[2]))
view = PlanningView(legal.map, (float(pose[0]), float(pose[2])), goal_xz)
gfield = view.field([goal_xz])
i, j = view.to_local(float(pose[0]), float(pose[2]))
here = gfield[i, j]
P = to_plane([pose[0], pose[2]])
b = body_cost(legal.map, P, float(pose[3]))
print('pose', np.round(pose, 2), 'here field', here*view.res, 'body cost', b, 'state at robot', view.state[i, j])
for mx in (15000, 60000):
    for gf in (max(here-1.0/view.res, 0.), max(here-0.3/view.res, 0.)):
        plan = lattice.plan(view, gfield, pose, goal_xz, NATIVE.step_m, NATIVE.turn_rad, goal_tol=0.12, goal_field=gf,
                            start_budget=b, max_expand=mx)
        print(f'max_expand {mx} goal_field drop {(here-gf)*view.res:.2f} m: prims {len(plan["prims"])} reached {plan["reached"]} expanded {plan["expanded"]}')
for d in (math.pi/2, -math.pi/2, math.pi, -math.pi):
    print('turn_ok', round(d, 2), turn_ok(legal.map, P, float(pose[3]), d, True, b))
# count map states around
u, c = np.unique(view.state, return_counts=True)
print('view states', dict(zip(u.tolist(), c.tolist())))
rt.close()
