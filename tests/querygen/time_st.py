import time, numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES, STEP_SECONDS
from eventwm_graph.querygen.gridplan import PlanningView
from eventwm_graph.querygen.control import human_estimates, to_plane, body_cost
from eventwm_graph.querygen import stplan
spec = [s for s in load('dev') if s.key == 'hm3d/val/Qpor2mEya8F/22'][0]
rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'])
st = rt.run(rt.root, Query('goal', label='goal'), horizon=16)
st = rt.run(st, Query('goal', label='goal'), horizon=14)
for k in range(3):
    legal = st.legal; pose = legal.pose
    goal = (float(legal.goal[0]), float(legal.goal[2]))
    t0 = time.perf_counter()
    view = PlanningView(legal.map, (pose[0], pose[2]), goal); field = view.field([goal])
    t1 = time.perf_counter()
    hum = human_estimates(legal.tracks, legal.time)
    r = stplan.plan(view, field, pose, goal, 0.20833, 0.08333, hum, STEP_SECONDS,
                    start_budget=body_cost(legal.map, to_plane([pose[0], pose[2]]), pose[3]))
    t2 = time.perf_counter()
    print(f'view+field {1000*(t1-t0):.1f} ms, st plan {1000*(t2-t1):.1f} ms, expanded {r["expanded"]}, reached {r["reached"]}, horizon {r["horizon_hit"]}, prims {"".join(p[0] for p in r["prims"])}, humans {len(hum)}')
rt.close()
