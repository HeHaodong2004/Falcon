import sys, math, json
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
from eventwm_graph.querygen.suites import load
key = sys.argv[1]; n = int(sys.argv[2]); planner = sys.argv[3] if len(sys.argv) > 3 else 'orca'
meta = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
spec = [s for s in load('dev') if s.key == key][0]
rt = EpisodeRuntime(spec, planner, FAMILIES['native'], strict=False, trace=True)
st = rt.root; q = Query('goal', label='goal', meta=meta)
while not st.terminal and st.index < n:
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else q, horizon=1, cont=cont)
    ex = st.execution; t = rt.trace[-1]
    plan = ex.plan['prims'][ex.plan_i:ex.plan_i+12] if ex.plan else None
    print(st.index, t['prim'], t['mode'], 'pose', [round(v, 2) for v in t['pose'][:1]+t['pose'][2:3]], 'psi', round(math.degrees(t['pose'][3])),
          'plan', ''.join(p[0] for p in plan) if plan else plan, 'reached', ex.plan.get('reached') if ex.plan else None, 'rlen', None if ex.route_len is None else round(ex.route_len, 2))
rt.close()
