import math
import sys
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.control import to_plane

key = sys.argv[1] if len(sys.argv) > 1 else 'hm3d/val/3t8DB4Uzvkt/2'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ds, sp, sc, ep = key.split('/')
rt = EpisodeRuntime(EpisodeSpec(ds, sp, sc, ep, 102), 'orca', NATIVE, strict=False, trace=True)
st = rt.root
print('start pose', np.round(st.legal.pose, 3), 'goal', np.round(rt.task.goal, 3), 'geo', round(rt.start_geodesic, 3))
q = Query('goal', label='goal')
ex = rt.new_execution(q)
st = st.copy(); st.execution = ex
for k in range(n):
    rt._load(st) if k == 0 else None
    st = rt.run(st, None if k else q, horizon=1, cont=bool(k))
    t = rt.trace[-1]
    pl = ex = st.execution
    orca = getattr(ex.planner, 'last', None)
    r = ex.route
    seg = None if r is None else np.round(r[:3], 2).tolist()
    vd = orca['v'] if orca else None
    vdir = None if vd is None else round(math.degrees(math.atan2(vd[1], vd[0])), 1)
    from eventwm_graph.querygen.control import body_cost, body_cells_state, heading_vec
    Pp = to_plane([st.legal.pose[0], st.legal.pose[2]]); ps = st.legal.pose[3]; g = st.legal.map
    bc = (body_cost(g, Pp, ps), body_cost(g, Pp, ps+1/12), body_cost(g, Pp, ps-1/12), body_cost(g, Pp+0.2083*heading_vec(ps), ps))
    cells = ''.join('.FB'[v] for v in body_cells_state(g, Pp, ps))
    hp = rt.tape.positions[st.index]
    dd = np.hypot(hp[:, 0]-st.legal.pose[0], hp[:, 2]-st.legal.pose[2]) if len(hp) else []
    tr = [(t_['track_id'][-2:], round(float(np.hypot(t_['position'][0]-st.legal.pose[0], t_['position'][2]-st.legal.pose[2])), 2), t_['visible_now'], t_['velocity_valid']) for t_ in st.legal.tracks]
    if k < int(sys.argv[3]) if len(sys.argv) > 3 else False:
        continue
    print(k, 'true d', np.round(np.sort(dd)[:2], 2).tolist(), 'tracks', tr, end=' | ')
    print(k, t['prim'], t['mode'], 'psi', round(math.degrees(st.legal.pose[3]), 1), 'v*', None if vd is None else np.round(vd, 2).tolist(),
          'v*dir', vdir, 'lines', orca['n_lines'] if orca else None, 'obst', orca['n_obst'] if orca else None, 'route', seg, 'rlen', None if ex.route_len is None else round(ex.route_len, 2))
rt.close()
