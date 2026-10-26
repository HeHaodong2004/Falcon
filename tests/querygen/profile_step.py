import cProfile, pstats, sys
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import NATIVE
rt = EpisodeRuntime(EpisodeSpec('hm3d', 'val', '5cdEh9F2hJL', '3', 102), 'orca', NATIVE, strict=False)
st = rt.root
pr = cProfile.Profile(); pr.enable()
q = Query('goal', label='goal')
while not st.terminal and st.index < 240:
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else q, horizon=16, cont=cont)
pr.disable()
print('steps', st.index, {k: round(v, 2) for k, v in rt.perf.items()})
pstats.Stats(pr).sort_stats('tottime').print_stats(22)
rt.close()
