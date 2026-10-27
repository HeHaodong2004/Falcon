from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
spec = [s for s in load('dev') if s.key == 'hm3d/val/SByzJLxpRGn/13'][0]
dyn = FAMILIES['native']
rt = EpisodeRuntime(spec, 'orca', dyn)
gen = QueryGenerator(dyn, GeneratorConfig(select='compact', budget=6, tc_variants=['st_goal'], st_goal=True, maneuver_spawn_only=False))
st = rt.root
for i in range(3):
    qs, diag = gen.propose(st.legal, st.execution)
    print('t', st.index, [q.label for q in qs], 'stuck', diag.get('stuck'))
    st = rt.run(st, qs[0] if qs[0].kind != 'stop' else qs[1], horizon=16)
gen2 = QueryGenerator(dyn, GeneratorConfig(tc_variants=['st_goal'], st_goal=True, maneuver_spawn_only=False))
qs, diag = gen2.propose(st.legal, st.execution)
print('K10 t', st.index, [q.label for q in qs])
rt.close()
