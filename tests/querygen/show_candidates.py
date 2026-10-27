"""Print the generator's candidate labels (quota vs compact K=6) along a few decisions of a result."""
import json, sys
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
r = json.load(open(sys.argv[1]))
spec = EpisodeSpec(**r['spec']); dyn = FAMILIES[r['dynamics']]
rt = EpisodeRuntime(spec, r['planner'], dyn)
gq = QueryGenerator(dyn); gc = QueryGenerator(dyn, GeneratorConfig(budget=6, select='compact'))
st = rt.root
for d in r['decisions'][:8]:
    a, _ = gq.propose(st.legal, st.execution); b, diag = gc.propose(st.legal, st.execution)
    print(st.index, 'quota  ', [q.label for q in a]); print(st.index, 'compact', [q.label for q in b], 'stuck', diag['stuck'], 'conf', diag['conflicts'])
    q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'], speed=d['speed'], label=d['label'], meta=d.get('meta', {}))
    st = rt.run(st, None if d['cont'] else q, horizon=d['horizon'], cont=d['cont'])
    if st.terminal: break
rt.close()
