"""Re-run the frozen Oracle on one episode (deterministic edge budget), then re-execute the found
decision sequence (a) inside the same runtime (search-populated private sensor cache) and (b) in a
fresh runtime, and compare with the Oracle branch actions."""
import json, sys
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator
from eventwm_graph.querygen.oracle import Oracle, OracleConfig
from eventwm_graph.querygen.dynamics import FAMILIES
r = json.load(open(sys.argv[1]))
spec = EpisodeSpec(**r['spec']); dyn = FAMILIES[r['dynamics']]
rt = EpisodeRuntime(spec, r['planner'], dyn, strict=True)
res = Oracle(rt, QueryGenerator(dyn), OracleConfig()).search()
st = res['state']
oracle_actions = ''.join(a[0] if isinstance(a, str) else 'v' for a in st.actions)
print('oracle', res['success'], len(oracle_actions))

def execute(rt_):
    s = rt_.root
    for d in st.decisions:
        s = rt_.run(s, None if d['cont'] else d['query'], horizon=d['horizon'], cont=d['cont'])
        if s.terminal:
            break
    return ''.join(a[0] if isinstance(a, str) else 'v' for a in s.actions)
same = execute(rt)
rt.close()
rt2 = EpisodeRuntime(spec, r['planner'], dyn, strict=True)
fresh = execute(rt2)
rt2.close()
print('same-process re-execution', len(same), same == oracle_actions)
print('fresh-process re-execution', len(fresh), fresh == oracle_actions)
