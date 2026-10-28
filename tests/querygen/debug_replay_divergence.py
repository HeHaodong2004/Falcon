"""Re-execute a result's decisions in the branch simulator and in the authoritative replay;
report the first step where actions differ and the executor's stop-test inputs there."""
import json, sys, math
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime, replay
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import FAMILIES
r = json.load(open(sys.argv[1]))
spec = EpisodeSpec(**r['spec'])
dyn = FAMILIES[r['dynamics']]
rt = EpisodeRuntime(spec, r['planner'], dyn, strict=True)
st = rt.root
decs = []
for d in r['decisions']:
    q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'],
              speed=d['speed'], label=d['label'], meta=d.get('meta', {}))
    st = rt.run(st, None if d['cont'] else q, horizon=d['horizon'], cont=d['cont'])
    decs.append(dict(query=q, cont=d['cont'], horizon=d['horizon']))
    if st.terminal:
        break
branch = ''.join(a[0] if isinstance(a, str) else 'v' for a in st.actions)
print('branch re-execution steps', len(branch), 'outcome', st.outcome)
rep = replay(spec, r['planner'], dyn, decs)
ra = ''.join(a[0] if isinstance(a, str) else 'v' for a in rep['actions'])
print('replay steps', len(ra), rep['metrics'].get('success'))
k = next((i for i, (a, b) in enumerate(zip(branch, ra)) if a != b), min(len(branch), len(ra)))
print('first divergence at step', k, 'branch', branch[max(0, k-5):k+3], 'replay', ra[max(0, k-5):k+3])
print('cont flags', [d['cont'] for d in r['decisions']])
rt.close()
