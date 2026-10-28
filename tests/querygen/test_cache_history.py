"""Branch-isolation check for the legal sensor's private support cache: re-execute a result's
decision sequence (a) in a fresh runtime and (b) after other branches have populated the shared
private cache; compare actions and base heights seen."""
import json, sys
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.generator import QueryGenerator
from eventwm_graph.querygen.dynamics import FAMILIES
r = json.load(open(sys.argv[1]))
spec = EpisodeSpec(**r['spec']); dyn = FAMILIES[r['dynamics']]

def execute(rt):
    st = rt.root
    for d in r['decisions']:
        q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'],
                  speed=d['speed'], label=d['label'], meta=d.get('meta', {}))
        st = rt.run(st, None if d['cont'] else q, horizon=d['horizon'], cont=d['cont'])
        if st.terminal:
            break
    return ''.join(a[0] if isinstance(a, str) else 'v' for a in st.actions), st

rt = EpisodeRuntime(spec, r['planner'], dyn, strict=True)
a_fresh, st_fresh = execute(rt)
ys = sorted({round(float(p[1]), 4) for p in st_fresh.legal.poses})
print('fresh:', len(a_fresh), st_fresh.outcome, 'distinct base heights along branch:', ys[:10], '...' if len(ys) > 10 else '')
# pollute: explore other branches from several states along the path
gen = QueryGenerator(dyn)
st = rt.root
for d in r['decisions'][:6]:
    q = Query(d['kind'], via=tuple(tuple(v) for v in d['via']), heading=d['heading'], duration=d['duration'],
              speed=d['speed'], label=d['label'], meta=d.get('meta', {}))
    for alt in gen.propose(st.legal, st.execution)[0]:
        try:
            rt.run(st, alt, horizon=16)
        except Exception:
            pass
    st = rt.run(st, None if d['cont'] else q, horizon=d['horizon'], cont=d['cont'])
    if st.terminal:
        break
a_poll, st_poll = execute(rt)
k = next((i for i, (x, y) in enumerate(zip(a_fresh, a_poll)) if x != y), min(len(a_fresh), len(a_poll)))
print('after other branches:', len(a_poll), st_poll.outcome, 'identical:', a_fresh == a_poll, 'first diff at', k)
# compare legal maps at the end
m1, m2 = st_fresh.legal.map, st_poll.legal.map
print('cache entries', len(rt.private_cache))
rt.close()
