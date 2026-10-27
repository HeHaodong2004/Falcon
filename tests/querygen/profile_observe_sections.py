"""Section timing of FastGeometryObserver.observe during Oracle-like branch execution
(line-level timers via sys.settrace-free wrapping of the helper calls)."""
import sys, time, collections
import numpy as np
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator
from eventwm_graph.querygen.dynamics import NATIVE
from eventwm_graph.querygen.executor import Query
from eventwm_graph.partial import fast_observer as fo
T = collections.Counter(); N = collections.Counter()
def wrap(obj, name, label):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); T[label] += time.perf_counter()-t; N[label] += 1; return r
    setattr(obj, name, g)
key = sys.argv[1]
spec = [s for s in load('dev') if s.key == key][0]
rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=False)
obs = rt.sensor.observer
wrap(fo, 'in_frustum', 'in_frustum')
wrap(obs, '_clear_many', 'clear_many(support+human)')
wrap(obs.batch, 'nearest_hits', 'nearest_hits(wall)')
wrap(obs.tracker, 'update', 'tracker.update')
idx = obs.cache  # noqa
orig_observe = obs.observe
def observe(ts=None):
    t = time.perf_counter(); r = orig_observe(ts); T['observe_total'] += time.perf_counter()-t; N['observe_total'] += 1; return r
obs.observe = observe
from eventwm_graph.querygen import legal as L
gen = QueryGenerator(NATIVE)
st = rt.root
for _ in range(5):
    cont = st.execution is not None and st.execution.status == 'running'
    st = rt.run(st, None if cont else Query('goal', label='goal'), horizon=16, cont=cont)
T.clear(); N.clear()
queries, _ = gen.propose(st.legal, st.execution)
for q in queries:
    rt.run(st, q, horizon=16)
n = N['observe_total']
for k in sorted(T, key=lambda k: -T[k]):
    print(f'{k:28s} {1000*T[k]/n:7.2f} ms/step  ({N[k]} calls)')
rt.close()
