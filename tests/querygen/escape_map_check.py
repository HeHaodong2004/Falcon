"""Build the Oracle's privileged escape map for an episode and report it (diagnostic)."""
import sys
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.dynamics import FAMILIES
from eventwm_graph.querygen.oracle import EscapeMap
for key in sys.argv[1:]:
    spec = [s for s in load('dev') if s.key == key][0]
    rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'])
    em = EscapeMap(rt)
    print(key, 'blocked_early', em.blocked_early, 'active', em.active, 'first', em.first, 'states', em.states,
          'sims', em.sims, 'seconds %.1f' % em.seconds, flush=True)
    rt.close()
