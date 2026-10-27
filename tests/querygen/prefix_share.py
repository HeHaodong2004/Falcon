"""How much simulation could sibling queries share? (diagnostic)

From a few decision states, run every generator candidate and build the trie of executed command
prefixes: shared = 1 - unique prefix steps / total steps.
"""
import sys, json
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
from eventwm_graph.querygen.oracle import OracleConfig, horizon_for
from eventwm_graph.querygen.dynamics import FAMILIES
from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
key = sys.argv[1]
gen_kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
spec = [s for s in load('dev') if s.key == key][0]
dyn = FAMILIES['native']
rt = EpisodeRuntime(spec, 'orca', dyn)
gen = QueryGenerator(dyn, GeneratorConfig(**gen_kw))
oc = OracleConfig()
frontier = [rt.root]
tot = uniq = 0
for depth in range(3):
    nxt = []
    for node in frontier[:3]:
        qs, _ = gen.propose(node.legal, node.execution)
        trie = set(); steps = 0
        kids = []
        for q in qs:
            ch = rt.run(node, q, horizon=min(horizon_for(q, oc), MAX_EPISODE_STEPS-node.index))
            a = ch.actions[node.index:]
            a = tuple(x if isinstance(x, str) else tuple(x) for x in a)
            steps += len(a)
            for k in range(1, len(a)+1):
                trie.add(a[:k])
            if ch.outcome is None:
                kids.append(ch)
        tot += steps; uniq += len(trie)
        print(f'depth {depth} t={node.index} K={len(qs)} steps {steps} unique {len(trie)} shared {1-len(trie)/max(1,steps):.2f}', flush=True)
        nxt += kids
    frontier = nxt[::max(1, len(nxt)//3)]
print(f'TOTAL steps {tot} unique {uniq} shared {1-uniq/max(1,tot):.2f}')
rt.close()
