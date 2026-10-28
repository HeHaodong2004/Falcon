"""Can the legal queries do what a native witness does?  (privileged diagnostic)

Follows a native witness (exact primitives, executed through the search runtime so the legal state is the one a
query-level branch would have there) and, every ``--every`` steps, executes each candidate query of the generator
from that witness state for one or two decision intervals (continue after the first).  Prints per probe state the
witness's own geodesic progress over the same number of steps and each query's progress / outcome.  If no query keeps
pace anywhere along the witness, the query vocabulary or the executor lacks the capability (not the search).

  python scripts/querygen_run.py tests.querygen.witness_probe RESULT_JSON WITNESS_JSON [--every 10] [--intervals 2]
RESULT_JSON gives the episode, planner and generator configuration (any official run record).
"""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('witness')
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--intervals', type=int, default=2)
    ap.add_argument('--gen', default='', help='JSON overrides of the generator configuration')
    a = ap.parse_args()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig, primitive_queries
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for, OracleConfig
    r = json.load(open(a.result))
    w = json.load(open(a.witness))
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[r['dynamics']]
    gkw = dict(r['generator']); gkw.update(json.loads(a.gen) if a.gen else {})
    gkw['primitives'] = False
    gen = QueryGenerator(dyn, GeneratorConfig(**gkw))
    rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False); rt.sim.check_static = False
    ocfg = OracleConfig()
    prims = {q.meta['action'][0]: q for q in primitive_queries()}
    acts = [c for c in w['actions'] if c != 's']
    chain = [rt.root]
    for c in acts:
        chain.append(rt.run(chain[-1], prims[c], horizon=1))
        if chain[-1].terminal:
            break
    geo_w = [float(s.geodesic) for s in chain]
    print(json.dumps(dict(key=spec.key, witness_steps=len(acts), start_geodesic=round(geo_w[0], 2))))
    for k in range(0, len(chain)-1, a.every):
        st = chain[k]
        qs, diag = gen.propose(st.legal, None)
        rows = []
        for q in qs:
            if q.kind in ('stop', 'continue'):
                continue
            s = rt.run(st, q, horizon=horizon_for(q, ocfg))
            for _ in range(a.intervals-1):
                if s.terminal or s.execution is None or s.execution.status != 'running':
                    break
                s = rt.run(s, None, horizon=ocfg.horizon, cont=True)
            n = s.index-st.index
            wk = min(len(geo_w)-1, k+n)
            rows.append((q.label, n, round(st.geodesic-s.geodesic, 2) if s.outcome is None or s.outcome == 'success' else None,
                         s.outcome or '', round(geo_w[k]-geo_w[wk], 2)))
        best = max((x for x in rows if x[2] is not None), key=lambda x: x[2] - x[4], default=None)
        print(f"t={k:3d} geo {geo_w[k]:.2f}  stuck={diag.get('stuck')}  humans={diag.get('humans')}  "
              f"best: {best[0] if best else '-'} {best[2] if best else ''} (witness {best[4] if best else ''})")
        for x in rows:
            print(f"     {x[0]:16s} steps {x[1]:3d}  progress {x[2]}  {x[3]:16s} witness over same steps {x[4]}")
    rt.close()


if __name__ == '__main__':
    main()
