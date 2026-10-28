"""Resumable portfolio search: (a) one call with budget B gives the same search as before (run once with the old code
via QG_SNAPSHOT and once with the tree; compare the printed digests); (b) a search stopped at B/2 and resumed to B
equals a single call with budget B (same edges, outcomes, decisions, actions).

  python scripts/querygen_run.py tests.querygen.test_resumable_portfolio KEY SUITE BUDGET [--resume]
"""
import hashlib
import json
import sys
from dataclasses import replace


def digest(res):
    st = res['state']
    acts = ''.join(a[0] if isinstance(a, str) else 'v' for a in st.actions)
    s = res['stats']
    payload = dict(success=res['success'], children=s['children'], outcomes=s['outcomes'],
                   by_label={k: v['n'] for k, v in s['by_label'].items()}, steps=int(st.index), actions=acts,
                   decisions=[d['query'].label for d in st.decisions])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16], payload


def main():
    key, suite, budget = sys.argv[1], sys.argv[2], int(sys.argv[3])
    resume = '--resume' in sys.argv
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.oracle import Oracle, OracleConfig
    from eventwm_graph.querygen.dynamics import FAMILIES
    spec = [s for s in load(suite) if s.key == key][0]
    dyn = FAMILIES['native']
    gen_kw = dict(tc_variants=('goal', 'st_goal'), st_goal=True, maneuver_spawn_only=False)

    def oracle(b):
        rt = EpisodeRuntime(spec, 'orca', dyn, strict=False)
        rt.sim.check_static = False
        cfg = OracleConfig(heading_h=True, escape_penalty=80, prefix_cache=48, max_edges=b, max_seconds=40000,
                           portfolio=[{'weight': 3.5}, {'st_radius': 0.6, 'st_radius_opt': 0}])
        return rt, Oracle(rt, QueryGenerator(dyn, GeneratorConfig(**gen_kw)), cfg)
    rt, o = oracle(budget)
    d1, p1 = digest(o.search())
    rt.close()
    print(json.dumps(dict(single=d1, success=p1['success'], children=p1['children'])), flush=True)
    if resume:
        rt, o = oracle(budget//2)
        r_half = o.search()
        if not r_half['success']:
            o.cfg = replace(o.cfg, max_edges=budget)
            r_half = o.search()
        d2, p2 = digest(r_half)
        rt.close()
        print(json.dumps(dict(resumed=d2, success=p2['success'], children=p2['children'])), flush=True)
        print('RESUME_IDENTICAL' if d1 == d2 else 'RESUME_MISMATCH')


if __name__ == '__main__':
    main()
