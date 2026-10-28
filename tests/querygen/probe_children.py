"""Diagnostic: execute a fixed decision prefix, then every generator candidate from that state.

  python scripts/querygen_run.py tests.querygen.probe_children KEY --prefix 'goal:16,goal:16' [--gen JSON]
     [--depth2]   also expand every surviving child once more (all its candidates)

Prints each candidate's via-points, outcome, end index/pose, geodesic and min rectangle distance.
"""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--gen', default='{}')
    ap.add_argument('--horizon', type=int, default=16)
    ap.add_argument('--depth2', action='store_true')
    ap.add_argument('--trace', default='', help='comma-separated candidate labels: print their per-step trace')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--rules', default='official', choices=['official', 'declared'])
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.oracle import OracleConfig, horizon_for
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    dyn = FAMILIES[a.dynamics]
    rt = EpisodeRuntime(spec, a.planner, dyn, trace=bool(a.trace), strict=(a.rules == 'declared'))
    want = set(x for x in a.trace.split(',') if x)
    gen = QueryGenerator(dyn, GeneratorConfig(**json.loads(a.gen)))
    ocfg = OracleConfig()
    st = rt.root
    for item in [x for x in a.prefix.split(',') if x]:
        lab, n = item.split(':')
        qs, _ = gen.propose(st.legal, st.execution)
        q = next((x for x in qs if x.label == lab), None)
        if q is None:
            print(f'prefix label {lab} not offered at t={st.index}: {[x.label for x in qs]}')
            return
        st = rt.run(st, q, horizon=int(n))
        p = st.legal.pose
        print(f'prefix {lab}:{n} -> t={st.index} outcome={st.outcome} pose=({p[0]:.2f},{p[2]:.2f},{math.degrees(p[3]):.0f}) '
              f'geo={st.geodesic:.2f}', flush=True)
        if st.outcome:
            return

    def expand(node, indent=''):
        qs, diag = gen.propose(node.legal, node.execution)
        print(f'{indent}t={node.index} stuck={diag.get("stuck")} conflicts={diag.get("conflicts")} raw={diag.get("raw")}')
        kids = []
        for q in qs:
            if rt.trace is not None:
                rt.trace.clear()
            ch = rt.run(node, q, horizon=min(horizon_for(q, ocfg), MAX_EPISODE_STEPS-node.index))
            if q.label in want:
                print('     trace', q.label, ' '.join(f"{t['index']}:{t['prim'][0] if isinstance(t['prim'], str) else 'v'}/{t['mode']}"
                                                     f"({t['pose'][0]:.2f},{t['pose'][2]:.2f},{math.degrees(t['pose'][3]):.0f})r{t['rect']:.2f}"
                                                     for t in rt.trace))
            p = ch.legal.pose
            via = [(round(x, 2), round(z, 2)) for x, z in q.via]
            print(f'{indent}  {q.label:16s} via={via} -> {ch.outcome or "ok":15s} t={ch.index:3d} '
                  f'pose=({p[0]:.2f},{p[2]:.2f},{math.degrees(p[3]):4.0f}) geo={ch.geodesic:.2f} '
                  f'minrect={ch.min_rect_dist:.2f}', flush=True)
            if ch.outcome is None:
                kids.append((q.label, ch))
        return kids
    kids = expand(st)
    if a.depth2:
        for lab, ch in kids:
            print(f'--- after {lab}')
            expand(ch, '    ')
    rt.close()


if __name__ == '__main__':
    main()
