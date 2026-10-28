"""Scan: at each episode's reset position, for t = 0..240 step 10, does the privileged space-time model declare
the goal unreachable (st_fallback applies)?  Compares human radii.

  python scripts/querygen_run.py tests.querygen.st_unreachable_scan --radii 0.6 0.5 [--only-feasible]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radii', nargs='+', type=float, default=[0.6, 0.5])
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import SpaceTimeHeuristic
    dev_inf = {'yr17PDCnDDW', '7Ukhou1GxYi', 'Dd4bFSTQ8gi', '5cdEh9F2hJL'}
    un_inf = {'sT4fr6TAbpF', 'D8bT1ambLFc', 'FgswoxWb3uN', 's19Uyn7AWwv', 'JeFG25nYj2p', 'VLzqgDo317F', 'gDDiZeyaVc2'}
    specs = [s for s in load('dev') if s.scene not in dev_inf]+[s for s in load('unseen') if s.scene not in un_inf]
    for spec in specs:
        rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'], strict=True)
        p = rt.root.legal.pose
        row = dict(key=spec.key)
        for r in a.radii:
            st = SpaceTimeHeuristic(rt, radius=r)
            vals = [st.value(p[0], p[2], t) for t in range(0, 250, 10)]
            row[str(r)] = ''.join('x' if v is None else '.' for v in vals)
        rt.close()
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
