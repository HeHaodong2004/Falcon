"""Check the privileged space-time slack (forced waiting ahead) at the reset position over time.

  python scripts/querygen_run.py tests.querygen.slack_check KEY [--suite dev]
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--radius', type=float, default=0.6)
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import SpaceTimeHeuristic
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rt = EpisodeRuntime(spec, 'orca', FAMILIES['native'], strict=True)
    st = SpaceTimeHeuristic(rt, radius=a.radius)
    p = rt.root.legal.pose
    print(a.key, 'radius', a.radius, 'free cost-to-go at reset', st.free[int((p[2]-st.z0)/st.cell), int((p[0]-st.x0)/st.cell)])
    print('  t: value / slack', [(t, st.value(p[0], p[2], t), st.slack(p[0], p[2], t)) for t in range(0, 160, 10)])
    rt.close()


if __name__ == '__main__':
    main()
