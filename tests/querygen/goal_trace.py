"""Trace the default (goal-query) policy for N steps: per-step primitive, pose, geodesic and executor reason.

  python scripts/querygen_run.py tests.querygen.goal_trace KEY [--suite dev] [--planner dwa] [--steps 120]
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--unstick', type=int, default=0)
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rt = EpisodeRuntime(spec, a.planner, FAMILIES[a.dynamics], strict=False, trace=True)
    st = rt.root
    q = Query('goal', label='goal', meta=dict(unstick=a.unstick) if a.unstick else {})
    while not st.terminal and st.index < a.steps:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else q, horizon=16, cont=cont)
        ex = st.execution
        print(f't={st.index:3d} geo={st.geodesic:5.2f} pose=({st.legal.pose[0]:.2f},{st.legal.pose[2]:.2f},'
              f'{st.legal.pose[3]*57.3:4.0f}) status={ex.status if ex else None} reason={ex.reason if ex else None} '
              f'outcome={st.outcome}', flush=True)
    acts = ''.join(x if isinstance(x, str) and len(x) == 1 else (x[0] if isinstance(x, str) else 'v') for x in
                   [s if isinstance(s, str) else 'v' for s in st.actions])
    print('actions', ''.join(c[0] for c in acts))
    tr = rt.trace or []
    print('last trace rows:', [(r['index'], r['prim']) for r in tr[-20:]])
    rt.close()


if __name__ == '__main__':
    main()
