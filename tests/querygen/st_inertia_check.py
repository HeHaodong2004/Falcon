"""Check the momentum-aware st_goal (stplan.plan_inertia) under DWA x inertia: run 'st_goal' and plain 'goal'
from the reset state (continued every decision until termination or a step cap) and report outcome, steps,
dynamics violations of the executed commands and plan timing.

  python scripts/querygen_run.py tests.querygen.st_inertia_check KEY [KEY ...] [--suite dev] [--cap 300]
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--cap', type=int, default=300)
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES, dynamics_violations
    dyn = FAMILIES['inertia']
    specs = {s.key: s for s in load(a.suite)}
    for key in a.keys:
        rt = EpisodeRuntime(specs[key], 'dwa', dyn, strict=True)
        for lab, kind in (('st_goal', 'stgoal'), ('goal', 'goal')):
            st = rt.root
            t0 = time.perf_counter()
            q = Query(kind, label=lab)
            while not st.terminal and st.index < a.cap:
                cont = st.execution is not None and st.execution.status == 'running'
                st = rt.run(st, None if cont else q, horizon=16, cont=cont)
            dt = time.perf_counter()-t0
            v = dynamics_violations(dyn, st.actions)
            calls = st.execution.lattice_calls if st.execution is not None else 0
            print(f'{key} {lab:8s} outcome={st.outcome} steps={st.index} geo={st.geodesic:.2f} '
                  f'violations={v} min_rect={st.min_rect_dist:.2f} wall={dt:.1f}s lattice_calls(last)={calls}', flush=True)
        rt.close()


if __name__ == '__main__':
    main()
