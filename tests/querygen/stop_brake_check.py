"""Check: a robot with momentum brakes to rest at its brake limit before STOP (v10 executor).

  python scripts/querygen_run.py tests.querygen.stop_brake_check [KEY] [--suite dev] [--steps 12]

Runs a goal query for ``--steps`` steps under DWA x inertia (the robot accelerates), then the STOP
query, and prints the per-step speed level and command.  Expected: levels fall by ``brake`` per step
to 0, then 'stop'.  The native robot (no velocity state) stops on the first step.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key', nargs='?', default='hm3d/val/AWUFxHEyV3T/0')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--steps', type=int, default=12)
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    ok = True
    for planner, dname in (('dwa', 'inertia'), ('orca', 'native')):
        dyn = FAMILIES[dname]
        rt = EpisodeRuntime(spec, planner, dyn, strict=True, trace=True)
        st = rt.run(rt.root, Query('goal', label='goal'), horizon=a.steps)
        lvl0 = int(st.dyn_state) if not isinstance(st.dyn_state, tuple) else 0
        rt.trace.clear()
        done = rt.run(st, Query('stop', label='stop'), horizon=8)
        prims = [t['prim'] for t in rt.trace]
        print(f'{planner} x {dname}: level before STOP {lvl0}; stop steps {len(prims)} prims {prims} '
              f'outcome {done.outcome}', flush=True)
        if dyn.stateful:
            ok &= len(prims) == 1+dyn.stop_steps(lvl0) and prims[-1] == 'stop'
        else:
            ok &= prims == ['stop']
        rt.close()
    print('OK' if ok else 'MISMATCH')


if __name__ == '__main__':
    main()
