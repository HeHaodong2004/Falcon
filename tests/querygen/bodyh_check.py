"""Check the privileged body-aware cost-to-go (bodyh.py) at spawn states: value at the reset pose, after k
in-place turns either way, and after one forward step; build time and table size.

  python scripts/querygen_run.py tests.querygen.bodyh_check KEY [--suite dev]
"""
import argparse
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.bodyh import BodyCostToGo
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    dyn = FAMILIES['native']
    rt = EpisodeRuntime(spec, 'orca', dyn, strict=True)
    B = BodyCostToGo(rt)
    p = list(map(float, rt.root.legal.pose))
    print(f'{a.key}: table {B.shape} built in {B.seconds:.1f} s; start geodesic {rt.start_geodesic:.2f} m '
          f'(= {rt.start_geodesic/dyn.step_m:.0f} straight steps)')
    print(f'  reset pose value {B.value(p)}')
    for k in (4, 8, 16, 24, 32, 38):
        l = B.value([p[0], p[1], p[2], p[3]+k*dyn.turn_rad])
        r = B.value([p[0], p[1], p[2], p[3]-k*dyn.turn_rad])
        print(f'  after {k:2d} turns: left {l}  right {r}')
    f = B.value([p[0]+dyn.step_m*math.cos(p[3]), p[1], p[2]-dyn.step_m*math.sin(p[3]), p[3]])
    print(f'  after one forward step: {f}')
    rt.close()


if __name__ == '__main__':
    main()
