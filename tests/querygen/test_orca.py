"""ORCA port checks (no simulator).

1. Against constant-velocity non-reactive agents, a holonomic robot executing
   full-responsibility ORCA velocities keeps every pairwise distance >= the
   combined radius (minus discretisation slack) whenever it starts outside.
2. The solution satisfies every half-plane when the LP is feasible, and the
   preferred velocity is returned unchanged when it is already admissible.
3. (Observation, not an assertion) with frequent re-solving, reciprocal 0.5
   responsibility also stayed outside the combined radius in these holonomic
   scenarios; full responsibility is still the declared P1 semantics because
   humans provably never take their half.
"""
import math
import numpy as np
from eventwm_graph.querygen.orca import ORCASolver, ORCAConfig, linear_program2, det


class H:
    def __init__(self, p, v):
        self.position = np.asarray(p, float); self.velocity = np.asarray(v, float); self.age = 0.


def run(seed, responsibility=1.0, n=3, steps=400, dt=0.05):
    rng = np.random.default_rng(seed)
    cfg = ORCAConfig(robot_radius=.5, human_radius=.3, margin=0., age_inflation=0., responsibility=responsibility)
    solver = ORCASolver(cfg)
    P = np.zeros(2); V = np.zeros(2); goal = np.array([10., 0.])
    hs = []
    for _ in range(n):
        p = np.array([rng.uniform(2, 8), rng.uniform(-3, 3)])
        target = np.array([rng.uniform(0, 10), rng.uniform(-3, 3)])
        v = (target-p); v = v/np.linalg.norm(v)*rng.uniform(.3, 1.2)
        hs.append(H(p, v))
    if min(np.linalg.norm(h.position-P) for h in hs) < 1.2:
        return None
    dmin = math.inf
    for _ in range(steps):
        d = goal-P
        v_pref = d/np.linalg.norm(d)*2.0 if np.linalg.norm(d) > .1 else np.zeros(2)
        v, lines, _ = solver.solve(P, V, v_pref, 2.0, hs, [], dt)
        P = P+v*dt; V = v
        for h in hs:
            h.position = h.position+h.velocity*dt
            dmin = min(dmin, float(np.linalg.norm(h.position-P)))
    return dmin


def test_full_responsibility_avoids():
    r = .8
    worst = math.inf
    count = 0
    for seed in range(300):
        d = run(seed, 1.0)
        if d is None:
            continue
        count += 1
        worst = min(worst, d)
    print('full responsibility: scenarios', count, 'min distance', round(worst, 4), 'combined radius', r)
    assert worst >= r-0.02, worst


def report_reciprocal_with_nonreactive():
    hits = 0; count = 0
    for seed in range(300):
        d = run(seed, 0.5)
        if d is None:
            continue
        count += 1; hits += d < .8-0.02
    print('reciprocal 0.5: scenarios', count, 'violations', hits)


def test_lp_feasible_and_identity():
    solver = ORCASolver(ORCAConfig(margin=0., age_inflation=0.))
    P = np.zeros(2); V = np.array([1., 0.])
    hs = [H([3., .2], [-1., 0.])]
    v, lines, _ = solver.solve(P, V, np.array([1., 0.]), 2., hs, [], .05)
    assert max(det(l.direction, l.point-v) for l in lines) <= 1e-6
    # far, irrelevant neighbour: preferred velocity is admissible -> unchanged
    hs = [H([0., 9.], [0., 1.])]
    v, lines, _ = solver.solve(P, V, np.array([1., 0.]), 2., hs, [], .05)
    assert np.allclose(v, [1., 0.])


if __name__ == '__main__':
    test_lp_feasible_and_identity()
    test_full_responsibility_avoids()
    report_reciprocal_with_nonreactive()
    print('ORCA_TESTS_PASSED')
