"""Space-time infeasibility certificate: humans block every route (privileged accounting, never used online).

A sound relaxation of the strict task, so "goal unreachable in the relaxation" proves the episode infeasible:

* the robot base (x, z) stays on the navmesh -- the simulator's step filter projects every motion onto it
  (``native._base_velocity``) -- and moves at most ``STEP_MAX`` = 25/120 m per step (native forward; the
  velocity families' top level is the same); turning does not translate;
* strict PASS requires the robot rectangle (``ROBOT_BOUNDS``) >= ``BODY_PROXY_M`` from every same-floor
  human centre at every step; the rectangle contains the disc of radius min|bounds| = 0.2382 m around
  the base, so the base must stay >= 0.25 + 0.2382 m from every same-floor human centre;
* success needs STOP with geodesic < 0.2 m, so Euclidean distance < 0.2 m, within 500 steps.

Grid of ``cell`` m: a cell is passable if any point of it may be navigable (fine top-down view dilated by
one fine cell), blocked at step t only if *all* its points are too close to a human, a goal cell if any of
its points is within 0.2 m of the goal, and a move may reach every cell whose centre is within
STEP_MAX + cell*sqrt(2) (any pair of points of the two cells closer than STEP_MAX).  Forward reachability
from the start cell over steps 1..499; if no goal cell is ever reached the episode is certified infeasible.

Status (2026-10-28): sound but loose -- the relaxed point robot moves ~1.7x faster than Spot at cell 0.1 and
ignores turning; it certified none of the unresolved unseen episodes (kdFEfVoT1WE, nMeXfQU4PMS,
W16Bm4ysK8v reach a goal cell at steps 29/66/24) and, as it must, none of the witness-feasible checks.
Their goal discs are never fully covered by human exclusion discs either (exact check).
"""
import json
import math
from pathlib import Path

import numpy as np

from .native import BODY_PROXY_M, ROBOT_BOUNDS, SAME_FLOOR_M, SUCCESS_DISTANCE, MAX_EPISODE_STEPS

STEP_MAX = 25./120.


def _disc(r_cells):
    n = int(math.ceil(r_cells))
    a = np.arange(-n, n+1)
    return [(int(i), int(j)) for i in a for j in a if math.hypot(i, j) <= r_cells+1e-9]


def human_block_certificate(rt, cell=0.1, fine=0.05):
    from scipy.ndimage import binary_dilation
    task = rt.task
    pf = task.sim.pathfinder
    y = float(task.robot.base_pos[1])
    navf = np.asarray(pf.get_topdown_view(fine, y), dtype=bool)
    navf = binary_dilation(navf, structure=np.ones((3, 3), dtype=bool))
    lb = pf.get_bounds()[0]
    x0, z0 = float(lb[0]), float(lb[2])
    k = int(round(cell/fine))
    nzf, nxf = navf.shape
    nz, nx = (nzf+k-1)//k, (nxf+k-1)//k
    pad = np.zeros((nz*k, nx*k), dtype=bool)
    pad[:nzf, :nxf] = navf
    nav = pad.reshape(nz, k, nx, k).any(axis=(1, 3))
    zc = z0+(np.arange(nz)+.5)*cell
    xc = x0+(np.arange(nx)+.5)*cell
    half_diag = cell*math.sqrt(2)/2
    r_min = BODY_PROXY_M+min(abs(b) for b in ROBOT_BOUNDS)
    g = task.goal
    goal = np.hypot(xc[None, :]-g[0], zc[:, None]-g[2]) < SUCCESS_DISTANCE+half_diag
    p = rt.root.legal.pose
    si, sj = int((p[2]-z0)/cell), int((p[0]-x0)/cell)
    reach = np.zeros((nz, nx), dtype=bool)
    reach[min(max(si, 0), nz-1), min(max(sj, 0), nx-1)] = True
    offsets = _disc((STEP_MAX+2*half_diag)/cell)
    H = rt.tape.positions
    T = min(len(H), MAX_EPISODE_STEPS)
    first = None
    max_cells = 0
    for t in range(1, T):
        grown = np.zeros_like(reach)
        ii, jj = np.nonzero(reach)
        for di, dj in offsets:
            a, b = ii+di, jj+dj
            ok = (a >= 0) & (a < nz) & (b >= 0) & (b < nx)
            grown[a[ok], b[ok]] = True
        blocked = np.zeros((nz, nx), dtype=bool)
        for h in H[t]:
            if abs(float(h[1])-y) >= SAME_FLOOR_M:
                continue
            blocked |= np.hypot(xc[None, :]-h[0], zc[:, None]-h[2]) < r_min-half_diag
        reach = grown & nav & ~blocked
        max_cells = max(max_cells, int(reach.sum()))
        if (reach & goal).any():
            first = t
            break
        if not reach.any():
            break
    return dict(key=rt.spec.key, cell=cell, r_min=r_min, step_max=STEP_MAX, goal_cells=int(goal.sum()),
                first_goal_step=first, max_reachable_cells=max_cells, certified_infeasible=first is None and
                int(goal.sum()) > 0, rule='human_block relaxation (stcert.py)')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from .suites import load
    from .episode import EpisodeRuntime
    from .dynamics import FAMILIES
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {s.key: s for s in load(a.suite)}
    for key in a.keys:
        rt = EpisodeRuntime(specs[key], 'orca', FAMILIES['native'], strict=True)
        try:
            rec = human_block_certificate(rt)
        finally:
            rt.close()
        (out/f"humanblock_{key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
