"""Diagnostic: is a verified witness admissible on the robot's *own legal map*?

Re-executes the witness actions in the branch simulator with the legal sensor (the same
observation stream a query branch would see) and reports, per step, the number of body-axis
samples on observed-BLOCKED cells (`control.body_cost`, the executor's static admissibility
test) and whether the generator's goal route exists from that legal state.  A witness pose with
body_cost > 0 lies outside every legal plan the executor can produce (legal-map conservatism),
which no query choice can fix.  Privileged; diagnostic only.

  python scripts/querygen_run.py tests.querygen.witness_legal KEY --witness DIR [--every 4] [--route]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--witness', required=True)
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--route', action='store_true', help='also run the generator at every reported step')
    ap.add_argument('--plot', default='', help='PNG: final legal map, witness path, body axes at blocked steps')
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.witness import _decode
    from eventwm_graph.querygen.control import body_cost, to_plane
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.legal import BLOCKED
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    w = json.loads((Path(a.witness)/f"{spec.key.replace('/', '_')}.json").read_text())
    dyn = FAMILIES[w.get('dynamics', 'native')]
    rt = EpisodeRuntime(spec, 'orca', dyn, strict=True)
    gen = QueryGenerator(dyn, GeneratorConfig())
    st = rt.root.copy()
    rt._load(rt.root)
    from scipy import ndimage
    from eventwm_graph.querygen.control import body_axis_points, to_world

    def eroded_cost(gmap, P, psi, k):
        B = gmap.state == BLOCKED
        if k:
            B = ndimage.binary_erosion(B, iterations=k, border_value=1)
        n = 0
        for x, z in to_world(body_axis_points(P, psi)):
            i, j = gmap.index(x, z)
            n += int(0 <= i < B.shape[0] and 0 <= j < B.shape[1] and B[i, j])
        return n
    worst = 0
    bad = []
    bad1 = []
    for t, ch in enumerate(w['actions']):
        p = st.legal.pose
        bc = body_cost(st.legal.map, to_plane(p[[0, 2]]), float(p[3]))
        worst = max(worst, bc)
        bc1 = eroded_cost(st.legal.map, to_plane(p[[0, 2]]), float(p[3]), 1) if bc else 0
        if bc:
            bad.append(t)
        if bc1:
            bad1.append(t)
        if t % a.every == 0 or bc:
            line = f'{t:3d} {ch} pose ({p[0]:6.2f},{p[2]:6.2f},{math.degrees(p[3]):6.1f}) body_blocked={bc} eroded1={bc1}'
            if a.route and t % a.every == 0:
                qs, diag = gen.propose(st.legal, None)
                goal = next((q for q in qs if q.kind == 'goal'), None)
                line += f" stuck={diag.get('stuck')} kinds={[q.label for q in qs]}"
            print(line, flush=True)
        if ch == 's':
            break
        res = rt.sim.step(_decode(ch), compute_geodesic=False)
        row = rt.sensor.observe(res.time)
        st.legal.observe(row, _decode(ch))
    print(f'steps {len(w["actions"])} worst body_blocked {worst} blocked steps {len(bad)}: {bad[:60]}')
    print(f'eroded by 1 cell: blocked steps {len(bad1)}: {bad1[:60]}')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from eventwm_graph.querygen.legal import FREE
        g = st.legal.map
        PS = np.asarray(st.legal.poses)
        img = np.full(g.state.shape+(3,), 1.)
        img[g.state == FREE] = (.82, .9, .82)
        img[g.state == BLOCKED] = (.25, .25, .25)
        x0 = g.ox*g.res; z0 = g.oz*g.res
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower', interpolation='nearest',
                  extent=(x0, x0+g.state.shape[0]*g.res, z0, z0+g.state.shape[1]*g.res))
        ax.plot(PS[:, 0], PS[:, 2], 'b.-', ms=3, lw=.6)
        for t in bad:
            p = PS[t]
            ax_pts = to_world(body_axis_points(to_plane(p[[0, 2]]), float(p[3])))
            ax.plot(ax_pts[:, 0], ax_pts[:, 1], 'r-', lw=.8, alpha=.6)
        ax.plot(st.legal.goal[0], st.legal.goal[2], 'r*', ms=14)
        lo = PS[:, [0, 2]].min(0)-1.5; hi = PS[:, [0, 2]].max(0)+1.5
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_title(f'{spec.key}: legal map after the witness; red = body axis at blocked steps')
        fig.tight_layout(); fig.savefig(a.plot, dpi=100); plt.close(fig)
    rt.close()


if __name__ == '__main__':
    main()
