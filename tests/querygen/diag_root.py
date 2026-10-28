"""Privileged diagnostic: what every root candidate does, drawn over the navmesh.

  python scripts/querygen_run.py tests.querygen.diag_root KEY [--escape ACTIONS] [--out PNG] [--horizon 16]

Runs each generator candidate from the reset state (strict rules) and prints
outcome, steps, first static links; plots the navmesh top-down view, the legal
map's BLOCKED cells, the robot rectangle, candidate trajectories (red = failed)
and an optional primitive escape sequence (e.g. from a feasibility certificate).
"""
import argparse
import math
import numpy as np


def rect(pose):
    from eventwm_graph.querygen.native import ROBOT_BOUNDS
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    c, s = math.cos(pose[3]), math.sin(pose[3])
    pts = []
    for lx, lz in ((xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax), (xmin, zmin)):
        # local forward (c, -s), right (s, c) in (x, z)
        pts.append((pose[0]+lx*c+lz*s, pose[2]-lx*s+lz*c))
    return np.asarray(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--escape', default='')
    ap.add_argument('--out', default=None)
    ap.add_argument('--horizon', type=int, default=16)
    ap.add_argument('--suite', default='dev')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator
    from eventwm_graph.querygen.dynamics import NATIVE
    from eventwm_graph.querygen.legal import BLOCKED
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rt = EpisodeRuntime(spec, 'orca', NATIVE, strict=True)
    root = rt.root
    gen = QueryGenerator(NATIVE)
    queries, diag = gen.propose(root.legal, None)
    print('diag', {k: v for k, v in diag.items() if k != 'raw'}, 'raw', diag.get('raw'))
    trajs = []
    for q in queries:
        ch = rt.run(root, q, horizon=a.horizon)
        P = np.asarray(ch.legal.poses)
        disp = math.hypot(P[-1][0]-P[0][0], P[-1][2]-P[0][2])
        print(f'{q.label:16s} kind={q.kind:7s} outcome={str(ch.outcome):17s} steps={ch.index:3d} disp={disp:.2f} '
              f'geo={ch.geodesic:.2f} actions={"".join(x[0] if isinstance(x, str) else "v" for x in ch.actions)}')
        trajs.append((q.label, P, ch.outcome))
    # escape replay
    esc = None
    if a.escape:
        sim = rt.sim
        sim.load(root.robot, 0)
        P = [rt.task.robot_pose()] if hasattr(rt, 'task') else []
        poses = []
        for ch in a.escape:
            r = sim.step({'f': 'forward', 'l': 'left', 'r': 'right'}[ch])
            poses.append(r.pose)
        esc = np.asarray(poses)
    pf = rt.task.sim.pathfinder
    y = float(root.legal.pose[1])
    mpp = 0.05
    td = np.asarray(pf.get_topdown_view(mpp, y))
    lb = pf.get_bounds()[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    # navigable = light green, non-navigable = grey (rows = z, cols = x)
    rgb = np.where(td[..., None], np.array([.85, .95, .85]), np.array([.6, .6, .6]))
    ax.imshow(rgb, origin='lower', extent=(lb[0], lb[0]+td.shape[1]*mpp, lb[2], lb[2]+td.shape[0]*mpp))
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(root.legal.pose[:3], dtype=np.float32)
    sp.requested_end = np.asarray(rt.task.goal, dtype=np.float32)
    if pf.find_path(sp):
        gp = np.asarray(sp.points)
        ax.plot(gp[:, 0], gp[:, 2], 'c--', lw=1.5, label='navmesh geodesic path')
    g = root.legal.map
    bi = np.argwhere(g.state == BLOCKED)
    if len(bi):
        ax.plot((bi[:, 0]+g.ox+.5)*g.res, (bi[:, 1]+g.oz+.5)*g.res, 's', color='k', ms=2, alpha=.5)
    p0 = root.legal.pose
    R = rect(p0)
    ax.plot(R[:, 0], R[:, 1], 'g-', lw=2)
    goal = rt.task.goal
    ax.plot(goal[0], goal[2], 'r*', ms=16)
    for lab, P, oc in trajs:
        col = 'tab:red' if oc not in (None, 'success') else 'tab:blue'
        ax.plot(P[:, 0], P[:, 2], '-', color=col, lw=1)
        ax.text(P[-1][0], P[-1][2], lab, fontsize=6, color=col)
    if esc is not None:
        ax.plot(esc[:, 0], esc[:, 2], 'm.-', lw=1.5, label='certificate escape')
        R = rect(esc[-1]); ax.plot(R[:, 0], R[:, 1], 'm-', lw=1)
        ax.legend()
    span = max(4., float(np.hypot(goal[0]-p0[0], goal[2]-p0[2]))*0.6)
    cx, cz = (p0[0]+goal[0])/2, (p0[2]+goal[2])/2
    ax.set_xlim(cx-span-1, cx+span+1); ax.set_ylim(cz-span-1, cz+span+1)
    ax.set_aspect('equal')
    ax.set_title(f'{spec.key} root candidates (red = failed)')
    out = a.out or f"output/runs/diag_{spec.key.replace('/', '_')}.png"
    fig.savefig(out, dpi=90)
    print('saved', out)
    rt.close()


if __name__ == '__main__':
    main()
