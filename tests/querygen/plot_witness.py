"""Plot a verified witness against the true human tape on the navmesh (privileged diagnostic).

  python scripts/querygen_run.py tests.querygen.plot_witness KEY --witness DIR --out PNG [--t0 0 --t1 200]
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
    ap.add_argument('--out', required=True)
    ap.add_argument('--t0', type=int, default=0)
    ap.add_argument('--t1', type=int, default=None)
    ap.add_argument('--every', type=int, default=10)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.native import NativeTask, HumanTape, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.witness import load_witness_poses
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    w = json.loads((Path(a.witness)/f"{spec.key.replace('/', '_')}.json").read_text())
    P = load_witness_poses(spec, w['actions'], w.get('dynamics', 'native'))
    task = NativeTask(spec)
    tape = HumanTape.record(task, MAX_EPISODE_STEPS)
    pf = task.sim.pathfinder
    y = float(P[0, 1])
    mpp = 0.05
    top = pf.get_topdown_view(mpp, y)
    lo, hi = pf.get_bounds()
    T = len(P)-1
    t1 = a.t1 if a.t1 is not None else T+20
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(top, origin='upper', cmap='Greys', alpha=.35,
              extent=(lo[0], lo[0]+top.shape[1]*mpp, lo[2]+top.shape[0]*mpp, lo[2]))
    ax.invert_yaxis()
    sel = slice(a.t0, min(t1, T)+1)
    sc = ax.scatter(P[sel, 0], P[sel, 2], c=np.arange(len(P))[sel], cmap='viridis', s=10, zorder=4)
    for t in range(a.t0, min(t1, T)+1, a.every):
        ax.annotate(str(t), (P[t, 0], P[t, 2]), fontsize=7, color='b', zorder=6)
        c, s = math.cos(P[t, 3]), -math.sin(P[t, 3])
        ax.arrow(P[t, 0], P[t, 2], .25*c, .25*s, width=.01, color='b', zorder=5)
    H = tape.positions
    for h in range(H.shape[1]):
        hp = H[a.t0:min(t1, len(H)-1)+1, h]
        same = np.abs(hp[:, 1]-y) < 1.0
        if not same.any():
            continue
        ax.plot(hp[same, 0], hp[same, 2], '-', color='tab:orange', alpha=.6, lw=1)
        for t in range(a.t0, min(t1, len(H)-1)+1, a.every):
            q = H[t, h]
            if abs(q[1]-y) < 1.0:
                ax.plot(q[0], q[2], 'o', color='tab:orange', ms=3)
                ax.annotate(f'{t}', (q[0], q[2]), fontsize=6, color='tab:red')
    ax.plot(P[0, 0], P[0, 2], 'gs', ms=10, zorder=6)
    ax.plot(task.goal[0], task.goal[2], 'r*', ms=16, zorder=6)
    pts = np.concatenate([P[:, [0, 2]], np.asarray(task.goal)[None, [0, 2]]])
    m = pts.min(0)-2.5; M = pts.max(0)+2.5
    ax.set_xlim(m[0], M[0]); ax.set_ylim(M[1], m[1])
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, shrink=.6, label='witness step')
    ax.set_title(f'{spec.key} witness {T} steps (t {a.t0}-{t1})')
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    task.close()


if __name__ == '__main__':
    main()
