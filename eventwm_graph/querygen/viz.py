"""Diagnostic plots (offline): legal map, robot trace, privileged human tracks."""
import math
import numpy as np


def plot_branch(runtime, state, path, trace=None, title=None, extra_points=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .legal import FREE, BLOCKED
    g = state.legal.map
    poses = np.asarray(state.legal.poses)
    tape = runtime.tape
    goal = runtime.task.goal
    pts = [poses[:, [0, 2]], goal[None, [0, 2]]]
    known = np.argwhere(g.state != 0)
    if len(known):
        kx = (known[:, 0]+g.ox+.5)*g.res; kz = (known[:, 1]+g.oz+.5)*g.res
        pts.append(np.stack([kx, kz], 1))
    allp = np.concatenate(pts)
    lo = allp.min(0)-1.; hi = allp.max(0)+1.
    fig, ax = plt.subplots(figsize=(9, 9))
    img = np.full(g.state.shape+(3,), 1.)
    img[g.state == FREE] = (.82, .9, .82)
    img[g.state == BLOCKED] = (.25, .25, .25)
    x0 = g.ox*g.res; z0 = g.oz*g.res
    ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower',
              extent=(x0, x0+g.state.shape[0]*g.res, z0, z0+g.state.shape[1]*g.res), interpolation='nearest')
    n = min(state.index+1, len(tape))
    for h in range(tape.positions.shape[1]):
        hp = tape.positions[:n, h]
        ax.plot(hp[:, 0], hp[:, 2], '-', color='tab:orange', alpha=.45, lw=1)
        ax.plot(hp[-1, 0], hp[-1, 2], 'o', color='tab:orange', ms=5)
    ax.scatter(poses[:, 0], poses[:, 2], c=np.arange(len(poses)), cmap='viridis', s=6, zorder=3)
    ax.plot(poses[0, 0], poses[0, 2], 'gs', ms=9, zorder=4)
    ax.plot(goal[0], goal[2], 'r*', ms=16, zorder=4)
    p = poses[-1]
    ax.arrow(p[0], p[2], .4*math.cos(p[3]), -.4*math.sin(p[3]), width=.03, color='k', zorder=5)
    if trace:
        st = [t for t in trace if t['static']]
        if st:
            sp = np.asarray([t['pose'] for t in st])
            ax.plot(sp[:, 0], sp[:, 2], 'rx', ms=5, zorder=4, label=f'static contact ({len(st)})')
        hc = [t for t in trace if t['human_collision'] or t['overlap']]
        if hc:
            hpz = np.asarray([t['pose'] for t in hc])
            ax.plot(hpz[:, 0], hpz[:, 2], 'mX', ms=10, zorder=6, label='human collision/overlap')
    if state.execution is not None and state.execution.route is not None:
        r = state.execution.route
        ax.plot(r[:, 0], r[:, 1], 'b--', lw=1.2, zorder=4)
    if extra_points:
        for (x, z), style in extra_points:
            ax.plot(x, z, style, ms=8, zorder=6)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('z')
    ax.set_title(title or f'{runtime.spec.key} outcome={state.outcome} steps={state.index}')
    if trace:
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
