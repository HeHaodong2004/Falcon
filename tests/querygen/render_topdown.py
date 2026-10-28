"""Top-down navigation video (CPU; no renderer needed): navmesh, the robot's oriented body footprint, humans
(recorded tape positions + short trails), path so far, goal, and step / action / distance / outcome.

  python scripts/querygen_run.py tests.querygen.render_topdown RUN_JSON [RUN_JSON:LABEL ...] --suite S --out MP4
         [--fps 12] [--every 1]
One panel per run (e.g. Oracle query branch | default policy), sharing the human motion.
"""
import argparse
import json
import math
import os

import numpy as np

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
HUMAN = '#6f6d68'
BODY_X = (-0.646495, 0.584202)
HALF_W = 0.243


def body_corners(x, z, psi):
    c, s = math.cos(psi), math.sin(psi)
    pts = []
    for lx, ly in ((BODY_X[1], HALF_W), (BODY_X[1], -HALF_W), (BODY_X[0], -HALF_W), (BODY_X[0], HALF_W)):
        # plane frame (x, -z): forward = (cos, sin), left = (-sin, cos)
        px = x+lx*c-ly*s
        pz_plane = -z+lx*s+ly*c
        pts.append((px, -pz_plane))
    return np.asarray(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    import imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, MAX_EPISODE_STEPS
    runs = []
    for item in a.runs:
        path, label = (item.rsplit(':', 1) + [''])[:2] if ':' in item else (item, '')
        r = json.load(open(path))
        if 'replay' in r:
            tr = r['replay']['trace']
            rows, actions, m = tr['trajectory'], tr['actions'], r['replay']['metrics']
        else:                                   # failed search: replay the best branch's decisions
            from eventwm_graph.querygen.episode import replay
            from eventwm_graph.querygen.executor import Query
            from eventwm_graph.querygen.dynamics import FAMILIES
            dec = [dict(query=Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'],
                                    duration=x['duration'], speed=x['speed'], label=x['label'], meta=x['meta'] or {}),
                        cont=x['cont'], horizon=x['horizon']) for x in r['decisions']]
            if r.get('generator'):              # process-wide legal-map options of the recorded run
                from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
                QueryGenerator(FAMILIES[r['dynamics']], GeneratorConfig(**r['generator']))
            rp = replay(EpisodeSpec(**r['spec']), r['planner'], FAMILIES[r['dynamics']], dec)
            rows, m = rp['trajectory'], rp['metrics']
            actions = ''.join(x[0] if isinstance(x, str) else 'v' for x in rp['actions'])   # as compact_trace
        traj = np.asarray([row[:4] for row in rows], dtype=float)      # x, z, yaw, geodesic
        outcome = 'success' if m.get('success') == 1.0 else ('human collision' if m.get('human_collision') else
                                                              r.get('branch', {}).get('outcome') or
                                                              ('best branch, no success' if 'oracle' in r else 'failure'))
        runs.append(dict(label=label or r.get('planner', ''), traj=traj, actions=actions, outcome=outcome,
                         metrics=m, spec=EpisodeSpec(**r['spec'])))
    spec = runs[0]['spec']
    task = NativeTask(spec)
    try:
        cache = os.environ.get('QG_TAPE_CACHE')
        tape = HumanTape.cached(task, spec, MAX_EPISODE_STEPS, cache) if cache else HumanTape.record(task, MAX_EPISODE_STEPS)
        pf = task.sim.pathfinder
        p0 = task.robot_pose()
        goal = np.asarray(task.goal, dtype=float)
        mpp = 0.05
        top = np.asarray(pf.get_topdown_view(mpp, float(p0[1])), dtype=bool)
        lo = pf.get_bounds()[0]
    finally:
        task.close()
    T = max(len(r['traj']) for r in runs)
    H = tape.positions
    allp = np.concatenate([r['traj'][:, :2] for r in runs]+[[[goal[0], goal[2]]]])
    xlim = (allp[:, 0].min()-2.0, allp[:, 0].max()+2.0)
    ylim = (allp[:, 1].max()+2.0, allp[:, 1].min()-2.0)
    ext = (lo[0], lo[0]+top.shape[1]*mpp, lo[2]+top.shape[0]*mpp, lo[2])
    n = len(runs)
    frames = []
    # panel size follows the plotted region's aspect ratio (wide corridors are not padded into squares)
    aspect = (ylim[0]-ylim[1])/max(1e-6, xlim[1]-xlim[0])
    pw = 5.4
    ph = min(7.0, max(2.8, pw*aspect))+0.75+(0.35 if a.title else 0.)
    import textwrap
    title = '\n'.join(textwrap.wrap(a.title, width=int(13*pw*n))) if a.title else ''
    for t in list(range(0, T, a.every))+[T-1]:
        fig, axes = plt.subplots(1, n, figsize=(pw*n, ph), dpi=100, squeeze=False)
        for k, (ax, r) in enumerate(zip(axes[0], runs)):
            c = SERIES[k % len(SERIES)]
            tt = min(t, len(r['traj'])-1)
            ax.imshow(np.where(top, 0.97, 0.80), cmap='gray', vmin=0, vmax=1, extent=ext, interpolation='nearest')
            hp = H[min(tt, len(H)-1)]
            for h in range(hp.shape[0]):
                if abs(hp[h, 1]-p0[1]) > 0.5:
                    continue
                trail = H[max(0, tt-24):tt+1, h]
                ax.plot(trail[:, 0], trail[:, 2], color=HUMAN, lw=1.0, alpha=0.5)
                ax.add_patch(Circle((hp[h, 0], hp[h, 2]), 0.25, color=HUMAN, alpha=0.85, zorder=4))
            P = r['traj'][:tt+1]
            ax.plot(P[:, 0], P[:, 1], color=c, lw=1.8, zorder=3)
            x, z, psi = r['traj'][tt][:3]
            ax.add_patch(Polygon(body_corners(x, z, psi), closed=True, color=c, alpha=0.9, zorder=5))
            ax.plot(goal[0], goal[2], '*', ms=15, color='#0b0b0b', mec='white', zorder=6)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            done = tt >= len(r['traj'])-1
            act = r['actions'][tt-1] if 0 < tt <= len(r['actions']) else '-'
            act = {'f': 'forward', 'l': 'turn left', 'r': 'turn right', 's': 'STOP'}.get(act, act)
            status = r['outcome'] if done else f'{act}'
            ax.set_title(f"{r['label']}\nstep {tt:3d}   distance {r['traj'][tt][3]:5.2f} m   {status}", fontsize=9,
                         color='#0b0b0b', loc='left')
        if title:
            fig.suptitle(title, fontsize=10, color='#0b0b0b', x=0.01, ha='left')
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        H16, W16 = -(-img.shape[0]//16)*16, -(-img.shape[1]//16)*16     # pad to the codec's block size
        pad = np.full((H16, W16, 3), 255, dtype=np.uint8)
        pad[:img.shape[0], :img.shape[1]] = img
        frames.append(pad)
        plt.close(fig)
    frames += [frames[-1]]*a.fps
    imageio.mimsave(a.out, frames, fps=a.fps, codec='libx264', quality=7, macro_block_size=8)
    print(json.dumps(dict(out=a.out, frames=len(frames), runs=[(r['label'], r['outcome'], len(r['traj'])-1) for r in runs])))


if __name__ == '__main__':
    main()
