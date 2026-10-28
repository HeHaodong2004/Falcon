"""Top-down episode figure: navmesh, human paths (recorded tape), and one robot path per run JSON
(Oracle solution, default policy, ...), with start, goal and the end state of each run.

  python scripts/querygen_run.py tests.querygen.viz_episode KEY --suite S --out PNG RUN_JSON:LABEL [...]
"""
import argparse
import json
import math

import numpy as np

# reference categorical palette (dataviz skill, light mode): fixed slot per entity, never cycled
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
HUMAN = '#8f8d87'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('runs', nargs='+', help='RUN_JSON:LABEL')
    a = ap.parse_args()
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.native import NativeTask, HumanTape, MAX_EPISODE_STEPS
    spec = [s for s in load(a.suite) if s.key == a.key][0]
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
    runs = []
    for item in a.runs:
        path, label = item.rsplit(':', 1)
        r = json.load(open(path))
        if 'actions' in r and 'found' in r:                      # a witness JSON (privileged search)
            from eventwm_graph.querygen.witness import load_witness_poses
            P = load_witness_poses(spec, r['actions'], r.get('dynamics') or 'native')
            traj = np.asarray([[p[0], p[2], p[3], 0.] for p in P], dtype=float)
            runs.append((label, traj, 'success (witness)', len(traj)-1, {}))
            continue
        rep = r.get('replay', {})
        rows = rep.get('trace', {}).get('trajectory', [])
        if not rows and r.get('decisions'):                      # failed branch: replay its decisions
            from eventwm_graph.querygen.episode import replay
            from eventwm_graph.querygen.executor import Query
            from eventwm_graph.querygen.dynamics import FAMILIES
            dec = [dict(query=Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'],
                                    duration=x['duration'], speed=x['speed'], label=x['label'], meta=x['meta'] or {}),
                        cont=x['cont'], horizon=x['horizon']) for x in r['decisions']]
            rp = replay(spec, r['planner'], FAMILIES[r['dynamics']], dec)
            rows = rp['trajectory']
            rep = dict(metrics=rp['metrics'], steps=rp['steps'])
        traj = np.asarray([row[:4] for row in rows], dtype=float)
        m = rep.get('metrics', {})
        outcome = r.get('branch', {}).get('outcome') or ('success' if m.get('success') == 1.0 else
                                                          'human collision' if m.get('human_collision') else
                                                          'best branch (no success)')
        runs.append((label, traj, outcome, int(rep.get('steps', len(traj)-1)), m))
    T = max([len(t) for _, t, _, _, _ in runs]+[2])
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    ext = (lo[0], lo[0]+top.shape[1]*mpp, lo[2]+top.shape[0]*mpp, lo[2])
    ax.imshow(np.where(top, 0.97, 0.80), cmap='gray', vmin=0, vmax=1, extent=ext, origin='upper', interpolation='nearest')
    H = tape.positions[:T]
    same = np.abs(H[..., 1]-p0[1]) < 0.5 if H.size else None
    for n in range(H.shape[1] if H.size else 0):
        sel = same[:, n]
        if sel.sum() < 2:
            continue
        ax.plot(H[sel, n, 0], H[sel, n, 2], color=HUMAN, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(H[sel, n, 0][-1], H[sel, n, 2][-1], 'o', ms=4, color=HUMAN, zorder=2)
    for k, (label, traj, outcome, steps, m) in enumerate(runs):
        if len(traj) < 2:
            continue
        c = SERIES[k % len(SERIES)]
        ax.plot(traj[:, 0], traj[:, 1], color=c, lw=2.0, zorder=4, solid_capstyle='round')
        end = traj[-1]
        mk = 'X' if 'collision' in outcome else ('o' if outcome == 'success' else 's')
        ax.plot(end[0], end[1], mk, ms=9, color=c, mec='white', mew=1.5, zorder=5)
        txt = f"{label}: {outcome.replace('_', ' ')}, {steps} steps" + (f", SPL {m.get('spl', 0):.2f}" if outcome == 'success' else '')
        ax.plot([], [], color=c, lw=2.0, label=txt)
    ax.plot(p0[0], p0[2], 's', ms=10, color=TEXT, mec='white', mew=1.5, zorder=6, label='start')
    ax.plot(goal[0], goal[2], '*', ms=16, color=TEXT, mec='white', mew=1.0, zorder=6, label='goal')
    ax.plot([], [], color=HUMAN, lw=1.0, label='human paths (recorded)')
    pts = np.concatenate([t[:, :2] for _, t, _, _, _ in runs if len(t)]+[np.asarray([[p0[0], p0[2]], [goal[0], goal[2]]])])
    pad = 1.5
    ax.set_xlim(pts[:, 0].min()-pad, pts[:, 0].max()+pad)
    ax.set_ylim(pts[:, 1].max()+pad, pts[:, 1].min()-pad)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(a.title or spec.key, fontsize=10, color=TEXT, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.04), fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
