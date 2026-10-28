"""Record a navigation video: chase-camera RGB (Falcon's third-person sensor) + top-down minimap, replaying a run's
recorded native action sequence in the unchanged habitat.Env.  The camera only renders; the replay is checked to
reproduce the run's robot poses exactly.

  python scripts/querygen_run.py tests.querygen.render_video RUN_JSON --suite S --out MP4 [--label TEXT] [--fps 12]
"""
import argparse
import json
import math

import numpy as np

SERIES = '#2a78d6'
HUMAN = '#8f8d87'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    import imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import habitat
    import torch
    import random
    from habitat.config import read_write
    from habitat.config.default_structured_configs import ThirdRGBSensorConfig
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    # rendering needs the renderer and textures: register Falcon's actions/measures WITHOUT the headless patch
    # that the simulation code installs (native._install -> install_headless disables the renderer)
    import habitat_sim  # noqa: F401
    import falcon.additional_action  # noqa: F401
    import falcon.additional_metric  # noqa: F401
    from eventwm_graph.querygen.native import EpisodeSpec, benchmark_config, ACTION_NAMES
    r = json.load(open(a.run))
    spec = EpisodeSpec(**r['spec'])
    acts = r['replay']['trace']['actions'] if 'replay' in r else r['actions']
    names = {'f': 'forward', 'l': 'left', 'r': 'right', 's': 'stop'}
    actions = [names[c] for c in acts]
    traj_ref = np.asarray([row[:3] for row in r['replay']['trace']['trajectory']], dtype=float) if 'replay' in r else None
    cfg = benchmark_config(spec)
    import os
    with read_write(cfg):
        if os.environ.get('QG_RENDER_GPU') is not None:
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = int(os.environ['QG_RENDER_GPU'])
        cfg.habitat.simulator.agents.agent_0.sim_sensors = {
            'third_rgb_sensor': ThirdRGBSensorConfig(height=480, width=640, position=[0.0, 2.6, 2.8],
                                                     orientation=[-0.6, 0.0, 0.0], hfov=90)}
    full = PointNavDatasetV1(cfg.habitat.dataset)
    eps = [e for e in full.episodes if str(e.episode_id) == str(spec.episode_id)]
    ds = PointNavDatasetV1(); ds.episodes = eps
    random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed)
    env = habitat.Env(config=cfg.habitat, dataset=ds)
    env.seed(spec.seed)
    import falcon.additional_action as fa
    obs = env.reset()
    fa.play_i = 0
    hn = int(eps[0].info.get('human_num', 0))
    hacts = tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1, hn+1))
    sim = env.sim
    robot = sim.get_agent_data(0).articulated_agent
    humans = [sim.get_agent_data(i).articulated_agent for i in range(1, hn+1)]
    goal = np.asarray(eps[0].goals[0].position, dtype=float)
    pf = sim.pathfinder
    y0 = float(robot.base_pos[1])
    mpp = 0.05
    top = np.asarray(pf.get_topdown_view(mpp, y0), dtype=bool)
    lo = pf.get_bounds()[0]
    key = [k for k in obs if 'third_rgb' in k][0]
    path, hpaths, frames = [], [[] for _ in humans], []
    mismatch = 0

    def frame(t, act, m):
        rgb = np.asarray(obs[key])[..., :3]
        fig = plt.figure(figsize=(10.4, 4.8), dpi=100)
        ax1 = fig.add_axes([0.0, 0.0, 0.615, 1.0]); ax1.imshow(rgb); ax1.axis('off')
        ax2 = fig.add_axes([0.63, 0.12, 0.36, 0.78])
        ax2.imshow(np.where(top, 0.97, 0.80), cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                   extent=(lo[0], lo[0]+top.shape[1]*mpp, lo[2]+top.shape[0]*mpp, lo[2]))
        for hp in hpaths:
            if len(hp) > 1:
                H = np.asarray(hp); ax2.plot(H[:, 0], H[:, 1], color=HUMAN, lw=0.8, alpha=0.6)
                ax2.plot(H[-1, 0], H[-1, 1], 'o', ms=4, color=HUMAN)
        P = np.asarray(path)
        ax2.plot(P[:, 0], P[:, 1], color=SERIES, lw=1.8)
        ax2.plot(P[-1, 0], P[-1, 1], 'o', ms=6, color=SERIES, mec='white', mew=1)
        ax2.plot(goal[0], goal[2], '*', ms=12, color='#0b0b0b', mec='white')
        c = np.concatenate([P, [[goal[0], goal[2]]]])
        ax2.set_xlim(c[:, 0].min()-2, c[:, 0].max()+2); ax2.set_ylim(c[:, 1].max()+2, c[:, 1].min()-2)
        ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        d = m.get('distance_to_goal', float('nan'))
        fig.text(0.63, 0.94, a.label, fontsize=10, color='#0b0b0b')
        fig.text(0.63, 0.05, f'step {t:3d}   action {act:8s}   distance to goal {d:5.2f} m', fontsize=9, color='#52514e')
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def record():
        p = robot.base_pos
        path.append((float(p[0]), float(p[2])))
        for k, h in enumerate(humans):
            hp = h.base_pos
            if abs(float(hp[1])-y0) < 0.6:
                hpaths[k].append((float(hp[0]), float(hp[2])))
    record()
    frames.append(frame(0, 'reset', env.get_metrics()))
    for t, act in enumerate(actions, 1):
        args = {n: np.array([1.]) for n in hacts}
        obs = env.step({'action': (ACTION_NAMES[act], *hacts), 'action_args': args})
        record()
        if traj_ref is not None and t < len(traj_ref):
            p = robot.base_pos
            mismatch += int(math.hypot(float(p[0])-traj_ref[t][0], float(p[2])-traj_ref[t][1]) > 1e-3)
        if t % a.every == 0 or env.episode_over:
            frames.append(frame(t, act, env.get_metrics()))
        if env.episode_over:
            break
    m = env.get_metrics()
    for _ in range(a.fps):                     # hold the last frame for a second
        frames.append(frames[-1])
    imageio.mimsave(a.out, frames, fps=a.fps, codec='libx264', quality=7)
    env.close()
    print(json.dumps(dict(out=a.out, frames=len(frames), success=m.get('success'), spl=m.get('spl'),
                          human_collision=m.get('human_collision'), psc=m.get('psc'), pose_mismatch_steps=mismatch)))


if __name__ == '__main__':
    main()
