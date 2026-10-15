"""Run one isolated native FALCON evaluation with crowd and recording audits."""
import argparse
import copy
import gzip
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts/controlled')]
from controlled_matrix import digest, read, write, scene_path

# Also executed when multiprocessing imports this entry point as __mp_main__.
# Registration is limited to this runner and its workers, not upstream imports.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
import habitat_sim  # noqa: F401,E402
import runtime  # noqa: F401,E402


def build_config(root, case, attempt, steps, checkpoint):
    # Import the native bindings first (required by the local Habitat-Sim build).
    import habitat_sim  # noqa: F401
    from habitat_baselines.config.default import get_config
    from habitat.config import read_write
    from omegaconf import open_dict, OmegaConf
    import falcon  # noqa: F401

    scene = next(s for s in read(root / 'scenes.json') if s['name'] == case['scene'])
    ep = read(root / case['episode'])
    if ep['info']['human_num'] != case['humans']:
        raise ValueError('Episode count differs from case manifest')
    # Absolute paths exist only in resolved runtime files; portable source inputs
    # remain relative. PointNav accepts absolute scene paths without prefixing.
    ep['scene_id'] = str(scene_path(root, scene))
    dataset = scene['scene_dataset']
    if dataset != 'default':
        dataset = str((REPO / dataset).resolve())
    ep['scene_dataset_config'] = dataset
    ep['info']['controlled_navmesh'] = str(root / scene['navmesh'])
    episode_file = attempt / 'episode.json.gz'
    with gzip.open(episode_file, 'wt') as f:
        import json
        json.dump({'episodes': [ep]}, f)
    cfg = get_config('social_nav_v2/falcon_hm3d.yaml')
    third = copy.deepcopy(cfg.habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor)
    with read_write(cfg), open_dict(cfg.habitat.task):
        cfg.habitat.task.type = 'ControlledPointNavTask-v0'
        cfg.habitat.task.controlled_human_count = case['humans']
        cfg.habitat.task.controlled_audit_path = str(attempt / 'crowd_audit.json')
        cfg.habitat.simulator.type = 'ControlledRearrangeSim-v0'
        cfg.habitat.simulator.scene_dataset = dataset
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        cfg.habitat.dataset.data_path = str(episode_file)
        cfg.habitat.dataset.scenes_dir = str(REPO / 'data/scene_datasets')
        cfg.habitat.dataset.content_scenes = ['*']
        cfg.habitat.dataset.split = 'val'
        cfg.habitat.environment.max_episode_steps = steps
        cfg.habitat.seed = case['seed']
        sensors = cfg.habitat.simulator.agents.agent_0.sim_sensors
        rgb = copy.deepcopy(sensors.jaw_rgb_sensor)
        rgb.height = rgb.width = 256
        with read_write(third):
            third.height = third.width = 256
        # Install display cameras directly on Spot. Empty extra_sim_sensors keeps
        # the unmodified trainer from duplicating them on all six human slots.
        cfg.habitat.simulator.agents.agent_0.sim_sensors = {
            'jaw_depth_sensor': sensors.jaw_depth_sensor,
            'jaw_rgb_sensor': rgb, 'third_rgb_sensor': third,
        }
        cfg.habitat.gym.obs_keys = list(cfg.habitat.gym.obs_keys) + [
            'agent_0_articulated_agent_jaw_rgb', 'agent_0_third_rgb']
        cfg.habitat_baselines.eval.extra_sim_sensors = {}
        cfg.habitat_baselines.rl.policy.agent_0.name = 'ControlledPointNavResNetPolicy'
        cfg.habitat_baselines.eval.video_option = ['disk']
        cfg.habitat_baselines.eval.split = 'val'
        cfg.habitat_baselines.eval.use_ckpt_config = False
        cfg.habitat_baselines.evaluate = True
        cfg.habitat_baselines.num_environments = 1
        cfg.habitat_baselines.test_episode_count = 1
        cfg.habitat_baselines.torch_gpu_id = 0
        cfg.habitat_baselines.verbose = False
        cfg.habitat_baselines.eval_ckpt_path_dir = str(checkpoint)
        cfg.habitat_baselines.video_dir = str(attempt / 'native_video')
        cfg.habitat_baselines.tensorboard_dir = str(attempt / 'tb')
        cfg.habitat_baselines.checkpoint_folder = str(attempt / 'checkpoints')
        # Work disables the redundant initialization checkpoint before loading
        # the complete official evaluation state. Keep the same inference path.
        cfg.habitat_baselines.rl.ddppo.pretrained = False
        cfg.habitat_baselines.video_fps = 10
    OmegaConf.save(cfg, attempt / 'resolved_config.yaml', resolve=True)
    return cfg


def recording(attempt, case, steps):
    import cv2
    import numpy as np
    import imageio.v2 as imageio
    import habitat_baselines.rl.ppo.falcon_evaluator as evaluator
    from habitat.utils.visualizations import maps
    original_image = evaluator.observations_to_image
    original_video = evaluator.generate_video
    keys = ['agent_0_articulated_agent_jaw_depth',
            'agent_0_articulated_agent_jaw_rgb', 'agent_0_third_rgb']

    def four_views(observations, info):
        # Explicit sensor order: depth, robot RGB, third-person RGB, map.
        cameras = original_image({k: observations[k] for k in keys}, {})
        panel = np.full((256, 256, 3), 255, dtype=np.uint8)
        if 'top_down_map' in info:
            view = maps.colorize_draw_agent_and_fit_to_height(info['top_down_map'], 256)
            scale = min(256 / view.shape[0], 256 / view.shape[1])
            view = cv2.resize(view, (max(1, round(view.shape[1]*scale)), max(1, round(view.shape[0]*scale))))
            y, x = (256-view.shape[0])//2, (256-view.shape[1])//2
            panel[y:y+view.shape[0], x:x+view.shape[1]] = view
        return np.concatenate([cameras, panel], axis=1)

    def video(*args, **kwargs):
        frames, metrics = kwargs['images'], kwargs['metrics']
        write(attempt / 'metrics.json', dict(tag=case['tag'], humans=case['humans'],
              step_limit=steps, frames=len(frames), metrics=metrics, benchmark_equivalent=False))
        for i in sorted({0, len(frames)//3, 2*len(frames)//3, max(0, len(frames)-2)}):
            imageio.imwrite(attempt / f'frame_{i:04}.png', frames[i])
        result = original_video(*args, **kwargs)
        native = next((attempt / 'native_video').glob('*.mp4'))
        outcome = ('SUCCESS' if metrics['success'] else 'HUMAN COLLISION'
                   if metrics['human_collision'] else 'STEP LIMIT'
                   if metrics['num_steps'] >= steps else 'STOP OUTSIDE GOAL')
        title = f"FALCON | {case['tag']} | {outcome} | {int(metrics['num_steps'])}/{steps} steps"
        filters = 'scale=1536:384,pad=1536:480:0:80:color=0x101823'
        labels = [(8, 7, 22, title),
                  (8, 35, 16, 'Official checkpoint | controlled episode | full rollout at 10 fps'),
                  (8, 59, 16, 'DEPTH + LIVE METRICS'), (392, 59, 16, 'ROBOT RGB'),
                  (776, 59, 16, 'THIRD-PERSON RGB'), (1160, 59, 16, 'TOP-DOWN + TRAJECTORIES'),
                  (8, 464, 12, 'First-frame map unavailable; final black sensor frame carries terminal metrics.')]
        for x, y, size, label in labels:
            filters += f",drawtext=text='{label}':x={x}:y={y}:fontsize={size}:fontcolor=white"
        font_cache = attempt / '.cache'
        (font_cache / 'fontconfig').mkdir(parents=True, exist_ok=True)
        encoder_env = dict(os.environ, XDG_CACHE_HOME=str(font_cache))
        subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(native),
                        '-vf', filters, '-c:v', 'libx264', '-threads', '1', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(attempt / 'four_panel.mp4')],
                       check=True, env=encoder_env)
        return result
    evaluator.observations_to_image = four_views
    evaluator.generate_video = video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tag', required=True)
    p.add_argument('--attempt', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--config-only', action='store_true')
    args = p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    root, attempt = args.output.resolve(), args.attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    case = next(c for c in read(root / 'cases.json') if c['tag'] == args.tag)
    os.chdir(REPO)
    cfg = build_config(root, case, attempt, args.steps, args.checkpoint.resolve())
    if args.config_only:
        print('CONFIG_ONLY', case['tag'])
        return
    import torch
    import habitat_sim
    from habitat_baselines.run import execute_exp
    torch.set_num_threads(1)
    write(attempt / 'provenance.json', dict(
        python=sys.version, torch=torch.__version__, habitat_sim=habitat_sim.__version__,
        platform=platform.platform(), checkpoint_sha256=digest(args.checkpoint),
        episode_sha256=digest(root / case['episode']),
        navmesh_sha256=digest(root / next(s['navmesh'] for s in read(root / 'scenes.json')
                                         if s['name'] == case['scene'])),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    (attempt / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True))
    source_dir = attempt / 'pipeline_source'
    source_dir.mkdir()
    for source in [REPO / 'scripts/controlled_matrix.py', *sorted((REPO / 'scripts/controlled').glob('*.py'))]:
        shutil.copy2(source, source_dir / source.name)
    recording(attempt, case, args.steps)
    workdir = attempt / 'workdir'
    workdir.mkdir()
    for name in ['data', 'pretrained_model', 'habitat-lab', 'habitat-baselines', 'falcon']:
        (workdir / name).symlink_to(REPO / name, target_is_directory=True)
    os.chdir(workdir)
    execute_exp(cfg, 'eval')
    print('FALCON_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
