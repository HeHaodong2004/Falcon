"""Lightweight checks: no simulator construction, GPU work, checkpoint load or rollout."""
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts/controlled')]
import habitat_sim  # native import order
import numpy as np
import runtime
from run_case import build_config, recording
from controlled_matrix import read, validate, read_probe
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from gym import spaces


def registry_worker(queue):
    from habitat.core.registry import registry
    queue.put(registry.get_task('ControlledPointNavTask-v0').__name__)


class HumanAction:
    num_goals = 2

    def reset(self, episode, task):
        self.human_num = task._human_num
        if self.slot <= task._human_num:
            self.goals = [episode.info[f'human_{self.slot-1}_waypoint_{k}_position'] for k in (1, 2)]


class ControlledTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = REPO / 'evaluation/controlled50'
        self.scenes, self.cases = validate(self.root)

    def test_exact_reference_matrix_and_config(self):
        self.assertEqual(len(self.scenes), 10)
        self.assertEqual(len(self.cases), 50)
        for case in self.cases[:5]:
            attempt = Path(self.temp.name) / case['tag']
            attempt.mkdir()
            cfg = build_config(self.root, case, attempt, 500, REPO / 'pretrained_model/falcon_noaux_25.pth')
            self.assertEqual(cfg.habitat.task.controlled_human_count, case['humans'])
            self.assertEqual(len(cfg.habitat.simulator.agents), 7)
            self.assertEqual(len(cfg.habitat_baselines.eval.extra_sim_sensors), 0)
            self.assertEqual(len(cfg.habitat.simulator.agents.agent_0.sim_sensors), 3)
            self.assertEqual(cfg.habitat.simulator.habitat_sim_v0.gpu_device_id, 0)
            self.assertEqual(cfg.habitat_baselines.torch_gpu_id, 0)
            dataset = PointNavDatasetV1(cfg.habitat.dataset)
            self.assertEqual(len(dataset.episodes), 1)
            self.assertEqual(dataset.episodes[0].info['human_num'], case['humans'])
            self.assertTrue(Path(dataset.episodes[0].scene_id).is_file())

    def make_task(self):
        task = runtime.ControlledPointNavTask.__new__(runtime.ControlledPointNavTask)
        agents = [SimpleNamespace(articulated_agent=SimpleNamespace(base_pos=[0, 0, 0], base_rot=0)) for _ in range(7)]
        task._sim = SimpleNamespace(num_articulated_agents=7, get_agent_data=lambda i: agents[i],
                                    reset=lambda: None, add_perf_timing=lambda *a: None,
                                    cur_runtime_perf_scope=[])
        task._use_episode_start_goal = True
        task._sim_reset = True
        task._should_place_articulated_agent = True
        task._articulated_agent_pos_start = None
        task.measurements = SimpleNamespace(get_metrics=lambda: {})
        task.actions = {}
        for i in range(1, 7):
            action = HumanAction()
            action.slot = i
            action.goals = [np.zeros(3), np.zeros(3)]
            task.actions[f'agent_{i}_oracle_nav_randcoord_action_obstacle'] = action
        task._config = SimpleNamespace(controlled_human_count=0,
                                       controlled_audit_path=str(Path(self.temp.name) / 'crowd.json'))
        return task

    def test_recording_cameras_excluded_from_policy(self):
        observation_space = spaces.Dict({
            'articulated_agent_jaw_depth': spaces.Box(0, 1, (256, 256, 1), np.float32),
            'articulated_agent_jaw_rgb': spaces.Box(0, 255, (256, 256, 3), np.uint8),
            'third_rgb': spaces.Box(0, 255, (256, 256, 3), np.uint8),
            'pointgoal_with_gps_compass': spaces.Box(-100, 100, (2,), np.float32),
        })
        filtered = runtime.policy_observations(observation_space)
        self.assertEqual(set(filtered.spaces), {'articulated_agent_jaw_depth', 'pointgoal_with_gps_compass'})
        import torch
        torch.set_num_threads(1)
        cfg = build_config(self.root, self.cases[0], Path(self.temp.name), 500,
                           REPO / 'pretrained_model/falcon_noaux_25.pth')
        policy = runtime.ControlledPointNavResNetPolicy.from_config(
            cfg, observation_space, spaces.Discrete(4), agent_name='agent_0')
        self.assertEqual(policy.net.visual_encoder._n_input_channels, 1)
        self.assertEqual(policy.net.visual_encoder.visual_keys, ['articulated_agent_jaw_depth'])

    def episode(self, count):
        ep = copy.deepcopy(read(self.root / self.cases[0]['episode']))
        ep['info']['human_num'] = count
        ep['info']['controlled_navmesh'] = 'test.navmesh'
        ep['goals'] = [SimpleNamespace(**g) for g in ep['goals']]
        return SimpleNamespace(**ep)

    def test_real_upstream_reset_placement_and_decreasing_counts(self):
        task = self.make_task()
        # Execute the real parent reset and placement methods on a fake simulator.
        for count in (6, 4, 2, 1, 0):
            task._config.controlled_human_count = count
            task.reset(self.episode(count), fetch_observations=False)
            audit = read(task._config.controlled_audit_path)
            self.assertEqual(audit['actual_active_humans'], count)
            self.assertEqual(sum(p['active'] for p in audit['positions']), count)

    def test_mismatch_and_wrong_placement_fail(self):
        task = self.make_task()
        with self.assertRaises(ValueError):
            task.reset(self.episode(6), fetch_observations=False)
        with patch.object(task, '_set_articulated_agent_start', lambda i: None):
            with self.assertRaises(RuntimeError):
                task.reset(self.episode(0), fetch_observations=False)

    def test_forkserver_registration(self):
        ctx = multiprocessing.get_context('forkserver')
        queue = ctx.Queue()
        child = ctx.Process(target=registry_worker, args=(queue,))
        try:
            child.start()
        except PermissionError as exc:
            self.skipTest(f'Sandbox denies forkserver socket: {exc}')
        child.join(30)
        if child.is_alive():
            child.terminate()
            child.join()
            self.fail('Registration worker timed out')
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(queue.get(timeout=2), 'ControlledPointNavTask')

    def test_four_panel_encoding_with_synthetic_test_frames(self):
        import habitat_baselines.rl.ppo.falcon_evaluator as evaluator
        original_image, original_video = evaluator.observations_to_image, evaluator.generate_video
        self.addCleanup(setattr, evaluator, 'observations_to_image', original_image)
        self.addCleanup(setattr, evaluator, 'generate_video', original_video)
        attempt = Path(self.temp.name)
        recording(attempt, dict(tag='synthetic_encoding_test', humans=0), 2)
        observations = {
            'agent_0_articulated_agent_jaw_depth': np.full((256, 256, 1), .5, np.float32),
            'agent_0_articulated_agent_jaw_rgb': np.full((256, 256, 3), [255, 0, 0], np.uint8),
            'agent_0_third_rgb': np.full((256, 256, 3), [0, 255, 0], np.uint8),
        }
        frame = evaluator.observations_to_image(observations, {})
        self.assertEqual(frame.shape, (256, 1024, 3))
        np.testing.assert_array_equal(frame[128, 300], [255, 0, 0])
        np.testing.assert_array_equal(frame[128, 600], [0, 255, 0])
        evaluator.generate_video(video_option=['disk'], video_dir=str(attempt / 'native_video'),
                                 images=[frame.copy() for _ in range(3)], scene_id='synthetic_test',
                                 episode_id='test', checkpoint_idx=0,
                                 metrics=dict(num_steps=2, success=0, human_collision=0),
                                 tb_writer=None, fps=10, verbose=False)
        probe = read_probe(attempt / 'four_panel.mp4')
        self.assertEqual((probe['width'], probe['height']), (1536, 480))
        self.assertEqual(int(probe['nb_read_frames']), 3)


if __name__ == '__main__':
    unittest.main()
