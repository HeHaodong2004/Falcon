"""Minimal headless Habitat-Sim render check using an official test scene."""

from pathlib import Path
import sys

import habitat_sim


scene = Path("data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")
assert scene.is_file(), f"Missing test scene: {scene}"

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = str(scene)
sim_cfg.gpu_device_id = 0  # CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 here.

rgb = habitat_sim.CameraSensorSpec()
rgb.uuid = "rgb"
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution = [64, 64]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb]

with habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg])) as sim:
    first = sim.reset()["rgb"]
    second = sim.step("move_forward")["rgb"]
    assert first.shape == (64, 64, 4), first.shape
    assert second.shape == (64, 64, 4), second.shape
    print(f"PASS scene={scene} rgb_shape={second.shape} rgb_mean={second[..., :3].mean():.2f}")

if "--video" in sys.argv:
    from habitat.utils.visualizations.utils import images_to_video

    images_to_video([first[..., :3], second[..., :3]], ".local_notes",
                    "habitat_render_smoke", fps=10, verbose=False)
    print("PASS: .local_notes/habitat_render_smoke.mp4 encoded")
