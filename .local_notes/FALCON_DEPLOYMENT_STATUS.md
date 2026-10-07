# Deployment status — 2026-10-05

Overall: **PARTIALLY READY**. Official A*, ORCA and pretrained FALCON evaluation
paths work on one public HM3D training episode. Full validation evaluation remains
blocked by licensed scene assets. No training or evaluation sweep was run.

Repository: `main` at `d15ef512c4b1451f9daeaf6d4d7ecc702ed57430` with the local
compatibility edits listed in `FALCON_ENVIRONMENT.txt`. Host: `cv-w06`.

| Gate | Status | Verified evidence / limitation |
|---|---|---|
| Conda env | PASS | Existing `falcon`: Python 3.9.19, CMake 3.14.0; no reinstall on continuation |
| Imports | PASS | torch, habitat, habitat_sim, habitat_baselines, falcon and its registered extensions; clean process exit |
| torch CUDA | PASS | CUDA available, one visible RTX 3090 with GPU 1 selected; actual CUDA tensor operation |
| Habitat-Sim import | PASS | Version 0.3.1, exact headless/Bullet build; `built_with_bullet=True` |
| EGL/context | PASS | Headless OpenGL 4.6.0 on NVIDIA 595.91.07; no host EGL/driver changes |
| Minimal simulation | PASS | Official Skokloster Castle scene, 64x64 RGBA render, movement step, clean close |
| HM3D assets: public sample | PASS | Official public example assets; matching CFVBbU9Rsyb training episode ran |
| HM3D assets: validation | BLOCKED | All 56 referenced validation scene files missing; authorized scene access required |
| MP3D assets | BLOCKED | All 11 referenced validation scene files missing; licensed Habitat scene downloads required |
| SocialNav episodes | PASS | Both public ZIPs downloaded and extracted with CRC checks; 1,087 HM3D and 317 MP3D val episodes parsed |
| Spot/humanoid assets | PASS | Public Spot URDF/meshes, walking CSV, six configured humanoid models/motions; instantiated by successful episodes |
| FALCON checkpoint | PASS | `falcon_noaux_25.pth` loaded and used; `pretrained_habitat3.pth` also loaded during construction |
| A* command prepared | PASS | HM3D/MP3D configs compose cleanly; one public HM3D episode completed, exit 0 |
| ORCA command prepared | PASS | HM3D/MP3D configs compose cleanly; one public HM3D episode completed, exit 0 |
| FALCON command prepared | PASS | HM3D/MP3D configs compose cleanly; pretrained public HM3D episode completed, exit 0 |
| Video command prepared | PASS | Official `[disk]` option produced an MP4; H.264 decoding verified |
| Full HM3D/MP3D validation runs | BLOCKED | Scene access/downloads; not claimed verified |

Training and multi-GPU execution were NOT VERIFIED. Generic `hab3-episodes` and
`hab3_bench_assets` were not downloaded: these SocialNav configs consume the
downloaded SocialNav episodes and explicit Spot/humanoid assets instead.

## Exit-134 root cause and fix

The full original failure outputs are retained in `astar_config_stdout.log` and
`astar_config_stderr.log`. The configuration printed completely, followed by duplicate
static importer-plugin warnings and `malloc_consolidate(): unaligned fastbin chunk detected`.
No evaluation episode had begun.

Isolation before any additional dependency changes:

| Minimal invocation | Observed exit |
|---|---|
| `import magnum; import habitat_sim` | 134, native heap/shutdown abort |
| `import habitat_sim; import magnum` | 0 |
| `import habitat_baselines.run` | 134 |
| `import habitat_sim; import habitat_baselines.run` | 0 |
| `import habitat` before the fix | 134 |

This isolates the failure to loading Magnum before Habitat-Sim's 0.3.1 native
bindings. Early Habitat-Sim imports were added to `habitat/__init__.py` and
`falcon/__init__.py`. No dependency changed to resolve exit 134. Afterward the
A* config-only command, all six algorithm/dataset config compositions, standalone
imports and actual A*/ORCA/FALCON episodes exited 0. Post-fix logs are retained.

A separate verified PyTorch 1.12 compatibility issue was fixed by removing three
unsupported `weights_only=False` arguments. Full checkpoint loading remains the
default in this pinned PyTorch version. Earlier MKL repair used the existing CI pin.
Task actions, planners/policy, metrics, termination rules and checkpoint tensors
were not changed.

## Episode and video evidence

The verified episodes use `social_nav_v2/{algorithm}_hm3d.yaml`, evaluation mode,
one environment, one episode, `eval.split=train`, and
`content_scenes=[CFVBbU9Rsyb]`. They use supplied scene CFVBbU9Rsyb, episode 0,
with four humans. These are public training-episode smoke results, not held-out benchmarks.

| Algorithm | Episode completed | Steps | Success | PSC | Exit |
|---|---|---:|---:|---:|---:|
| A* | Collision termination | 96 | 0 | 0.9583 | 0 |
| ORCA | Episode step limit | 500 | 0 | 1.0000 | 0 |
| FALCON | Collision termination | 93 | 0 | 0.9355 | 0 |

FALCON generated one video under `evaluation/falcon/hm3d/video/` with the official
`habitat_baselines.eval.video_option=[disk]` override. Verified output:
932,822 bytes, H.264 / yuv420p, 1360x512, 10 FPS, 9.4 seconds; first frame decoded.
No additional benchmark videos were collected.

Logs: `health_check.log`, `astar_episode_smoke.log`, `orca_episode_smoke.log`,
`falcon_episode_smoke.log`, `video_verified.log`, and six `*_config_verified.log` files.

The exact fresh-terminal activation commands in `LOCAL_START_HERE.md` were also
executed successfully; `fresh_terminal_verified.log` records imports, the selected
RTX 3090 and a successful CUDA tensor operation (exit 0).

## Asset acquisition and remaining work

Downloaded public sources:

- Habitat downloader: `habitat_test_scenes`, `hab_spot_arm`, `habitat_humanoids`,
  `hm3d_example_habitat`, `hm3d_example_configs`.
- Upstream Spot walking CSV from the README URL.
- Upstream Google Drive SocialNav files: `social-hm3d.zip`, `social-mp3d.zip`.
- Upstream Google Drive weights: `falcon_noaux_25.pth`, `pretrained_habitat3.pth`.

Remaining manual acquisitions: authorized HM3D Habitat-format validation scenes
and/or MP3D Habitat scene files. Exact paths are in `missing_validation_assets.log`;
sources, placement, credentials-based downloader command and verification commands
are in `../LOCAL_START_HERE.md`. The licensed scene requirements were not bypassed.

The public sample alias is:
`data/scene_datasets/hm3d/train/00337-CFVBbU9Rsyb -> ../example/00337-CFVBbU9Rsyb`.
All assets and the environment are on the NVMe filesystem; no HDD operations occurred.

## Relevant source/config paths

- Entry: `habitat-baselines/habitat_baselines/run.py`: `main()`, `execute_exp()`.
- A*/ORCA evaluation: `rl/ppo/orca_trainer.py::ORCANoTrainer._eval_checkpoint()`
  and `rl/ppo/orca_evaluator.py::ORCAEvaluator.evaluate_agent()`.
- FALCON evaluation: `rl/ppo/ppo_trainer.py::PPOTrainer._eval_checkpoint()`
  (registered as `ddppo`) and `rl/ppo/falcon_evaluator.py::FALCONEvaluator.evaluate_agent()`.
- Configs: `habitat-baselines/habitat_baselines/config/social_nav_v2/`
  contains `astar_hm3d.yaml`, `orca_hm3d.yaml`, `falcon_hm3d.yaml`, their MP3D
  counterparts, and `falcon_hm3d_train.yaml`.
- Task defaults: `habitat-lab/habitat/config/benchmark/nav/socialnav_v2/`
  contains `{astar,orca,falcon}_{hm3d,mp3d}_task.yaml` and `falcon_hm3d_task_for_train.yaml`.
- Task details: `habitat-lab/habitat/config/habitat/task/{astar_task_detail,
  orca_task_detail,socialnav_task_detail,falcon_task_detail}.yaml`.
- Dataset defaults: `habitat-lab/habitat/config/habitat/dataset/social_nav_v2/`
  contains `social-hm3d.yaml` and `social-mp3d.yaml`; `eval.split` overrides their
  initial `train` split for evaluation.
- Registered simulator: `falcon/rearrange_sim_v2.py::RearrangeSim_v2` (`RearrangeSim-v2`).
- Task: `habitat-lab/habitat/tasks/rearrange/social_nav_v2/socialnav_task.py::MultiAgentPointNavTask`.
- Planners: `habitat-baselines/habitat_baselines/rl/ddppo/policy/{astar_policy,
  orca_policy}.py`; FALCON uses `resnet_policy.py::PointNavResNetPolicy`.
- Extensions/actions/metrics: `falcon/{additional_action,additional_sensor,
  additional_metric,auxiliary_tasks}.py`.

For a one-episode evaluation, Hydra loads the config and selects the registered
trainer; `HabitatVectorEnvFactory.construct_envs()` creates the gym/vector wrappers,
`PointNavDatasetV1` episodes, `habitat.Env`, registered simulator and task.
The evaluator calls `envs.reset()`, which selects an episode, reconfigures the scene,
places agents from episode metadata and obtains visual/task observations. `MultiPolicy.act()`
combines the robot policy/planner with fixed humanoid policies. CPU actions pass to
`envs.step()`, gym action decoding, `MultiAgentPointNavTask.step()` and the registered
discrete/oracle actions. `Env.step()` updates measures and termination/step limits;
the wrappers return observations/reward/done/info. The evaluator records scalar
metrics and, when requested, calls `generate_video()` at termination. VectorEnv
automatically resets after terminal steps; episode statistics refer to the captured
pre-step episode ID.
