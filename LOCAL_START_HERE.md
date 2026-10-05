# FALCON on cv-w06

Deployment status: **PARTIALLY READY**. The environment, GPU rendering, A*/ORCA episodes,
pretrained FALCON episode and official video output are READY. Full validation
evaluation requires licensed HM3D or MP3D scene downloads. Public SocialNav episodes,
robot/humanoid assets and both required evaluation weight files are already installed.

The source is branch `main`, commit `d15ef512c4b1451f9daeaf6d4d7ecc702ed57430`, with
the small compatibility edits documented below. No training or evaluation sweep was run.

## A. Fresh terminal — READY

```bash
cd ~/workspace/baselines/Falcon
source ~/miniforge3/etc/profile.d/conda.sh
conda activate falcon
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

Physical GPU 1 is PCI bus `B3:00.0`, UUID
`GPU-0ef701aa-76c7-12aa-c21b-f40f44c8e1b0`. Inside this process it is GPU **0**;
keep `habitat_baselines.torch_gpu_id=0` and
`habitat.simulator.habitat_sim_v0.gpu_device_id=0`.
No additional EGL, LD_LIBRARY_PATH, ROS or CUDA-toolkit variables were needed.
Run from the repository root so the local `falcon` package and relative asset paths resolve.

Use the installed module entry point `python -m habitat_baselines.run`; it invokes
the vendored `habitat-baselines/habitat_baselines/run.py`.

## B. Quick health checks — READY

```bash
python .local_notes/health_check.py
cmake --version
python .local_notes/habitat_smoke.py
python .local_notes/verify_assets.py --dataset hm3d --split train --scene CFVBbU9Rsyb
```

Expected: Python 3.9.19, PyTorch 1.12.1 / CUDA runtime 11.3, CUDA available,
one visible RTX 3090, Habitat-Sim 0.3.1 with Bullet, CMake 3.14.0, and `PASS`.
The simulator check loads the official Skokloster Castle test scene, renders RGB,
takes a movement step and closes cleanly. Its missing-semantic-annotation warnings
are expected for this RGB test scene.

These checks need a normal workstation terminal with GPU access. Restricted agent
sandbox checks can report CUDA unavailable or fail EGL even when the host checks pass.

Check the licensed validation assets after downloading them:

```bash
python .local_notes/verify_assets.py --list-missing
```

It currently reports 56 missing HM3D validation scenes and 11 missing MP3D validation
scenes and exits with status 2. This is an asset requirement, not a failed installation.

## Public sample episode available now — READY for A*, ORCA and FALCON

The official public HM3D example `00337-CFVBbU9Rsyb` matches the supplied
Social-HM3D **train** content file. A symlink makes that same scene available at its
episode-declared path. These commands run evaluation on one supplied training episode;
they are deployment checks, not validation benchmark results.

```bash
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/astar_hm3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=train \
  'habitat.dataset.content_scenes=[CFVBbU9Rsyb]'

python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=train \
  'habitat.dataset.content_scenes=[CFVBbU9Rsyb]' \
  'habitat_baselines.eval.video_option=[disk]'
```

For the same verified ORCA sample, replace `astar_hm3d.yaml` with `orca_hm3d.yaml`
in the first command. ORCA completed at the 500-step limit with success 0.

A* and FALCON completed with collision termination (success 0); that is a valid
completed smoke episode. The verified FALCON video is in `evaluation/falcon/hm3d/video/`.

## C. Official A* validation — REQUIRES MANUAL DOWNLOAD of scenes

Both configs compose successfully. Use the matching command after its scene check passes:

```bash
# Social-HM3D
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/astar_hm3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val

# Social-MP3D
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/astar_mp3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val
```

The explicit `evaluate=True` is required: this commit's A*/ORCA YAML files inherit
`evaluate=False`. Without the override, `run.py` selects the training path.
The one-environment, one-episode overrides limit work and preserve the selected
task, planner, actions, metrics and 500-step episode limit.

## D. Official ORCA validation — REQUIRES MANUAL DOWNLOAD of scenes

```bash
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/orca_hm3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val

python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/orca_mp3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val
```

ORCA is implemented in the vendored `ORCAPolicy`; no external ORCA/RVO package
is required by this implementation. Both configs and their video overrides were composed.

## E. Official FALCON validation — REQUIRES MANUAL DOWNLOAD of scenes

```bash
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val

python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_mp3d.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=val
```

Both configs already set evaluation mode. Keep both weight files:
`pretrained_model/falcon_noaux_25.pth` is the evaluation checkpoint, while
`pretrained_model/pretrained_habitat3.pth` is also read during policy construction
because `rl.ddppo.pretrained=True`. Both were downloaded and loaded successfully.

## F. Official video output — READY on the public sample

Add the following override to any evaluation command above:

```bash
'habitat_baselines.eval.video_option=[disk]'
```

For example, the fully available public FALCON sample with an explicit output directory:

```bash
python -u -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.split=train \
  'habitat.dataset.content_scenes=[CFVBbU9Rsyb]' \
  'habitat_baselines.eval.video_option=[disk]' \
  habitat_baselines.video_dir=evaluation/falcon/hm3d/video
```

The evaluator adds third-person RGB sensors, assembles frames and metrics, then
uses the repository's `generate_video()` and imageio/FFmpeg pipeline. The produced
MP4 was decoded successfully. No desktop/display window is required.

## G. Changing run settings

| Setting | Location or CLI override |
|---|---|
| Algorithm and dataset | `--config-name=social_nav_v2/{astar,orca,falcon}_{hm3d,mp3d}.yaml` |
| Evaluation split | `habitat_baselines.eval.split=val` (use `train` only for the public sample) |
| Episode file | `habitat.dataset.data_path=.../{split}/{split}.json.gz` |
| Scene root | `habitat.dataset.scenes_dir=data/scene_datasets` |
| Scene selection | `'habitat.dataset.content_scenes=[CFVBbU9Rsyb]'`; use a content-file stem in the selected split |
| Episode count | `habitat_baselines.test_episode_count=1`; `-1` evaluates all loaded episodes |
| Parallel environments | `habitat_baselines.num_environments=1` |
| GPU | `CUDA_VISIBLE_DEVICES=1`; logical torch/simulator IDs stay 0 |
| Evaluation checkpoint | `habitat_baselines.eval_ckpt_path_dir=pretrained_model/falcon_noaux_25.pth` |
| Initial pretrained weights | `habitat_baselines.rl.ddppo.pretrained_weights=pretrained_model/pretrained_habitat3.pth` |
| Video | `'habitat_baselines.eval.video_option=[disk]'` |
| Video directory / FPS | `habitat_baselines.video_dir=...`, `habitat_baselines.video_fps=10` |
| TensorBoard/results | `habitat_baselines.tensorboard_dir=...`; scalar results are also printed in the run log |
| Human count | Stored per episode in `episode.info['human_num']` plus human waypoints; there is no supported count-only CLI override |

The official configs instantiate Spot as agent 0 and up to six humanoids as
agents 1–6. Selecting existing episodes preserves their crowd count and paths.
Changing the count alone would require consistent episode/agent changes and would
change the benchmark. Do not use `habitat.simulator.scene` to select a benchmark
scene: the episode's `scene_id` replaces it during reset.

## H. Important paths

| Item | Path | Status |
|---|---|---|
| Repository | `/home/haodong/workspace/baselines/Falcon` | READY |
| Conda environment | `/home/haodong/miniforge3/envs/falcon` | READY |
| Vendored Habitat-Lab | `habitat-lab/` (editable install) | READY |
| Vendored Baselines | `habitat-baselines/` (editable install) | READY |
| FALCON extensions | `falcon/` | READY from repository root |
| Social-HM3D episodes | `data/datasets/pointnav/social-hm3d/{train,minival,val}/` | READY |
| Social-MP3D episodes | `data/datasets/pointnav/social-mp3d/{train,val}/` | READY |
| Original episode archives | `data/downloads/social-{hm3d,mp3d}.zip` | READY |
| Official RGB test scenes | `data/scene_datasets/habitat-test-scenes/` | READY |
| Public HM3D examples | `data/scene_datasets/hm3d/example/` | READY |
| Public sample alias | `data/scene_datasets/hm3d/train/00337-CFVBbU9Rsyb` | READY, symlink to the matching example |
| Licensed HM3D validation | `data/scene_datasets/hm3d/val/` | REQUIRES MANUAL DOWNLOAD |
| Licensed MP3D scenes | `data/scene_datasets/mp3d/{scene}/{scene}.glb` | REQUIRES MANUAL DOWNLOAD |
| Spot URDF and meshes | `data/robots/hab_spot_arm/` | READY |
| Spot walking CSV | `data/robots/spot_data/spot_walking_trajectory.csv` | READY |
| Humanoid URDFs, meshes and motion files | `data/humanoids/humanoid_data/` | READY |
| Dataset downloader storage | `data/versioned_data/` | READY |
| Weights | `pretrained_model/{falcon_noaux_25,pretrained_habitat3}.pth` | READY |
| Evaluation configs | `habitat-baselines/habitat_baselines/config/social_nav_v2/` | READY, six configs verified |
| Benchmark task configs | `habitat-lab/habitat/config/benchmark/nav/socialnav_v2/` | READY |
| Dataset configs | `habitat-lab/habitat/config/habitat/dataset/social_nav_v2/` | READY |
| Default videos | `evaluation/{astar,orca,falcon}/{hm3d,mp3d}/video/` | READY for verified FALCON sample; others created when requested |
| TensorBoard | `evaluation/{algorithm}/{dataset}/tb/` | READY when its evaluation runs |
| Deployment records and smoke logs | `.local_notes/` | READY |

The simulator may generate Spot-specific navmeshes in
`data/scene_datasets/navmeshes/` on first use. Dataset downloader paths are
symlinks into `data/versioned_data/`; preserve their targets when moving assets.
These paths are on the workstation NVMe filesystem. The 8 TB HDD was not touched.

## Manual scene downloads

### HM3D validation — REQUIRES MANUAL DOWNLOAD / license credentials

Use the official [HM3D access instructions](https://github.com/matterport/habitat-matterport-3dresearch)
to obtain access. Download the Habitat-format validation scenes (`.basis.glb`),
preferably v0.2, and their configs. Preserve the split/scene folder structure:

```text
data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
data/scene_datasets/hm3d/val/00859-3t8DB4Uzvkt/3t8DB4Uzvkt.basis.glb
```

After accepting the applicable terms and obtaining credentials, the installed
official downloader supports these exact UIDs:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_val_habitat_v0.2 hm3d_val_configs_v0.2 \
  --data-path data --no-replace \
  --username YOUR_AUTHORIZED_USERNAME --password YOUR_AUTHORIZED_PASSWORD
python .local_notes/verify_assets.py --dataset hm3d --split val --list-missing
```

For one episode, obtain just one listed validation scene and select its content
stem with `habitat.dataset.content_scenes`. Full evaluation needs all 56 referenced
validation scenes. HM3D training requires its training scenes separately. The supplied
Social-HM3D `minival` is a competition split and is **not** the HM3D scene `minival`
bundle; use the verifier to determine its actual scene requirements.
RGB/depth navigation and video do not require semantic annotations.

### MP3D validation — REQUIRES MANUAL DOWNLOAD / license access

Follow the official [Matterport3D download instructions](https://github.com/facebookresearch/habitat-sim/blob/v0.3.1/DATASETS.md#matterport3d-mp3d-dataset)
and request the Habitat scene assets. For a minimal run, obtain one validation
scene listed by the verifier; for full evaluation obtain all 11 referenced scenes.
Place each `.glb` with its associated assets at:

```text
data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb
```

```bash
python .local_notes/verify_assets.py --dataset mp3d --split val --list-missing
```

No further episode or evaluation checkpoint download is currently needed.
The exact missing scene paths are also recorded in `.local_notes/missing_validation_assets.log`.

## Compatibility notes and troubleshooting

- Python 3.9, CMake 3.14.0 and Habitat-Sim 0.3.1 headless/Bullet match the README.
- PyTorch is unpinned in normal installation (`torch>=1.3.1`, unconstrained torchvision).
  This deployment uses the repository CI pair: torch 1.12.1, torchvision 0.13.1,
  CUDA runtime 11.3, and MKL 2021.4.0. Newer MKL caused a verified `iJIT_NotifyEvent`
  import failure; applying the CI MKL pin fixed it.
- Exit 134 was reproduced with `import magnum; import habitat_sim` and with the A*
  config-only command. Loading Habitat-Sim first eliminated the native shutdown abort
  and duplicate-plugin warnings. Early imports in `habitat/__init__.py` and
  `falcon/__init__.py` enforce this order; package versions were not changed for this fix.
- Three `weights_only=False` arguments were removed from checkpoint loaders because
  CI-pinned torch 1.12.1 does not support that keyword. Its default is full checkpoint
  loading, preserving the original behavior. Use this pinned environment with these edits.
- Headless EGL and CUDA were verified on the existing NVIDIA 595.91.07 driver.
  No host driver, global CUDA, system Python, Conda base dependencies or ROS changes
  were made. The environment contains its own CUDA runtime.
- If CUDA fails in a fresh terminal, verify `which python`, `conda activate falcon`,
  `CUDA_VISIBLE_DEVICES`, and `nvidia-smi`. If only EGL fails, inspect the host NVIDIA
  EGL/libglvnd libraries and vendor selection; do not reinstall the driver.
- If FALCON waits indefinitely instead of loading a checkpoint, verify
  `test -f pretrained_model/falcon_noaux_25.pth`: the trainer polls when that file is absent.

Training was NOT VERIFIED and was not run. Its upstream single-GPU config is
`habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train.yaml`,
using `falcon_hm3d_task_for_train.yaml` and `falcon_task_detail.yaml`. The upstream
`single_node_falcon.sh` hardcodes four GPUs; do not use it on this two-GPU workstation unchanged.
