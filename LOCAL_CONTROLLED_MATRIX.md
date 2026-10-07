# Local controlled FALCON matrix

The implementation uses this local checkout and its existing compatibility fixes.
All additions are in `scripts/controlled_matrix.py`, `scripts/controlled/`, and
`tests/controlled_matrix/`. **No existing FALCON/Habitat source file was changed by
this port.** In particular, `falcon/__init__.py` retains only your prior changes.
The Work installers and patches were inspected, not applied.

The exact 50 reference episodes, ten navmeshes, and three static scene instances
have already been copied into **`evaluation/controlled50/`**. No evaluation or
asset download was launched. This directory is ignored by the existing Git rules;
retain it with your experiment data. The source bundle is only needed to prepare
another copy, not to run this prepared matrix.

## Commands to launch yourself

Activate the existing environment first. Physical GPU 1 is exposed as logical GPU
0 to both PyTorch and Habitat-Sim. Runs are sequential, with one environment;
there is no multi-GPU/DDP launch.

```bash
cd ~/workspace/baselines/Falcon
source ~/miniforge3/etc/profile.d/conda.sh
conda activate falcon
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

**One episode: castle with zero humans, up to 500 steps.**

```bash
python scripts/controlled_matrix.py check --scenes castle --humans 0
python scripts/controlled_matrix.py run --scenes castle --humans 0
python scripts/controlled_matrix.py verify --scenes castle --humans 0
```

For two humans, replace `--humans 0` with `--humans 2`. The count selects an actual
saved episode and is checked against the running task, positions and controllers.

**One scene with 0, 1, 2, 4 and 6 humans.**

```bash
python scripts/controlled_matrix.py check --scenes castle
python scripts/controlled_matrix.py run --scenes castle
python scripts/controlled_matrix.py verify --scenes castle
```

**Full ten-scene × five-count matrix, after installing the missing assets below.**

```bash
python scripts/controlled_matrix.py check
python scripts/controlled_matrix.py run
python scripts/controlled_matrix.py verify
```

These commands use `evaluation/controlled50` by default. Add `--output PATH` to
all commands to select another prepared matrix. `--scenes castle,apartment_1`,
`--humans 1,4`, and `--tags 003_castle_h2` provide explicit subsets. Invalid or
empty selections fail. Count values supported by this checkpoint setup are
0, 1, 2, 4, 6; supporting more than six requires a different agent configuration.

To print all 50 commands without launching anything or requiring the missing
assets, use:

```bash
python scripts/controlled_matrix.py run --dry-run
```

Successful completed cases are verified and skipped on repeat invocation. Failed
attempts get a new timestamped directory on retry. A failure produces a nonzero
exit status; navigation failures such as collisions are valid completed results.
`--rerun` deliberately evaluates selected cases again and retains old attempt
folders. Changed episodes, navmeshes, pipeline sources, local tracked diff,
checkpoint or step limits prevent silently reusing a completed run. Do not launch
two orchestrators against the same output directory concurrently.

For an optional short **user-launched** smoke episode, use `run --scenes castle
--humans 0 --steps 2`. That is a truncated test, not a 500-step experiment. Use
`--rerun --steps 500` afterwards to replace the completion marker with a full run.

## Missing assets and preflight

At port time, castle, apartment_1 and all three HM3D public examples are present.
The full-matrix check reports these missing paths without substituting scenes:

```text
data/scene_datasets/17DRP5sb8fy/17DRP5sb8fy.glb
data/scene_datasets/mp3d.scene_dataset_config.json
data/scene_datasets/coda.glb
data/replica_cad/replicaCAD.scene_dataset_config.json
```

Scene meshes/configs must include their referenced textures, object templates and
meshes. `check` verifies top-level scene/config/navmesh/robot/human/checkpoint paths
and FFmpeg tools; Habitat performs full scene loading when you launch evaluation.
Both `ffmpeg` and `ffprobe` are already available in the `falcon` Conda environment.
Activate that environment before checking or launching.

The following are optional download commands for **you to run when ready**,
matching the public asset sources used by Work. They were not executed by this
port. Existing files are retained (`unzip -n`, downloader `--no-replace`).

```bash
mkdir -p data/downloads/controlled_matrix data/scene_datasets
curl -fL https://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip \
  -o data/downloads/controlled_matrix/mp3d_example.zip
unzip -n data/downloads/controlled_matrix/mp3d_example.zip -d data/scene_datasets
curl -fL https://dl.fbaipublicfiles.com/habitat/coda_v1.0.zip \
  -o data/downloads/controlled_matrix/coda.zip
unzip -n data/downloads/controlled_matrix/coda.zip -d data/scene_datasets
python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset --data-path data --no-replace
python scripts/controlled_matrix.py check
```

The bundled static ReplicaCAD scene instances are already in the experiment
folder. They reference the real ReplicaCAD assets above; they are not substitute
geometry. The original `apt_0/1/2.scene_instance.json` files are never overwritten.
If assets are installed at different paths, update the prepared `scenes.json`
entries (`scene_id`, `scene_dataset`) and matching episode scene IDs consistently,
or supply the expected paths using your own symlinks. Paths in portable manifests
are relative to the repository; `local_scene` and `navmesh` are relative to the
experiment directory. Runtime files resolve paths on the current workstation.

## What Work implemented, and what is preserved

| Scene index | Manifest name | Sampling and policy seed |
|---|---|---|
| 1 | castle | 2301 |
| 2 | apartment_1 | 2302 |
| 3 | mp3d_17DRP5sb8fy | 2303 |
| 4 | coda | 2304 |
| 5 | hm3d_CFVBbU9Rsyb | 2305 |
| 6 | hm3d_NBg5UqG3di3 | 2306 |
| 7 | hm3d_GLAQ4DNUx5U | 2307 |
| 8 | replicacad_apt_0 | 2308 |
| 9 | replicacad_apt_1 | 2309 |
| 10 | replicacad_apt_2 | 2310 |

These are ten scene configurations. The last three are different furniture
layouts in the same apartment shell. They are custom controlled episodes, not
paper benchmark validation episodes. No Work success-rate target is imposed.

Work's `create_scene_manifest.py` explicitly selects the first four scenes,
adds the three sorted public HM3D examples, and creates static furniture copies
of ReplicaCAD apt_0/1/2. `prepare_matrix.py` recomputes a Spot navmesh with radius
0.25 m, height 1.41 m, and static objects included. It selects the largest indoor
island, seeds Habitat and NumPy with `2300 + scene index`, and samples a connected
robot path with geodesic length strictly between 4 and 10 m and vertical difference
below 0.25 m. Human samples stay within 0.3 m of the robot floor, at least 1.2 m
from the robot start and 0.8 m from the supplied avoidance points. Six starts are
mutually separated; each gets two further route waypoints and random rotations.

All five cases share the robot start, quaternion, goal, all six human waypoint
sets, sampling seed and policy seed. Only `episode_id` and `info.human_num` differ.
The IDs are `3000 + case index`; count order is 0/1/2/4/6. Exact JSON and navmesh
bytes are copied for replay; regenerating would not be exact input reproduction.
The generator's quaternion is retained verbatim, including the local upstream
task's existing quaternion conversion behavior. It is not silently corrected.

The 200-input reference uses the same mechanism but selects forty distinct
ReplicaCAD staging layouts (four families × ten even-numbered variants), seed
`8300 + scene index`, and episode ID `10000 + case index`. Its scene instances
live in `scene_instances/` instead of the 50 bundle's `scene_configs/`. The importer
supports both layouts. The second bundle was inspected; none of its runs were
launched or mixed into the requested ten-scene matrix.

Work executes `social_nav_v2/falcon_hm3d.yaml`, the official `falcon_noaux_25.pth`,
one environment, one episode, 500-step cap, and native stochastic policy sampling
(`deterministic=False`). Its wrapper disables redundant `ddppo.pretrained`
initialization before loading the full evaluation checkpoint. This port does the
same, so the separate `pretrained_habitat3.pth` is not needed for this runner.
The native evaluator, actions, human navigation, metrics and termination rules
remain in use. Equal seeds do not guarantee identical actions across counts or
across Work's CPU/PyTorch 2.2.2 environment and this GPU/PyTorch 1.12.1 environment.

## Runtime crowd verification and isolation

`runtime.py` registers experiment-only simulator, task and policy subclasses.
Registration occurs when the standalone runner or its worker imports the module;
normal upstream entry points do not import it.

The task checks the requested count against `episode.info.human_num`, requires
Spot plus the six native humanoid slots, then calls the actual upstream reset and
placement methods. It reads each humanoid's simulator `base_pos`: active slots
must match waypoint 0; inactive slots must match the upstream parking location
`[-100 + slot] * 3`. The observed nonparked count must equal the requested count.
It also checks `_human_num`, every controller's crowd count and active controllers'
waypoint 1/2 goals. Any discrepancy aborts the run. A runtime record contains all
six positions and prints, for example:

```text
CONTROLLED_CROWD requested=2 actual_active=2 allocated_slots=6 episode=3003
```

`crowd_audit.json` is written at each reset, including the evaluator's automatic
reset after termination. It verifies placement/configuration, not how many people
are currently visible in the robot camera. Six allocated humanoid slots are
retained for checkpoint/action-space compatibility; only the requested prefix
occupies the scene. Stale start attributes and unused controller routes are
cleared when a reused task transitions to a lower count.

The simulator subclass loads the experiment's exact navmesh; it never replaces
the upstream `data/scene_datasets/navmeshes` cache. Absolute runtime scene paths
and `scene_dataset_config` avoid the PointNav prefix problem that Work solved
using `data/scene_datasets/../replica_cad` paths.

Recording cameras are added directly to Spot. The normal trainer's extra-camera
list stays empty, which avoids Work's global trainer patch that skipped all
human cameras. The thin policy subclass excludes the two display RGB sensors
before calling the native policy constructor. Its model, parameter names and
inference behavior are unchanged; the checkpoint still receives one depth
channel plus point-goal input. Local early native-binding imports and PyTorch
compatibility edits are preserved. No `os._exit(0)` shutdown bypass is used.

## Videos, metrics and reproduction records

Every attempt is under:

```text
evaluation/controlled50/runs/<tag>/<timestamp>/
  run.log                    # native evaluator output + runtime crowd checks
  execution.json             # exact command, status, duration, CUDA visibility
  episode.json.gz            # actual one-episode dataset with resolved paths
  resolved_config.yaml       # composed local configuration
  crowd_audit.json            # actual simulator positions/count/controller checks
  provenance.json            # versions, Git HEAD, input/navmesh/checkpoint hashes
  source.diff                # local tracked source changes at run time
  pipeline_source/           # standalone pipeline source used for this attempt
  metrics.json               # terminal native metrics + frame count
  frame_*.png                # selected native frames
  native_video/*.mp4         # full native evaluator recording
  four_panel.mp4             # full labeled 1536×480 recording at 10 fps
  workdir/output/actions.json
  workdir/output/result.json
  tb/                        # evaluator TensorBoard output
```

The panes are depth with live metrics, robot RGB, third-person RGB and top-down
map with trajectories. The three cameras render at 256×256. The port fits the map
to a fixed square before encoding, preserving its aspect ratio. This avoids
Work's variable map width and post-encoding crop assumptions. The first frame has
a blank map; the native final black sensor frame carries terminal metrics. No
frames are omitted, and unsuccessful episodes are retained. Titles report the
final outcome; overlays show each step's current metrics.

The orchestrator requires a zero process return code, runtime crowd audit, native
result/actions, metrics and a decoded video before recording completion. It checks
runtime scene/episode identity, action count equals `num_steps`, and video frame
count equals `num_steps + 1`; `ffprobe` verifies 1536×480 and the decoded count.
Per-case `complete.json` points to the latest successful attempt. Top-level
`results.json` and `results.csv` collect completed cases, including native success,
SPL, distance, PSC and human-collision metrics. Partial progress is allowed during
execution; `verify` fails if any selected case lacks a verified completion.

To reproduce one exact saved input as a new attempt:

```bash
python scripts/controlled_matrix.py run --tags 003_castle_h2 --rerun
python scripts/controlled_matrix.py verify --tags 003_castle_h2
```

Inspect that attempt's `execution.json` for its exact command and `provenance.json`
for checkpoint/input hashes. `resolved_config.yaml` is an audit record; the runner
resolves portable inputs afresh after relocation. Preserve the environment,
checkpoint, scene assets and source snapshot when comparing runs.

## Preparing another matrix or adding scenes

The already-prepared default output should not be prepared again. The importer
refuses a nonempty output directory to protect prior results. For a fresh exact
copy:

```bash
python scripts/controlled_matrix.py prepare \
  --reference ~/workspace/handoffs/falcon_work/FALCON_50_artifacts/falcon50 \
  --output evaluation/controlled50_fresh
python scripts/controlled_matrix.py check --output evaluation/controlled50_fresh
```

For a future exact 200-input import (preparation only):

```bash
python scripts/controlled_matrix.py prepare \
  --reference ~/workspace/handoffs/falcon_work/FALCON_200_repro_inputs/falcon200 \
  --output evaluation/controlled200
```

To intentionally generate new episodes, supply a JSON list of scenes with `name`,
`scene_id` and `scene_dataset` (use `"default"` for ordinary GLB scenes). Paths are
relative to this repository; names must contain letters, digits, underscores or
hyphens. For ReplicaCAD, point at the original scene-instance file: the generator
creates its own STATIC furniture copy. Keep the scene order stable because it
defines the seed offset. Example `new_scenes.json`:

```json
[
  {
    "name": "castle",
    "scene_id": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "scene_dataset": "default"
  }
]
```

```bash
python scripts/controlled_matrix.py generate --scene-manifest new_scenes.json \
  --seed-base 2300 --output evaluation/new_controlled
python scripts/controlled_matrix.py check --output evaluation/new_controlled
```

Generation uses a real simulator without a renderer and the Work sampling rules;
it was not executed during this port. The manifests/episodes can later be shared
with other baseline runners. This implementation currently evaluates FALCON;
changing baseline names alone is not a supported adapter.

## Validation performed during the port

Run the lightweight checks again with:

```bash
python tests/controlled_matrix/test_controlled.py
python -m compileall -q scripts/controlled_matrix.py scripts/controlled tests/controlled_matrix
python scripts/controlled_matrix.py check --scenes castle,apartment_1,hm3d_CFVBbU9Rsyb,hm3d_NBg5UqG3di3,hm3d_GLAQ4DNUx5U
python scripts/controlled_matrix.py run --dry-run
```

Tests cover all 50 controlled input relationships, all five count configurations,
PointNav dataset decoding, the actual upstream reset/placement methods using a
simulator test double, decreasing counts, mismatch rejection, RGB exclusion from
the native depth policy, and encoding/decoding three explicitly synthetic test
frames in a temporary directory. These test frames are not experiment outputs.
The forkserver registration test is skipped only when the sandbox denies its
Unix socket. No real GPU simulator reset, checkpoint inference, expensive matrix
or asset download was performed, so end-to-end GPU execution remains to be
validated by your first episode command.
