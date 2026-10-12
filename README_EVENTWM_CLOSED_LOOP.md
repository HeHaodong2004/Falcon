# EventWM closed-loop SocialNav

EventQueryWM-M evaluates executable timed motion queries during real Habitat
navigation. The accepted system repeatedly observes, predicts, selects,
executes a short prefix, and replans. Gaussian CV is an external baseline and
is absent from this robot policy. No retraining is required.

```text
Habitat -> legal observation adapter -> human history tracker
        -> goal-conditioned timed queries -> EventWM P(safe)
        -> risk-aware selector -> timed waypoint controller
        -> native robot actions -> next observation / replanning
```

Human observations use current-state visibility gating, a 90-degree field of
view, a 10 m range, ray-tested occlusion, timestamps, and velocity estimated
from past detections. This initial simulator-backed perception adapter is
replaceable; hidden states, future trajectories, and outcome labels are not
policy inputs. Known static geometry, robot odometry, and goal information are
permitted. Privileged visualization witnesses stay diagnostic-only.

The controller executes MOVE, SLOW, and zero-velocity WAIT through native
continuous BaseVelAction. STOP is terminal and used near the goal. It respects
query timing, executes about 0.5 s per prefix, and bounds repeated waiting.
Official SocialNav task termination and metric implementations are retained.
The continuous robot action extension differs from Falcon's published discrete
policy interface; this pilot is not a claim of parity with published baselines.

## Requirements and external assets

The validated workstation environment uses Python 3.9, Habitat-Sim 0.3.1 with
Bullet/headless EGL, PyTorch 1.12.1, NumPy 1.23.5, this repository's Habitat-Lab
and Habitat-Baselines, OmegaConf, OpenCV, Pillow, imageio, matplotlib, FFmpeg,
and DejaVu Sans. Follow the original README for environment and licensed asset
setup. Do not alter the shared environment or GPU drivers to reproduce this run.

Required external assets are HM3D scenes, official Social-HM3D validation
episodes, Spot assets, humanoids, and their animation files. They are not
included in Git. On the validated workstation they reside under
`/home/haodong/workspace/baselines/Falcon/data`.

Supply the validation-selected 188,265-query seed-20261013 checkpoint separately:

```text
/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_v4_full_scaling_20261012/runs/all_full_seed20261013/best.pt
SHA256: 6478820b4fb1a109d5ba62282064257be98dd3fb6f1a3db0195eb1eff03ec09e
EventQueryWM-M: 1,035,649 parameters; checkpoint step 2560.
```

The launcher verifies this identity. Checkpoints, datasets, rollout logs,
videos, native frame caches, and graphics libraries are excluded from this
publication. Compact numerical evidence is in `research_results/`.

These launchers retain the accepted workstation paths and C24 safety guard.
They default to `/home/haodong/miniforge3/envs/eventwm-falcon/bin/python`,
physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`), logical CUDA/EGL device 0, one CPU
thread, nice 19, idle I/O priority, and one simulator at a time. Asset/project
defaults are in `scripts/eventwm_closed_loop.py`; rendering defaults are in
`scripts/eventwm_native_render.py`. Another host requires explicit path
configuration and equivalent assets. The guard requires the current C24
read-only status database and progress log; it only stops research.

## Reproduce an episode or matched eight-episode evaluation

Run from this checkout, with a fresh output directory outside source control:

```bash
RUN_OUT=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_closed_loop_reproduction
scripts/run_eventwm_closed_loop.sh freeze --from-manifest configs/eventwm_closed_loop_20261012.manifest.json --output "$RUN_OUT"
# One complete EventWM episode, without rerunning the entire panel:
scripts/run_eventwm_closed_loop.sh episode --output "$RUN_OUT" --policy eventwm --tag ep001_YRUkbU5xsYj_6
# Matched Static and EventWM arms, identical membership and seeds:
scripts/run_eventwm_closed_loop.sh start --output "$RUN_OUT"
scripts/run_eventwm_closed_loop.sh summarize --output "$RUN_OUT"
```

The frozen manifest contains episode identities and source hashes, not the
episode dataset. Headless evaluation is the default. Episode actions,
observations, queries, predictions, timing, and official outcomes are recorded
under the selected output. Completed episodes are skipped on resume.

```bash
scripts/run_eventwm_closed_loop.sh status --output "$RUN_OUT"
scripts/run_eventwm_closed_loop.sh stop --output "$RUN_OUT"
scripts/run_eventwm_closed_loop.sh resume --output "$RUN_OUT"
```

Stop requests affect only research. Resume restarts an interrupted episode
from its fixed seed; it is not simulator-state checkpoint restoration. No
larger benchmark is needed to reproduce this milestone.

## Native Habitat RGB videos and EGL workaround

Native replay uses recorded actions and native human controllers. Every step
checks robot pose, active human base positions, legal tracks, clock, terminal
state, and scalar metrics before composing video. The three accepted replays
matched exactly on those checks. Human joint poses were not independently
recorded and compared. Rendering does not change policy actions or metrics.

On this workstation, kernel graphics version 595.91.07 and installed EGL
libraries 595.99.02 were incompatible. CUDA inference worked, but EGL exposed
Mesa devices without NVIDIA CUDA-device association. The fix extracts the
already cached matching NVIDIA GL package into a private research output,
selects its EGL vendor JSON and library path only in the renderer subprocess,
and retains CUDA_VISIBLE_DEVICES=1 with logical gpu_device_id=0. No driver,
CUDA, Conda, or production installation is changed. No proprietary libraries
are bundled. `prepare` intentionally asserts the workstation's matching
kernel/package version; another host needs a matching local graphics runtime.

```bash
VIZ_OUT=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_native_visualization_reproduction
scripts/run_eventwm_native_visualization.sh prepare --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh diagnose --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh replay --source "$RUN_OUT" --output "$VIZ_OUT" --policy eventwm --tag ep001_YRUkbU5xsYj_6
scripts/run_eventwm_native_visualization.sh replay --source "$RUN_OUT" --output "$VIZ_OUT" --policy eventwm --tag ep004_bCPU9suPUw9_10
scripts/run_eventwm_native_visualization.sh replay --source "$RUN_OUT" --output "$VIZ_OUT" --policy static --tag ep004_bCPU9suPUw9_10
scripts/run_eventwm_native_visualization.sh compose --source "$RUN_OUT" --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh verify --source "$RUN_OUT" --output "$VIZ_OUT"
```

Alternatively, use the existing accepted rollout directory as `--source`.
Videos show native Falcon RGB and a jaw-camera inset, Habitat NavMesh maps,
visible/remembered humans, field of view, candidates, selected/executed paths,
risk scores, speed/state, history availability, latency, and terminal status.
The matched comparison synchronizes by simulation time and labels held
terminal frames. `--show-ground-truth` enables explicitly privileged diagnostic
markers only. Outputs are A_eventwm_success.mp4, B_eventwm_collision.mp4,
C_static_vs_eventwm.mp4, and representative PNG screenshots.

## Current results and limits

| Policy | Navigation success | SPL | PSC | Human collisions |
|---|---:|---:|---:|---:|
| Static, WM disabled | 4/8 | 0.50 | 0.875 | 4/8 |
| EventWM | 4/8 | 0.50 | 0.895 | 4/8 |

EventWM changed only **2 executable action prefixes across 86 decisions**.
Both policies failed on the same four episodes. The PSC difference includes
an earlier EventWM collision; it does not establish improved safety. This is
an **engineering milestone**, not demonstrated performance improvement or a
SOTA result. Query-level safe-execution SR is distinct from navigation success.
The eight episodes are a development panel, not a powered generalization test.

See [closed-loop evidence](EVENTWM_CLOSED_LOOP_RESEARCH_20261012.md) and
[native visualization evidence](EVENTWM_NATIVE_VISUALIZATION_20261012.md).
The earlier closed-loop report's abstract videos were superseded by the native
visualization milestone. Planning and model capacity have not been optimized
after these results.

## Lightweight checks and publication provenance

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/haodong/miniforge3/envs/eventwm-falcon/bin/python -B \
  -m unittest discover -s tests -p test_closed_loop_navigation.py -v
git diff --check
```

Accepted local implementation commits are
`81d03e4840db4e2604f1fd1ce038347f8abaa5a8` and
`c55d4d9b892a0916d1a58227c19bb506fa1db87d`. The clean publication branch is
`publish/eventwm-closed-loop-socialnav-20261012`, based on this fork's main
commit `d15ef512c4b1451f9daeaf6d4d7ecc702ed57430`. It preserves the implementation
without importing older offline research history that contains checkpoint
and prediction binaries. The original research branch/history remain local
and unchanged by the publication branch. The production guard was extracted
verbatim into a standalone module to avoid unrelated training dependencies;
robot policy, selector, query generation, controller, and rendering are intact.
