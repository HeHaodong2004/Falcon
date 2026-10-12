# Native Falcon rendering restored

The three requested MP4s now use real Habitat RGB: textured HM3D geometry,
native walking human meshes, and the robot executing the recorded navigation
actions. The visualization is an isolated, validated replay of the original
closed-loop episodes. Neural predictions and inference latency are the original
recorded policy telemetry. No model, selector, generator, controller, benchmark
definition, or production process was changed. No 32-episode benchmark ran.

Output directory:
`/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_native_visualization_20261012`.

| Deliverable | MP4 relative to output | Screenshot |
|---|---|---|
| A: complete WM success, YRUkbU5xsYj / 6 | `videos/A_eventwm_success.mp4` | `screenshots/A_eventwm_success.png` |
| B: complete WM collision, bCPU9suPUw9 / 10 | `videos/B_eventwm_collision.mp4` | `screenshots/B_eventwm_collision.png` |
| C: matched static versus WM, same collision episode | `videos/C_static_vs_eventwm.mp4` | `screenshots/C_static_vs_eventwm.png` |

`screenshots/C_static_vs_eventwm_final.png` shows both terminal outcomes.
All videos are 1280×768, H.264/yuv420p, 12 fps, with successful full decoding.
Playback is slower than native simulation time for readability; the native
clock is displayed. Each video has a two-second final-frame hold.

## EGL diagnosis and process-only fix

The running NVIDIA kernel module is **595.91.07**, whereas installed NVIDIA EGL
libraries are **595.99.02**. With the installed graphics stack, EGL returned
three Mesa devices and no `EGL_NV_device_cuda` association. CUDA `cuInit` still
succeeded, so PyTorch functionality did not establish EGL functionality.
Changing a GPU ordinal could not repair this enumeration failure.

Matching graphics libraries already existed in the package cache:

`/var/cache/apt/archives/libnvidia-gl-595_595.91.07-0ubuntu0.22.04.1_amd64.deb`

Package SHA256:
`de419223db9285fa075b9dec700730d51fcae5a884a5d947e252528ed8ea6552`.

`prepare` uses `dpkg-deb --extract` into the research output's
`graphics_runtime/`, **not package installation**. The rendering process alone
sets `LD_LIBRARY_PATH` to those matching libraries and
`__EGL_VENDOR_LIBRARY_FILENAMES` to a private vendor JSON, then re-execs before
loading Habitat. Shader caching also stays in the output directory.
System libraries, kernel modules, CUDA installations, shared Conda environments,
and production environment variables remain unchanged. `libcuda` continues to
come from the existing system installation.

With matching EGL, enumeration returns two NVIDIA devices. Under
`CUDA_VISIBLE_DEVICES=1`, physical GPU 1 (`card2`, `renderD129`, PCI b3:00.0)
is **logical CUDA device 0**, so Habitat `gpu_device_id=0` is correct. The other
physical GPU has no CUDA-visible ordinal in that process. With visibility
`0,1`, EGL CUDA attributes become 0 and 1. Saved system/private enumeration
JSONs record both cases. Native rendering reports RTX 3090, OpenGL 4.6.0 NVIDIA
595.91.07. No reboot, driver replacement, or environment installation occurred.

## Native pipeline and overlays

The original Falcon evaluator at
`habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py` uses
`observations_to_image` and `generate_video`/`images_to_video`. The new code
reuses Habitat RGB sensors, its original third-person camera configuration,
`observations_to_image`, native NavMesh map/projection utilities, and
`images_to_video`. Geometry is not drawn by a replacement abstract renderer.
RGB resolution is increased only for diagnostic capture; the jaw-depth
calibration and the original observation FoV are retained.

A/B layout:

- Left: actual native Falcon third-person RGB, with a jaw-RGB inset. Both are
  diagnostic views; this checkpoint consumes legal tracked states, not pixels.
- Top right: a cropped **Habitat NavMesh map**, robot heading, known goal,
  observation FoV, yellow currently visible tracks, hollow remembered tracks
  with age, gray candidate paths, cyan selected nominal path, and white executed
  motion. Off-map goals are explicitly marked. Static geometry is known; the
  cone shows geometric FoV, while actual detections also respect occlusion.
- Bottom right: simulation time, goal distance/progress, selected query, last
  executed action/speed, WAIT/TURN/SLOW/MOVE/STOP state, history availability,
  recorded WM latency, each candidate's full-query collision risk, and terminal
  native SPL/PSC. Query risk is distinct from navigation Success Rate.

Ground-truth map positions are **off by default**. The optional
`--show-ground-truth` flag adds explicitly labeled magenta diagnostic crosses
for current GT. It cannot affect policy inputs or actions. No future human
trajectories are drawn. Native third-person RGB naturally reveals its own
diagnostic camera view and must not be mistaken for the robot's legal tracker.

The comparison uses the same episode initialization and seed 20261016. It
samples each native replay at or before a shared 0.048 s simulation timeline,
with actual timestamps shown separately. A terminated arm holds its final
frame with an explicit label. Insets show selected and executed trajectories.
The first differing robot action is #31, observed at **1.504 s**; WM's selected
path differs from static. WM collides at 3.256 s; static collides at 4.280 s.
This demonstrates an earlier WM collision, not improved safety.

## Replay validity and unchanged metrics

Replay steps the actual native environment with the original action dictionaries
and native human controllers; it does not teleport actors or fabricate motion.
Every native step checks robot XYZ/yaw, both active humans' base XYZ, simulation
time, legal track identity/visibility/validity/velocity/history fields, terminal
state, and all recorded scalar metrics. All measured differences are **zero**
for all three completed replays. Human joint poses were not recorded in the old
logs and are not independently compared; animation uses the native actors.
Failed or nonmatching replays are rejected before final video composition.

| Replay | Actions | Nav success | SPL | PSC | Human collision | Native time |
|---|---:|---:|---:|---:|---:|---:|
| WM success | 141 | 1 | 1 | 1 | 0 | 6.856 s |
| WM collision | 67 | 0 | 0 | .761194 | 1 | 3.256 s |
| Static collision | 88 | 0 | 0 | .602273 | 1 | 4.280 s |

The checkpoint is unchanged: seed 20261013, validation-selected 188k model,
SHA256 `6478820b4fb1a109d5ba62282064257be98dd3fb6f1a3db0195eb1eff03ec09e`.
Existing architecture/execution modules have no diff from accepted commit
`81d03e4840db4e2604f1fd1ce038347f8abaa5a8`. Headless evaluation stays available
through the existing launcher. Rendering and inference are separated by the
immutable recorded-action/telemetry interface.

`replay_validation_summary.json`, per-replay `validation.json`, and
`playability.json` contain exact errors, scalar metrics, action hashes, codec
metadata, video hashes, and successful full-decode checks. Compact evidence is
committed under `research_results/eventwm_native_visualization_20261012/`.
Videos, RGB frame caches, graphics binaries, and shader caches remain outside
Git. Representative images were inspected for native geometry, moving humans,
robot visibility, readable labels, action divergence, and terminal outcomes.

C24 stayed alive with PID 3296093, cap 24, source
`eac1c27813a753d26fc1756671d8d0bd198a65c6`, and unchanged pipeline settings and
all five protected source fingerprints. A final read-only sample contained
217,965 retained queries and 6,605 completed roots. All replay/encoding work ran
at nice 19, idle I/O priority, one CPU thread and one simulator at a time, with
the production publication guard. No collector control API was called.

## Reproduce only these three videos

Use a new output directory: validated existing replays are preserved. The
graphics package and kernel assertion intentionally target the current
workstation's matching 595.91.07 runtime.

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_closed_loop_checkout
VIZ_OUT=/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_native_visualization_reproduction_20261012
scripts/run_eventwm_native_visualization.sh prepare --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh diagnose --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh replay --output "$VIZ_OUT" --policy eventwm --tag ep001_YRUkbU5xsYj_6
scripts/run_eventwm_native_visualization.sh replay --output "$VIZ_OUT" --policy eventwm --tag ep004_bCPU9suPUw9_10
scripts/run_eventwm_native_visualization.sh replay --output "$VIZ_OUT" --policy static --tag ep004_bCPU9suPUw9_10
scripts/run_eventwm_native_visualization.sh compose --output "$VIZ_OUT"
scripts/run_eventwm_native_visualization.sh verify --output "$VIZ_OUT"
```

No fresh model inference or benchmark expansion is required. Recompute overlays
from existing validated RGB frames with the default-output `compose` command;
that does not run more navigation episodes.

```bash
scripts/run_eventwm_native_visualization.sh diagnose --system-egl
scripts/run_eventwm_native_visualization.sh diagnose --cuda-visible 0,1
scripts/run_eventwm_native_visualization.sh status
scripts/run_eventwm_native_visualization.sh stop
```

`stop` touches only the selected visualization output's `STOP_RESEARCH` file.
Replay/overlay generation checks it at safe boundaries. Existing partial frame
captures are diagnostic only; completed validation is required for composition.
To reproduce after an interruption, use a fresh output directory and the same
three fixed source episodes. Do not launch concurrent simulator jobs.
