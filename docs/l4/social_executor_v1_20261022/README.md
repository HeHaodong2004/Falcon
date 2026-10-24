# L4 native social executor: reproduction and handoff

> Historical conservative release. Current progress-first implementation and
> contract: [2026-10-23](../progress_first_20261023/README.md). Results below are
> preserved historical measurements, not the current executor's behavior.

This branch delivers a runnable non-learning executor and its research evidence.
It is **not approved for L3 RL training**. Read [FINAL_REPORT.md](FINAL_REPORT.md)
for the acceptance decisions, [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for
matched native results, and [FAILURE_LOG.md](FAILURE_LOG.md) for failed attempts.

Worktree: `/research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022`.
Branch: `research/l4-social-executor-20261022`.
Starting source: `cf1971228c37517202cc9e5787f785a3876a0514`.
Frozen executor evaluation source: `e19431fb61b0c525d5d5711329052d546c92c54b`.
The later delivery commit adds reports, diagnostics and artifacts. B3 was
evaluated at `a2155778`; its Falcon policy implementation is identical in the
release source, and its original complete source receipt is retained.

## Environment and assets

Use `/research-user/miniforge3/envs/eventwm-falcon/bin/python` (Python 3.9.19).
The pinned environment contains Habitat-Sim 0.3.1, Torch 1.12.1, NumPy 1.23.5,
Numba 0.59.1, Gym 0.23.0 and Matplotlib 3.3.2. `habitat_sim` must be imported
before Habitat on this installation; the runners do so. No model training or
package upgrades are needed.

The checkpoint is
`/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt`.
Its SHA256 is
`215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`.
The loaded version is `l2-intermittent-1-fc2c500cda1b243c7526`. The checkpoint
and the source branch are unchanged.

The worktree's `data` link points to
`/research-user/workspace/projects/eventwm-falcon/data`, and `output/stage0_benchmark`
points to the primary checkout's corresponding directory. These assets are
read only. Protocol JSON files record exact scene, dataset, certified-map and
navmesh paths/hashes. All new writable outputs stay in this worktree. The
28-worker collector remains running; do not change it to reproduce these runs.

## Commands

Use new output destinations. The runners refuse existing ones.

```bash
cd /research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022
source scripts/l4_env.sh

# Protocol and real-checkpoint adapter tests: 19 tests.
"$L4_PY" -m unittest discover -s tests/l4 -v

# Repeat the pre-development native/L2 gate.
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_baseline_smoke.py --output output/l4_gate_replay
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_encounter_smoke.py --output output/l4_encounter_replay

# Frozen analytic fixtures and deterministic speed variations.
nice -n 19 "$L4_PY" -m eventwm_graph.l4.scenarios --output output/l4_analytic_replay --randomized 64

# Native scenarios, followed by full episode execution on the public panel.
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/social_executor_v1_20261022/native_scenarios.json \
  --variants B0,B1,B2 --output output/l4_scenarios_replay
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/social_executor_v1_20261022/navigation_public12.json \
  --variants B0,B1,B2 --output output/l4_navigation_replay

# Warm control profiling and the retained static-contact regression.
nice -n 19 "$L4_PY" scripts/l4_profile.py --output output/l4_profile_replay
nice -n 19 "$L4_PY" scripts/l4_static_regression.py \
  --witness docs/l4/social_executor_v1_20261022/artifacts/static_contact_witness.json \
  --output output/l4_contact_replay.json
```

Run B3 in a **fresh process**. The nonvisual adapter intentionally removes
pixel sensors process-wide, so it cannot precede Falcon in the same interpreter.
The machine currently has system NVIDIA libraries from 595.99 with a 595.91.07
kernel module. The already cached matching libraries restore rendering locally:

```bash
export LD_LIBRARY_PATH=/research-user/workspace/projects/eventwm-falcon/output/eventwm_runtime/nvidia-595.91.07/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=/research-user/workspace/projects/eventwm-falcon/output/eventwm_runtime/nvidia-595.91.07/usr/share/glvnd/egl_vendor.d/10_nvidia.json
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/social_executor_v1_20261022/navigation_public12.json \
  --variants B3 --output output/l4_falcon_replay
```

Falcon uses the existing `falcon_noaux_25.pth`, CPU policy inference and native
depth rendering. Its absent auxiliary training heads are removed from the
checkpoint configuration before strict weight loading. No policy weights,
native actions, human controllers or metric definitions are changed.

```bash
"$L4_PY" scripts/l4_summarize.py \
  --protocol docs/l4/social_executor_v1_20261022/navigation_public12.json \
  --roots output/l4_navigation_replay output/l4_falcon_replay \
  --output output/l4_navigation_summary_replay.json
"$L4_PY" scripts/l4_audit_traces.py \
  --roots output/l4_scenarios_replay output/l4_navigation_replay output/l4_falcon_replay \
  --output output/l4_trace_audit_replay.json
nice -n 19 "$L4_PY" scripts/l4_visualize.py \
  --scenarios output/l4_scenarios_replay --analytic output/l4_analytic_replay \
  --output output/l4_figures_replay
```

For true JIT-cold profiling, set `NUMBA_CACHE_DIR` to a new directory inside
your output tree before launching the profiler. Warm workers before repeated
control. The reported mesh fallback tails remain a limitation even after warmup.

## Runtime interface

The canonical package is `eventwm_graph.l4`. The complete contract is
[EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md), version
`eventwm.l25l4.contract.v1`. L2.5 should consume this compiler and its immutable
reference hash. L4 requires no L2.5 service in its action loop.

```python
from eventwm_graph.l4 import QuerySpec, compile_reference, Executor, RuntimeContext

query = QuerySpec(
    query_id="selected-query-17",
    reference_route=selected_ordered_world_xyz_route,
    start_pose=actual_packet["robot_pose"],
    terminal_orientation=selected_terminal_yaw,
    timestamp=actual_packet["timestamp"],
    execution_limit=480,
    duration_limit=26.88,
)
reference = compile_reference(query, supplied_legal_static_geometry)
executor = Executor()
handle = executor.start(query, reference, RuntimeContext(supplied_legal_static_geometry))
decision = executor.act(handle, actual_packet, actual_packet, forecast_view=None)
# If action is None, inspect status and hand control back; do not invent a wait.
# Otherwise execute exactly one native action and acknowledge the actual result:
feedback = executor.on_transition(handle, {
    "action": decision.action,
    "timestamp": next_actual_packet["timestamp"],
    "robot_pose": next_actual_packet["robot_pose"],
    "observation_ref": durable_actual_observation_id,
    "human_collision": actual_human_collision,
    "static_collision": actual_static_contact,
    "episode_termination": actual_terminated,
    "episode_truncation": actual_truncated,
})
```

This example assumes a legal ordered query and supplied geometry. Static
geometry adapters are `StaticGrid.from_partial` and the explicitly privileged
`StaticGrid.from_navigation`. The native runner additionally installs the
verified ego-motion model and optional mesh/FK fallback. Production integration
must preserve that boundary and observe articulation when using the fallback.
The unified actual-observation owner maintains L2 belief; `ForecastCache` merely
shares short-horizon forecasts. `CompiledReference.l2_segment()` exposes nominal
segments to L2/L2.5 without inventing actual observations.

## Evidence locations

The output root is
`/research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022/output/l4_social_executor_v1_20261022/`.
Final evidence is in `release_scenarios/`, `release_navigation/`, `final_falcon/`,
`release_analytic/`, `release_profile/`, `profile_cold/` and `figures/`.
Each native case has `case.json`, `reference.json` where supported, `actual.json`
and `report.json`. Receipts bind protocol/source hashes and verify source
stability. Complete attempted runs and their failures are retained separately.
Committed `artifacts/` holds summaries, receipts, per-case tables, regression
evidence, figures and a SHA256 manifest of full native result files.

The [native trajectory panel](artifacts/figures/native_scenarios.png) shows
head-on avoidance, crossing/yielding, a blocked narrow passage, multiple humans,
static escape/rejoining and limited visibility with PSC events. White cells
denote certified grid support; gray cells are unsupported by that grid, not a
literal wall raster. The separate mesh fallback may support some gray regions.
Human dots are actual legal measurements, with timestamp-dependent color.

The [L2 comparison](artifacts/figures/l2_comparison.png) overlays a sampled
prediction from the first supported bank, not ground truth. The
[head-on animation](artifacts/figures/head_on_comparison.gif) displays accumulated
legal measurements and actual robot paths, not a rendered camera video. The
[reobservation diagnostic](artifacts/figures/reobservation_analytic.png) is
explicitly analytic and shows the configured 0.85 s observation gap.

No production/main merge was performed. L2.5's working tree was not modified.
