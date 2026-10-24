# Baseline verification

PASS for runnable native execution and frozen L2 inference, verified before navigator development on 2026-10-22. Source and initially clean worktree: `cf1971228c37517202cc9e5787f785a3876a0514` on `research/l4-social-executor-20261022`. Original L2 worktree was clean. No collector, training, checkpoint, confirmation-data or L2.5 changes.

Interpreter: `/research-user/miniforge3/envs/eventwm-falcon/bin/python`. Imports use this checkout through `PYTHONPATH=.:habitat-lab:habitat-baselines`; numerical threads are one, research jobs run at nice 19 / idle I/O priority. Shared scene data and certified Stage 0 assets are read only; writable output is local.

Checkpoint: `/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt`. SHA256 `215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`. Loaded model: `l2-intermittent-1-fc2c500cda1b243c7526`.

Native evidence: `output/l4_social_executor_v1_20261022/baseline_smoke.json` and `encounter_smoke.json`. The MP3D encounter executed an existing native orientation query, turns and translations, acquired 9 visible-human measurements, initialized L2 from the legal history and obtained a [16, 41, 1, 2] joint future. Positions are world X,Z; timestamps are absolute native seconds. Forecasts remain distinct from actual observations. The 1.92-second/16-world prediction took 35.15 ms in that loaded process. STOP demonstrably terminated both episodes.

The initial HM3D prefix executed 131 actions. Its nominal 48 ms clock is not exact: 121 transitions took 48 ms and 10 took 56 ms. Unobstructed translation was 0.208333 m; yaw increments are 0.083333 rad. Actual timestamps are authoritative. STOP also advances the clock. Control integration and physics time are distinct.

The first attempt rejected the absence of a partial-graph query; its log is retained. The repeated inherited prefix produced zero visible humans and 43 static-contact frames. This is protocol evidence, not safe navigation. We exercised the existing certified-map orientation compiler and a separately declared head-on encounter without modifying the environment. Full-body support is stricter than center-path certification.

Authority: `eventwm_graph/stage1/runtime.py`, `partial/observer.py`, `partial/map.py`, `partial/query.py`, `benchmark/motion.py`, `falcon/additional_action.py`, `falcon/additional_metric.py`, `l2_intermittent/model.py`. Configured robot actions: forward/left/right/stop; pause and backward source classes are disabled. Episode limit: 500 actions. Human collision and STOP terminate; static contact alone does not.

Reproduce: `source scripts/l4_env.sh`; run `$L4_PY scripts/l4_baseline_smoke.py --output output/l4_baseline_replay` and `$L4_PY scripts/l4_encounter_smoke.py --output output/l4_encounter_replay`. Fresh destinations are required. The nonfatal HM3D semantic-descriptor warning did not prevent physics, sensing or execution.
