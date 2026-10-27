# Query Generator + privileged Oracle on native Falcon — final report

(Draft: sections are filled from the run artefacts listed under each heading; numbers are
copied from `report.py` / study summaries, never typed by hand.)

## 1. Verdict
TBD after the frozen HOLDOUT run (protocol: `docs/querygen/FINAL_PROTOCOL.md`).

## 2. What was built (deliverables 1-4)
- Native harness and legality: `native.py`, `legal.py` (see PROGRESS.md "Verified facts").
- Planners: full-responsibility ORCA (RVO2 port, `orca.py`, verified by `tests/querygen/test_orca.py`),
  SFM, DWA with exact-dynamics rollouts, path follower (`planners.py`); body-aware static lattice
  (`lattice.py`).
- Dynamics families + calibration (`dynamics.py`, `calibrate.py`, `output/runs/calibration`).
- Query generator (`generator.py`), executor semantics (`executor.py`), Oracle (`oracle.py`),
  authoritative replay (`episode.replay`), evaluation driver (`evaluate.py`).

## 3. Replay evidence (deliverable 5)
Per-episode result JSONs hold the replay block (official metrics, identical actions, shadow pose
mismatch count, static/escape/overlap steps, trace).

## 4. Iteration report (deliverable 6)
DEV v0 -> v3 -> v4/v5 with failure taxonomy (PROGRESS.md).

## 5. Diversity / runtime / sandbox (deliverable 7)

## 6. Untouched HOLDOUT (deliverable 8)

## 7. Commits and reproduction (deliverable 9)

## 8. Readiness for Layer 3 (deliverable 10)

## Appendix A. Reproduction commands
All commands run from the worktree root with
`PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python`, `CUDA_VISIBLE_DEVICES=1`, `nice -n 10`
(Habitat runs headless on CPU; `output -> /var/tmp/falcon-querygen-claude`).

```bash
# native equivalence / unit checks
$PY scripts/querygen_run.py tests.querygen.native_equivalence
$PY scripts/querygen_run.py tests.querygen.test_orca
$PY scripts/querygen_run.py tests.querygen.test_velocity_action
$PY scripts/querygen_run.py tests.querygen.test_canonical_contacts
$PY scripts/querygen_run.py tests.querygen.test_sandbox_score
# dynamics calibration
$PY scripts/querygen_run.py eventwm_graph.querygen.calibrate --out output/runs/calibration
# Oracle + replay on a suite (frozen code snapshot per run, deterministic edge budget)
$PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite dev --out output/runs/<run> --workers 9 \
    --oracle '{"max_edges": 6000, "max_seconds": 5400, "block_weight": 1.0}'
# executor-only baseline (goal query only)
$PY scripts/querygen_run.py eventwm_graph.querygen.evaluate suite dev --policy executor --planner orca \
    --dynamics native --out output/runs/<run> --workers 4
# certificates (privileged diagnostics)
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --depth 60          # start escape
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --goal              # goal region
$PY scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR --progress 0.3 \
    --exact-gait --max-depth 500 --budget 14000                                                   # progress
# studies on a run's PASS records
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout diversity output/runs/<run> --out DIR
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout sandbox_study output/runs/<run> --out DIR
$PY scripts/querygen_run.py eventwm_graph.querygen.fanout l3_interface output/runs/<run> --out DIR
# metrics
$PY scripts/querygen_run.py eventwm_graph.querygen.report output/runs/<run> --cert output/runs/feasibility_v2
```
