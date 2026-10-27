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
