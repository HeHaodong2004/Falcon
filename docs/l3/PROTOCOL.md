# Layer 3 evaluation protocol (declared 2026-10-29, before any Layer 3 training or evaluation)

## Data separation (`docs/l3/suites.json`, built by `eventwm_graph/l3/suites.py`, sha256 `fefe3414...`)

| suite | content | use |
|---|---|---|
| `train` | 4,028 episodes from 557 Social-HM3D / Social-MP3D **train** scenes (<= 6 per HM3D scene, <= 20 per MP3D scene, human_num >= 2, salted-hash choice) | RL training |
| `train_querygen` | 334 querygen unseen-suite episodes (train split; dev scenes excluded) with v14 Oracle solutions | Oracle-guided warm-up, RL training |
| `dev` | 72 episodes from 66 train scenes that no querygen suite used (60 HM3D, 6 reserved MP3D; never in training) | model selection, ablations, all development numbers |
| `test` | 275 episodes of the official **val** splits (3 per Social-HM3D val scene, 12 per Social-MP3D val scene) | final held-out configuration matrix, run once after the freeze |
| `test_full` | all 1,404 val episodes (1,087 HM3D + 317 MP3D) | final headline numbers for the official configurations |

* The Falcon paper trains on `train` and tests on `val` (`dataset_split_explanation.txt`); `test` / `test_full` are that
  official test data. Social-HM3D `minival` scenes are excluded from training and dev.
* The querygen generator was developed on the querygen DEV suite (28 HM3D val episodes); final results are also
  reported without the 28 querygen-DEV scenes.
* **`test` and `test_full` are never run before the Layer 3 freeze.** Every selection decision (checkpoint, ablation,
  hyper-parameter) uses `dev` (or training statistics) only. Every evaluation run is listed in the report.

## Metrics and rules
Official Falcon definitions (`docs/querygen/OFFICIAL_RULES.md`): success = STOP within 0.2 m (geodesic), <= 500
steps, the episode ends at the first official human contact; SPL; PSC (share of steps with every human >= 1 m);
human-collision rate.  Every suite episode is in the denominator (no infeasibility exclusions for Layer 3 numbers;
the Oracle's certified-infeasible episodes are listed separately).  Evaluation runs use the exact branch simulator
(the `habitat.Env.step` mirror); the decision sequences are re-executed with the unchanged `habitat.Env.step`
(`evaluate.py replay`) and official replay results are the numbers of record for the official configurations.

## Configurations
Official benchmark configurations use the native discrete Falcon action set (dynamics `native`): ORCA, DWA (and SFM)
planners with the K=10 / K=6 candidate sets.  Every other dynamics family (`slowturn`, `inertia`, `slowturn_inertia`,
`heavy_brake`, `slowturn_heavy`) uses the continuous base-velocity action: **modified robot dynamics, outside the
official benchmark**, reported separately.  Held-out (never trained) combinations are declared in `docs/l3/REPORT.md`
before the first training run.

## Observation legality (amended 2026-10-29 19:00 at the user's direction)
Primary benchmark = the legal 90-degree jaw camera (people and static map certified only inside its frustum with
occlusion), the robot's own legal history, and Layer 2 predictions from legal inputs.  Not allowed in the primary
benchmark: privileged future information, hidden scene geometry / navmesh, unrestricted 360-degree sensing
(`QG_HUMAN_360` runs are analysis only).  The Oracle and every privileged quantity (counterfactual labels, the human
tape) are training supervision or analysis only, never policy input.  Layer 3 may ACT to observe (scan queries:
in-place rotations); what it then sees is legal.  Milestone dev evaluations: `scripts/l3_milestone.sh RUN VERSION
LAYOUT WORKERS [evaluate.py args]` (all 14 configurations; selection by mean SR over the 7 seen ones; margin and value
weights are dev-selected settings of the same checkpoint).

## Execution-loop option (amended 2026-10-29 21:10)
From R10 on, every Layer 3 process -- actors, labellers, evaluations AND the official replay of Layer 3 episodes --
runs with `L3_INHERIT=1` (`QueryExecution.INHERIT`): a goal query issued while a goal query is running inherits the
running execution's stall / unstick / stuck-STOP history (measured: without it, interrupting a running goal query
cost 4-5.5 SR points on dev).  The option is part of the Layer 3 interface (how a selection is applied), identical in
the branch runtime, the Sandbox imagination and the official replay; released components default to OFF.  Oracle
solution replays keep it OFF (their sequences were found without it).
