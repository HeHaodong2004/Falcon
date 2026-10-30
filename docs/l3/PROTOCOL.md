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

## Imagination fidelity options (amended 2026-10-30 04:00)
Legal (may be used by a primary-benchmark selector): `perceive.VisibleWorld` -- the imagined planner perceives the
people of a Stage 1 world only through the robot's own camera model (90 deg / 10 m, line of sight on the legal or
predicted map, the tracker's 2 s memory) from the imagined pose; `perceive.Discovery` with a PREDICTED truth map
(`perceive.predicted_truth`: the legal map plus Layer 2 / Stage 2 predicted cells for the imagined paths;
`L3_MW_DISCOVER=1`); continuation fidelity (`SANDBOX_CONT_BOUNDARY` / `SANDBOX_CONT_INHERIT`); `L3_MW_RECT` /
`L3_MW_CONTACT_STEPS` (scoring rules).  Analysis only (privileged): every run with `L3_ORACLE_BANK` (the people's true
future), `fidelity.py` (exact branch-simulator rollouts, real end-of-window maps as truth, shifted human tapes) and
`rolloutpolicy.py` (incl. `--shifts`).  Released components default to OFF for all of these.

## Stage 2 map-stream options (amended 2026-10-30 16:00)
Legal, default off, read by `heuristics._discovery_rollouts` (selector `mwroll` with `L3_MW_DISCOVER=1|union`), the
closed-loop Sandbox map loop and the probe's `camS2` variants: `L3_S2_PB` / `L3_S2_PF` -- reveal a predicted cell as
BLOCKED / FREE only if Stage 2's probability that the legal camera certifies it so along the path exceeds the
threshold (`perceive.s2_states`; unset = Stage 2's own states, i.e. its class rule 0.575 / 0.475; `L3_S2_PF=1.01`
reveals no predicted FREE cell); `L3_S2_REVEAL=observer` -- predicted cells are certified as the legal sensor certifies
(`perceive.ObserverReveal`: FREE halo inside the 90 deg frustum at 1.0-9.6 m with line of sight, BLOCKED through the
35 wall rays to 10 m and the 0.25 m band around each hit, the jaw camera's mount; parameters fitted on TRAIN legal
map histories, `l3/revealfit.py`) instead of `Discovery`'s 5 m frustum; `L3_S2_ROUNDS=n` -- Stage 2 again on the paths
of the previous pass (no fidelity gain measured).  `L3_S2_OBS` overrides the observer distances (analysis).

## Stage 2 checkpoint option (amended 2026-10-30 20:00)
Legal, default off: `L3_S2_CKPT=<path>` (`layer2/model.py`, `Layer2WorldModel.from_release`) makes Layer 2 load that
Stage 2 checkpoint instead of the release's -- same `LegalVoxelPredictor` interface, legal inputs and candidate domain;
the checkpoint's own SHA-256 replaces the manifest's in `release['stage2']` (with `override='L3_S2_CKPT'`), is written
to every Sandbox server's ready file (`server<i>.ready`, key `stage2`) and is carried by every prediction's
provenance.  A Sandbox server reads the option when it starts, so set it in the environment of the `evaluate` /
`fleet` command (`fleet` passes its environment to the servers).  The released checkpoint and the defaults are
unchanged.  Fine-tuned checkpoints for the Sandbox's imagined map stream (`l3/s2ft.py`) are trained on TRAIN episodes
only (`l3/s2data.py`: floor decisions, every candidate's exact real branch as the label, privileged for supervision
only); their operating point (the class rule's BLOCKED threshold) is chosen on scene-disjoint TRAIN validation and
stored in the checkpoint, so `L3_S2_PB` / `L3_S2_PF` unset use it and `L3_S2_PF=1.01` gives its walls-only stream.
DEV is used once, for the frozen checkpoint's map metrics and fidelity probe.

## People forecaster options (amended 2026-10-30 19:40)
Legal, default off (`docs/l3/PEOPLE_FORECASTER.md`): `L3_MW_BANK=fc1|fck` replaces the constant-velocity people bank
with the map-aware multimodal forecaster of the TRACKED people (`l3/peoplemap.py`; model `L3_FC_MODEL`, default
`output/agents/people/fcmap/v1`).  Its inputs are the robot's legal track list, legal track log, legal map and clock.
Tracks without a detection in the last 2 s keep constant velocity, and untracked people do not exist in the bank.
`fc1` is one world: the weighted mean of the modes, chosen on TRAIN (`L3_FC1=top` gives the top mode).  `fck` gives
the `L3_MW_WORLDS` most probable joint mode combinations with their weights; `L3_FC_MINW` drops light ones.
`L3_FC_LOG=<jsonl>` is an analysis log.

The model was trained on TRAIN replays only: the `output/runs/R10` spool, 131 TRAIN scenes.  Model selection and the
`fc1` choice used a scene-grouped TRAIN validation split.  Labels come from the recorded tape, matched by the track's
last detection; the older estimate-matched collection is used only for tracks detected at most 0.25 s ago.  The
point-forecast NLL calibration of the CV / Stage 1 baselines was fitted on TRAIN.  DEV data (the DEV state bank and
replays of the DEV floor run, `output/agents/people/dev_floor`) was used only for the final evaluation of the frozen
model.
