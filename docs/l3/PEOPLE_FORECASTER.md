# Map-aware multimodal people forecaster for the Sandbox (Layer 2 v2 people module, 2026-10-30)

Status: built, trained on TRAIN, evaluated offline on DEV, wired into the Sandbox selector as `L3_MW_BANK=fc1|fck`
and smoke-run on 2 DEV episodes per option.  Every runtime input is legal: the robot's legal track list, its legal
track log and its legal map.  The recorded people tape gives training labels and evaluation only.  Default off:
released checkpoints, defaults and the best legal configuration are unchanged.

Code: `eventwm_graph/l3/peoplemap.py` (collector, model, training, calibration, DEV evaluation, `FCBank`).  `env.py`
adds `L3_MW_BANK=fc1|fck`.  `sbeval.py` adds the variants `cl_frozen_fc1` / `cl_frozen_fck`, the analysis variant
`cl_frozen_truetrk` and `--only-tracked`.  Final model: `output/agents/people/fcmap/v1`.

**Summary.**
* **Model.** `v1` is a 'small' CNN on a 6.4 m patch of the legal map plus an MLP on the 2 s track history.  It
  predicts 6 weighted trajectory modes to 3.07 s.  It was trained on 37,232 TRAIN instances (108 training + 15
  validation scenes; checkpoint by validation NLL).
* **Offline accuracy on DEV** (6,538 detection-labelled tracked instances).  Against constant velocity, today's legal
  bank:
  - 1.5 s median / p90 error of the weighted mean 0.97 / 2.42 m vs 1.25 / 3.00 m;
  - 3 s 1.87 / 3.96 m vs 2.67 / 5.10 m;
  - turn-or-stop subset at 1.5 s 0.93 vs 1.52 m;
  - NLL 1.87 vs 2.75 at 1.5 s.

  All of these are significant under an episode-clustered bootstrap.  It is level with Stage 1's mean world and the
  history-only MDN v2 in point error, and better calibrated than MDN v2 (NLL -0.35 at 1.5 s).
* **The map** adds almost nothing.  The robot's legal map is 82% unknown around the people it tracks.
* **Closed loop.**  In the Sandbox (116 DEV states), the learned bank does not significantly improve contact
  prediction:
  - collision AUC at the 1.54 s window: fc1 0.576, fck 0.611, CV 0.596, TRUE tracked futures 0.769;
  - the choice of the outcome rule barely moves, even with TRUE tracked futures.

  The lead's privileged ceiling run with TRUE tracked futures at 1.5 s reaches 59.7% SR vs 61.1% for CV (3 / 4 paired
  wins / losses).
* **Integration.**  Both options are legal: the bank reads only the legal tracks, track log, map and clock.  The bank
  costs 5 ms per decision.  `fck` costs about 3.7x CV's imagination when people are tracked.  Both ran 2 DEV episodes
  end to end, and the 11 Sandbox tests pass.
* **Recommendation** (section 7).  Do not expect an SR gain from tracked-people forecasting at 1.5 s.  If one SR run
  is spent here, make it `fck` with `L3_MW_WORLDS=6`, not `fc1`.

## 1. Why

The legal Sandbox imagines tracked people at constant velocity (`oraclebank.CVBank`).  At 1.5-3 s its large errors
come from people who turn or stop.  Stage 1's mean world is a better point forecast but did not raise SR.  A
history-only MDN (`peoplefc.py`, `output/peoplefc/v2`) is no better than the Stage 1 mean.  This module adds the
robot's LEGAL map around the person.  It offers the forecast to the multi-world selector either as one world (`fc1`)
or as K weighted worlds (`fck`).

## 2. Data (TRAIN only)

* Source: the R10 spool, i.e. episodes of the Layer 3 RL run (legal policies) on the TRAIN suite.  They are replayed
  in querygen's exact branch simulator with the legal sensor (the robot's recorded decisions; `collect`).  Every
  4 steps (0.19 s), every legal track with a detection in the last 2 s gives one instance: the legal inputs at t, plus
  the person's true positions over the next 64 steps (the tape: labels only).
* Label person: the true person nearest to the track's last detection, at the detection's own step of the tape
  (`--match detection`; match distance 0 m for every instance).  The first collection (`output/agents/people/data`,
  527 episodes) used the older estimate match: the person nearest to the constant-velocity estimate at t within
  0.75 m.  For tracks lost for a while, that match drops the people who turned or stopped while unseen.  Training
  therefore keeps from it only tracks detected at most 0.25 s ago, where both matches pick the same person
  (`fit --estimate-max-age 0.25`).
* Size: 1,176 episodes (527 + 649 in `output/agents/people/data2`).  They cover 131 TRAIN scenes: none of the 66 DEV
  scenes, none of the 10 minival scenes and no test (val) scene (checked against `docs/l3/suites.json`).  The 7
  configurations are orca / dwa x native / inertia / slowturn variants, k6 / k10.  There are 46,186 instances, of which
  43,909 have the full 3.07 s future.  The training set has 37,232 instances: 33,379 train and 3,853 validation, from
  108 / 15 scenes.
* What the legal map shows around a tracked person (TRAIN, 8,923 instances): 82% of the 9.6 m patch is UNKNOWN, and
  68% of the 3 m x 2 m box ahead of the person.  In 73% of the instances, more than half of that box is unknown.  The
  robot certifies the space it looks at, not the space the person walks into.

## 3. Model

* Inputs, in the person's heading frame (the tracker velocity if faster than 0.1 m/s; else the direction of the
  detection history; else world axes):
  - the last 2 s of detections (10 points every 0.2 s, relative to the last detection, with validity flags);
  - the tracker velocity and speed, the age of the last detection, the frame type and the share of valid points
    (38 numbers, MLP 38 -> 128 -> 128);
  - a 6.4 m x 6.4 m crop of the legal map around the last detection (64 x 64 cells of 0.1 m, UNKNOWN / FREE /
    BLOCKED one-hot), averaged to 0.2 m cells.  Map encoder `small` (`make_net(cnn_arch='small')`): three blocks of
    3 x 3 convolution + ReLU + 2 x 2 max-pool (16 / 32 / 32 channels) -> 4 x 4 x 32 -> 64 features, dropout 0.5.
    47 k of the model's 198 k parameters.
* Output: K = 6 modes.  Each mode is a trajectory of 17 knots every 4 steps (0 to 3.07 s), predicted as residuals on
  constant velocity.  Each knot has an isotropic spread (0.03 to 6 m).  One weight per mode (a mode is one coherent
  future).
* Loss: the negative log-likelihood of the recorded future under the trajectory mixture (averaged over the knots),
  plus 0.5 x the best-of-K ADE (keeps the modes apart).
* Training: AdamW (weight decay 1e-3), one-cycle LR 2e-3, 25 epochs, batch 256, dropout 0.2 on the joint features.
  The checkpoint is the epoch with the best validation NLL.  Augmentation uses exact symmetries only: a lateral mirror,
  plus random 90 deg turns when the frame has no heading.
* Validation: scene-grouped, 12% of the TRAIN scenes held out (`scene_split`).  The single world of `fc1` (the
  weighted mean of the modes or the top mode) is chosen on the same split, by the mean over 0.77 / 1.5 / 3 s of
  median + p90 error.  It is stored with the model.

Model selection on TRAIN (one data snapshot, 38,862 instances with the earlier mixed labels, the same
scene-grouped split for every variant: 34,102 / 4,760 instances, 100 / 14 scenes).  Val NLL is per knot (lower is
better).  Errors are median / p90 in m of the weighted mean of the modes, and best-of-6.  Seed noise of the val NLL is
about 0.006 (A vs A seed 1).

| variant | best epoch | val NLL | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|---|
| A history only | 21 | -0.632 | 0.33 / 1.25 | 0.75 / 2.05 | 1.63 / 3.65 | 0.27 / 0.96 | 0.61 / 1.75 | -0.62 / 0.68 / 1.80 |
| A, seed 1 | 21 | -0.638 | 0.33 / 1.25 | 0.74 / 2.05 | 1.63 / 3.65 | 0.28 / 0.97 | 0.61 / 1.76 | -0.64 / 0.66 / 1.79 |
| D 'v1' CNN (0.36 M map parameters), dropout 0.2 | 8 | -0.608 | 0.32 / 1.24 | 0.71 / 2.03 | 1.48 / 3.61 | 0.24 / 0.98 | 0.59 / 1.73 | -0.55 / 0.67 / 1.80 |
| D, seed 1 | 8 | -0.568 | 0.32 / 1.23 | 0.72 / 2.02 | 1.48 / 3.58 | 0.27 / 1.07 | 0.61 / 1.89 | -0.58 / 0.69 / 1.81 |
| C 'v1' CNN + ray descriptor, dropout 0.2 | 9 | -0.552 | 0.33 / 1.23 | 0.70 / 2.05 | 1.44 / 3.68 | 0.26 / 1.06 | 0.61 / 1.93 | -0.55 / 0.70 / 1.84 |
| E = C with dropout 0.3, weight decay 1e-3 | 8 | -0.564 | 0.33 / 1.26 | 0.74 / 2.02 | 1.55 / 3.66 | 0.27 / 1.12 | 0.67 / 1.92 | -0.55 / 0.68 / 1.82 |
| G 'v1' CNN, crop 6.4 m, dropout 0.2 | 9 | -0.555 | 0.33 / 1.24 | 0.73 / 2.02 | 1.51 / 3.60 | 0.26 / 1.13 | 0.63 / 1.92 | -0.57 / 0.72 / 1.82 |
| S1 'small' CNN, crop 6.4 m, map dropout 0.5 | 21 | -0.635 | 0.33 / 1.25 | 0.75 / 2.05 | 1.63 / 3.62 | 0.28 / 0.97 | 0.61 / 1.74 | -0.62 / 0.67 / 1.80 |
| S2 'small' CNN, crop 9.6 m, map dropout 0.5 | 20 | -0.633 | 0.33 / 1.24 | 0.75 / 2.05 | 1.64 / 3.64 | 0.27 / 0.97 | 0.61 / 1.74 | -0.63 / 0.67 / 1.79 |
| **S4** = S1 + dropout 0.2, weight decay 1e-3 | 14 | **-0.646** | 0.33 / 1.23 | 0.72 / 2.03 | 1.50 / 3.60 | 0.26 / 0.95 | 0.61 / 1.73 | -0.62 / 0.65 / 1.77 |

* The 'v1' CNN memorises the training scenes.  Its training NLL keeps falling (to -1.1) while the validation NLL is best
  at epoch 8-9 and then degrades to -0.18.  It improves the 3 s point error but not the likelihood.
* The low-capacity 'small' encoder does not overfit.  With dropout and weight decay (S4) it is the best variant, so S4
  is the recipe of the final model.  Its margin over history only (+0.008 to +0.014) is 1-2 seed-noise units.

Final models (all TRAIN data with consistent labels; the same split for all: 33,379 / 3,853 instances, 108 / 15
scenes).  The ablation `v1_hist` has v1's regularisation without the map.  CV at 1.5 s: 0.75 / 2.54, at 3 s:
1.97 / 4.76.

| model | best epoch | val NLL | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|---|
| **v1** (final): 'small' CNN on the map + MLP on the track | 17 | -0.775 | 0.30 / 1.28 | 0.72 / 2.10 | 1.49 / 3.67 | 0.23 / 0.93 | 0.54 / 1.83 | -0.81 / 0.59 / 1.77 |
| v1_hist: no map | 22 | -0.777 | 0.32 / 1.28 | 0.78 / 2.13 | 1.54 / 3.78 | 0.23 / 0.92 | 0.56 / 1.88 | -0.79 / 0.60 / 1.81 |
| v1_cnnbig: the 'v1' CNN | 8 | -0.751 | 0.33 / 1.28 | 0.76 / 2.12 | 1.52 / 3.75 | 0.24 / 0.92 | 0.56 / 1.85 | -0.74 / 0.63 / 1.81 |

* On TRAIN-val the map does not change the trajectory likelihood (-0.775 vs -0.777).  It lowers the error of the
  weighted mean a little at every horizon (1.5 s median 0.72 vs 0.78 m; 3 s p90 3.67 vs 3.78 m).
* Turn or stop subset (TRAIN-val): at 1.5 s the weighted mean errs 0.86 / 2.38 m, against CV 1.36 / 2.59 m; at 3 s
  1.74 / 3.43 m against CV 3.13 / 5.38 m.
* At 0.77 s, CV has the lower median error (0.18 vs 0.30 m), but the higher p90 (1.50 vs 1.28 m).  The modes hedge
  toward stopping, turning and starting.  That pulls the weighted mean off the many straight walkers and standing
  people.
* `fc1` world (chosen on TRAIN-val by the mean over the horizons of median + p90): the weighted mean (3.19 vs top
  mode 3.43).
* Mode weights are flat: on the DEV instances the median top weight is 0.26 (uniform: 0.17), with entropy 1.74 of a
  possible 1.79.  The modes are distinct, 0.96 m from their centroid at 1.5 s, with a spread of 0.39 m each.  From
  legal inputs, the model cannot tell which way a person will go.

## 4. Offline evaluation on DEV (frozen model)

Two DEV sets, final evaluation only.  The model, its `fc1` world and the point-forecast calibration were frozen
on TRAIN before any DEV number was computed.

* **DEV floor replays** (primary; `output/agents/people/dev_floor`).  These are the default-policy floor run on the
  DEV suite (`output/eval/floor_dev24`, the 216 native-dynamics k10 episodes: orca / dwa / sfm), replayed with the
  TRAIN collector and labelled by the detection match.  6,538 tracked instances from 3,662 states and 57 DEV episodes
  with tracked people.  Stage 1 is not available on these states.
* **DEV state bank** (`output/statebank/orca_dev`, ORCA; Stage 1's banked worlds).  171 tracked instances from 104
  states.  They are labelled by the estimate match, because the bank's truth starts at t: lost tracks of people who
  turned are missing.

Errors are median / p90 in m.  "mean" is the weighted mean of the modes (the `fc1` world), "top" the most probable
mode.  "NLL" is the per-horizon marginal NLL of the 2-D position.  CV and Stage 1 are point forecasts; they get an NLL
from a 3-component isotropic Gaussian scale mixture around the point, fitted on TRAIN (`calib`: CV on the TRAIN
collection, Stage 1 on the TRAIN state bank's 159 tracked instances; spreads floored at the forecaster's 0.03 m).
The subsets are defined on the TRUE future (`maneuvers`).  'turn_stop': moving faster than 0.5 m/s, then turning more
than 45 deg or stopping (less than 0.5 m covered between 1.5 and 3 s).  'near': within 2.5 m of the robot's position
at t within 1.5 s.

**DEV floor replays (6,538 instances).**

all, n = 6538:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.46 / 1.58 | 0.97 / 2.42 | 1.87 / 3.96 | 1.19 / 2.77 | 0.40 / 1.08 | 0.78 / 1.86 | 0.46 / 1.87 / 3.08 |
| v1_hist (no map) | 0.46 / 1.56 | 0.99 / 2.43 | 1.88 / 3.98 | 1.14 / 2.71 | 0.38 / 1.10 | 0.76 / 1.93 | 0.47 / 1.86 / 3.08 |
| v1_cnnbig | 0.47 / 1.61 | 0.99 / 2.44 | 1.86 / 3.99 | 1.06 / 2.63 | 0.39 / 1.11 | 0.79 / 1.92 | 0.49 / 1.91 / 3.12 |
| MDN v2 (history only) | 0.45 / 1.66 | 0.95 / 2.52 | 1.90 / 3.98 | 1.07 / 2.67 | 0.41 / 1.24 | 0.82 / 2.05 | 1.03 / 2.21 / 3.09 |
| CV (today) | 0.50 / 1.97 | 1.25 / 3.00 | 2.67 / 5.10 | - | - | - | 1.46 / 2.75 / 3.95 |

turn_stop, n = 2058:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.57 / 1.77 | 0.93 / 2.53 | 1.67 / 3.36 | 1.30 / 2.77 | 0.42 / 1.12 | 0.81 / 1.59 | 1.51 / 2.68 / 3.82 |
| v1_hist (no map) | 0.59 / 1.80 | 0.99 / 2.56 | 1.58 / 3.29 | 1.19 / 2.76 | 0.41 / 1.14 | 0.78 / 1.65 | 1.49 / 2.64 / 3.83 |
| v1_cnnbig | 0.63 / 1.87 | 1.03 / 2.50 | 1.63 / 3.34 | 1.09 / 2.70 | 0.42 / 1.13 | 0.79 / 1.66 | 1.50 / 2.68 / 3.85 |
| MDN v2 (history only) | 0.53 / 1.97 | 0.93 / 2.70 | 1.66 / 3.43 | 1.11 / 2.77 | 0.46 / 1.32 | 0.87 / 1.77 | 2.45 / 3.13 / 3.91 |
| CV (today) | 0.72 / 2.18 | 1.52 / 3.00 | 3.08 / 5.26 | - | - | - | 2.65 / 3.77 / 5.09 |

near, n = 3348:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.46 / 1.40 | 0.90 / 2.32 | 1.69 / 3.91 | 1.12 / 2.68 | 0.37 / 1.08 | 0.73 / 1.87 | 0.35 / 1.75 / 3.00 |
| v1_hist (no map) | 0.48 / 1.42 | 0.94 / 2.33 | 1.69 / 3.97 | 1.08 / 2.59 | 0.37 / 1.10 | 0.73 / 1.92 | 0.39 / 1.77 / 3.01 |
| v1_cnnbig | 0.47 / 1.45 | 0.95 / 2.34 | 1.68 / 3.96 | 0.99 / 2.56 | 0.38 / 1.11 | 0.74 / 1.91 | 0.42 / 1.81 / 3.03 |
| MDN v2 (history only) | 0.45 / 1.52 | 0.91 / 2.41 | 1.72 / 3.98 | 1.03 / 2.61 | 0.41 / 1.25 | 0.81 / 2.04 | 0.89 / 2.14 / 3.01 |
| CV (today) | 0.49 / 1.84 | 1.26 / 2.89 | 2.71 / 4.91 | - | - | - | 1.40 / 2.70 / 3.90 |

**DEV state bank (171 instances).**

all, n = 171:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.43 / 1.20 | 0.77 / 2.15 | 1.82 / 3.51 | 0.94 / 2.30 | 0.36 / 1.04 | 0.71 / 1.72 | -0.10 / 1.47 / 2.92 |
| v1_hist (no map) | 0.46 / 1.17 | 0.93 / 2.13 | 1.88 / 3.59 | 0.93 / 2.29 | 0.36 / 1.00 | 0.64 / 1.78 | -0.08 / 1.48 / 2.91 |
| v1_cnnbig | 0.45 / 1.23 | 0.86 / 2.19 | 1.83 / 3.69 | 0.83 / 2.31 | 0.37 / 1.03 | 0.76 / 1.75 | -0.01 / 1.53 / 2.98 |
| MDN v2 (history only) | 0.41 / 1.14 | 0.82 / 2.04 | 1.90 / 3.42 | 0.90 / 2.21 | 0.40 / 0.90 | 0.74 / 1.69 | -0.25 / 1.31 / 2.79 |
| Stage 1 mean world | 0.39 / 1.19 | 0.76 / 2.06 | 1.54 / 3.72 | - | best world 0.18 / 0.70 | best world 0.36 / 1.40 | 1.05 / 2.41 / 3.67 |
| CV (today) | 0.36 / 1.33 | 1.05 / 2.47 | 2.50 / 4.65 | - | - | - | 0.74 / 2.25 / 3.80 |

turn_stop, n = 50:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.49 / 1.21 | 0.76 / 2.11 | 1.49 / 2.71 | 1.21 / 2.24 | 0.36 / 1.09 | 0.72 / 1.39 | 0.77 / 2.26 / 3.50 |
| v1_hist (no map) | 0.52 / 1.24 | 0.87 / 2.03 | 1.42 / 2.80 | 0.89 / 2.01 | 0.40 / 1.01 | 0.68 / 1.39 | 0.80 / 2.27 / 3.51 |
| v1_cnnbig | 0.52 / 1.25 | 0.96 / 2.05 | 1.46 / 2.73 | 0.84 / 2.08 | 0.39 / 1.00 | 0.70 / 1.41 | 0.88 / 2.34 / 3.57 |
| MDN v2 (history only) | 0.46 / 1.15 | 0.81 / 1.84 | 1.39 / 2.71 | 0.84 / 2.09 | 0.43 / 0.85 | 0.73 / 1.28 | 1.01 / 2.23 / 3.55 |
| Stage 1 mean world | 0.51 / 1.25 | 0.88 / 1.80 | 1.18 / 3.25 | - | best world 0.19 / 0.67 | best world 0.41 / 1.31 | 1.60 / 2.40 / 3.34 |
| CV (today) | 0.53 / 1.29 | 1.37 / 2.35 | 2.74 / 4.95 | - | - | - | 1.96 / 3.48 / 4.91 |

near, n = 92:

| method | 0.77 s mean | 1.5 s mean | 3 s mean | 1.5 s top | 1.5 s best-of-6 | 3 s best-of-6 | NLL 0.77 / 1.5 / 3 s |
|---|---|---|---|---|---|---|---|
| **v1** (map + track) | 0.44 / 1.26 | 0.71 / 2.28 | 1.70 / 3.40 | 0.93 / 2.41 | 0.29 / 1.18 | 0.64 / 1.71 | -0.02 / 1.47 / 2.98 |
| v1_hist (no map) | 0.47 / 1.32 | 0.71 / 2.34 | 1.76 / 3.58 | 0.75 / 2.45 | 0.32 / 1.22 | 0.58 / 1.77 | 0.04 / 1.52 / 2.97 |
| v1_cnnbig | 0.46 / 1.30 | 0.79 / 2.28 | 1.58 / 3.55 | 0.76 / 2.50 | 0.29 / 1.15 | 0.63 / 1.69 | 0.03 / 1.48 / 3.01 |
| MDN v2 (history only) | 0.44 / 1.20 | 0.74 / 2.18 | 1.84 / 3.33 | 0.89 / 2.29 | 0.35 / 1.09 | 0.75 / 1.65 | -0.04 / 1.49 / 2.96 |
| Stage 1 mean world | 0.40 / 1.25 | 0.65 / 2.24 | 1.52 / 3.46 | - | best world 0.14 / 0.71 | best world 0.34 / 1.29 | 1.18 / 2.40 / 3.55 |
| CV (today) | 0.28 / 1.56 | 0.81 / 2.70 | 2.39 / 4.79 | - | - | - | 0.77 / 2.29 / 3.93 |

Paired differences, v1 minus the other method, with an episode-clustered bootstrap (95% CI; DEV episodes as clusters,
the three planner configurations pooled).  Values are the median error of the weighted mean (m) and the mean NLL.
Negative means v1 is better.

| DEV set, subset | vs | median error 1.5 s | median error 3 s | NLL 1.5 s | NLL 3 s |
|---|---|---|---|---|---|
| floor, all (6,538) | CV | -0.28 [-0.38, -0.15] | -0.79 [-1.00, -0.60] | -0.89 [-1.05, -0.72] | -0.87 [-1.03, -0.72] |
| floor, turn_stop (2,058) | CV | -0.59 [-0.69, -0.49] | -1.41 [-1.65, -1.26] | -1.09 [-1.37, -0.70] | -1.27 [-1.46, -1.08] |
| floor, near (3,348) | CV | -0.36 [-0.45, -0.18] | -1.02 [-1.22, -0.82] | -0.95 [-1.09, -0.79] | -0.90 [-1.05, -0.74] |
| floor, all | MDN v2 | +0.02 [-0.02, +0.06] | -0.03 [-0.10, +0.04] | -0.35 [-0.52, -0.20] | -0.01 [-0.10, +0.08] |
| floor, turn_stop | MDN v2 | +0.01 [-0.06, +0.06] | +0.02 [-0.08, +0.10] | -0.45 [-0.63, -0.29] | -0.09 [-0.18, -0.02] |
| floor, all | v1_hist (no map) | -0.02 [-0.05, +0.01] | -0.00 [-0.06, +0.05] | +0.01 [-0.04, +0.06] | -0.00 [-0.04, +0.03] |
| floor, turn_stop | v1_hist (no map) | -0.06 [-0.09, +0.00] | +0.10 [+0.02, +0.16] | +0.04 [-0.07, +0.19] | -0.01 [-0.06, +0.04] |
| bank, all (171) | CV | -0.28 [-0.48, -0.02] | -0.68 [-1.03, -0.22] | -0.78 [-0.94, -0.59] | -0.88 [-1.09, -0.71] |
| bank, turn_stop (50) | CV | -0.60 [-0.84, -0.38] | -1.25 [-1.66, -0.85] | -1.22 [-1.55, -0.77] | -1.40 [-1.73, -1.05] |
| bank, all | Stage 1 mean | +0.01 [-0.07, +0.17] | +0.28 [+0.03, +0.50] | -0.94 [-1.55, -0.37] | -0.74 [-1.59, -0.01] |
| bank, all | MDN v2 | -0.06 [-0.17, +0.09] | -0.08 [-0.22, +0.10] | +0.16 [-0.07, +0.33] | +0.13 [-0.01, +0.26] |
| bank, all | v1_hist (no map) | -0.16 [-0.22, +0.01] | -0.06 [-0.18, +0.14] | -0.02 [-0.08, +0.05] | +0.01 [-0.05, +0.06] |

* **Against CV (today's legal bank)** the forecaster is much better at 1.5 and 3 s on both DEV sets.  At 1.5 s the
  median error falls by 0.28 m (22-27%), and on the turn-or-stop subset by 0.6 m (39-45%).  The p90 falls by 0.6 m on
  the floor set.  The NLL improves by 0.8-1.4 nats.  At 0.77 s, CV's median is lower on the DEV bank (0.36 vs
  0.43 m), where tracks are mostly fresh; on the floor set it is 0.50 vs 0.46 m.
* **Against Stage 1's mean world** (DEV bank only) it is level at 1.5 s.  Stage 1 is better at 3 s in median
  (+0.28 m), but has a worse p90 (3.72 vs 3.51 m).
* **Against the history-only MDN v2** the point errors are level everywhere.  The NLL is much better on the
  detection-labelled floor set (-0.35 at 1.5 s; -0.57 at 0.77 s): MDN v2 was trained with estimate-matched labels and
  is overconfident on people who turned while unseen.  On the small estimate-labelled bank, MDN v2's NLL is not
  significantly better (+0.16 [-0.07, +0.33]).
* **The map** (v1 vs v1_hist, same data and regularisation) changes nothing significant overall.  It is mixed on
  turn-or-stop: -0.06 m at 1.5 s (CI touches 0), +0.10 m at 3 s.  The bigger 'v1' CNN is no better.  Training on the
  older mixed labels (`v1mix`) gives the same DEV numbers within 0.04 m / 0.03 nats.

## 5. Sandbox reliability on the DEV state bank (`sbeval`, closed loop, frozen map, exact candidate outcomes)

Setup: 116 DEV states where the robot tracks someone (`--only-tracked`; 76 with a tracked person within 3 m).
Every candidate query is imagined in the closed-loop Sandbox with the camera model and a frozen map, with
`ClosedLoopConfig(window=64, map_loop='off')` and 1 world for CV / fc1 / truth, 6 for fck.  Scoring uses the
exact candidate outcomes of the state bank's probes: 1,160 candidates, with 59 / 109 / 192 real collisions within
0.77 / 1.54 / 3.07 s.

* **Collision AUC**: the imagined contact probability against the real collision.
* **Category = best**: the share of states where the outcome rule picks a candidate with the best real outcome
  category.
* **Critical**: states where the default collides but some candidate does not.  "Fixed" counts those where the rule
  picks a non-colliding candidate.
* **Broke**: the rule picks a colliding candidate where the default is safe.
* `truetrk` is privileged analysis: the same tracked people at their TRUE future (`sbeval.TrueTracksCV`).

| people bank | AUC 0.77 s | AUC 1.54 s | AUC 3.07 s | category = best 1.54 s / 3.07 s | critical fixed / broke, 1.54 s | deviations 1.54 s | CPU s / state (*) |
|---|---|---|---|---|---|---|---|
| CV (today's legal bank) | 0.664 | 0.596 | 0.554 | 0.888 / 0.819 | 0 of 6 / 3 | 0.26 | 2.4 |
| fc1 (weighted mean, 1 world) | 0.656 | 0.576 | 0.566 | 0.897 / 0.784 | 0 of 6 / 2 | 0.27 | 2.5 |
| fck (6 worlds) | 0.699 | 0.611 | 0.608 | 0.897 / 0.810 | 0 of 6 / 2 | 0.31 | 9.8 |
| TRUE future of the tracked people (analysis) | 0.857 | 0.769 | 0.688 | 0.905 / 0.802 | 1 of 6 / 2 | 0.28 | 5.7 |

(*) CV and TRUE were run earlier on a differently loaded machine; fc1 and fck ran together.

AUC difference to CV, state-clustered bootstrap (500 resamples, 95% CI):

| | 0.77 s | 1.54 s | 3.07 s |
|---|---|---|---|
| fc1 - CV | -0.008 [-0.018, -0.002] | -0.020 [-0.049, +0.000] | +0.011 [-0.026, +0.048] |
| fck - CV | +0.035 [-0.019, +0.107] | +0.015 [-0.042, +0.072] | +0.053 [-0.006, +0.114] |
| TRUE tracked - CV | +0.193 [+0.022, +0.394] | +0.173 [+0.064, +0.283] | +0.134 [+0.058, +0.207] |

* The better point forecast does not make the Sandbox predict contacts better.  fc1 is no better than CV, and slightly
  worse at 0.77 s, where CV's median error is lower (section 4).
* fck's six weighted worlds rank contacts a little better than CV at every window.  The gain is not significant,
  and fck costs four times the imagination.
* Even TRUE futures of the tracked people change the rule's choice in at most 2 of 116 states at 1.54 s.  The
  contact prediction improves a lot (AUC +0.17), but the selection hardly moves.  This matches the SR result of the
  lead's ceiling run (section 7).

## 6. Integration: legality, latency, selector smoke runs

**Legality** (`FCBank` built from a proxy of each DEV state-bank `LegalState` that records every attribute read; 283
states, 116 with tracks).  `FCBank` reads exactly `tracks` (the legal track list), `track_log` (the legal track
history), `map` (the legal map) and `time`.  It never sees the tape, the truth, the human count or the state bank's
privileged fields.  Its track keys equal `CVBank`'s.  People without a detection in the last 2 s are bit-identical to
`CVBank` in every world.  The output is deterministic.  The model was trained on TRAIN scenes only (section 2).  The
Sandbox regression tests pass: 11 of 11 listed tests in `tests/sandbox/test_sandbox.py`.

**Latency** (CPU, one torch thread, per decision = per bank build; DEV states with tracks, median / p95 / max):
`fc1` 5.2 / 11.1 / 30.9 ms, `fck` 5.3 / 11.2 / 31.9 ms, `CVBank` 0.1 / 0.3 / 4.5 ms.  The forward pass is part of
this (198 k parameters, 1.7 tracks per state).  The imagination cost is what changes: `fck` imagines 6 worlds whenever
someone is tracked.  `sbeval` CPU time per state is CV 2.4 s, fc1 2.5 s and fck 9.8 s.

**Selector smoke runs** (2 DEV episodes, ORCA native k10, the best legal recipe `S_cv32fPS2` with the bank swapped;
`output/agents/people/smoke/{cv,fc1,fck}_dev2`).  Both episodes collide in the lead's best legal run; they were chosen
so that the forecaster is exercised.  The three runs overlapped on the same machine.  Selector time = wall - setup -
simulator time.

| bank | 2XVvKEDd54w_1468 | bHKTDQFJxTw_8182 | selector s / decision | banks with forecast people | worlds |
|---|---|---|---|---|---|
| CV (this branch) | collision, step 128, 8 decisions | collision, step 75, 5 decisions | 5.76 | - | 1 |
| fc1 | collision, step 133, 9 decisions | collision, step 75, 5 decisions | 6.52 | 10 of 14 | 1 |
| fck (L3_MW_WORLDS=6) | collision, step 93, 6 decisions | collision, step 75, 5 decisions | 21.50 | 7 of 11 | 6 when forecasting (top weight median 0.27) |

* The CV run on this branch reproduces the lead's `S_cv32fPS2_dev` on both episodes exactly (step 128 / 8 decisions;
  step 75 / 5 decisions).
* Both options run end to end without errors.  Bank build: median 4.1 ms, p95 14-19 ms.  Neither option turns these
  two collisions into successes.

## 7. Recommendation

**The ceiling says no.**  The lead's privileged ceiling run, `S_trk32fPS2_dev`, puts the TRUE future on the tracked
people inside the best legal machinery at 1.5 s.  It reaches 59.7% SR (43 / 72; 24 collisions, 5 timeouts),
against 61.1% (44 / 72) for legal CV `S_cv32fPS2_dev`.  Paired, it wins 3 episodes and loses 4 (sign test
p = 1.00).  A perfect forecaster of the tracked people therefore does not raise SR at 1.5 s in today's Sandbox, and
a learned one is not expected to.  Caveat (`LAYER2_V2_DESIGN.md` section 1): the arm's `OracleBank` matches tracks by
last-seen position within 0.75 m, so people lost more than about 0.6 s ago drop out of it.

**Offline, the forecaster is good.**  It is much better than CV, about level with Stage 1's mean world and MDN v2 in
point error, and calibrated (section 4).  In the closed-loop Sandbox it does not change the selection (section 5).
Even TRUE tracked futures move the rule in at most 2 of 116 DEV states.  The collisions sit in the Sandbox's own
robot loop (`LAYER2_V2_DESIGN.md` 1.1-1.2) and in people never seen.

**If one SR run at 1.5 s is spent on this module, run `fck`** (`L3_MW_BANK=fck L3_MW_WORLDS=6`, with the
`S_cv32fPS2` recipe of section 8):
* It is the only option that changes the decision rule and not just the point forecast.  The selector weighs six
  coherent futures of each tracked person, so a candidate that hits a likely turn or stop is penalised.  This
  mechanism is not bounded by the one-world truth ceiling.
* Offline it has the best legal contact AUC at every window: +0.015 at 1.54 s, +0.053 at 3.07 s over CV.  Neither
  gain is significant.
* Cost: about 3.7x CV's selector time per decision when people are tracked (21.5 vs 5.8 s per decision in the smoke
  runs; 9.8 vs 2.4 s per state in `sbeval`).  The 72-episode run takes roughly 2-3x as long as `S_cv32fPS2_dev`.
* Expectation: within noise of 61.1%.  Read the result paired against `S_cv32fPS2_dev`.  A gain would show as fewer
  collisions with tracked people, and possibly more timeouts.

**Do not spend a run on `fc1`.**  At the same cost as CV it predicts contacts slightly worse at 0.77 s (AUC -0.008,
CI excludes 0) and no better at 1.54 s.  The weighted mean hedges between modes.  On the bank's states, where tracks
are mostly fresh, its median error at 0.77 s is above CV's (0.43 vs 0.36 m).

## 8. How to enable

```
export L3_FC_MODEL=output/agents/people/fcmap/v1        # default of peoplemap.DEFAULT_MODEL
# best legal recipe (S_cv32fPS2) with the forecaster in place of constant velocity:
export L3_MW_CONT=16 L3_INHERIT=1 L3_LAYOUT=v4 L3_MW_PERCEIVE=1 L3_MW_RULE=outcome L3_MW_EPS=0 \
       SANDBOX_CONT_BOUNDARY=16 SANDBOX_CONT_INHERIT=1 SANDBOX_PHYSICS=core L3_MW_DISCOVER=union
export L3_MW_BANK=fc1 L3_MW_WORLDS=1        # one world: the weighted mean of the modes (L3_FC1=top: the top mode)
export L3_MW_BANK=fck L3_MW_WORLDS=6        # the 6 most probable joint mode combinations, weighted (L3_FC_MINW=0.01)
python scripts/l3_run.py eventwm_graph.l3.evaluate run --suite dev --configs orca-native-k10 --limit 72 \
    --selector mwroll --out output/eval/<name> --workers 6 --servers 1 --server-devices 1
```

* `fck` worlds are the most probable joint mode combinations of the forecast people: a beam search over the people,
  each combination weighted by the product of the mode weights and renormalised.  Combinations below `L3_FC_MINW` of
  the total are dropped.  `heuristics.multiworld_index` takes the `L3_MW_WORLDS` heaviest worlds and renormalises
  their weights.  The contact probability and the outcome value are weight sums over the worlds.  The imagined planner
  perceives each world's people through the camera model (`perceive.VisibleWorld`).  When nobody is forecast, the bank
  has one world, so imagination costs no more than with CV.
* A track without a detection in the last 2 s of the legal track log continues at constant velocity in every world,
  exactly as in `CVBank`.  Untracked people do not exist in this bank.
* `L3_FC_LOG=<jsonl>` (analysis): one line per bank with the tracks, forecast people, worlds, weights and build time.

Reproduce:
```
L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.peoplemap collect --run output/runs/R10 \
    --out output/agents/people/data2 --shard i --shards 10 --every 4 --match detection --skip output/agents/people/data
python scripts/l3_run.py eventwm_graph.l3.peoplemap fit --data output/agents/people/data,output/agents/people/data2 \
    --estimate-max-age 0.25 --save output/agents/people/fcmap/v1 --device cuda:1 \
    --cnn-arch small --crop 64 --map-drop 0.5 --drop 0.2 --wd 1e-3          # ablations: --no-cnn | --drop 0.2 (v1 CNN)
python scripts/l3_run.py eventwm_graph.l3.peoplemap calib                   # TRAIN calibration of the CV / Stage 1 NLL
L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.peoplemap collect --run output/eval/floor_dev24 \
    --out output/agents/people/dev_floor --shard i --shards 6 --every 4 --match detection --grep=-native-k10
python scripts/l3_run.py eventwm_graph.l3.peoplemap evaluate --model output/agents/people/fcmap/v1 \
    --model output/agents/people/fcmap/v1_hist [--data output/agents/people/dev_floor] --json <out.json>
L3_FC_MODEL=output/agents/people/fcmap/v1 python scripts/l3_run.py eventwm_graph.l3.sbeval run \
    --bank output/statebank/orca_dev --variant cl_frozen_fc1 --out output/agents/people/sbeval --only-tracked
```

## 9. Limits

* Only TRACKED people are forecast.  43% of the people who come within 2 m of the robot within 3.1 s were never seen
  (lead's TRAIN measurement), and 7 of the 23 collisions of the best legal run involve people never seen.  This bank
  has nothing for them.
* The legal map rarely shows where a tracked person is going (section 2: 68% of the box ahead is unknown).  The map
  branch adds little: the val NLL ties history only, and the point error improves modestly.  A larger CNN memorises
  the 108 training scenes.  More scene diversity is the obvious next lever.  The history-only MDN v2 was trained on
  R10 replays from more than 230 scenes.
* The mode weights are flat (section 3), so `fck` with 2 or more forecast people keeps the 6 most probable joint
  combinations of a flat product.  These cover a small part of the joint distribution, and the weights after
  renormalisation are nearly uniform.  A rank-aligned or sampled world set would cover each person's modes better.
* The weighted mean (`fc1`) is not a physical path: it hedges between modes.  For fresh tracks its 0.77 s median
  error is above CV's (TRAIN-val 0.30 vs 0.18 m; DEV bank 0.43 vs 0.36 m).
* The DEV state bank's tracked instances are few (171) and labelled by the estimate match.  The DEV floor replays are
  larger and labelled by the detection match, but come from the default policy's trajectories, not the selector's.
* `sbeval` freezes the map and uses exact candidate outcomes on banked states.  It measures the Sandbox's
  contact-prediction and selection quality, not SR.
