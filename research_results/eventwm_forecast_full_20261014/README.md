# EventWM: full-data recurrent forecasting result

A trained, versioned forecasting model is available. It improves joint human
trajectory prediction over CV and the deterministic recurrent baseline on the
fixed reserved banks, with similar gains from a second training seed. It does
**not** establish reliable scene-belief uncertainty: calibration fails on the
older-observation natural bank, foreground occupancy AP trails CV, and changed-ID
association / existence dynamics remain unsupported.

This stage uses intrinsic forecasting criteria exclusively. It includes no
navigation benchmark, selector training, RL policy or action-ranking objective.

## Data and executed training

The unchanged frozen snapshot `eventwm_350k_belief_snapshot_20261014` contains
348,777 validated queries, 10,569 roots and 534 scenes. SHA256:
`5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27`.
Its eligible training reservation has 304,293 queries, 9,221 roots and 465 scenes.
The 33 queries per root are not independent human futures.

All 9,221 roots were compactly extracted and audited. The fixed protocol reserves
256 train-internal development roots and fits on 8,965 roots with 48,434 windows.
The two selected-family seeds each completed 24,576 updates, batch 64, and
1,562,598 root presentations (174.30 root epochs). Exact sampler replay confirms
both used every fitting root and window. Original full deterministic/stochastic
baselines completed 32,768 updates; a matched batch-64 control and the full map
model also completed 24,576. Scaling runs and bounded component ablations are
complete. Collector pauses preserved optimizer, sampler and RNG state.

The selected architecture has 530,084 stored parameters before calibration and
439,872 parameters trained by the physical objective. It uses a shared width-128
history GRU, six persistent scene motion modes, recurrent 0.25 s acceleration
residuals, and a small legal-history initial-state correction. The exported
per-agent variance model has 530,155 stored parameters, including unused
compatibility heads. No dense recurrent spatial hybrid was introduced.

## Reserved forecasting results

Selection was committed before opening 128 natural and 128 unseen-scene episodes.
All 30 panels completed, covering 15 precommitted model/probability-law variants.
The primary comparison below uses the same training-only per-agent persistent
variance calibration and the same support masks for every model.

| Model | Natural ES 1 s | Natural ES 3 s | Unseen ES 1 s | Unseen ES 3 s |
|---|---:|---:|---:|---:|
| CV | 1.4997 | 2.0219 | 0.5007 | 0.9572 |
| Filtered CV | 1.5385 | 2.0766 | 0.5252 | 0.9909 |
| Deterministic recurrent B | 1.4271 | 1.5767 | 0.4805 | 0.7707 |
| Original stochastic C | 1.3778 | 1.4770 | 0.4615 | 0.7541 |
| Improved seed 14, selected | 0.8712 | 1.2686 | 0.3402 | 0.6811 |
| Improved seed 15 | 0.8441 | 1.2314 | 0.3463 | 0.6837 |
| Map model, single seed | 1.3669 | 1.4586 | 0.4561 | 0.7238 |

Lower Energy Score is better. Three-second identity support is only 45 roots /
41 scenes on the natural bank and 73 roots / 37 scenes on unseen scenes; missing
labels are not scored as zero. P14 minus CV 3 s ES is -0.7533, scene-clustered
95% interval [-1.2323, -0.3112], and -0.2761 [-0.4475, -0.1277], respectively.
Mean ADE changes 3.2561 to 2.2719 m and 1.5274 to 1.1513 m. P14 improves over B
on both banks. Its unseen improvement over original C is inconclusive, while
P15's paired interval excludes zero. Do not pool the two seeds as independent
episodes.

Exact likelihood, all trajectory metrics, fixed-K comparisons, occupancy,
uncertainty and supported 5 s extrapolation results are in
[CONFIRMATION.md](CONFIRMATION.md). The 256-root selection benchmark is separate:
[DEVELOPMENT_BENCHMARK.md](DEVELOPMENT_BENCHMARK.md).

## What passed and what failed

Learned recurrence contributes to prediction under a physical transition
intervention. The initial-belief correction improves short-horizon forecasting
in both full-data seeds. Map conditioning improves the matched 3 s development
comparison, but combining map and initial-belief heads did not improve the 1k
pilot. Cross-agent attention failed its capacity-matched 1k ablation. Multimodal
benefits are supported at short horizons in one controlled head comparison,
not universally across architectures. Scaling from 1k to 4k roots helps C;
further gains largely plateau, with optimization budget a stated confound.

All 256 development roots passed composition checks: direct, 1+2 s, and repeated
0.25 s imagination have zero measured mean/covariance discrepancies; persistent
sample paths agree with identical noise. Query perturbations change human motion
by zero. Actual arrived observations reduce remaining-horizon ADE 1.6366 to
0.9330 m on 26 stable-ID roots, including 17 reacquisition cases. Changed-ID
reacquisition is not covered by that claim. The suite passes 45 tests and exact
training-resume checks.

The primary uncertainty law improves density calibration relative to Gaussian
OU but remains unreliable under distribution shift. On the natural bank its
nominal finite-rank 89.23% region covers 62.2%, interval [48.9%, 75.0%]; unseen
coverage is 83.6% [73.1%, 92.3%]. Natural-bank supported observations are much
older: median agent age 0.920 s versus 0.048 s in development. Sixteen of its
45 supported roots contain an age ≥3 s, versus 4/147 in development. This is a
post-confirmation diagnosis, not a new tuning target.

Foreground occupancy AP remains below CV on both reserved banks. Never-observed
humans receive only a common anonymous marginal prior; no exact identity paths
are invented. Physical tails, geometry violations, changed-ID association and
existence/birth uncertainty remain open. These findings prevent calling the
model a validated general scene-belief backbone. It can be frozen as a tracked-
human forecasting research reference; future RL experiments must not assume
reliable uncertainty or complete unseen-human state.

## Checkpoints and artifacts

The immutable release is
`/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_20261014/release_forecast_v1_20261015/`.

- `primary.pt`: preselected seed 14 with persistent per-agent variance categories;
  SHA256 `d092b5d05d38be4619251e009d2aa2dafc691f3f57440473d72e87acf7887970`.
- `replicate.pt`: independently trained seed 15;
  SHA256 `2faaeff8b19f7298b39eea506354d20e18247d4607ba2ff683c00758a38e6063`.
- `gaussian_alternative.pt`: identical seed-14 physical means with OU residuals;
  SHA256 `ec4e3b0790531f7b688116ccf5be00b73aca0c26e6ed36cb8c4d1d94b231f1e6`.
- `manifest.json`: exact recipes, provenance, selection identity and limitations.

Release inference succeeded using only the bundled arrived history and sensor
configuration (`release_inference_smoke/forecast.npz`). Training raw best/final
checkpoints and exact hashes remain in each original versioned run directory.
The selected raw checkpoint is update 17,408; its full run completed update
24,576. All output locations below are under
`output/eventwm_forecast_full_20261014/`.

Representative artifacts include `qualitative_full_posterior_agent_scale_v2/`,
`correction_full_posterior_agent_scale/`, `calibration_full_selected_figure/`, and
`curves_full_posterior_completed/`. They contain standalone PNG/PDF files and
selection/provenance JSON. Examples deliberately include successes and failures;
error extrema are labeled as illustrative, not representative averages.

Measured 3 s forecast/readout p95 on one six-track, 129-frame scene is 149 ms
with CPU legal preprocessing and GPU inference; batches of 8/32 repeated scenes
are 126/196 ms. These shared-workstation timings include recomputation and
transfer of legal features, exclude IO/sensors, and are not a navigation latency
claim. Preprocessing parity is within 1.44e-6 m. The original GPU-preprocessing
path remains measured separately.

## Reproduction and scientific scope

Worktree: `output/eventwm_forecast_full_checkout`.
Branch: `research/eventwm-forecast-full-20261014`.
Starting commit: `27d52801378639436eb80bc0b736103f6dc5e2c8`.
Frozen selection commit: `3ed2d2e`; evaluated implementation: `745d177`.
Final report commit is recorded in `release_validation_receipt.json` in the
experiment output directory and in Git history.

- [DATA.md](DATA.md): deduplication, target support, missingness and reservations.
- [MODEL.md](MODEL.md): dimensions, probability law and encode/imagine/observe contracts.
- [TRAINING.md](TRAINING.md): actual steps, roots, time, memory, seeds and recovery.
- [ABLATIONS.md](ABLATIONS.md): controlled positive and negative findings.
- [CONFIRMATION.md](CONFIRMATION.md): complete fixed reserved evaluation.
- [REPRODUCE.md](REPRODUCE.md): launch/status/stop/resume, inference and separate 500k scaling.
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md): historical interim findings and superseded commands.

All scheduled research training/evaluation has completed. Production collection
was not stopped, reconfigured or used as a mutable training index. The final
500k comparison is not executed: it requires a separate audited snapshot and
unchanged evaluation membership. No automatic incorporation or restart occurs.

The final read-only resource receipt, `resource_final_release_20261015.json`,
records the original collector PID 3719917 and source commit still running:
419,331 collected queries / 12,707 completed roots, 393.1 GB free disk, and
23.5 / 24.6 GB free GPU memory. These live collected counts are not a newly
validated training snapshot and were not added to this experiment.
The two final resource receipts measure 2,838 additional queries over 51.4
minutes, about 3,312 queries/hour. No forecasting worker remained active.

The next forecasting priorities follow the measured failures: root-balanced
uncertainty fitting across later/older observation windows; full-data replication
of simpler interaction dynamics; legal changed-ID association and existence
support; and better anonymous occupancy/physically constrained sampling. Any
model tuned after this confirmation must use a new untouched confirmation bank.
These are forecasting requirements, not conclusions drawn from a planner.
