# Historical development log

This preserves interim findings and superseded queue commands. Use README.md and REPRODUCE.md for the current result and operations.

# EventWM forecasting-only development

The original full-pool deterministic and six-mode recurrent fits are complete:
32,768 optimizer steps, all 8,965 fitting roots, 1,045,444 root presentations
(116.61 root epochs). Active training durations were 6,825.7 and 6,885.9 s;
peak allocated GPU memory was 1.64 GB each. Selected steps are 31,232 and
26,624 respectively. Full-pool means and training-only calibration are now
being compared on the unchanged train-internal development membership.

The completed six-mode full-data model with OU residual calibration obtains
Energy Score 0.3929 / 0.7179 / 0.9787 at 1 / 3 / 5 s. Corresponding mean ADE is
0.6141 / 1.2093 / 1.6494 m. The benchmark used a frozen interim copy, but its
checkpoint hash is identical to the final selected checkpoint:
`3513675c116deccadb895876cee7d2540ecae69ed42c0c06b41a60373b9504da`.
The exact completed result is `benchmark_interim_full_correlated_ou/result.json`.
The name records when the copy was made, not an incomplete final fit.

The matched original full-data panel is now complete. All rows below use the
same training-only per-agent persistent variance-mixture calibration and the
same development support (156 / 147 / 99 identity roots at 1 / 3 / 5 s).
Five-second forecasts extrapolate beyond the three-second training horizon.

| Physical mean model | ES 1 s | ES 3 s | ES 5 s | NLL/coordinate 3 s | ADE 3 s (m) | Occupancy AP 3 s |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.4111 | 0.8926 | 1.3646 | -0.9453 | 1.4149 | 0.1065 |
| Filtered CV | 0.4232 | 0.8991 | 1.2867 | -0.9374 | 1.4422 | 0.0991 |
| Deterministic recurrence | 0.3827 | 0.7241 | 0.9778 | -0.8909 | 1.1962 | 0.0938 |
| Six-mode recurrence | 0.3839 | 0.7488 | 0.9891 | -0.9242 | 1.2093 | 0.0958 |

Occupancy uses a separate complete-scene mask: 250 / 240 / 167 roots at
1 / 3 / 5 s. AP and positive-cell log loss are defined on 197 / 188 / 124
roots with foreground positives. Empty exposed scenes contribute Brier/log
loss, not invented zero AP. All models use the same masks and human-center
cell target, rather than full body-volume occupancy.

The learned means improve trajectory Energy Score but do not dominate the
kinematic baselines on likelihood or occupancy AP. With Gaussian OU residuals,
CV / deterministic / six-mode 3 s ES is 0.8894 / 0.7078 / 0.7179. The six-mode
minus CV difference is -0.1715, with scene-clustered 95% interval
[-0.2369, -0.1114]. The deterministic versus six-mode ES difference is
inconclusive under that residual law. These are development findings, not
untouched confirmation or established multimodality gains.

Per-agent variance calibration improves the six-mode model's 3 s NLL from
-0.7226 to -0.9242 while worsening ES from 0.7179 to 0.7488. Joint density
coverage is 85.7% at nominal finite-sample 89.23%, with scene interval
[80.0%, 91.0%]. Marginal 90% coverage is 95.4%, and 2.45% of sampled body-center
displacements exceed 3 m/s. Broad tails, imperfect marginal calibration and
weak occupancy AP remain material limitations. The corresponding full-data
calibration figure is `calibration_full_original_agent_scale_figure/`.
Completed results are `benchmark_full_{cv,filtered,deterministic,correlated}_agent_scale/`
and `uncertainty_full_{cv,filtered,deterministic,correlated}_agent_scale/`.
The Gaussian comparison is `comparison_full_original_ou/result.json`.

The full six-mode per-agent checkpoint also passed standalone legal-input
inference in `inference_full_correlated_agent_scale/forecast.npz`. Its SHA256 is
`583a360925ae7658aeff09b353684b63d6b891a8c3c11beae15dfb6ea250bad4`.
The compact `example_legal_inputs/` directory contains only arrived history,
sensor configuration and legal static geometry, with source hashes. It contains
no future-state targets or candidate outcomes. This is a completed reference
checkpoint, not the final improved-model selection.

The first full-pool posterior-initial-belief seed completed 24,576 steps at
batch 64: all 8,965 fit roots, 1,562,598 root presentations and 174.30 root
epochs. Its selected checkpoint is step 17,408, raw SHA256
`8a044785a5a30b2fc2631c3f339afd39a26212aa041c4f6247851978466308a0`.
Accumulated optimizer-loop wall time was 5,679.6 s, including evaluation and
checkpointing but excluding resource pauses/startup. Reported peak allocation
was 3,408,641,024 bytes in the final process session; this is not a proven
maximum across all resumed sessions.

With training-only OU calibration, the completed seed scores ES
0.3467 / 0.6876 / 0.9575 and mean ADE 0.5843 / 1.1682 / 1.5936 m at 1 / 3 / 5 s.
The result is `benchmark_full_posterior_s20261014_ou_reuse/result.json` and its
calibrated `result.model.pt` SHA256 is
`ec4e3b0790531f7b688116ccf5be00b73aca0c26e6ed36cb8c4d1d94b231f1e6`.
Standalone legal-only inference succeeded in
`inference_full_posterior_s20261014_ou/forecast.npz`.

The second seed also completed all 24,576 steps with the same 1,562,598 root
presentations. Its selected step is 16,640, raw development 3 s ES 0.7136,
and raw best SHA256
`0624b21e127ac575e488a0620e2f0866b5277c4f337d380ebfd2f878c85ef0b6`.
Accumulated optimizer-loop time is 5,527.0 s; final-session peak allocation is
3,378,640,896 bytes. Its calibration, the variance-mixture comparisons and
the full map experiment remain pending or in progress under the collector
guard. These are development results; the clean confirmation panel is still
unopened.

The older step-4,352 diagnostic remains immutable at
`benchmark_interim_full_posterior_s20261014_ou/result.json`; it must not be
substituted for the completed-seed result.

The 4,096-root run completed 16,384 steps (128 root epochs). With matched
Gaussian residual calibration, 3 s ES improves from 0.7786 at 1,024 roots to
0.7603; paired scene-clustered 95% difference interval is [-0.0367, -0.0021].
Mean ADE improves from 1.2608 to 1.2236 m. This comparison changes both data
and convergence budget; it is not an isolated data-only effect.

A physical mechanism ablation preserves the 1k model's encoded belief, mixture
weights, and noise but replaces learned future transitions with constant
velocity. Its 3 s Energy Score worsens to 0.9043 and ADE to 1.4451 m. The paired
scene-clustered Energy Score difference is +0.1257 [0.0757, 0.1823]. Thus the
learned recurrent transition contributes to forecasting, independently of any
planner. This ablation is not a separately fitted baseline.

The completed scalar-map ablation scores 0.7752 at 3 s but does not resolve
geometry violations: on supported cells, mean-path non-walkable fractions are
0.2971 without map conditioning and 0.3192 with it. Native human targets score
0.1414 on this eroded robot-navmesh proxy, so these are qualified geometry
diagnostics, not human collision rates. The 25-point directional local-map
encoder has completed its 8,192-step, 1,024-root pilot. With the same OU
calibration it obtains 3 s ES 0.7202 versus 0.7466 without map conditioning;
the paired scene-clustered difference is -0.0264 [-0.0467, -0.0082]. Mean ADE
improves 1.2608 to 1.2153 m. The mean nonwalkable proxy fraction remains
0.2859 versus 0.1414 for targets. Its sampled fraction is 0.4653, with sampled
speed above 3 m/s on 0.21% of displacements. Forecasting improved, but geometry
compliance is not established. See `comparison1024_map_patch/` and
`physical1024_map_patch_ou/`.

The initial-belief diagnostic finds +0.25 s position error of 0.176 m for fresh
tracks (111 roots), 0.995 m for tracks aged 1–3 s (20 roots), and 5.072 m for
tracks aged at least 3 s (only four roots). A 9,412-parameter legal-history head
is being trained to revise stale initial position/velocity before recurrence.
It does not re-encode history during imagination. Long-occlusion evidence is
too sparse for a broad robustness claim.

The initial-belief pilot has now completed 8,192 steps on all 1,024 roots,
using 262,144 root presentations and 1,726.8 active training seconds. Its best
raw checkpoint is step 4,608. With the same training-only OU calibration,
1 s Energy Score improves from 0.4004 to 0.3664; the paired scene-clustered
difference interval is [-0.0562, -0.0153]. The 3 s change, 0.7466 to 0.7321,
has interval [-0.0411, 0.0122] and is not established. Mean ADE is 0.6176 m
at 1 s and 1.2585 m at 3 s. The calibrated checkpoint is
`benchmark1024_posterior_damped/result.model.pt`, SHA256
`1e81fbfb4b7c28298505b884acc8e409a12c7b00a60e49b2bd03a38149531aee`.
It is a development candidate, not the final full-data checkpoint.

A 45-step GPU throughput measurement on frozen windows obtains 169.7, 325.2,
and 626.9 root presentations/s at batch sizes 32, 64, and 128. Peak allocated
memory is 0.37, 1.08, and 2.15 GB on that measured prefix; these are not
worst-case full-pool memory bounds. New matched full-pool comparison configs
use batch 64 for memory margin, 24,576 steps, a 128-step warmup, and the same
root-balanced objective. This planned budget is about 175 root epochs and is
distinct from the original batch-32/32,768-step runs. Queued configurations
are not completed experiments.

Calibration remains qualified. `calibration1024_figure/` shows that a single
90% coverage number concealed substantial distortion at other joint-density
levels. A finite persistent scene-scale mixture fits seven variance-component
probabilities by exact training-only likelihood, holding physical means fixed.
For the 1k recurrent means it obtains 3 s ES 0.7528, NLL -0.8590, and nominal
89.23% joint coverage 85.7% [80.0%, 91.1% scene-clustered interval]. However,
marginal 90% coverage is 96.2% and 3.0% of sampled finite-difference speeds
exceed 3 m/s. This improves one joint diagnostic but is not a reliable
uncertainty claim. The shared variance category persists across imagined
chunks and is updated by actual observations; it is distinct from the six
physical motion modes. No arbitrary raster smoothing is used.

The confirmation-history audit found that the old full-V4 scalar report had
already inspected 478 natural and 497 unseen roots. The first new 512-root
membership overlapped these by 211 and 204 roots respectively. It remains
unchanged as a historical diagnostic membership; its partial cache was stopped
and no forecast scores were inspected. It must not be described as untouched.
A separately frozen clean membership selects 128 later-added natural roots
(105 scenes) and 128 later-added unseen roots (50 scenes), excluding every
old full-V4 root, the complete pilot/extension plan, and any shared historical
episode/fixture/trajectory identity. Additional physical-identity overlaps are
zero. Its membership SHA256 is
`c35502fe8f9c83afd12cc7c025250ca853c071f904b150debc905f486c8c44e1`.
Model/checkpoint selection must finish before any scores from this clean bank
are read. Original reservations and the old stopped raw extension remain intact.

This stage replaces planner-based promotion criteria with intrinsic forecasting
quality. No selector, RL policy, action-ranking experiment, or navigation benchmark
is part of this stage. Earlier reports remain historical records.

## Frozen data and independent units

The unchanged `eventwm_350k_belief_snapshot_20261014` contains 348,777 validated
queries, 10,569 roots, and 534 scenes. Its SHA256 is
`5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27`.
The eligible training reservation contains 304,293 queries, 9,221 roots, and
465 scenes. Thirty-three queries from one root are **not** independent physical
futures. A source audit finds 9,221 distinct episode IDs and 7,606 distinct non-null
scene-human-trajectory IDs; 1,615 roots have no humans.

The new immutable protocol selects 256 train-internal development roots, excludes
the previous pilot's training roots from development, and explicitly records
nested 128/1,024/4,096/8,965 fit memberships. All 9,221 roots have been extracted
and hash-audited. A later all-root refit must use a previously selected recipe and
budget, without selecting checkpoints on the included development roots.
Original validation/natural/unseen reservations are unchanged. The old 106-root
validation bank and 16 exploratory unseen roots are not untouched confirmation
data. The stopped raw unseen extension has not been resumed.

Each compact payload verifies raw source hashes and the canonical
`(branch_id, query_id)` labels, then extracts the longest precontact physical
branch once. This uses the previously audited exogenous simulator factorization;
it does not claim a new exhaustive 33-way parity audit. Multiple windows start at
actual arrived observations approximately one second apart. Inputs contain only
history available at that origin. Future physical positions are targets only.
Missing labels and first-contact censoring remove support; they never create
empty-occupancy targets. Root-uniform sampling followed by window-uniform sampling
prevents roots with more windows or queries from dominating the objective.

Measured support in the complete 9,221-root cache (256 development + 8,965 fit):

| Support | 1 s | 3 s | 5 s |
|---|---:|---:|---:|
| Roots with complete native scene exposure at origin | 8,911 | 8,563 | 5,641 |
| Roots with an associated, fully exposed tracked trajectory at origin | 5,707 | 5,392 | 3,244 |
| Roots with at least one complete identity window | 6,363 | 5,948 | 3,326 |
| Supported physical windows | 41,400 | 24,196 | 7,472 |
| Supported identity windows | 27,125 | 15,363 | 4,116 |

The cache has 49,857 correlated windows and 25,439 physical agent trajectories;
1,615 roots have zero humans, 1,528 have never-observed humans, 4,062 contain lost
tracks, and 3,972 contain at least three humans. All 561,574 supported window-tick exposures
have complete physical identity coverage. Windows overlap: this is an exposure
occurrence count, not 561,574 distinct physical times or independent futures. The fit partition has
48,434 windows; 6,410 fit roots have any supported 0.25 s tracked transition and
5,782 have any complete 3 s identity window. Presenting all 8,965 roots during
training does not imply all provide tracked-motion gradients. Zero-human and
never-observed scenes still contribute to the anonymous occupancy prior.

The separate `association_census/result.json` audits all cached windows without
changing inputs. It records 70,368 track-window occurrences, 59,087 supported
associations, and 1,348 switched-ID occurrences. These are correlated window
counts. Positively observed tracker slots exceed the recorded physical
population in 593 roots (153 at the initial origin), covering 2,295 windows.
The maximum is 12 legal track slots versus at most six physical humans. In the
development partition, 11 roots have excess slots somewhere in their history
and three have excess slots at origin. The diagnostic counts cannot identify
which slots are aliases and are never used as an inference-time count or filter.
Changed-ID/duplicate-track repair remains an implementation limitation; this
census does not establish that it causes the aggregate occupancy-AP gap.

`audit_full/result.json` is the measured census. The complete compact manifest
SHA256 is `d8df96f6cfd61c6677987ab22eef3d53eb1e29a426651fc0a313846f8efc43d7`.
The 9,221 payloads and four manifests are read-only, with a separate receipt in
`cache_full_readonly.json`; runtime guard/status files remain writable. No raw
dataset was copied or modified.

## Forecast distribution and interfaces

The starting model uses a shared history GRU, width 128, two masked interaction
layers at recurrent transitions, and six persistent joint scene modes. It has
520,672 stored parameters and 430,460 trainable parameters in this stage. Old
event, anonymous, and learned correction heads are frozen and unused by the
physical objective. Human dynamics do not read robot queries.

The mean transition predicts bounded acceleration residuals over a causal
constant-velocity estimate at a 0.25 s timestep. One categorical mode describes
the entire scene trajectory. For the correlated uncertainty variant, each
agent/axis has a Gaussian position/velocity residual:

```
u[t+1] = [[1, dt], [0, 1]] u[t] + [dt²/2, dt] a[t]
a[t] ~ Normal(0, q²)
observed body-center position = recurrent mean + u_position + Normal(0, r²)
```

The four learned noise scales specify initial position, initial velocity,
acceleration, and readout uncertainty. Residuals are conditionally independent
across agents; the shared categorical mode provides joint dependence. Exact
mixture likelihood uses Kalman innovations, marginalizing missing observations.
Likelihood computation may inspect targets; free-running transitions never do.
The Energy Score samples this same distribution, including within-mode variance.
The IID-noise model is a separately identified probabilistic baseline.

`encode_history()` returns physical and latent states, legal track support,
observation ages, mode weights, covariance, and a persistent clock. `imagine()`
returns a directly reusable successor belief. For identical sampled paths across
segments, pass the returned sample residual state and the same RNG to the next
forecast's `sample()` call. `observe()` performs a Gaussian correction using
actual arrived measurements and updates scene-mode weights. A delay of at most
one canonical tick uses an explicitly approximate linear observation operator.
Changed-ID association and births are rejected rather than silently duplicated.

The age-aware ablation adds no parameters: initial covariance propagates over
the legally known time since observation. The map ablation supplies existing
per-agent clearance/walkability features from a known navmesh. This eroded map at
robot height is a geometry covariate, not exact human contact supervision.

Important limits: identity/existence probabilities are not calibrated; anonymous
occupancy is currently a common training-only marginal prior, not a joint unseen
human dynamics model. Ambiguous identity associations are masked. Five-second
forecasts extrapolate beyond the three-second training objective and are scored
only where complete physical exposure exists.

## Executed preliminary results

All values below are exploratory development results from one seed, not final
confirmation or evidence of navigation performance. The three-second trajectory
population contains 147 of the 256 development roots. Lower scores are better.

| Completed 128-root run | Steps | Best step | Selection Energy Score | Training seconds |
|---|---:|---:|---:|---:|
| Legacy discrete-center objective | 1,024 | 704 | 0.8107 | 166.2 |
| Consistent IID distribution objective | 1,024 | 704 | 0.8134 | 178.8 |
| Correlated trajectory distribution | 1,024 | 320 | 0.8111 | 214.4 |

These objectives are tied at this scale. The legacy run's final score worsens to
0.8592 while NLL improves slightly, demonstrating that the metrics can disagree;
this does not establish the cause of the old C256/C4096 behavior. The new pilot
also uses different fit roots and multiple windows, so it is not an exact
reproduction of that historical run.

The later matched historical diagnosis (`historical_diagnosis_v2/result.json`)
does support overfitting in the original run. Between C256 and C4096, proper
3 s Energy Score on its original training roots changes 0.8374 to 0.2436 (80
supported roots), while the new development score worsens 0.8871 to 1.0087
(147 roots). Mean ADE changes 1.3398 to 0.3753 m on training and 1.4042 to
1.4705 m on development. All 128 human histories match exactly. Forty-two ego
histories differ only in previous angular command, which the physical encoder
does not read; a poisoned-command invariance test passes. This comparison does
not isolate learning rate, auxiliary losses, or limited root diversity as the
sole cause. Both checkpoints use the same current target support masks.

Training-only probabilistic calibration matters. On the same development roots,
the 0.75 s filtered kinematic model obtains three-second Energy Score 0.9753 with
IID residuals and 0.9287 with correlated residuals. Its mean ADE is unchanged at
1.4422 m. Corresponding normalized joint NLL changes from 1.8308 to -0.6766 nats
per coordinate. Negative continuous-density NLL is valid. The latter model's
90% scalar intervals cover 94.8% overall, with mean width 5.31 m; visible-only
roots cover 98.8% and roots with lost tracks cover 92.9%. It is overdispersed on
visible tracks, motivating the controlled observation-age experiment.

Completed 32-root native mechanics diagnostic for the correlated 128-root model:
mean, covariance, and persistent-noise sample composition have maximum difference
zero; poisoned robot queries change human forecasts by zero. Five roots satisfy
the conservative actual-observation correction criteria. Their remaining-horizon
ADE changes from 1.746 m to 1.377 m. This small selected supported subset validates
the mechanism, not broad correction reliability or changed-ID recovery.

The 1,024-root deterministic IID and stochastic correlated runs have completed
8,192 steps each, using all 1,024 fit roots and 5,556 windows. Each presented
262,144 roots (256 root epochs); durations were 1,326.1 s and 1,605.0 s, excluding
resource pauses and initial preparation. Peak allocated GPU memory was about
1.28 GB. The deterministic correlated baseline also completed 8,192 steps.

Matched training-only calibrated results on the same 147 development roots:

| 1k-root comparison, 3 s | Energy Score | Mean ADE (m) | Mean FDE (m) | Joint NLL / coordinate |
|---|---:|---:|---:|---:|
| CV, correlated uncertainty | 0.9200 | 1.4149 | 2.5329 | -0.6865 |
| Filtered kinematics, correlated uncertainty | 0.9134 | 1.4422 | 2.5253 | -0.6807 |
| Deterministic recurrent, IID uncertainty | 0.7802 | 1.2536 | 2.0618 | 1.5833 |
| Deterministic recurrent, correlated uncertainty | 0.7807 | 1.2573 | 2.0818 | -0.6716 |
| Stochastic recurrent, correlated uncertainty | 0.7786 | 1.2608 | 2.0873 | -0.6715 |
| Same stochastic means, Student scene-scale uncertainty | 0.7718 | 1.2608 | 2.0873 | -0.8131 |

The stochastic correlated model's Energy Score difference versus CV is -0.1414,
with exploratory scene-clustered 95% interval [-0.2017, -0.0840]. Its one-second
interval crosses zero. The comparison with the deterministic model does **not**
yet establish a benefit from multiple scene modes. The completed deterministic
correlated comparison is effectively tied with the six-mode model, isolating
this question from the residual law.

Uncertainty is not yet reliably calibrated. The Gaussian correlated model's 90%
marginal intervals cover 95.6%, but its nominal 90% **joint trajectory density
region** covers only 70.7% (147 roots; finite 64-draw rank target 89.23%). Its
mode-center separation is only 0.202 m RMS despite prior effective mode count
5.65. These diagnostics motivate heteroscedastic and heavy-tailed uncertainty
experiments; entropy alone is not evidence of meaningful multimodality.

The Student variant adds one shared inverse-gamma variance latent for an entire
scene trajectory. Conditional on that scale, the same Gaussian state process
applies. Exact multivariate-t likelihood and conjugate actual-observation scale
updates preserve temporal dependence. Sample scale persists across segments.
Occupancy integrates the shared scale before combining agents/axes; multiplying
independent Student marginals would be incorrect. A 256-step training-only
calibration fit selected degrees of freedom 2.257 in the preliminary 1k model.
This improves NLL and foreground occupancy AP (3 s: 0.0744 to 0.0919) without
changing physical means. Heavy tails require separate physical-plausibility and
joint-coverage assessment; this is not a claim of reliable uncertainty.

The completed joint diagnostic raises density-region coverage from 70.7% to
93.2%, but sampled body-center displacements above the mean model's 3 m/s bound
increase from 3.7% to 8.6% (these finite differences include readout noise).
The Student variant therefore fails promotion as a physically satisfactory
forecaster in its current form. The equally calibrated Student CV baseline
obtains 3 s Energy Score 0.8935 and NLL -0.8384; the learned model improves Energy
Score while still losing slightly on NLL. These are useful diagnostics, not a
uniform probabilistic superiority claim.

An exact integrated Ornstein-Uhlenbeck residual is a smaller alternative:
`dv = -lambda*v*dt + q_c*dW`, `dp=v*dt`, with `q_c=q*sqrt(0.25)`.
It adds one learned decay-rate parameter to the Gaussian model. Exact transition
and covariance matrices define the likelihood, samples and fractional-age
initialization consistently. Mean reversion acts on residual velocity around the
recurrent mean; it does not clip samples or force the physical mean to stop.

With the 1k stochastic physical means frozen and noise fitted only on training
roots, OU changes 3 s ES from 0.7786 to **0.7466**, NLL from -0.6715 to
**-0.7033**, and 90% marginal coverage/width from 95.6%/4.75 m to **90.2%/3.45 m**.
Sampled speed above 3 m/s falls from 3.7% to **0.17%**. Joint density-region
coverage remains **68.0%**, below the finite-rank nominal 89.23%; this is not
reliable joint calibration. An equally fitted CV+OU obtains ES **0.8885** and
NLL **-0.7030**. Age-aware means with OU obtain ES **0.7419**, NLL **-0.7936**,
and marginal coverage **85.8%**. These are single-seed development diagnostics,
not a selected final backbone. Conditional uncertainty remains a research gap.

Known-map diagnostics are deliberately qualified: on the same 147-root support,
the eroded robot-floor navmesh marks 14.1% of target positions nonwalkable, versus
29.7% of stochastic mean predictions and 52.3% of Gaussian samples. Outside-grid
positions are excluded and reported separately. This proxy exposes a geometry
deficiency but is not a human collision/contact label.

The full 256-root composition check has zero maximum differences in means,
covariance, persistent sampled paths, and query invariance. On 26 supported
stable-ID correction roots, remaining-horizon ADE changes from 1.717 m to
1.023 m. Changed-ID and birth cases remain excluded and explicitly unsupported.
The actual GPU training loop passed a 32-step recovery test: interrupting after
step 8 reproduced uninterrupted parameters, optimizer, sampler order, and all
CPU/CUDA RNG states bit for bit on this machine.

Substantial 4k-root and full-pool jobs are separately versioned. Consult
their `result.json` files for completion; a checkpoint or a queued job is not a
completed experiment. Collector guard pauses preserve optimizer, sampler,
root-order, and CPU/CUDA RNG states. Full original deterministic/stochastic training results are complete; final
calibrated comparisons and the improved replicated runs remain in progress.
The exact fixed-step full-pool checkpoints are preserved under
`checkpoint_archive/`. At step 4,096, raw development ES was 0.7704 and 0.7578
for deterministic and stochastic models. Their final selected raw scores are
0.7439 and 0.7476. Neither result establishes a multimodality advantage.
`curves_1024/` contains exported smaller-run learning curves.

## Reproduction and artifacts

Worktree: `output/eventwm_forecast_full_checkout`.
Branch: `research/eventwm-forecast-full-20261014`.
Experiment root: `output/eventwm_forecast_full_20261014` (absolute paths are in
the committed configuration files). Starting commit:
`27d52801378639436eb80bc0b736103f6dc5e2c8`.

```
cd /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_checkout
bash scripts/run_eventwm_forecast.sh status configs/eventwm_forecast_workflow_v9_launch.json
bash scripts/run_eventwm_forecast.sh stop configs/eventwm_forecast_workflow_v9_launch.json
bash scripts/run_eventwm_forecast.sh resume configs/eventwm_forecast_workflow_v9_launch.json
```

Stop the owning workflow first when stopping a managed child job; otherwise the
queue intentionally resumes resource-paused children. For an independent job,
use `launch`, `status`, `stop`, or `resume` with its `_launch.json` configuration.
Completed result directories are not rerun. The workflow never absorbs live data
or creates a new snapshot. It respects the original three-window throughput
recovery condition; it does not reset the guard to improve research throughput.

```
export PYTHONPATH=scripts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
$PY -B -m unittest discover -s tests -p 'test_forecast*.py' -v
$PY -B -m eventforecastwm.train_cached --config configs/eventwm_forecast_correlated_full_cached.json
$PY -B -m eventforecastwm.benchmark --help
$PY -B -m eventforecastwm.inference --help
$PY -B -m eventforecastwm.scaling --help
```

Direct training is for a free GPU after preflight; use detached launch controls
for long jobs. `best.pt` and `last.pt` retain exact recipe/data/source identities;
calibrated benchmark checkpoints additionally pin calibration roots. Minimal
inference reads legal history JSON, never a future-target cache. Existing
`c128_correlated/best.pt` is a preliminary mechanics reference, not a selected
full-data backbone.

Queue v3 fixes an observed research retry bug: resource-failed children wait at
least 65 s between attempts so their three recovery samples can span 120 s.
Guard thresholds and production settings are unchanged. Parameter-free legal
preprocessing is cached incrementally and hash-verified across models/resumes;
learned GRU embeddings are always recomputed.
Queue v4 prioritizes the two full-pool fits. With extraction finished and measured
memory available, it allows at most two compact jobs per GPU, retaining at least
8 GiB free before each launch. Each job uses one CPU thread. The collector's
original thresholds and three-window recovery rule remain unchanged.

For 500k scaling, first create a **separate audited frozen snapshot** with the
350k snapshot as its explicit parent. `eventforecastwm.scaling` checks the new
snapshot hash and unchanged inherited joins/reservations, retains exact old
subset/development IDs, and can hard-link compact verified cache payloads into
a new cache. It does not copy raw data, modify production, or launch training.
Use new experiment directories and matched optimization/root-exposure views.

Queue v5 owns recovery of the full-pool B/C/IID runs and the map-patch pilot,
then runs two explicitly seeded full-data posterior-belief fits (24,576 steps,
batch 64), a one-mode posterior pilot, a capacity-matched cross-agent attention
ablation, and a matched batch-64 original-model control. These are planned
budgets, not completed results. Queue v4 was cooperatively stopped; only one
monitor samples the shared collector guard.

Exported checkpoint loading applies `protect_observe`: masked sensor slots may
contain NaN placeholders without contaminating a belief. Actual detections still
require finite positions and acquisition timestamps. Stable slot identity remains
required; this numerical fix does not implement birth or changed-ID association.
The forecasting suite passes 43 unit/schema/probability tests.

A 63-parameter conditional variance-probability head was tested on the 1k
training pool, holding recurrent means and the global OU law fixed. Its 3 s
normalized NLL changes only -0.8590 to -0.8595 while ES worsens 0.7528 to
0.7572. It is retained as a negative ablation, not promoted. The other conditional
head configurations are unexecuted. Tests verify exact mixture likelihood,
useful probability gradients, persistent sampling and Bayesian correction.

`evaluation_queue_v4` runs at most one CPU benchmark, gated on completed training
artifacts. It never opens confirmation data or modifies shared guard history.
`benchmark_reuse` hashes fitting membership, immutable data and target/prior
source before reusing a model-independent anonymous occupancy prior. Its stop
command is cooperative; an interrupted benchmark repeats evaluation, while
training resumes its exact optimizer state. Completed artifacts are immutable.
The queue configuration's `iid_gp` path is a historical filename: its saved
recipe remains the IID model and reports must label it IID, not GP.

New static-map sidecars do not change frozen physical targets.

The full geometry sidecar is now complete: 9,221 supported roots, manifest SHA256
`8a24bf6fb74793fba43ef39a7474a3618b67c883fb415e840d92e7cc9d5b9116`.
It reuses 1,280 compact payloads; 7,941 new payloads and the new manifest are
read-only. Shared earlier payload permissions were left unchanged. All payloads
are hash-verified on use. This adds known-map covariates, not new physical targets.

`data_distribution/result.json` records exact subset and scene counts. The full
fit uses 8,965 roots across all 465 training scenes, with 48,434 windows. There
are 1–7 windows per root (median 6) and 1–77 roots per scene (median 15). The
256-root internal development bank covers 172 scenes. `utilization.py` replays
the frozen sampler without training; final RNG, order, cursor and presentation
counts must match the checkpoint. The completed posterior 1k pilot visited all
5,556 available windows and all 1,024 roots (`utilization_posterior1024/`).
Window coverage includes unsupported/no-human windows, not only motion targets.

The posterior+OU pilot's actual-observation correction diagnostic uses the same
strict 26-root stable-ID support as the earlier mechanics test. Seventeen roots
contain stable-ID reacquisition. Remaining-horizon ADE changes 1.6244 to 1.0095 m;
the paired scene-clustered difference is -0.6149 m [-0.8809, -0.3605].
`correction1024_posterior/correction.png` and `.pdf` deliberately include gain
and failure extrema. Changed-ID association and births remain unsupported.

Workflow v7 owns both training and the sole CPU evaluation queue. One command
stops all managed children cooperatively, preserving training checkpoints:

```bash
bash scripts/run_eventwm_forecast.sh status configs/eventwm_forecast_workflow_v9_launch.json
bash scripts/run_eventwm_forecast.sh stop configs/eventwm_forecast_workflow_v9_launch.json
bash scripts/run_eventwm_forecast.sh resume configs/eventwm_forecast_workflow_v9_launch.json
bash scripts/run_eventwm_forecast.sh status configs/eventwm_forecast_evaluation_queue_v6_launch.json
```

Resume the owner, not its children independently. It monitors the original
collector guard during a pause and resumes child jobs only after recovery.
The previous workflows, two-worker evaluation queues and guard keepers were
cooperatively stopped. Do not resume those historical controllers concurrently.
The owner now remains alive through CPU evaluation, so a separate guard keeper
is unnecessary. Training resumes exact optimizer/RNG state; short calibration
and evaluation jobs restart computation against their unchanged inputs.

A persistent per-agent variance experiment retains the shared physical scene
mode but replaces a single scene variance category with one category per agent:
`sum_z pi_z product_n sum_l rho_nl Normal(Y_n; mu_zn, v_l K_zn)`.
All variance categories persist through time. Likelihood, sampling, occupancy
and observation correction use this same factorization. A bounded comparison
fits (i) seven shared-scene probabilities, (ii) seven unconditional per-agent
probabilities, or (iii) those probabilities plus 63 legal-context parameters.
Physical means and Gaussian covariance parameters are frozen. The scene control
uses the same 512 full-batch training-only updates. An equivalently calibrated
CV control is included. These configurations are experiments, not established
improvements. Tests cover analytic joint density, masked targets, persistent
samples, useful probability gradients and Bayesian actual-observation updates.

The capacity-matched no-cross-agent ablation initially failed before training
because its configuration omitted `prepared_cache`. That failed directory is
preserved. The corrected `nointeraction_1024_v2` configuration uses the same
physical recipe, data and budget and a new experiment directory.

Workflow v6 retains unfinished exact checkpoints from v5, replaces only the
pretraining-failed interaction configuration with its corrected version, and
adds two evidence-motivated experiments: full-pool map conditioning (24,576
steps, batch 64) and a 1,024-root map-plus-initial-belief pilot (8,192 steps,
batch 32). The combined model has 534,836 stored parameters; it adds no new
recurrent spatial grid. Calibration adapters preserve its learned physical
mean exactly. The 40-test suite verifies mean preservation and composition.
The two components' separate gains do not establish their combined benefit.

`curves_full_original/` exports completed full-data learning curves. Development
Energy Score keeps improving beyond 4,096 steps, while joint NLL largely levels
off earlier. No gradient became nonfinite. `utilization_full_completed/` exactly
replays all 32,768 stochastic optimizer-step samples, confirming all 8,965 roots
and all 48,434 windows were visited. It matches checkpoint sampler state, order,
cursor and presentation counts. This is utilization evidence, not a claim that
unsupported windows supplied motion gradients.

[MODEL.md](MODEL.md) specifies input/output contracts, every probability law,
parameter counts, reusable sampling and the approximate nonlinear observation
update. It distinguishes supported tracked-human forecasting from unresolved
identity/existence and anonymous-dynamics claims.

`observability.py` reports intrinsic trajectory scores on legal age groups,
tracks visible in every recorded history frame, and stable-ID reacquisition
within history. Each score marginalizes the joint model to fully supported
agents in the group and then averages roots. Groups overlap; an unobserved track
is retained rather than interpreted as physically absent. These diagnostics
must be explicitly executed for a chosen checkpoint; implementation alone does
not establish performance on a stratum.

The completed full original deterministic+OU model obtains 1 / 3 / 5 s Energy
Scores 0.3963 / 0.7078 / 0.9687. Its 3 s mean ADE is 1.1962 m and normalized
joint NLL -0.7123. Against six-mode C+OU, the 3 s ES difference is -0.0101 with
scene-clustered interval [-0.0235, 0.0031]. Neither model establishes superiority
here, and multiple modes remain unjustified by this comparison alone. See
`comparison_full_original_modes/`.

The full C+OU observability diagnostic has 3 s mean ADE 1.057 m on fresh-track
subsets (111 roots), 1.637 m at observation age 1–3 s (20 roots), and 4.965 m at
age at least 3 s (only four roots). The latter regime is difficult and too small
for a broad robustness claim. `observability_full_correlated_ou/` reports
supported counts and scene-clustered intervals for every stratum and horizon.
Its Energy Score uses a reset per-root diagnostic RNG and should not be mixed
numerically with the main benchmark's continuous batched RNG.

The clean confirmation geometry sidecar is complete and hash-verified for all
256 roots, with no unsupported maps. New payloads and manifest are read-only;
`geometry_confirmation_clean_freeze.json` records the identity. This extracts
only legally known static geometry at observed robot frames. No forecasting
score from the clean confirmation bank has yet been read.

The per-agent variance experiment has now run. Compared with a seven-parameter
shared-scene control with the same 512 full-batch updates, its 3 s NLL improves
-0.8593 to -0.8957, while ES worsens 0.7596 to 0.7678. Its joint density CDF at
nominal 10 / 25 / 50 / 75 / 90 percent is 10.2 / 23.1 / 51.0 / 71.4 / 86.4
percent (147 roots, 64 Monte Carlo draws). The conditional 63-parameter addition
changes NLL to -0.8978 and ES to 0.7664, with no compelling gain in the other
diagnostics; it is not the scaling candidate. Sampled speed above 3 m/s remains
2.3%, so physical tails remain imperfect. `calibration1024_agent_scale_figure/`
compares the full calibration curves, not only one coverage threshold.

The equivalent conditional CV control obtains 3 s ES 0.8921 and NLL -0.9488.
Learned dynamics improves Energy Score here but does not beat CV on likelihood.
No uniform probabilistic-superiority claim is supported. Full-data comparisons
use the same simpler unconditional per-agent variance procedure for every
baseline and candidate, preserving this distinction.

After repeated throughput pauses and measured host-load saturation, workflow v7
reduces concurrency to one training process per GPU plus one CPU evaluation
worker in total. It preserves all optimizer budgets and all original guards.
Explicit device dependencies prevent a new pilot from taking a slot while a
higher-priority full-data seed's private guard recovers. `evaluation_queue_v6`
serializes calibration and diagnostics for the original and improved physical
models. These full-pool uncertainty fits are pending, not inferred from 1k data.
The calibration implementation removes unsupported agent padding only after
legal contexts are constructed; an analytic test verifies unchanged joint
objective and gradients. This reduces compact-calibration memory without
changing sampling weights or interpreting missing targets as negatives.

`window_evaluate.py` is a secondary diagnostic for all actual-arrival windows
within the fixed development roots. It averages supported windows within each
root before computing scene-clustered intervals. Its membership and sampling
population are explicitly distinct from the primary root-origin checkpoint
selection protocol. No all-window measurement is claimed until executed.

## Initial-belief mode and objective controls

The matched 1,024-root initial-belief ablation has completed. With the same
training-only OU calibration, six versus one motion mode gives 1 s Energy Score
0.3664 versus 0.3860: paired scene-clustered difference -0.0196
[-0.0352, -0.0068]. The 3 s difference is -0.0140 [-0.0382, 0.0107],
which remains inconclusive. Joint NLL improves by -0.0972 at 1 s and -0.0310
at 3 s, with intervals excluding zero. This is a single-seed development
ablation, not a general proof of multimodality. See
`comparison1024_posterior_modes/result.json`.

Two additional 8,192-step, 1,024-root controls are queued: Energy Score only
and trajectory likelihood only. Architecture, root membership, seed, optimizer
schedule and checkpoint-selection metric match the existing mixed-loss control.
The likelihood-only gradient check passed; these training runs are not yet
completed results. Their configurations are `eventwm_forecast_correlated_energy_1024.json`
and `eventwm_forecast_correlated_nll_1024.json`. Gaussian and OU residual
calibration receive the same stages as the reference.

The current owner is `workflow_v8`, with `evaluation_queue_v6` as its sole CPU
worker. Version 7 and queue 5 exited cooperatively before version 8 started;
never resume both controller generations together. The extension adds the
objective controls without changing full-pool budgets, production guards or
resource limits.

## Completed full-data seeds and current execution

Both initial-belief seeds completed all 24,576 optimizer updates. Exact sampler
replay confirms all 8,965 fitting roots and all 48,434 available windows were
visited by each. The audits are `utilization_posterior_full_s20261014/` and
`utilization_posterior_full_s20261015/`; completed curves are in
`curves_full_posterior_completed/`. The second seed now has calibrated OU
Energy Score 0.3449 / 0.6908 / 0.9752 at 1 / 3 / 5 s, compared with
0.3467 / 0.6876 / 0.9575 for the first seed. These are development results.

Workflow v9 and its sole CPU evaluation queue v7 now own the remaining jobs.
Workflow v8 and queue v6 exited before this replacement started. The additional
jobs are a bounded full-root covariance-fit check, a matched 4k OU calibration,
and full-model composition, correction, observability, physical and later-origin
window diagnostics. All use the unchanged collector guard; no confirmation
panel has been opened. The suite passes 45 tests.

See [DATA.md](DATA.md) for supervision/support counts and
[REPRODUCE.md](REPRODUCE.md) for current commands. Older queue descriptions above
record execution history and must not be used to launch concurrent owners.
