# Ranked hypothesis ledger

This is a prospective ledger, not a table of claimed improvements. All results
will be appended with exact manifests and root-level comparisons.

| Rank / intervention | Failure and mechanism | Current evidence | Expected benefit | Cost / complexity | Failure condition | Minimal controlled experiment |
|---|---|---|---|---|---|---|
| 1. Root-balanced all-window conditional calibration | Root-origin noise fitting misses older beliefs; estimate heteroscedastic state and process uncertainty from legal context | Calibration code uses only windows[0], although physical training samples all windows; v1 natural ages shift | Older-history NLL/coverage, useful sharpness | Minutes to tens of minutes; small probability changes | Coverage improves only by indiscriminate widening or ES/physical tails regress | Freeze v1 means; origin vs all-window weights, unconditional vs age/context; unchanged dev-origin and later-window masks |
| 2. Occupancy population and readout audit | Full-scene AP may mix tracked, unassociated and never-observed populations; mean shifts and sharp stationary modes may affect cell ranking | Existing tracked_occupancy diagnostic is scored against all-human targets; full-scene coordinate formula appears consistent | Correct attribution and a testable occupancy intervention | Minutes; diagnostics first | No mismatch, or a relabeling gives no model improvement | Exact point/raster sanity checks, associated/anonymous targets, tracked-only/total strata, cell mass and tails |
| 3. Missingness-aware encoding | Zero-filled post-retention histories can erase useful observed motion; preserve last observed representation and explicit elapsed time | prepare_history retains ever-seen support, but clean becomes zero after tracker deletion and GRU runs on every frame | Aged state estimation and uncertainty | One matched 1k run, moderate | Fresh forecasts regress or no aged benefit | Existing encoder vs causal missingness features / last-valid-state encoder; no invented detections |
| 4. Simpler interaction + interpretable motion branches | Cross-agent layers may oversmooth individual dynamics; persistent CV/stationary alternatives can retain easy cases | v1 capacity-matched attention ablation was negative; CV wins occupancy AP despite worse ES | Point paths, occupancy ranking and runtime | One or two 1k runs, moderate | Only likelihood/ES change from extra noise, or modes collapse | Self-only vs shared interaction, then motion-prior modes with matched stochastic control |
| 5. Occupancy auxiliary likelihood | Trajectory-only objective gives little weight to calibrated cell mass or anonymous humans | No occupancy loss trains the selected physical model; anonymous prior is global | Occupancy AP and log score | Small controlled loss weights, moderate | Trajectory/covariance degeneration or contradictory readouts | Differentiable exact tracked occupancy BCE on properly supported targets; separate anonymous conditional prior |
| 6. Better motion dynamics and horizon weighting | CV-biased stale initialization, recurrent turning errors and short horizon dominance | Initial correction already helped strongly; 5s is untrained | Turning and long-gap trajectory quality | 1k plus intermediate runs, moderate | No gain over frozen-mean uncertainty control | Bounded turning/speed residual or velocity prior; compare 1/3s horizon weighting and optional 5s training |
| 7. Legal missing-arrival augmentation / window sampling | Training information may underrepresent no-update forecasting and rare turns | Root/window uniform sampler is already root-balanced; actual aged support not yet counted | Robustness on old history | Moderate; requires legal protocol tests | Impossible histories, importance bias or fresh regression | Prefix-only no-arrival continuation with correct clock/targets; matched original-distribution evaluation |
| 8. Local geometry/social context | Existing integration may waste predictive geometry | Directional-map gain at 1k and full matched C; map+posterior pilot negative | Turns and physical plausibility | One bounded capacity-controlled test only if audit warrants | Repeats prior negative result without new mechanism | Simpler relative-neighbor or local-map integration vs capacity control |

Existing strengths to retain: legal timestamp regression, root-balanced data,
initial-state correction, recurrent mean transitions, persistent modes and residual
paths, exact supported likelihood, scene-aware evaluation, and exact training resume.
`age_aware=false` disables covariance aging, not all age features: both the mean
transition and posterior head already consume legal age. Tracker TTL is 2 seconds;
forecast support persists for tracks positively observed earlier in history. Thus
belief age can exceed TTL without any collector error. No cross-ID work is planned.

## Additional mechanism from code audit, before physical training

`correct_initial` gates velocity revision by max(1-exp(-age),1-velocity_valid).
Fresh, supported finite-difference velocities therefore receive no correction,
although they are estimates rather than observations. The 0.75s trailing window
can lag stops/turns. A capacity-identical `posterior_velocity_v2` control removes
only that gate, retaining exact fresh positions and the existing speed bound.
A separate 0.15s-window posterior control isolates preprocessing from learned
velocity correction. Existing CV versus filtered-CV development occupancy AP
(0.10647 versus 0.09908) supports testing this mechanism; it does not establish
that either learned-model intervention will help. Both are bounded 1k controls,
bringing the prospective physical pilot count to seven (maximum eight).

## Final bounded controls, registered before results

The eighth and final physical pilot changes only the window proposal: existing
windows with any legal ever-seen track at least one second old receive four
sampling slots, other windows one. Roots remain uniformly shuffled. Inverse
proposal weights restore the original within-root objective, with explicit
finite-minibatch self-normalization bias over supported roots. Original legal
histories are unchanged. `sampling.json` records actual windows separately from
duplicated sampling slots. The baseline 1k posterior is the matched control.

The anonymous occupancy control fits root-balanced conditional empirical cell
probabilities, shrunk toward the existing global prior by 128 or 512 equivalent
roots. Legal context consists of ever-observed track count (capped at three),
any current visibility, and any track age >=1s. A separate control additionally
conditions on known static navmesh walkability/clearance; no cells are hard
masked. The four combinations share targets and fitting weights. This component
predicts anonymous occupancy marginals, not exact identities or coherent anonymous
paths. Evaluate anonymous and total AP/Brier/log score on matching support.
The frozen trajectory model receives its original unavailable-geometry inputs;
passing newly available maps into its untrained geometry weights would silently
change the supposedly frozen physical means and is explicitly prevented.

## Bounded probability-law follow-up

The baseline-selected median long-gap visualization retains a nearly stationary
future while the learned mean continues moving. Together with the slow-motion
occupancy deficit, this motivates one frozen-parameter stopped-motion alternative.
Fit an 18-parameter two-branch gate from the existing eight legal context features,
using root-balanced complete-three-second NLL, 512 updates, first on 1,024 fitting
roots. The learned branch and a constant last-legally-observed-position branch
share the parent's age-aware OU residual law and variance probabilities. Agent
branch choices persist across the entire future, under the same shared scene mode.
This changes the distribution mean but trains no physical transition parameters.
Both comparisons use 48 stratified paths per bank and the same geometry occupancy
prior. Promote to full-root fitting only for a useful proper-score or occupancy
gain without the preregistered regressions. A positive result must separately earn
a coherent recurrent observation-update implementation; the offline experiment
is not a deployable World Model. This is not a ninth physical-training pilot.

The first stopped-law control completed with substantial likelihood/AP gains but
ES regressions above the promotion tolerance. One fixed half-probability follow-up
is allowed before rejecting or promoting this family: multiply its fitted stop
probability by .5, reassign remaining mass to the learned branch, and use the exact
same frozen gate, covariance, scenes and Monte Carlo seeds. This is a proper
mixture distribution, not occupancy-only sharpening or clipping. No grid search,
additional gate optimization, or new physical pilot is planned at this stage.

The 4,096-root stage also includes a fresh matched original-posterior control at
16,384 updates/batch 32. The existing 4k experiment used another mean family and
is not a capacity/architecture control for self-only versus posterior dynamics.

## One bounded horizon-calibration follow-up

The completed age-aware three-second fit improves older-history coverage/NLL,
but adding the stopped regime puts all-window one-second ES just beyond the 3%
regression tolerance. The fitting census also establishes that many real
one-second windows lack complete three-second support. During full physical
training, permit one frozen-incumbent probability control using complete 1s and
3s targets: average windows within each root/horizon, average available horizons
inside each root, then average supported roots. Overlapping horizon terms are
not independent futures. Keep the same 70 probability parameters, age propagation,
512 Adam steps, initialization, regularization, legal contexts, physical means,
noise parameters and fixed-K development evaluation as `cal_all_age_cond`.

This tests additional supervised information and horizon weighting, not a new
physical architecture or a variance-inflation objective. A useful result must
meet the existing ES/NLL/coverage progression rule at both horizons without a
material occupancy regression. The ordinary 3s-only fit remains the control.
There is no horizon/weight grid: only the predeclared pair {1s, 3s} with equal
within-root horizon weights. If it fails, retain the existing calibration. If
it earns promotion, verify it on the selected full physical family and replicate
seed before release. Do not delay the gated full physical fits for this control.
## Occupancy reference with the same frozen geometry prior

One bounded readout control applies the already selected map-only strength-128
anonymous prior to the calibrated CV reference, using CV's declared .15s legal
velocity preparation. Compare only unconditional versus that fixed prior; no new
counts, probability parameters, geometry thresholds or hyperparameters are fit.
This separates gains in the learned trajectory law from gains a simpler motion
reference can obtain from the same anonymous context. It is a diagnostic reference,
not a ninth physical pilot or a new tuning grid. Run in an available CPU analysis
slot after the priority physical runs, under the user's revised adaptive guard;
do not delay full-data physical training.
