# Fixed-mean probability controls: development evidence

All results below are completed, training-only calibration experiments. Physical
means and mode weights are bitwise unchanged. Each fit used 512 Adam updates;
only variance-category logits/context weights changed. Global OU parameters were
held fixed. All-window fits average supported windows within each fitting root:
5,782 roots / 14,920 windows versus 5,245 roots / windows at root origin. The
eligible fitting population remains 8,965 roots; repeated windows are dependent.

| Probability law | Origin ES 1s | Origin ES 3s | Origin NLL 3s | Origin AP 3s | All-window ES 1s | All-window ES 3s | All-window NLL 3s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released primary | .34762 | .70319 | -.91303 | .09311 | .61583 | .81346 | -.90544 |
| Origin conditional | .34813 | .70422 | -.91474 | .09430 | .61534 | .81159 | -.90720 |
| All-window unconditional | .34763 | .70543 | -.91316 | .09448 | .61586 | .81559 | -.90580 |
| All-window conditional | .34671 | .70335 | -.91497 | .09630 | .61422 | .81310 | -.90770 |
| All-window conditional + aged initial covariance | .34890 | .70214 | -.93363 | .09705 | .62437 | .81856 | -.97947 |
| All-window unconditional + aged initial covariance | .34906 | .70531 | -.93176 | .09613 | .62598 | .82195 | -.97785 |

ES/NLL lower is better. AP higher is better. Means/ADE/FDE are exactly unchanged.
The original root-origin protocol uses 147 supported 3s trajectory roots; all
windows provide 166. Occupancy AP uses foreground-positive supported roots and
therefore a different population; probabilities also include empty scenes.

The conditional aged-covariance intervention has a clear all-window likelihood gain: candidate-minus-v1
NLL is -.20032 at 1s (scene 95% interval [-.23433,-.16888]) and -.07403 at 3s
([-.09542,-.05269]). Its ES changes are +.00854 at 1s (1.39%, interval
[-.00339,.01999]) and +.00510 at 3s (0.63%, [-.00261,.01342]). These ES intervals
span zero; do not call the small origin ES reduction a proven forecasting gain.
Origin occupancy AP increases .003945 ([.001337,.007089]), but remains below
the CV reference. Brier improves by .00000534; occupancy log-score change is
inconclusive. These are paired root comparisons with 1,000 scene bootstraps.

## Conditional coverage explains the tradeoff

64-draw joint density ranks give a finite nominal level of 58/65 = 89.23%.
These are complete supported joint marginals for each age stratum, evaluated
on unchanged development histories, averaging windows inside each root first.
The conditional diagnostic uses the same seeds across models but a separate
stream from the primary root-origin benchmark.

| Horizon / legal age | Roots | Released coverage | Aged-covariance conditional coverage |
|---|---:|---:|---:|
| 1s / <.25s | 169 | 92.84% | 91.85% |
| 1s / 1–2s | 145 | 76.67% | 88.05% |
| 1s / 2–3s | 129 | 55.56% | 87.86% |
| 1s / >=3s | 108 | 30.86% | 91.82% |
| 3s / <.25s | 153 | 88.89% | 86.93% |
| 3s / 2–3s | 67 | 66.42% | 88.81% |
| 3s / >=3s | 21 | 57.14% | 90.48% |

The >=3s age group improves one-second coverage by 60.96 percentage points
(paired scene interval [53.57,68.63]) and NLL by .69177 nats/coordinate
([.61146,.77471]). Its 3s coverage estimate has only 21 independent roots and
20 scenes; the gain interval is [14.29,55.00] points. This is substantially
stronger evidence for corrected stale-state uncertainty than overall origin
coverage, which drops from 85.03% to 82.31%. Sampled body-center speed >3m/s
falls to .96%; finite-difference speed includes readout jitter and is not the
same quantity as bounded mean-state velocity.

The cached-density unconditional aged control also completed. Most of the
likelihood gain remains without context-dependent weights. Conditional weights
give small additional ES/AP gains. With conditional weights but no covariance
aging, >=3s-age one-second coverage reaches only 37.11%, compared with 30.86%
for v1 and 91.82% with covariance aging. This supports uncertainty propagation
across missing intervals as the main measured mechanism.

The same conditional aged-covariance fit also completed on the released second
physical seed. Origin ES1 changes from .347225 to .349847, ES3 from .710418 to
.708606, NLL3 from -.925287 to -.954442, and AP3 from .093133 to .097536.
All-window ES1 changes from .616537 to .633184 (2.70% higher), ES3 from .816018
to .825417 (1.15% higher), and NLL3 reaches -.994931. Thus the likelihood/AP
mechanism replicates while the Energy Score tradeoff is larger on this seed.
This is probability fitting over a second completed v1 backbone. The separate
new full physical seeds have also completed and are compared in
DEVELOPMENT_BENCHMARK.md.

Decision: retain aged initial covariance as a serious Pareto alternative, not
an unconditional replacement, and combine only with physical models that earn
promotion. The five fixed-mean controls and all three initial conditional
coverage diagnostics are complete. A single preregistered follow-up fits the
same conditional law jointly at one and three seconds, balancing roots and
supported horizons. Its density cache now contains 26,338 unique supported
windows, 41,258 window/horizon pairs and 6,180 supported roots out of 8,965
eligible roots. Optimization and the fixed-protocol evaluation are complete; see
HORIZON_FOLLOWUP.md for both its ES gain and central-coverage regression.
No horizon-weight grid is planned.
No confirmation population has been opened for these comparisons.

Before that control completes, bound its possible follow-up: only if it earns
the existing practical development progression thresholds, apply the same
one/three-second calibration to the development-leading physical family and
its independent seed. Evaluate the age law alone and with the already fixed
geometry prior / half-weight stopped law. If the variance law changes, refit
the same 18-parameter stopped gate on fitting-only density inputs; reusing a
gate optimized against a different variance law would confound the comparison.
Keep 512 updates, the existing regularizer, and stopped weight .5. This is one
replicated probability-recipe follow-up, with no new physical architecture,
gate-weight search, horizon-weight search, or confirmation access.

Receipts: `cal_*_evaluation/result.json`, `*_conditional_coverage/result.json`,
and `calibration_comparison.json` under the versioned v2 output root. The latter
pins source/result hashes. The final development choice is frozen in SELECTION.md. Confirmation and
release are separate; no already-opened reserved bank informed these fits.

The three-second age/geometry law is selected as a documented tradeoff after
two-seed verification. At origin 3s, NLL differences versus each corresponding
v1 seed are −.020596 [−.035388, −.007272] and −.029155
[−.053814, −.007393]. AP gains are separately attributed to both variance
fitting and the map-conditioned anonymous prior. All-window ES1 costs are
1.39%/2.70%; the second seed's paired interval excludes zero. Old-history
58/65 coverage improves strongly, while central one-second sets can be too
broad. This is a conditional probabilistic benefit, not universal calibration.
The bounded horizon/stopped follow-up was a complementary-law diagnostic,
not a waiver of its measured central-coverage deficit or primary promotion.
It did not resolve that deficit and its search is closed.
