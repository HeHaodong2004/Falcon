# Persistent stopped-motion probability control

This is a development finding, not a completed full-data physical-model release.
The frozen backbone is the existing full-data primary with all-window age-aware
calibration. A learned alternative stays at the last legally observed position,
with the same coherent OU residual law. An 18-parameter gate uses the existing
eight legal context features; targets are used only for likelihood fitting.
The initial gate was fitted on 1,024 eligible roots, of which 656 supplied 1,685
complete-three-second windows, for 512 full-batch optimizer updates. Window
likelihoods average within root before averaging roots. No backbone, noise,
variance-category or anonymous-map parameters were fitted in this control.

The NLL-fitted gate places about 44% stationary probability on root origins and
47% across all three-second windows. It improves likelihood and occupancy but
harms ES beyond the promotion tolerance. A single predeclared follow-up halves
that probability, assigning the remaining mass to the learned branch. There was
no grid search or additional optimization of the gate. All models use the same
geometry-only anonymous occupancy prior, exact marginal cell probabilities,
and 48 coherent paths per stratified sample bank.

| Origin development control | ES 1s | ES 3s | NLL 3s | ADE 3s | Total AP 3s |
|---|---:|---:|---:|---:|---:|
| Frozen age-aware parent + geometry | .35581 | .70777 | -.93363 | 1.16822 | .104795 |
| NLL-fitted stopped mixture | .37201 | .73185 | -1.02121 | 1.21463 | .117236 |
| Fixed half-weight stopped mixture | .35980 | .71004 | -1.01879 | 1.17286 | .112342 |

These ES numbers share the control's all-window batch/RNG order, which differs
from the primary root-origin benchmark; they must not be subtracted from the
primary benchmark's ES numbers. Likelihood and occupancy are deterministic and
match the corresponding parent exactly. The CV development AP reference is
.10647; this candidate's AP exceeds that point estimate. The later independent
seed control below replicates this development finding. Protected confirmation
remains outstanding.

For the half-weight law, origin-three-second paired differences are ES +.002265
[-.01112, .01399], NLL -.08517 [-.13059, -.04875], AP +.007547 [.003502, .011324],
and Brier -.000002825 [-.000005397, -.000000469]. Intervals bootstrap scenes,
1,000 replicates, keeping root weights. Across all windows, three-second ES
changes +.28%, NLL improves .11253, AP improves .009288, and Brier/log score
both improve with intervals excluding zero. The largest ES regression is 1.97%
at one second across all windows, within the declared 3% tolerance.

This earns full-root gate fitting and recurrent implementation. It does not
substitute for the separately required new full-data physical training.
The full-weight alternative is retained as a likelihood/AP tradeoff, not selected
as primary. No further gate dilution search is planned.

The recurrent implementation keeps a learned branch and a stopped-branch OU
residual state for each legal identity under each shared scene mode. Branch and
variance identities persist across sampled time. Arrived observations update
the Gaussian states, per-agent branch/variance probabilities, and scene-mode
weights. The learned branch retains the original physical correction; the stopped
branch assimilates measurements into its residual mean. The shared unscaled
Gaussian covariance remains equal across branches because their dynamics/noise
are shared and variance scaling cancels in the Kalman gain. After an observation,
draw new posterior samples, as in the original interface.

Tests verify exact direct/composed means and sample paths before and after
correction, analytic two-agent Bayes updates, offline likelihood/occupancy parity,
CDFs, vectorized density ranks, query independence, missing observations, future
observation rejection, serialization, and anonymous occupancy composition. The
belief exposes its learned core plus explicit stopped state and branch weights;
its covariance property describes within-branch residual uncertainty. Forecast
position variance additionally includes between-branch spread. Exact CDFs and
cell occupancy integrate the full mixture, rather than treating its mean as a
single Gaussian. The full-root results and actual-arrival audit are recorded below.

Receipts: `stopped_control_1024/result.json` and
`stopped_half_control_1024/result.json`. Source: `performance_v2/stopped_control.py`
for the bounded experiment and `performance_v2/regime.py` for the recurrent law.

## Completed full-root gate fit

`stopped_control_full_incremental/result.json` completes the same 512-update,
18-parameter gate fit on all 8,965 eligible fitting roots: 5,782 roots supply
14,920 complete-three-second windows. The physical backbone remains frozen.
This is full-root probability fitting, not newly completed physical training.
The single predeclared half-weight law gives:

| Population | ES 1s | ES 3s | NLL 3s | ADE 3s | Total AP 3s |
|---|---:|---:|---:|---:|---:|
| Root origins | .35928 | .71033 | -1.01878 | 1.17298 | .112423 |
| All windows, root-balanced | .63677 | .81742 | -1.09196 | 1.38269 | .117326 |

These use the same control-specific RNG order as the earlier table. Against the
age-aware geometry parent, origin-three-second differences are ES +.002556
[-.01040, .01423], NLL -.085157 [-.13041, -.04874], and AP +.007628
[.003610, .011349]. All-window three-second Brier and occupancy log-score
differences are -.000003292 [-.000005702, -.000001054] and -.000033062
[-.000060124, -.000006205]. All-window one-second ES worsens 1.84%; its paired
interval [.000268, .022939] excludes zero. This is a real sharpness/proper-score
tradeoff despite the likelihood and occupancy gains. Final eligibility must be
checked against the original incumbent as well as this age-aware parent.

The full-root and 1k gate outcomes are close. Increasing gate-fitting data has
not produced a large additional gain. The following completed checks keep
standard-protocol scores separate from the control-specific RNG above.

The completed standard primary-protocol evaluation gives origin ES .350176,
.546068, .703454 at one/two/three seconds; NLL3 -1.018782; total AP3 .112423;
and nominal 58/65 joint coverage .823129. All-window ES1 .634354 and ES3 .827969
must be compared with the same standard-protocol incumbent (.615829 and .813458),
not the control-specific RNG numbers above. The all-window ES1 increase is just
over 3%, so this combined candidate is not automatically eligible as primary.

The actual-arrival audit uses 151 supported adjacent-window pairs from 71 roots
and 65 scenes. Observation correction reduces ES from .737906 to .533217, NLL
from -.544606 to -.794529, and ADE from 1.376343 to .916167. This demonstrates
useful correction; it does not establish better correction ES than the age-aware
parent. Direct/composed position and persistent-path discrepancies are exactly
zero in that audit. These checks exercise actual arrived observations and
preserve the legal track identities.

## Independent seed and conditional uncertainty

The identical probability intervention was fitted on the released second
physical seed, using all fitting roots and the same fixed half-weight law and
geometry prior. These are fresh probability fits over an independently trained
physical backbone; they are not the new v2 full-data physical training runs.

| Standard protocol | Seed 14 parent | Seed 14 regime + geometry | Seed 15 parent | Seed 15 regime + geometry |
|---|---:|---:|---:|---:|
| Origin ES 1s | .347622 | .350176 | .347225 | .352738 |
| Origin ES 3s | .703191 | .703454 | .710418 | .710687 |
| Origin NLL 3s | -.913030 | -1.018782 | -.925287 | -1.018127 |
| Origin total AP 3s | .093107 | .112423 | .093133 | .112296 |
| All-window ES 1s | .615829 | .634354 | .616537 | .642928 |
| All-window ES 3s | .813458 | .827969 | .816018 | .836716 |

Likelihood and occupancy gains replicate. The all-window one-second ES increases
by approximately 3.01% and 4.28% against the corresponding released parents,
exceeding the preregistered 3% tolerance. This combined frozen-backbone control
therefore remains a tradeoff alternative, not an automatically eligible primary.
The new full-data physical family must be assessed independently.

For seed 14, nominal 58/65 coverage among histories at least three seconds old
increases from .308642 to .9236 at a one-second horizon, using 108 roots in 85
scenes. At three seconds it increases from .5714 to .928571, but only 21 roots
in 20 scenes support that slice. Central 32/65 coverage across all one-second
windows is .6129, above its .4923 nominal level. Better tail coverage does not
establish calibration at every density level.

Physical-tail diagnostics show fewer samples above 3 m/s and 6 m/s², and better
temporal variogram scores. Spatial variogram quality is essentially unchanged;
the outside-grid fraction and coarse nonwalkable-map proxy worsen. See
`PHYSICS.md` for exact populations, values and the limits of the static-map proxy.
No sample was silently discarded or projected for these evaluations.

Receipts: `incumbent_s15_evaluation/result.json`,
`incumbent_s15_regime_geometry_evaluation/result.json`,
`regime_full_incremental_conditional_coverage/result.json`,
`regime_full_incremental_recurrent_audit/result.json`, and
`regime_full_incremental_physics/result.json`. A same-geometry-prior CV control
is preregistered separately; exceeding the original CV readout alone does not
attribute every occupancy gain to learned dynamics.
