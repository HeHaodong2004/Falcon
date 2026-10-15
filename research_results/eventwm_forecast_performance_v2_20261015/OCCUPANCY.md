# Occupancy diagnosis and completed readout controls

The original total-occupancy evaluation is spatially aligned correctly. The
32x32 grid spans +/-8m, with x in columns and z in rows. Targets and exact
mixture cell integrals agree on those axes. The original diagnostic called
`tracked_occupancy` used all-human truth; the v2 diagnostic instead compares
supported tracked-human marginals on identical predicted/target populations.
This correction does not change the valid primary total-occupancy metric.

On development root origins at 3s, the released primary has total foreground
AP .093107 versus CV .106467. Supported-tracked AP is .159743 versus .198067.
The paired primary-minus-CV tracked AP interval is [-.06283,-.01302], while
tracked occupancy log score improves by .000330. Thus the ranking gap is real,
and does not imply that every probability-quality measure is worse. Physical
trajectory accuracy also improves. A single occupancy-head explanation would
miss these distinctions.

## Root-balanced anonymous readout

Anonymous targets exclude uniquely associated supported human positions. All
physical exposure masks are retained, including empty scenes. One fitting root
contributes its mean over supported windows at each tick. No anonymous identity
trajectory is supplied or predicted. The full 8,965-root fitting population is
used; targets never construct the legal context or map features.

Conditional empirical cell probabilities are shrunk toward the global prior by
128 or 512 equivalent roots. Legal history bins describe capped ever-observed
track count, any current visibility, and any age >=1s. Map bins describe unknown,
nonwalkable, low-clearance walkable, and higher-clearance walkable cells. The
known navmesh is eroded for a navigation body: it is a covariate, not a definitive
human collision label. No cell is hard-masked to zero.

| Anonymous readout, frozen v1 trajectories | Total AP 3s | Total Brier | Total log score |
|---|---:|---:|---:|
| Global prior | .093107 | .00219316 | .01331440 |
| History context, shrinkage 128 | .093580 | .00219344 | .01340089 |
| History context, shrinkage 512 | .093968 | .00219321 | .01334284 |
| Context + map, shrinkage 128 | .098323 | .00218851 | .01262701 |
| Context + map, shrinkage 512 | .096847 | .00219049 | .01280824 |
| Map only, shrinkage 128 | .100704 | .00218561 | .01226916 |
| Map only, shrinkage 512 | .100062 | .00218659 | .01233880 |

These are completed development-origin measurements. Map-only/128 improves AP
by .007597 (paired scene 95% interval [.005988,.009189]), Brier by .00000755
([.00000608,.00000908]), and log score by .001045 ([.000883,.001202]). All-window
AP improves from .092854 to .100192. Context-only changes are inconclusive and
can worsen log score; retain the simpler geometry-only readout. The first map
control nearly doubled anonymous AP; total AP remains a distinct population.

Combining map-only/128 with the completed aged-covariance conditional law gives
total AP .104795, Brier .00218020, and log score .01225489. Trajectory ES/NLL
match the underlying age-aware checkpoint exactly. This is a measured 12.6%
relative AP improvement over v1, but it still falls short of CV. No reserved
forecast scores have been used to select this component.

## Released-interface candidate

`performance_v2.hybrid` stores 81,920 anonymous cell probabilities and a persistent
forecast clock. Direct and composed anonymous marginals agree exactly. It keeps
the map frame fixed through imagination and stable-ID observation correction;
re-encoding legal history may establish a new frame. Stable-track observations
correct the tracked distribution; no new information about anonymous identities
is invented. This component has no joint anonymous sample-path law.

The physical model receives its original unavailable-map inputs, preventing
untrained geometry weights from changing its means. Total occupancy uses
1-(1-p_tracked)(1-p_anonymous), an explicit conditional-independence approximation.
Anonymous supervision supports up to 5s where physical labels exist; tracked
trajectory dynamics remain trained to 3s. Outputs mark anonymous support after
5s unavailable rather than treating a missing prior as certified free space.

Receipts: `diagnose_primary`, `diagnose_cv_correct_cache`, `occupancy_baseline_comparison.json`,
`anonymous_occupancy`, `anonymous_geometry_control`, and `combined_age_geometry_evaluation`
under the versioned output root. The combined checkpoint is a development
candidate; it is not a frozen final release or a completed new physical-training run.

The preregistered same-prior CV control has completed, without changing any
trajectory parameter or fitting additional occupancy counts. Giving CV the same
geometry-only anonymous prior raises origin AP from .215445 to .223004 at one
second, and .106467 to .113897 at three seconds. The three-second paired AP
difference is .007430 [.005702, .009220], with 188 roots / 134 scenes. Brier
improves from .002190886 to .002183157 and occupancy log score from .013451428
to .012391399. All-window AP3 similarly rises .115573 to .122462.

Thus the anonymous-map benefit is available to CV as well as learned dynamics.
The existing full stopped-law candidate AP3 .112423 exceeds the original CV
reference but does not exceed this same-prior CV control. Report both references;
do not attribute the entire occupancy gain to improved human dynamics. Receipt:
`cv_geometry_readout_control/result.json` (38.66 seconds).

The completed horizon-balanced stopped alternative reaches origin AP3
.115125/.115078 across physical seeds, slightly above same-prior CV .113897
as point estimates. Its calibration regression prevents primary selection;
occupancy alone does not establish a better joint future distribution. The
preferred age/geometry family reaches .104795/.105357 and therefore remains
below CV. Both occupancy successes and the remaining ranking deficit are
retained in the frozen confirmation panel.
