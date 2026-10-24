# Scientific validation

**Forecasting/belief quality REVISE; descriptive development audit only.**

Frozen trained checkpoint, no fitting/tuning/confirmation access. 83 frozen development roots / 59 scenes. Predictions are serialized before offline truth is opened; only legal history/graph/references enter the compiler. All selected payload hashes match the original frozen snapshot.

| Target/alignment | Requested H s | Rows / roots / scenes | Positive | Scene/root-balanced Brier [95% CI] | AUROC | Within-root ranking | ECE |
|---|---:|---|---:|---|---:|---:|---:|
| nominal_envelope/exact | 0.5 | 307/58/48 | 79 | 0.0617 [0.0140,0.1259] | 0.903 | 0.970 | 0.0601 |
| nominal_envelope/exact | 1.0 | 103/46/40 | 7 | 0.0283 [0.0004,0.0838] | 0.759 | 0.500 | 0.0243 |
| nominal_envelope/exact | 2.0 | 103/46/40 | 7 | 0.0283 [0.0004,0.0838] | 0.759 | 0.500 | 0.0243 |
| nominal_envelope/exact | 3.0 | 103/46/40 | 7 | 0.0283 [0.0004,0.0838] | 0.759 | 0.500 | 0.0243 |
| nominal_envelope/interpolated | 0.5 | 471/83/59 | 127 | 0.0727 [0.0264,0.1277] | 0.924 | 0.974 | 0.0679 |
| nominal_envelope/interpolated | 1.0 | 471/83/59 | 147 | 0.0857 [0.0399,0.1423] | 0.919 | 0.956 | 0.0711 |
| nominal_envelope/interpolated | 2.0 | 471/83/59 | 155 | 0.0899 [0.0415,0.1477] | 0.922 | 0.965 | 0.0663 |
| nominal_envelope/interpolated | 3.0 | 471/83/59 | 158 | 0.0890 [0.0414,0.1449] | 0.924 | 0.959 | 0.0645 |
| social_horizontal_proxy/exact | 0.5 | 307/58/48 | 55 | 0.0339 [0.0054,0.0811] | 0.936 | 0.897 | 0.0226 |
| social_horizontal_proxy/exact | 1.0 | 103/46/40 | 3 | 0.0281 [0.0002,0.0836] | 0.440 | 0.500 | 0.0218 |
| social_horizontal_proxy/exact | 2.0 | 103/46/40 | 3 | 0.0281 [0.0002,0.0836] | 0.440 | 0.500 | 0.0218 |
| social_horizontal_proxy/exact | 3.0 | 103/46/40 | 3 | 0.0281 [0.0002,0.0836] | 0.440 | 0.500 | 0.0218 |
| social_horizontal_proxy/interpolated | 0.5 | 471/83/59 | 96 | 0.0499 [0.0142,0.0943] | 0.940 | 0.888 | 0.0353 |
| social_horizontal_proxy/interpolated | 1.0 | 471/83/59 | 125 | 0.0683 [0.0276,0.1146] | 0.932 | 0.903 | 0.0507 |
| social_horizontal_proxy/interpolated | 2.0 | 471/83/59 | 133 | 0.0742 [0.0309,0.1226] | 0.928 | 0.919 | 0.0490 |
| social_horizontal_proxy/interpolated | 3.0 | 471/83/59 | 137 | 0.0764 [0.0333,0.1257] | 0.930 | 0.914 | 0.0452 |

## Targets, time support and inference boundary

Collision target is the audited all-phase rectangle/disk enclosing-body overlap
of the **nominal reference** with the recorded exogenous human center paths.
Social target is horizontal Euclidean center distance <1 m. These targets match
the declared readouts. They are not native articulated mesh contact, installed
geodesic/XYZ PSC, L4 closed-loop completion or episode success. No old follower
completion label is reused as L4 supervision. No RL experiment occurred.

Exact targets require all nominal 48 ms endpoints to occur on the recorded
native tape. Some actual intervals are 56 ms. Censor ledger: {'EXACT_NATIVE_CLOCK_UNSUPPORTED': 1268, 'UNSUPPORTED_OR_TRUTH_HORIZON_CENSORED': 12}.
Exact 1/2/3 s rows coincide because only shorter queries retain complete exact
clock support; these are not independent long-horizon confirmations. The table's
H is the requested forecast cap; actual covered duration is min(query duration,
native horizon), retained per row in the compressed label ledger.

The separately named interpolated auxiliary target uses the existing L2 offline
`truth_at` linear interpolator at the **same absolute nominal times**, with no
extrapolation. It supports 471 legal candidates per requested horizon. It tests
the finite center-path/envelope proxy on time-aligned paths, not exact native
execution/contact. Both cohorts are reported rather than silently rounding
timestamps or discarding clock failures.

## Metrics and uncertainty

Brier intervals use 1,000 fixed-seed scene bootstrap samples: candidates average
within root, then roots within scene. AUROC/ECE/reliability are descriptive row
metrics; within-root ranking compares opposite-label candidates with half-credit
ties. Five predeclared probability bins, including empty-bin counts, are stored
in `artifacts/scientific.json`; the standalone reliability plot uses the auxiliary
interpolated nominal envelope target. Intervals exclude training-seed variation,
geometry error, interpolation error and selection reuse. Multiple correlated
horizons/candidates do not create independent evidence. No statistical claim
is made from the point AUROCs alone.

Sixteen root worlds provide coarse finite-bank support. Zero sampled collisions
does not mean safety. Effective sample support and an explicitly conditional
zero-hit iid reference bound are exposed; the iid coverage assumption is not
established for the inherited stochastic carrier. There is no guarantee against
model error, unknown populations or temporal aliasing. Social region `mass` is
weighted endpoint exposure count (multiply by .048 for duration), not an any-event
probability; collision and first-observation region masses are first-event mass.

## Inherited limitations and decision

The original 4.2% native-gap path ES decrease has paired difference −0.0247 and
95% CI [−0.0905,+0.0479]; improvement is **not statistically established**.
Belief/count support, duplicate/recycled ancestry, missing preferred tracks,
association/reobservation, speed/turning and broad hidden spatial priors remain
unresolved. The selected development scenes were already used by L2; this is
not fresh confirmation or a model-accuracy certification.

Coarse any/none likelihoods conserve their finite optical law and never mutate
actual belief, but do not prove legal exact visibility, native occlusion,
identity reobservation or useful Bellman transitions. Reobservation probability,
static revelation, exact sensor-conditioned belief, official contact/PSC,
L4 completion and full navigation success remain unsupported. Core compilation
works independently of these capabilities. Tests establish implementation
correctness for the declared law; scientific readiness remains REVISE.

The additional illustrative graph/outcome figures use a qualitative root chosen by maximum predicted candidate-risk contrast, without truth labels. See `illustration_selection.json` and `illustrative_decision_packet.json`; this does not change the fixed benchmark/scientific cohorts.
