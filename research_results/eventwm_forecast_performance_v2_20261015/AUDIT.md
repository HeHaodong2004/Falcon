# Completed baseline and data audit

All measurements here use fitting or existing internal development data. No v1
confirmation predictions were rerun or used for selection.

## Reproduction and information boundaries

The immutable checkpoint, compact-cache manifest, protocol and 350k snapshot
hashes match the supplied release. The original 45 tests pass. Re-evaluating
the primary on the 256 development roots reproduces all checked 1s/3s ES, NLL,
ADE/FDE, occupancy AP/Brier/log scores exactly (maximum absolute difference 0).
Receipt: `baseline_reproduction/result.json`; elapsed evaluation 28.79 seconds.
`baseline_inference/forecast.npz` was produced only from the bundled legal
history and sensor configuration, with no target-cache inference dependency.

The legal adapter replaces supplied velocity fields with causal timestamped
position regression. Numeric tracker IDs are bookkeeping. Future states and
simulator identities occur only in target construction/association support.
Encoding and recurrence do not consume candidate robot queries. History, native
supervision and supported evaluation remain separate.

The tracker deletes entries after 2 seconds since last observation. The history
adapter creates slots for all IDs positively observed anywhere in the arrived
history, and preserves their last acquisition clocks after deletion. The belief
age can therefore exceed 2 seconds. No collector bug or changed-ID repair is
implied. `age_aware=false` disables initial residual covariance propagation;
age already enters history features, the initial-state correction and future
mean transitions.

## Independent data and age support

The fitting population remains 8,965 roots / 48,434 windows and development
256 roots / 1,423 windows. Windows are overlapping observations of the same
physical root; their occurrences are not independent futures.

| Three-second identity support | Fitting origin | Fitting all windows | Development origin | Development all windows |
|---|---:|---:|---:|---:|
| Supported roots | 5,245 | 5,782 | 147 | 166 |
| Supported windows | 5,245 | 14,920 | 147 | 443 |
| Roots with age >=1s | 824 | 3,832 | 21 | 119 |
| Roots with age >2s | 376 | 1,904 | 10 | 69 |
| Roots with age >=3s | 204 | 759 | 4 | 21 |
| Agent-window age median (s) | 0.048 | 0.440 | 0.048 | 0.536 |

Occurrence quantiles are descriptive, not independent statistical samples.
`data_census/result.json` contains every root/window support record, including
1s/2s/3s/5s exposure, category, history length, current visibility and tracker
retention. The later-window development population provides materially more aged
support without using protected groups or creating a contaminated holdout.

Existing physical training is root-uniform then window-uniform. Existing noise
and scale calibration uses only `windows[0]` and complete 3s identities. Thus
simply adding all windows is a change to calibration population, not evidence
that the prior trainer never saw aged examples. Controlled fixed-mean comparisons
separate this change from conditional variance and covariance aging.

## Error and occupancy decomposition

On all development windows, averaging supported windows inside each root, the
incumbent's 3s mean ADE is 1.383 m. Conditional agent-group ADE is 0.998 m for
age <0.25s (153 roots), 1.660 m for 1–2s (115), 2.335 m for 2–3s (67), and
2.952 m for >=3s (21). The oldest subset remains small. At root origins it
contains only four roots and has 4.139 m ADE. Its already-large 1s error indicates
state-estimation error as well as future-process error. Future-chord speed and
turning strata use targets solely for diagnostics, never as inference features.

Total occupancy uses complete physical exposure, human-center cells, and the
unchanged 32x32 grid over ±8m. The coordinate formulas agree: both target and
probability integrate x into columns and y into rows. Cell AP handles tied scores
by thresholds. Empty exposed scenes enter probability scores, not fictitious AP.

The old diagnostic named `tracked_occupancy` compared a tracked-only prediction
against the **full-scene** target; its label did not denote a pure tracked-population
score. Primary total occupancy was correctly named and is unchanged. New
diagnostics explicitly compare the supported tracked marginal against exactly
that target, and separately compare anonymous probability against anonymous truth.
Inference support is never filtered by supervision; marginal support restriction
is only a diagnostic operation.

Incumbent 3s origin AP: total 0.09311 (188 positive roots); supported-tracked
marginal 0.15974 (145); anonymous prior 0.00905 (141). These populations differ
and their scores must not be compared as architecture gains. Total occupancy
log score is 0.013314; supported-tracked 0.007163; anonymous 0.009821. The
anonymous component is demonstrably weak. A matched CV population decomposition
is pending; its original total AP 0.10647 is already reproduced in v1 evidence.

Artifacts retain per-root, per-window and per-agent results with scene bootstrap
intervals. Secondary stratified ES uses explicitly reset diagnostic RNG and must
not replace the primary benchmark's ES, even at identical K.

## Operations

Initial host probe: about 194 GiB available RAM, 379.8 GB free project disk,
and 23.7/24.6 GB free GPU memory after initializing CUDA. The NVIDIA userspace
version differs from the loaded kernel. Extracting the cached matching compute
package under the new campaign runtime and pinning `LD_LIBRARY_PATH` restores
CUDA without any host driver or collector change.

The initial collector was running PID 3719917 at source
`8e318e794c30774aa8d25f350ccec4e4e3af5683`, with 425,931 live queries / 12,907
completed roots. This is not an audited new dataset. Research workers later
paused after three rolling windows below 65% of the original 3,615.965 q/h
reference; that initial policy required three windows above 85% spanning 120s.
The user's subsequent explicit revision replaced this throughput policy with
absolute thresholds and adaptive throttling; see RESOURCE_POLICY.md for the
effective settings and verified exact continuation.
No production process was stopped, restarted, reconfigured or signaled.
