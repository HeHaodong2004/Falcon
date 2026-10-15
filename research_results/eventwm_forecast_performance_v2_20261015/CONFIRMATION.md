# Completed frozen confirmation and release decision

Retain the original v1 primary and independent replicate. Keep both age-aware
candidate seeds and the two predeclared probability-law alternatives as named
alternatives. No model, calibration, hyperparameter, support or seed was changed
after opening the new confirmation banks. The review bound to immutable hashes is
`final_selection/confirmation_review.json` (hash binding is not a digital signature).

## Identity and support

The confirmation-only snapshot contains 444,774 validated queries / 13,478 roots,
including exact inherited 350k identities. Snapshot SHA256:
`59c18f61843f871c09c4249fabd879e4fb0e5b21cb0af731ed93a2873618ca42`.
No newly added confirmation root entered fitting. All 350k roots and their episode,
fixture and trajectory identities were excluded from new membership. Unseen
scenes are disjoint from fitting. Selection used a predeclared root hash, never
forecast targets, observation ages or model results.

Natural/unseen banks contain 128 roots each, spanning 102/44 scenes, selected
from 142/169 eligible new roots. Membership SHA256:
`842155bb96e81c4696d6e91fd4ee92d469638b9794cb820faa1564abbb165777`.
Cache manifest SHA256:
`f6c9f65993d53f9b4dd2d39ed2ad010db7e1af3d2579ec950493924d90c04a4a`.
Geometry manifest SHA256:
`1ee0ae89f6a9caa94991d15b4824ef63bdf2066ef89dc36fdb3b0460b76c030c`.

Development selection was committed at `af2e67f5d5965fdec939a08f5965cc687aa18fd8`
before target extraction. The complete checkpoint/cache panel was committed at
`7f2d75a98c389825953a5ad62668af493e670054` before scoring. Selection SHA256 is
`3042632c79a2640c14499d6ea05732fa9f8fa795b834081e18f2ab7b91e4520f`.
The completed 20-panel receipt SHA256 is
`fc4a26d9ae69020d36d32363b690fed979ed70cbe07c66bfcc9c90b0bb61b57b`.
The final panel completed in 646.97 seconds of its successful invocation.

| Horizon | Natural trajectory roots / scenes | Unseen trajectory roots / scenes |
|---|---:|---:|
| 1 s | 57 / 48 | 77 / 38 |
| 2 s | 55 / 46 | 75 / 38 |
| 3 s | 53 / 44 | 73 / 38 |

At three seconds, complete-scene occupancy has 120/119 roots (96/43 scenes),
while foreground AP has 84/92 roots (69/40 scenes). Empty scenes contribute
proper probability scores but not foreground AP. All model comparisons use
identical support. The low trajectory support is disclosed rather than treating
all 256 members as independent scored trajectories. The predeclared protocol
uses 48 stratified paths per ES sample bank, fixed K=6 minimum-distance samples,
and 64-draw density ranks. Intervals resample 1,000 scene clusters with fixed
seed 83199, retaining root weights. Five-second results are extrapolation on
complete supported labels, not in-distribution three-second results.

## Why the candidate was not promoted

Both candidate seeds improve natural NLL3 by .31348/.33332 nats per coordinate,
AP3 by .01517/.01698, occupancy Brier/log score and high-nominal joint coverage.
The corresponding paired intervals exclude zero. Unseen NLL3 and occupancy gains
also replicate. Candidate primary natural 58/65 coverage rises from .67925 to
.88679; replicate rises from .69811 to .86792. Point forecasts remain unchanged.

However, natural ES2 increases .092980 [.041049,.150451] and
.069431 [.003976,.138206], or 9.72% and 7.13%. Natural ES1 increases 3.79%/3.18%
as point estimates, with inconclusive intervals. Natural ES3 increases
1.22%/3.59%, also with inconclusive paired intervals. Unseen primary ES3 changes
+.008164 [-.010801,.035427]; replicate changes +.024310 [.005454,.049057], or
3.07%. These costs exceed the predeclared 3% ES tradeoff. No test-driven exception
to that ceiling is introduced.

High-level coverage alone hides central overdispersion. The independent fixed-RNG
conditional diagnostic finds natural-origin one-second central coverage .7193/
.6842 against nominal .4923, compared with v1 .4561/.4211. Natural three-second
central coverage is .6226/.5660, compared with .4340/.4528. Old-history 90%-level
coverage improves strongly, but its central sets become too broad. The diagnostic
RNG differs from the primary density-rank panel, so its finite-MC coverage values
must not be silently substituted for the primary table. Full conditional receipts
retain root counts, intervals and visibility/age strata.

The new physical self-only family has modest point-error gains but does not
establish the needed full-data multiobjective advantage. The horizon/stopped
alternatives remain alternatives regardless of favorable test point estimates.
The frozen role rule permits only retaining the candidate family or reverting
to the exact v1 primary/replicate. We choose the latter, preserving all useful
alternatives and negative results.

The complete ten-model tables, all four horizons, both seeds and paired
scene intervals are in [CONFIRMATION_TABLES.md](CONFIRMATION_TABLES.md).
Machine-readable scores and comparisons are under `confirmation_panel/` and
`confirmation_analysis/`. These populations are now opened and cannot be
called untouched in a future tuned campaign.

## Supplementary diagnostic correction

The original new-bank physics jobs failed before scoring because they loaded
old development IDs from the new confirmation cache. Their failure receipts
remain preserved. Commit `606dd30` changes only this diagnostic's population
resolver and result label, with a regression check. Corrected jobs use new output
directories and verify the already-completed frozen panel. No checkpoint,
training, inference, primary metric or confirmation support changed. The full
97-test forecasting suite and all four candidate/fallback standalone proofs were
rerun on the corrected source. Final archive inference has its own external receipt.
