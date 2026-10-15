# Development physical and dependence diagnostics

These completed diagnostics compare the immutable v1 primary with the full-root
half-weight stopped-motion law on the same frozen primary backbone. They do not
describe a newly completed full-data physical architecture or reserved confirmation.
All 256 development roots and their 1,423 actual windows are inspected. Accuracy
requires complete supported identity targets; sample-tail diagnostics include
every valid legal forecast slot, including slots lacking target association.

| All-window 3s diagnostic, root-balanced | Incumbent | Age-aware stopped mixture |
|---|---:|---:|
| Sample speed above 3 m/s | .015422 | .009255 |
| Mean within-window 95th speed percentile (m/s) | 2.353878 | 2.218914 |
| Sample acceleration above 6 m/s^2 | .009989 | .005137 |
| Mean within-window 95th acceleration percentile (m/s^2) | 4.364429 | 4.116320 |
| Sample positions outside +/-8 m grid | .050375 | .071220 |
| Nonwalkable-cell proxy, conditional on known grid support | .509684 | .529115 |
| Adjacent-tick displacement variogram score | .111108 | .101682 |
| One-second displacement variogram score | .300028 | .271774 |
| Simultaneous cross-agent distance variogram score | .258401 | .259274 |

The probability change reduces extreme speed/acceleration occurrences and improves
the temporal-dependence scores. Cross-agent dependence does not improve clearly;
outside-grid probability and the map proxy worsen. These are measured tradeoffs,
not evidence of a static-feasibility guarantee. The diagnostic receipts retain
root/scene intervals; differences in this table alone are not paired significance
claims. Full-model and seed-replication comparisons remain required.

Variograms use power .5 of Euclidean displacements and two independent weighted
48-path stratified banks. The product of their moment errors is an unbiased
estimate of the squared moment score and may be negative at finite K. This score
is proper but not strictly proper and supplements ES/NLL. It is not substituted
for the preregistered primary score. Speed and acceleration use finite differences
of complete sampled paths, including the explicitly modeled readout noise. No
sample is clipped, discarded, rejected or projected in these diagnostics.

The map is an eroded navigation mesh sampled at 0.5 m cell centers and the legal
robot height. It is not an exact human-body collision oracle. The separate
target-only audit finds that 16.89% of supported tracked target positions are
also in proxy-nonwalkable cells over all three-second windows (16.83% at origins).
Their outside-grid fraction is 3.33% (2.01% at origins). For all native target
humans, the corresponding all-window rates are 17.37% and 9.47%. These reference
populations differ from the forecast-tail population; subtracting their rates
does not yield a paired wall-violation excess. The result establishes a material
proxy limitation while preserving the evidence of broad sample dispersion.

The detailed accuracy records also cover horizons 1/2/3, acceleration, turning,
speed, proximity to another supported human, navmesh clearance, crowd density,
observation/history support, source scene family and category. Dynamic labels
define diagnostic groups only after forecasts are produced. Coherent samples
and their probabilities remain unchanged across groups.

Receipts: `incumbent_physics/result.json`,
`regime_full_incremental_physics/result.json`, and
`map_support_audit/result.json`. The four-root `physics_software_smoke` is a
software check only and is excluded from these model comparisons.

## Age/geometry candidate under final review

The primary candidate's all-window three-second speed >3 m/s fraction is
.010919 versus .015422 for released v1; acceleration >6 m/s² is .005429
versus .009989. The respective p95 values are 2.2674 versus 2.3539 m/s and
4.1485 versus 4.3644 m/s². These are improvements in marginal sample tails,
not a universal physical-validity guarantee. Outside-grid probability rises
from .050375 to .075748 and the nonwalkable-navmesh proxy from .509684 to
.540729. Coverage of known map cells falls accordingly. The broader aged-state
belief has a real geometric cost; no samples are clipped, projected or discarded.

Evidence: candidate_review_primary_physics/result.json and incumbent_physics/
result.json. Final replicate and confirmation diagnostics remain separately
identified and must not be inferred from these primary development numbers.

## Completed new confirmation diagnostics

Corrected protected-population receipts compare the same frozen primary pairs.
All-window three-second results are root averaged; no samples are removed.

| Bank / law | Speed >3 m/s | Acceleration >6 m/s² | Outside grid | Nonwalkable proxy |
|---|---:|---:|---:|---:|
| natural / v1 | 0.021617 | 0.010023 | 0.094795 | 0.536327 |
| natural / age alternative | 0.017222 | 0.005582 | 0.147714 | 0.570418 |
| unseen / v1 | 0.017054 | 0.010085 | 0.084995 | 0.510409 |
| unseen / age alternative | 0.012209 | 0.005570 | 0.119992 | 0.543951 |

The tail improvement and geometry deterioration both replicate across these banks. Eroded-navmesh nonwalkability is not an exact wall-contact oracle. Outside-grid probability is separately retained, not silently excluded from the geometry claim. The original wrong-population attempts failed before scores and remain preserved. No diagnostic change affected the frozen primary panel.
