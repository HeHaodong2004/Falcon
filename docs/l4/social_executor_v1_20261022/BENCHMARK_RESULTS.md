# Native navigation benchmark results

**Decision: REVISE.** The executor improves the small native encounter suite,
but its public navigation success rate is substantially below both references.
It must not be accepted on collision rate or PSC alone.

## Protocol and comparison limits

`navigation_public12.json` freezes the first three original public validation
episodes in each of four previously certified scenes: HM3D 3t8DB4Uzvkt and
5cdEh9F2hJL; MP3D 2azQ1b91cZZ and 8194nk5LbLH. Selection preceded evaluation;
no failed episode was removed. Each arm has 12 independent episode roots across
four scenes. The source dataset, episode, map and navmesh hashes and exact
seeds are in the protocol. All 48 public-arm executions have results.

These are **full closed-loop episodes on a 12-episode development panel**,
not the complete published validation corpus or a held-out confirmation study.
The inherited four-episode panel was also used during development; it is not
pooled with this panel as independent evidence. Broad failure already prevents
an acceptance claim; a larger frozen-corpus evaluation is still required.

B0 follows the deterministic nominal native schedule. B1 uses L4 with exact
legal visible-human measurements and short-term motion estimates. B2 adds the
unchanged 500k L2 through shared forecasts. B0/B1/B2 have identical high-level
static shortest-path references, exact query semantics, full static-map access,
native human controllers, seeds, action protocol and episode budget. The
aggregator verifies matching query records across all three arms.

B3 strictly loads the existing `falcon_noaux_25.pth` policy and runs its actual
depth/PointGoal interface, stochastic action sampling and recurrent state with
the same case seeds. It has no explicit map or structured-human detection
advantage. This is a reproducible local Falcon reference, **not an equal-input
policy comparison** or a reproduction of a historical paper score.

Every arm uses the same certified navmesh and unchanged installed native task,
including its WXYZ start-angle conversion of stored episode rotations. The
certified navmesh can differ from the original published Falcon navmesh. Native
STOP, 0.2 m success distance, human collision termination and 500-action limit
are unchanged. SPL, STL and PSC are the original installed evaluator classes.
The native STL reference formula is retained even though its speed convention
differs from measured action timing. Static contacts use the established
articulated contact evaluator with floor support excluded; static contacts do
not terminate the native episode automatically.

L4 queries have a 480-action / 26.88 s cap and a five-second meaningful-progress
watchdog. At completion or failure, the experiment driver explicitly STOPs and
retains the actual native result. No high-level objective is substituted.

## Public episode results

| Variant | Navigation SR | Human collisions | Static contacts | SPL | STL | PSC | Timeout | Blocked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 nominal | 6/12 (50.0%) | 5/12 | 8/12 | 0.5000 | 0.5000 | 0.9730 | 0/12 | 0/12 |
| B1 observed L4 | 1/12 (8.3%) | 2/12 | 0/12 | 0.0821 | 0.0833 | 0.9847 | 0/12 | 8/12 |
| B2 L4 + frozen L2 | 1/12 (8.3%) | 2/12 | 0/12 | 0.0799 | 0.0833 | 0.9847 | 0/12 | 8/12 |
| B3 Falcon | 5/12 (41.7%) | 5/12 | 11/12 | 0.3122 | 0.4167 | 0.9406 | 1/12 | 0/12 |

Successful-episode mean native completion time is 5.795 s, 6.616 s, 7.536 s and
12.242 s respectively. These are conditional means over different successful
subsets, not evidence that B1 is faster at completing the benchmark.

B1/B2 each have eight BLOCKED queries, two collision interruptions, one
REFERENCE_UNSUPPORTED episode and one completed query. Executor-limit events
are 0/12. Six blocks are no-meaningful-progress failures; two are finite-search
failures. The unsupported HM3D goal cannot be snapped to the requested certified
surface. It remains a navigation failure; no connectivity is invented. An
earlier attempt let that exception escape the reference wrapper; this was
repaired, with both attempts retained.

Both public human collisions occur with an empty legal human-track packet
before contact: HM3D 3t8DB4Uzvkt/public_1 and MP3D 2azQ1b91cZZ/public_1. Turning
oscillation leaves important space unobserved while humans approach. This is
an active-observation/progress failure; it cannot be repaired by claiming low
forecast occupancy is safe. The exact observation-only safety screen must not
be mistaken for protection against all unseen humans.

B1/B2 each record four detours, one completed rejoin and three episodes ending
before rejoining. The completed recovery lasts 2.184 s. Their only completed
query adds 0.107 m / 0.311 m and 2.632 s / 3.552 s relative to the nominal
reference. There are 1,159 / 1,171 adjacent turn reversals and a maximum
103-action turn run. These are serious progress failures, even though the
watchdog eventually hands control back. Two same-legal-track reobservation
gaps of 0.5–2 s occur per L4 arm; this does not validate physical identity
association or controlled reobservation performance.

Exact percentile scene-cluster resampling gives exploratory 95% SR intervals
of 33.3–66.7% (B0), 0–25.0% (B1/B2), and 0–83.3% (B3). Four scenes provide weak
uncertainty estimation. Repeated steps and outcomes from a scene are not
treated as independent trials. Zero static contacts are 0/12 observations,
not a universal guarantee.

## Frozen native encounter suite

| Variant | Completed/native SR | Human collisions | Static contacts | SPL | STL | PSC |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3/8 | 5/8 | 1/8 | 0.3739 | 0.3750 | 0.8366 |
| B1 | 7/8 | 0/8 | 0/8 | 0.8154 | 0.8750 | 0.9603 |
| B2 | 7/8 | 0/8 | 0/8 | 0.8083 | 0.8750 | 0.9688 |

The retained failed query is the permanently obstructed passage. Native
reobservation coverage is not inferred from a case name. Full case details
and narrower analytic results are in
[SAFETY_AND_PROGRESS_VALIDATION.md](SAFETY_AND_PROGRESS_VALIDATION.md).

The L2 augmentation changes some encounter maneuvers and improves small-suite
PSC slightly, but provides no public-panel SR, collision or PSC improvement.
It increases time and reduces SPL on the sole completed public query. Thus
`PREDICTION_AUGMENTATION_PASS` is not established. This is an ablation of this
specific soft-cost integration, not proof that all uses of the World Model
are ineffective.

## Evidence

`artifacts/navigation_summary.json`, `navigation_cases.csv`,
`scenarios_summary.json`, `scenario_cases.csv` and `trace_audit.json` contain
per-case outcomes and aggregate calculations. Source/protocol receipts are
retained. Full actual traces, legal observations, forecast banks and old failed
attempts remain in the absolute output directory documented in README.

All 72 final native traces pass the independent protocol/feedback audit. There
are no dropped cases or integration-error files in the final matrix. The
comparison source is frozen at e19431fb; the unchanged B3 policy's earlier
a2155778 receipt is retained. No claim of 75%, 85% or state-of-the-art navigation
is supported by these results.
