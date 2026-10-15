# Frozen physical forecasting data

The experiment uses `eventwm_350k_belief_snapshot_20261014`, SHA256
`5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27`.
It contains 348,777 validated queries, 10,569 roots and 534 scenes. The eligible
training reservation contains 304,293 queries, 9,221 roots and 465 scenes.
The 33 queries per root are counterfactual robot queries, not 33 independent
human futures. No newly collected query enters these experiments.

All artifact paths below are relative to
`/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_20261014`.

## Membership and independence

`protocol/protocol.json` fixes 256 train-internal development roots and 8,965
fitting roots. Its SHA256 is
`a70f5a57e06f4461308438150dac6cc01e53e6e2de2d3218f1e62c9c297b7307`.
The fitting subsets of 128, 1,024 and 4,096 roots are nested explicit ID lists.
Development was excluded from the previous 128-root training pilot. It is now
used for model selection and is not untouched confirmation data.

The 9,221 eligible roots have distinct recorded episode IDs. There are 7,606
nonnull scene-human-trajectory IDs, all distinct, and 1,615 zero-human roots.
There are 25,439 root-associated physical agent trajectories; humans within a
scene share interactions and are not independent statistical units. Dataset
coverage is 8,356 HM3D, 851 MP3D and 14 castle roots.

| Fitting population | Roots | Scenes | Available windows |
|---|---:|---:|---:|
| Nested small reference | 128 | 108 | 711 |
| Intermediate | 1,024 | 343 | 5,556 |
| Intermediate | 4,096 | 445 | 22,175 |
| Full fitting pool | 8,965 | 465 | 48,434 |
| Train-internal development | 256 | 172 | 1,423 |

Training samples roots uniformly, then samples a window within each root.
Full-pool original B/C runs visited every fitting root and window; exact sampler
replay agrees with saved order, cursor, RNG state and presentation count. The
two completed initial-belief runs also visited all 48,434 windows, each with
1,562,598 root presentations (174.30 root epochs). Sampling a window does not
imply that all its identity targets are supported.

## Target construction and compact storage

The approximately 1.5 GB compact cache is `cache_full/manifest.json`, SHA256
`d8df96f6cfd61c6677987ab22eef3d53eb1e29a426651fc0a313846f8efc43d7`.
It contains all 9,221 roots and 49,857 windows. Published cache payloads and
manifests are read-only. Raw trajectories were not duplicated.

Extraction chooses a longest supported pre-contact branch for each root,
preserving its canonical `(branch_id, query_id)` and source seal/hash. This
relies on the prior 250-root native exogeneity audit and unchanged simulator
configuration; it is not a new exhaustive equality audit of every query.
Physical motion comes from timestamped native positions. Rigid-body velocity
fields do not supply motion labels. Origins are actual arrived observation
histories, approximately one second apart. Targets use 0.25 s transitions.

Future positions, future identities, complete population count and native
query outcomes are targets or audit fields only. History preprocessing reads
legal past observations and exact acquisition times. Ambiguous, switched and
duplicate identity associations are masked from trajectory supervision.
First contact and early termination bound exposure. Missing future labels
neither imply empty occupancy nor contribute zero trajectory error.

## Supported horizons

Counts below describe the entire eligible training reservation, including the
256 roots held out for development. A complete identity horizon requires every
tick to be supported. Window occurrences overlap in physical time.

| Support | 0.25 s | 1 s | 3 s | 5 s |
|---|---:|---:|---:|---:|
| Complete physical exposure at root origin | 9,145 | 8,911 | 8,563 | 5,641 |
| Identity-supported roots at origin | 5,917 | 5,707 | 5,392 | 3,244 |
| Identity-supported agents at origin | 9,363 | 9,038 | 8,538 | 5,073 |
| Complete physical windows | 49,762 | 41,400 | 24,196 | 7,472 |
| Identity-supported windows | 32,882 | 27,125 | 15,363 | 4,116 |
| Identity agent-window occurrences | 58,963 | 48,276 | 26,483 | 6,456 |
| Roots with any supported identity window | 6,599 | 6,363 | 5,948 | 3,326 |

There are 561,574 supported **window-tick occurrences**, not unique physical
timestamps. Recorded history spans range from 0.48 to 10.704 s, with median
2.232 s. Training uses up to three seconds; five-second evaluations are
explicit extrapolation tests on the supported subset, not unsupported invented
targets. Exposure is conditional on recorded termination, so these results do
not estimate an uncensored distribution of every possible future.

Primary development trajectory support is 156 / 147 / 99 roots at 1 / 3 / 5 s.
Complete scene occupancy support is 250 / 240 / 167 roots. Foreground AP and
positive-cell log loss have 197 / 188 / 124 roots with positives. Empty exposed
scenes contribute Brier/log loss, but not a fabricated AP value. Every compared
model uses identical membership and support masks. Primary metrics average
roots; uncertainty intervals resample roots and scenes. Secondary all-window
metrics first average supported windows inside a root.

## Partial observability and identity limits

The cache includes 1,528 never-observed-human roots, 4,062 roots with a lost track
at origin and 3,972 roots with at least three physical humans. Complete native
physical exposure supports foreground scene occupancy targets even when no
identity-associated forecast can be scored. Never-observed identities are not
invented: their current output is a training-only anonymous marginal prior.

`association_census/result.json` measures 70,368 legal track-window occurrences,
59,087 accepted associations and 1,348 switched associations. In 593 roots,
the number of positively observed track slots exceeds the recorded physical
population in at least one window (153 at origin). There are 2,295 such windows
and 3,221 excess-slot occurrences. The maximum is 12 legal slots for a physical
population of at most six. The census diagnoses possible aliases; it does not
identify which slots should merge and never filters inference using target
counts. Only three development origins have this excess, so it cannot be
assumed to explain aggregate occupancy errors.

Changed-ID association and physical birth/death probabilities are not validated
by the present model. No nontrivial physical birth/death target coverage has
been established. Stable-ID actual-observation correction is evaluated
separately. Lost/never-seen targets are never used as inference observations.

## Evaluation reservations and scaling

Original reservations remain unchanged: 106 validation roots (already inspected),
606 natural roots and 636 unseen-scene roots. The earlier 16-root unseen panel
is exploratory. The stopped raw unseen extension has not been resumed.

`confirmation_clean_protocol/membership.json` fixes 128 later-added natural
episodes and 128 later-added unseen-scene episodes, excluding the old full-V4
evaluation index, pilot/extension plans and shared historical episode/fixture/
trajectory identities. Membership SHA256 is
`c35502fe8f9c83afd12cc7c025250ca853c071f904b150debc905f486c8c44e1`.
Its compact cache SHA256 is
`8eb661949427a790c2951f7ca30379c169346fd4eba641f82052c5914ecef60c`.
These are untouched forecast episodes until the precommitted panel is executed;
other episodes in some reserved scenes have been inspected previously.

The 500k extension must be a separately audited snapshot with this snapshot as
its declared parent. Scaling preserves exact development and nested fitting ID
lists, checks new scene/episode/trajectory overlaps, and incrementally reuses
compact payloads. A matching random seed alone is insufficient to preserve a
subset. Live collected-query counts are not validated training counts and never
trigger automatic retraining. See `scripts/eventforecastwm/scaling.py` and the
reproduction commands in the main report.

Audit evidence: `audit_full/result.json`, `data_distribution/result.json`,
`association_census/result.json`, `utilization_full_completed/result.json`, and
`utilization_posterior_full_s20261014/result.json` / `...s20261015/result.json`.
