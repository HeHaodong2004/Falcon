# Data identities, dependence and support

All completed fitting/development experiments use the immutable 350k snapshot.
No 500k fitting claim is made. Live collector counters are operational evidence,
not validated dataset membership.

| Artifact | SHA256 |
|---|---|
| `eventwm_350k_belief_snapshot_20261014/snapshot.json` | `5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27` |
| v1 `protocol/protocol.json` | `a70f5a57e06f4461308438150dac6cc01e53e6e2de2d3218f1e62c9c297b7307` |
| v1 `cache_full/manifest.json` | `d8df96f6cfd61c6677987ab22eef3d53eb1e29a426651fc0a313846f8efc43d7` |

The snapshot contains 348,777 validated queries, 10,569 roots and 534 scenes.
Its eligible training reservation contains 304,293 queries, 9,221 roots and 465
scenes. The approximately 33 robot-query branches per root share a physical
human future. Root, not query, is the independent training/evaluation unit;
scene clustering additionally reflects shared environments.

| Fixed membership | Roots | Scenes | Actual windows |
|---|---:|---:|---:|
| Pilot subset | 1,024 | 343 | 5,556 |
| Intermediate subset | 4,096 | 445 | 22,175 |
| Full fitting pool | 8,965 | 465 | 48,434 |
| Internal development | 256 | 172 | 1,423 |

These explicit ID lists are nested and unchanged. Development is used for
selection and cannot be described as untouched confirmation. Existing v1 natural
and unseen confirmation results have already been opened; no repeated selection
against those targets is performed in this campaign.

## Compact targets and legal histories

The existing approximately 1.5 GB compact cache is reused by hash. No raw archives
are copied or rebuilt. Extraction retains the longest supported pre-contact
branch per root, including its canonical query/branch identity and source seals.
This relies on the established native exogeneity audit and unchanged simulator,
not a new exhaustive assertion that all branch records are byte-identical.

Origins are actual arrived histories, approximately one second apart. Targets
use 0.25-second ticks. Timestamped native positions supply supervised future
coordinates. Native rigid-body velocities, goals, hidden current positions and
future observations are never inference features. Legal preprocessing estimates
velocity from arrived positions and exact acquisition clocks. The learned
models use the .75-second prepared sidecars; CV uses its .15-second sidecars.

Identity target support excludes ambiguous, switched and duplicate associations.
Missing labels, first contact and early termination restrict exposure. A complete
trajectory score requires support at every tick through its horizon; missing
positions contribute neither a zero error nor an empty-cell label. Complete
native scene exposure separately supports all-human occupancy, including exposed
empty scenes. Foreground AP is undefined for a root with no positives, while
Brier/log score still includes it. The raster predicts human-center cells, not
body contact or swept-volume occupancy.

Anonymous occupancy targets are formed from native humans not associated with
a supported legal slot, before cell rasterization. An anonymous human can share
a cell with a tracked human. This bookkeeping is supervision only. Never-seen
identity paths are not supplied to or invented by the model.

## Age and missingness

Age means forecast origin minus last actual acquisition, not elapsed time since
a tracker row was last serialized. The collector removes current-table entries
after two seconds. The forecasting history may still contain their earlier
positive observations; retaining those belief slots permits older ages legally.
The collector, tracker IDs and observation semantics are unchanged.

The new census covers every fitting/development window at 1, 2, 3 and 5 seconds.
Three-second fitting identity support grows from 5,245 root origins to 14,920
windows across 5,782 roots. Median agent-window age grows from .048 s at fitting
origins to .440 s across windows. Development medians are .048 s and .536 s.
Development roots with a supported age >=3 s grow from four at origin to 21
over three-second windows; one-second windows provide 108 such roots. The
existing natural-bank age median .920 s is historical diagnostic evidence,
not a target used to fit new calibration parameters.

Original physical training already samples actual windows uniformly inside
uniformly shuffled roots. Original uncertainty calibration used only origins.
The new all-window calibration averages supported windows within each root
before averaging roots. The older-window physical control changes only the
proposal over existing legal windows, with inverse-proposal weighting and
documented finite-batch self-normalization bias. Distinct-arrival encoding changes
how real packets update the GRU. Neither creates synthetic observations. Explicit
observation-dropout augmentation has not been executed.

## Exposure, confirmation and expansion

Completed pilots have 8,192 updates and 262,144 presentations each. Exposure
audits replay sampler RNG, shuffled order, cursor and per-window selections and
verify saved best/last checkpoint hashes. A selected early checkpoint and a
completed final checkpoint have different exposure; both are reported. The
8,965-root physical runs have both completed 24,576 updates and 1,562,598 root
presentations. Exact replay verifies all 48,434 windows were presented in both selected
and final checkpoints, with matching sampler RNG, root order and counters.
Full-root probability fitting on a frozen v1 backbone is reported separately.

The audited confirmation snapshot has completed with **444,774 validated queries
and 13,478 roots**. It inherits the exact 350k roots and query identities, reusing
10,569 parent audits and adding 2,909 new root audits. Its SHA256 is
`59c18f61843f871c09c4249fabd879e4fb0e5b21cb0af731ed93a2873618ca42`;
the direct parent remains
`5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27`.
The sealed payload is `confirmation_snapshot/frozen/` under the v2 output root.
It is not used to fit any model and is not a 500k training claim.

New confirmation membership selection excludes every
parent root and shared historical episode/fixture/trajectory identity. Its fixed
root hash chooses 128 per bank when available, otherwise the predeclared 64-root
fallback in both banks. No age, difficulty, model output or future outcome enters
membership selection. Unseen scenes must remain disjoint from fitting scenes.
Checkpoint family, seeds, calibration, source and hashes must be frozen before
any new confirmation forecast metric. Membership selection and target extraction
are separate gated stages; their completion is not implied by the snapshot receipt.

500k expansion requires at least 495,000 validated queries, inherited identities,
unchanged development/confirmation reservations and incremental extraction of
only missing compact windows. A smaller snapshot used solely to secure new
confirmation is explicitly not a 500k training dataset. The independent production
collector continues unchanged throughout.

## Final scaling status

All model fitting remains on the frozen 350k membership. The final read-only validated index contains 457,314 queries / 13,858 roots, below the unchanged 495,000 gate. The 444,774-query snapshot was used solely to form new excluded confirmation populations. The future continuation prepares incremental compact-cache extraction and matched-update/root-exposure recipes; no 500k training or positive end-to-end scaling run is claimed.
