# EventWM performance v2: preregistered development protocol

Starting source: `c8e85e73949b77edba7421cdefaa196f4e1a589d`. The immutable
v1 primary remains the incumbent. No production, tracker, query generation,
ReID, navigation, selector or planning changes are authorized in this campaign.

## Population and comparisons

Use the existing 350k snapshot and exact 8,965 fitting / 256 development root
membership. Reuse verified compact data and parameter-free history features.
The 33 queries per root are correlated branches, never independent examples.
Root-origin development remains the primary comparison. All actual-arrival
development windows are a required secondary population, averaged inside each
root first. Age groups are <0.25, 0.25–1, 1–2, 2–3, and >=3 seconds since last
actual acquisition. Report both per-agent strata and roots containing such agents;
do not count a root repeatedly as independent evidence. If later windows still
provide inadequate aged support, explicitly hold out a new fitting-internal fold
before fitting any v2 model on it; flag the incumbent's prior exposure to that fold.

Use identical supported targets for all models. Identity trajectory scores require
complete identity support through the indicated horizon; complete scene occupancy
requires native exposure, including empty scenes. Anonymous and associated-human
occupancy have separate targets; full-scene targets must not be labeled tracked-only.
Report human-center occupancy on the existing 32x32, ±8 m grid. Any additional
grid/body representation is secondary and never substituted into the comparison.

Primary horizons are 1, 2, 3 seconds at 0.25-second ticks. Five seconds is a separate
extrapolation diagnostic unless a version explicitly trains to that horizon.
Energy Score uses 48 stratified paths per A/B sample bank, normalized as in v1;
fixed-K minimum-distance diagnostics use K=6 independent scene scenarios. Joint
density ranks use 64 draws and nominal finite-rank levels floor(p*65)/65 for
p in {0.1,0.25,0.5,0.75,0.9}. Do not change K between compared models.
Likelihood is exact when claimed, per supported spatial coordinate. Report
ADE/FDE, conditional coverage, interval width, occupancy AP/Brier/log score,
positive-cell log score, sample speed/acceleration tails and qualified map proxies.
Pair root records; bootstrap scenes (1,000 replicates, seed 83199), retaining
root weights. Differences with intervals spanning zero are inconclusive.

## Selection before confirmation

Select exclusively on development. Keep the v1 incumbent and a Pareto set over
3s/1s ES, joint NLL, joint-density calibration, occupancy AP/log score, and physical
tails. Checkpoint selection inside physical training uses the existing 3s ES;
post-training probability calibration may reorder candidates and is evaluated
explicitly. Calibration fits fitting roots only, including any all-window variant.

An architecture is promoted to full training if an intermediate matched-budget
comparison improves at least one primary capability by a practically meaningful
amount (>=1% ES, >=5% occupancy AP, >=0.03 nats/coordinate likelihood, or >=5
percentage points conditional coverage error) without an unexplained material
regression (>3% ES, >5% AP, >0.03 likelihood, or >5 points coverage error) elsewhere.
These are progression heuristics, not significance claims. Consider confidence
intervals and supported root counts; a small noisy pilot cannot settle superiority.
If primary-origin and later-window evidence disagree, retain both and investigate.

The final primary must show a reproducible development benefit across two full
training seeds and satisfy legal input, composition, sampling, observation update,
numerical, occupancy-frame, support, resume and standalone-inference checks.
Prefer lower 3s ES among eligible nondominated candidates; accept an ES cost <=3%
only for a clear uncertainty/occupancy gain and name that tradeoff explicitly.
The first predeclared seed is primary, the second replication; never choose the
best reserved-test seed. Freeze model family, checkpoint hashes, calibration,
source and evaluation identities before confirmation. If no candidate qualifies,
v1 remains the best verified model; a finished new run alone is not an improvement.

Already-opened v1 natural/unseen banks are historical evidence, never a tuning
target. Seek new post-350k reserved roots using an independently audited freeze,
exclude all historical evaluation and physical identities, and keep targets closed
until selection. If unavailable, report the lack of untouched confirmation.

## Finite compute budget and stopping

Stage 1 reproduces v1 metrics, tests and legal inference and audits errors.
Stage 2 permits up to eight meaningful matched 1,024-root experiments (8,192
updates, batch 32) plus inexpensive fixed-mean probability/readout controls.
Stage 3 advances at most three families to 4,096 roots (16,384 updates), or
directly to full data when existing matched evidence justifies skipping a repeat.
Stage 4 trains the best family on all 8,965 roots, at least two seeds, each at
24,576 updates / batch 64, and necessary matched full-data controls. At most two
additional full-data architecture runs after replication. Extend a training budget
only with documented late-curve progress; do not silently change a resumed schedule.
Stop architecture search after two successive well-budgeted hypotheses fail
promotion, the declared budget is exhausted, or candidate improvements saturate;
then complete verification, confirmation and artifact delivery. Record unexecuted
ideas as unexecuted. A worker launch is never a completed experiment.

The 500k branch requires >=495,000 validated queries, exact inherited identities,
preserved memberships and incremental extraction. At initial probe collection is
425,931 live queries / 12,907 roots; this is not a validated 500k snapshot.
Do not wait for it. If ready before final selection, compare the best family on
both datasets, matched updates and root exposure where practical.

## Resource policy

One trainer per available GPU; initially at most one additional CPU analysis
worker, one CPU thread per worker, nice 19 / idle IO. After both physical runs
completed, the user's research-priority revision and measured CPU/RAM headroom
justify at most two CPU workers in the v15 finite queue. Scientific recipes
and GPU concurrency remain unchanged. Preserve 48 GiB RAM, 50 GiB disk and
8 GiB free GPU before launch. The initial policy preserved the 3,615.965
queries/hour reference, three windows below 65% stop, and three above 85%
spanning 120 seconds for recovery. **The user's explicit subsequent revision
supersedes those throughput thresholds.** The effective training-priority policy
targets 1,500 q/h, accepts 1,000–1,500, and throttles after three valid windows
below 1,000 instead of stopping all research. Two valid windows above 1,500
clear the legacy pause or throttling. RAM/disk/health/OOM remain hard constraints.
The full-data trainers take priority over auxiliary CPU/IO work. See
`RESOURCE_POLICY.md` for exact settings, measurements and unchanged-checkpoint
proofs. The v2 guard preserves the original measurement history for audit.
Controllers have finite 16-hour lifetimes and explicit stop/resume; no two owners
may manage a run. Never mutate an existing completed result.

The host has kernel driver 595.91.07 and userspace 595.99.02. A process-local
extraction of the cached matching compute package restores CUDA/NVML. Its path
is pinned in v2 launch manifests, including tmux/resume. No driver installation,
kernel reload, reboot or collector environment change is performed.

## Supplementary physical diagnostics

The completed primary metrics stay unchanged. Add fixed 48-path stratified banks
for finite-difference speed/acceleration tails and power-.5 displacement variogram
scores. Two independent banks give an unbiased estimate of the squared moment
score; finite-sample estimates may be negative. Report adjacent-tick and one-second
temporal displacements, plus simultaneous cross-agent distances. These complement
joint ES/NLL; they are not replacements or new architecture-selection thresholds.

Strata include target-only acceleration above 1 m/s^2, turns above 45 degrees,
speed below .25 or above 1.25 m/s, another supported human within 1 m, at least six
physical humans, histories below 1 s or at least 3 s, and no currently visible
legal human. Known-map clearance below .5 m uses the existing eroded-navmesh
proxy. Report proxy nonwalkability, missing-map coverage, and outside-grid mass
separately. Never label the proxy a true human wall/contact test or discard paths.
All dynamic labels are read after generating forecasts. Tail diagnostics include
all legal forecast tracks; accuracy strata use complete supported targets. Root
and scene identities remain the statistical units. Software smoke subsets are
explicitly flagged and cannot enter model comparisons.
