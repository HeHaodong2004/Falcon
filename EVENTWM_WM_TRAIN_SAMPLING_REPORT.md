# EventWM information-rich training-root sampling

## Scope and frozen mapping

The natural investigation is stopped. Its completed data is a reference/evaluation census, **not the preferred training distribution**. Targeted `wm_train` selects roots using legal arrived history only. All v3 labels, native physical contact detection, PSC threshold, humans/controllers, observation/tracker, branch replay, query timing/speed semantics and `eventwm-static-swept-v2.2` remain unchanged. N=8 remains the pilot query count, with 33 primary queries/root and existing audits. The prior N=8 native gate passed with zero static violations.

The existing mapping was verified in `collector.v3.SocialOutcome` and `social_outcome_labels`: class 0 SAFE_SUCCESS and class 1 SOCIAL_VIOLATION_SUCCESS require full normal completion without native physical human contact; class 2 HUMAN_COLLISION is native physical contact and may terminate early. The new **offline binary training view** maps class 2 to positive, classes 0/1 to negative, and retains class 1 as an explicit hard-negative stratum. It does not redefine or overwrite the categorical v3 dataset. Research-disc overlap alone is not class 2. Static violations and invalid outcomes never become training labels.

The benchmark-native human configuration is retained exactly. Humans may avoid other humans; robot agent 0 is excluded and the robot-distance activation switch remains -1. Thus humans are **ego-independent with respect to the robot, not independent of other humans**. The user's clarified requirement is fidelity to these defaults. No robot-reactive replacement pilot is required. See [the source/configuration audit](EVENTWM_BENCHMARK_HUMAN_DYNAMICS_AUDIT.md). The current pilot is not disqualified on dynamics grounds, although its controlled episode distribution remains narrower than the full benchmark.

## Natural archive

`datasets/raw/eventwm_social_outcome_v3_natural_pilot_01`: stopped by user after **64 attempts: 42 complete, 22 failed, 46 unattempted declared jobs**. Both running root processes finished cleanly before scheduler termination; their successful roots were sealed normally. Original jobs, failed attempts, logs and pre-finalization manifest are retained. No supplement or seek-comparison run followed the stop.

The 42 completed roots comprise 13 h0 and 29 human-present roots. Current track state: 0 visible, 1 lost, 41 no current track; among human-present roots, 28 no current track and 25 never observed in history. Two roots were already inside canonical PSC <1m. The 210 primary branches are 132 safe, 41 social success, 37 physical human collisions, 0 invalid/static. Human-present: 67/41/37 across 145 branches. Failures: 21 native-contact prefix terminations and one exact root-replay mismatch, excluded without relaxing integrity checks. No completed-root integrity contradiction was found.

No outcome balancing is applied to this holdout. Its N=1 queries and fixed Castle episode fixtures make it an initial reference set, not a complete N=8 deployment-calibration set. Future collection budgets should reserve approximately 10–15% for separately declared natural roots; the current archive is intentionally not trimmed to achieve that percentage. Shared environment/episode geometries limit claims of independent generalization even with disjoint root IDs/seeds.

## Legal targeted selection

`collector.wm_sampling` is the canonical selector. It receives only arrived structured history and declared sampling configuration. Inputs read are robot position/timestamp and legal track `valid`, `visible_now`, observed `position`, `observation_age`; history visibility establishes whether a human was ever observed. Configured human count is an explicit experimental condition, not a hidden-position observation. No collision, social metric, future state, hidden human position/velocity/goal, or label is read.

Root categories are mutually exclusive:

| Target | Requested share | Acceptance |
|---|---:|---|
| zero | 10% | Declared human_count=0 |
| unseen | 15% | Humans present; no human ever legally observed |
| visible | 30% | Current visible track(s), no lost track |
| lost | 30% | Current lost track(s), no visible track |
| visible_lost | 15% | Both current visible and current lost tracks |

A previously observed but expired track history is `observed_expired`, **not unseen**. Unavailable categories fail explicitly after the bounded prefix; no unseen fallback exists.

Visible substrata use nearest **legally observed 3D center distance**, not privileged PSC/geodesic distance: near [1,1.5)m, medium [1.5,3]m, far >3m; targets 30/50/20%. Lost strata use the **oldest currently retained lost track's observation_age**: recent <=0.5s, medium (0.5,1]s, stale (1,2]s; targets 50/35/15%. The tracker is unchanged: prediction/velocity gap 0.5s, maximum retention 2s. The 1s middle boundary is an explicit configurable sampling subdivision, not a new tracker rule.

The prefix reuses native WAIT/rotation commands. It scans in a declared seeded direction, holds visible observations while waiting for the requested distance, and holds a lost view while waiting for the requested age. It stops at the first matching legal state after the configured minimum step. Bounds: maximum 220 commands, minimum 10 for track categories and seeded 20–60 for zero/unseen. Root controls are frozen and replayed through the unchanged mechanism. No human policy is changed.

Visible distances below 1m are disabled in this pilot, conservatively targeting <=5% close cases without using privileged distance. **This does not certify that hidden humans are outside canonical PSC.** True initial-PSC prevalence is audited after selection; it cannot legally be used to discard/resample hidden-human roots. A high true rate is a reported limitation and a scale-up gate, not a reason to filter labels.

## Declared pilot

`configs/eventwm_wm_train_pilot.json`: 60 deterministic attempts, stop scheduling at 40 completed roots (an already running root may finish), Castle h0/h1/h2/h4/h6, two isolated root processes on GPUs 0/1. Multi-track targets use h2/h4/h6. Master seed 253000; each job's episode seed is master+index. The full shuffled target table is persisted before collection. Target counts: 6 zero, 9 unseen, 18 visible, 18 lost, 9 simultaneous. Visible/lost bucket assignments are fixed before any outcome is observed.

The bounded runner calls canonical `eventwm_collect.py collect`; no v3 logic is copied. N=8, all four temporal/speed variants, explicit WAIT, root-local audits and sealing remain unchanged. Static contract violations stop new scheduling. No images/video, neural model, training, production/resume scheduler, overnight run, or outcome-dependent branch rejection is introduced.

## Offline training index

`eventwm_wm_train_index.py` verifies completed seals, recomputes labels with the canonical function, and calls `load_wm_input` for every indexed primary query. Audit/repeat branches, partial/failed roots and invalid labels are excluded. Every valid primary branch remains in the raw dataset and index; raw data is not copied or relabeled.

Spatial groups use `(root_id, spatial_path_id)`. Both three-class mixed groups and binary-collision mixed groups are reported separately. Sampling aims for class masses 30/30/40% and 45% mass from three-class mixed timing/speed groups. When those margins are infeasible, the index reports missing classes and the attainable mixed interval; it never fabricates data. Within that three-class mixed quota, the index targets 40% overall mass from **binary-collision mixed groups**, prioritizing actual collision contrasts. It reports the feasible binary-mixed interval and projects the requested mass into that interval when necessary. Within each class/three-class-mixed/binary-mixed cell, mass is divided equally by root, then spatial group, then variants. This controls root dominance within a stratum; rare-class roots may still have more total mass, which is reported.

Weights describe replacement sampling from references, not additional independent examples. The index keeps legal track category/distance/age metadata, scene/count, query timing/speed, raw categorical label, binary target, hard-negative flag, and group/root IDs. Model inputs must still be obtained exclusively through `collector.core.load_wm_input`. Target labels/index diagnostics are not input features.

All targeted roots have training role; natural roots have a separate unbalanced holdout role. Root-ID intersections are rejected; `split_group=root_id` is persisted. No root's query branches are split across roles. Natural N=1 versus targeted N=8 gives different spatial-query coverage; mixed-group comparisons report fractions and first-path subsets, not raw group counts alone.

## Commands

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_wm_train_pilot.py --output datasets/raw/eventwm_social_outcome_v3_wm_train_pilot_01
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_wm_train_index.py --training datasets/raw/eventwm_social_outcome_v3_wm_train_pilot_01 --natural-holdout datasets/raw/eventwm_social_outcome_v3_natural_pilot_01 --output output/eventwm_wm_train_validation
```

## Validation and reproducibility

The expanded collector suite passes **128 tests**, including the original 77 v3 tests, legal-boundary/static/label regressions, natural and targeted root-selection tests, deterministic job-table checks, offline balancing tests and the completed native-defaults audit tests. All **6 controlled-environment regressions pass**. Logs: `output/eventwm_v3_natural_validation/wm_tests.txt` and `wm_controlled_tests.txt`. The native N=8 gate remains unchanged and passed before the pilots.

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python -m unittest discover -s tests/collector -p 'test*.py'
env PATH=/home/haodong/miniforge3/envs/eventwm-falcon/bin:/usr/bin:/bin /home/haodong/miniforge3/envs/eventwm-falcon/bin/python -m unittest discover -s tests/controlled_matrix -p 'test*.py'
```

The final offline index audit passed for sealed completed roots and verifies them, canonical labels and the legal loader for every indexed primary. Failed/partial roots remain on disk and outside the index. The generated `training_index.json`, `natural_holdout_index.json` and `sampling_audit.json` are under `output/eventwm_wm_train_validation/`. Source/config/tests/docs are versioned; raw datasets and generated validation output are ignored. No training or production-scale collection was run.


## Measured targeted-pilot results

Attempted **60**, completed **40**, failed **20** roots. All completed roots are Castle / wm_train / targeted. Primary branches: **1320**, valid indexed **1320**, invalid **0**. Unexpected static violations: **0**. Audit/repeat branches are excluded from these denominators.

### Requested and achieved root strata

| Category | Target share | Attempted | Completed | Achieved share | Failed/shortfall |
|---|---|---|---|---|---|
| zero | 10% | 6 | 6 | 15.0% | 0 |
| unseen | 15% | 9 | 8 | 20.0% | 1 |
| visible | 30% | 18 | 14 | 35.0% | 4 |
| lost | 30% | 18 | 12 | 30.0% | 6 |
| visible_lost | 15% | 9 | 0 | 0.0% | 9 |

Failure taxonomy: {'wm_train_stratum_unavailable': 5, 'prefix_human_contact': 15}. These failures are not converted into alternate categories. Actual branch labels never determine target assignment or acceptance.

### Visible distance substrata

| Bucket | Roots | Conditional share |
|---|---|---|
| far | 2 | 14.3% |
| medium | 10 | 71.4% |
| near | 2 | 14.3% |

### Lost age substrata

| Bucket | Roots | Conditional share |
|---|---|---|
| medium | 4 | 33.3% |
| recent | 8 | 66.7% |

### Human-count composition

| Human count | Completed roots | Share |
|---|---|---|
| 0 | 6 | 15.0% |
| 1 | 12 | 30.0% |
| 2 | 9 | 22.5% |
| 4 | 8 | 20.0% |
| 6 | 5 | 12.5% |

Canonical PSC <1m already at root: **1/40 (2.5%)**. This is post-selection privileged audit evidence. Legally observed <1m cases were disabled; hidden humans cannot be certified by that legal filter.

### Saved primary outcomes

| Condition | Branches | SAFE_SUCCESS | SOCIAL_VIOLATION_SUCCESS | HUMAN_COLLISION | INVALID |
|---|---|---|---|---|---|
| All | 1320 | 381 | 434 | 505 | 0 |
| Human-present | 1122 | 183 | 434 | 505 | 0 |
| zero | 198 | 198 | 0 | 0 | 0 |
| unseen | 264 | 72 | 37 | 155 | 0 |
| visible | 462 | 40 | 217 | 205 | 0 |
| lost | 396 | 71 | 180 | 145 | 0 |
| visible_lost | 0 | 0 | 0 | 0 | 0 |

| Human count | Branches | SAFE_SUCCESS | SOCIAL_VIOLATION_SUCCESS | HUMAN_COLLISION | INVALID |
|---|---|---|---|---|---|
| 0 | 198 | 198 | 0 | 0 | 0 |
| 1 | 396 | 61 | 176 | 159 | 0 |
| 2 | 297 | 31 | 127 | 139 | 0 |
| 4 | 264 | 64 | 80 | 120 | 0 |
| 6 | 165 | 27 | 51 | 87 | 0 |

### Query variants

| Query condition | Branches | SAFE_SUCCESS | SOCIAL_VIOLATION_SUCCESS | HUMAN_COLLISION | INVALID |
|---|---|---|---|---|---|
| movement | 1280 | 375 | 412 | 493 | 0 |
| WAIT | 40 | 6 | 22 | 12 | 0 |
| immediate_normal | 320 | 124 | 81 | 115 | 0 |
| half_speed | 320 | 86 | 88 | 146 | 0 |
| wait_0.5 | 320 | 92 | 109 | 119 | 0 |
| wait_1.0 | 320 | 73 | 134 | 113 | 0 |

Movement and WAIT partition branches; timing/speed rows subdivide movement only.

### Root-level outcome diversity

| Property | Roots |
|---|---|
| with_safe | 29 |
| with_social | 34 |
| with_collision | 33 |
| mixed_classes | 33 |

These root properties overlap; they are not independent branch trials.

### Counterfactual signal versus natural reference

| Dataset | Subset | Groups | 3-class mixed | Rate | Binary mixed | Rate | Roots with mixed |
|---|---|---|---|---|---|---|---|
| Natural N=1 | all spatial groups | 42 | 10 | 23.8% | 6 | 14.3% | 10 |
| Natural N=1 | human-present spatial groups | 29 | 10 | 34.5% | 6 | 20.7% | 10 |
| Natural N=1 | human-present first path | 29 | 10 | 34.5% | 6 | 20.7% | 10 |
| wm_train N=8 | all spatial groups | 320 | 141 | 44.1% | 101 | 31.6% | 33 |
| wm_train N=8 | human-present spatial groups | 272 | 141 | 51.8% | 101 | 37.1% | 33 |
| wm_train N=8 | human-present first path | 34 | 18 | 52.9% | 3 | 8.8% | 18 |

Raw group counts are affected by N and zero-human composition. The human-present/first-path subsets reduce those differences but do not make the sampling policies causally interchangeable. Mixed safe/social-only groups are informative hard negatives but contain no positive collision contrast.

### Indexed sampling distribution and root contribution

| Class | Requested probability | Achieved probability |
|---|---|---|
| SAFE_SUCCESS | 0.3 | 0.3 |
| SOCIAL_VIOLATION_SUCCESS | 0.3 | 0.3 |
| HUMAN_COLLISION | 0.4 | 0.4 |

Raw mixed-group sample fraction: 0.4273; requested sampler mixed mass 0.45; feasible interval [0, 1.0]; achieved 0.4500. Missing classes: []. Unique targeted roots: 40; maximum root sampling mass 0.0489; effective root count under sampler weights 32.89.

Sampler mass from binary-collision mixed groups: 0.4000. Counterfactual movement groups per complete root: 8; primary branches per root: 33.

Natural holdout: 42 roots / 210 primary branches, unbalanced. No root-ID overlap with training. These are reference indexes, not a training run. Visual files: 0.


## Gate interpretation and limitations

The generic `smoke_summary.json` status remains **FAIL** because its original four-fixture smoke criterion requires every attempted root to complete. This targeted experiment instead declares bounded requests with permitted stratum shortfalls. Its 20 failed attempts are 15 native-human-contact prefix terminations and 5 unavailable legal strata. None is replaced, relabeled, silently converted to another category or included in the training index. The **completed-data integrity gate passes**; all 40 roots have seals, legal inputs, unchanged canonical labels, 33 valid primaries and 3 audit/repeat branches each. All 1,320 valid primaries are indexed; the 120 audits are excluded from model sampling.

The diagnostic pilot succeeds at its stated information-yield goals: visible and lost roots, near/medium/far visible distances, all three outcome strata, zero static violations, and low initial PSC violation prevalence. Human-present roots with current visible/lost tracks increase from 1/29 (3.4%) in the natural reference to 26/34 (76.5%) here. Human-present unseen roots are 8/34 (23.5%), rather than the dominant category. No exact quota fit is claimed.

**Unfilled strata remain important:** zero simultaneous visible+lost roots from nine declared attempts, and zero stale-lost roots. Recent/medium lost coverage is demonstrated. These are coverage shortfalls under this Castle fixture/prefix matrix, not proof that the strata are impossible. Near/far visible buckets also undershoot their conditional targets. The sampler and targets remain unchanged; no hidden-human or future-outcome filter is added to compensate.

Overall and human-present spatial-group fractions improve for both three-class and binary contrasts. However, on the human-present first-path-only subset, binary mixed incidence is **3/34 (8.8%)**, versus **6/29 (20.7%)** in the natural reference. Thus the increased binary signal is a property of the **targeted N=8 pipeline as a whole**; these unpaired fixtures do not establish that root targeting alone improves every spatial query. The report preserves this counterexample rather than claiming a uniformly better natural-vs-targeted result.

Only Castle was used for this targeted pilot. Existing non-Castle static smoke results remain regression evidence, not proof of targeted coverage in additional scenes. Seeds vary inherited speed draws and root observations over existing episode fixtures; they do not provide new independent spawn/goal geometries. Root-grouped separation prevents query leakage but does not remove this scene/episode correlation.

The bounded runner deliberately has no production resume/retry scheduler. Its optional `--max-roots` limit is now persisted as job-declaration metadata and checked by the index; this bookkeeping correction was made after collection and does not alter the completed full-table pilot. The original run's source hashes remain in its manifest.

## Recommendation

**Recommend staged production scale-up of the validated sampler/query/index path: YES.** Preserve the declared five-stratum targets, visible-distance and lost-age subdivisions, N=8 variants, all raw branches, static hard stop, and separate unbalanced natural holdout. Report attempted and completed coverage independently; do not silently redistribute the missing simultaneous/stale categories or promise the current fixtures can meet those quotas.

This is readiness of the validated data path on the tested Castle configuration, not a claim that all coverage targets or a resumable production orchestrator have been validated. Use a separately bounded, declared production workload and concurrency-safe production orchestration; do not simply remove this diagnostic runner's limits for an overnight run. Future scene/fixture expansion should remain explicit and preserve benchmark dynamics. Reserve roughly 10–15% of the eventual collection budget for natural evaluation/reference roots without class balancing.

No production-scale collection or EventWM training was started. The current work stops at the pilot, indexes, report and source checkpoint.
