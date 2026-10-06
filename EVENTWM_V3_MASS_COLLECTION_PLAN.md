# EventWM v3 mass collection plan

Release identity is the immutable tag `eventwm-v3-mass-collection-ready`; resolve its exact commit with `git rev-parse eventwm-v3-mass-collection-ready^{commit}` or read the generated `release.json`. This avoids a self-referential commit hash in the committed report.

**Production collection has not been launched.** This document records scene inventory, bounded full-root benchmarks, storage measurements, and the manual launch/resume interface.

## Frozen collector and orchestration boundary

The requested strict version audit found no newer omitted local collector fix: both worktrees were at bf46195, all 41 scoped tracked sources/configs matched byte-for-byte, and no local ref was ahead. All 17 pilot collector hashes matched. New model/audit code exists only as original-worktree uncommitted additions. A historical pre-base pilot-launcher hash differs; its older source is unavailable, but that launcher is not used by the mass runner. See [the provenance audit](EVENTWM_MASS_COLLECTION_PROVENANCE_AUDIT.md) and its full hashes/branch evidence.

The collector, static checker v2.2, v3 labels, legal observer/tracker, fixed N=8 queries, human controller and EventWM input loader are unchanged from `bf46195517d33367af061c2982bfbf2ed6f58000`. Only orchestration, inventory, episode-recipe preparation, monitoring and tests were added in an isolated worktree. Existing pilot/natural/legacy data and the original worktree are untouched.

`structured_production` maps explicitly to the validated collector `production` retention profile plus v3 and `--no-render`, with debug visuals disabled. Root acceptance scans for forbidden image/video/array archives, checks the canonical integrity audits (excluding the scientific human-reactivity diagnostic), validates all 33 primary outcomes and static certificates, seals completed data, and emits role-separated per-root index shards.

## Scene and episode evidence

The inventory distinguishes dataset episode files, physical scene IDs, aliases, local meshes, regenerated NavMeshes and preparation failures. The full gzip census contains 4,455,818 MP3D episodes referring to 71 scenes, and 7,646,425 HM3D episodes referring to 843 scenes. These episode counts are not scene counts. Local meshes cover additional scenes without those episode files.

All local assets were searched under the available Habitat/FALCON asset locations. There is one local scene_datasets tree; aliases in versioned_data were deduplicated. The 993 local meshes produced NavMesh files using the unchanged controlled generator (Spot radius 0.25 m, height 1.41 m). Of these, 952 also passed the generator’s connected 4–10 m start/goal and separated-human waypoint checks. Exclusions are preparation evidence, not a claim that a different episode sampler could never use those meshes. Full dynamic v3 rollouts were smoke-tested on the bounded workloads, not exhaustively on all 952 scenes. Every production root still passes the canonical checks before commitment.

See [scene_inventory.json](reports/eventwm_mass_collection/scene_inventory.json), which retains every physical ID, asset/config/NavMesh path, aliases, scene role, annotation rooms and exact exclusion log. ReplicaCAD references remain in the inventory but required assets are missing; no fake/fallback geometry is used.

## Diversity and roles

Final preservation audit: the mass scheduler calls the existing `eventwm_wm_train_pilot.allocation` with frozen config proportions and the unchanged runtime `collector.wm_sampling` selector. An orchestration bookkeeping error was corrected during finalization: natural-reference slots no longer consume targeted-strata positions. Category-conditional visible/lost counters preserve 30/50/20 and 50/35/15. Fast tests verify exact 10/15/30/30/15 root masses over 4,000 targeted opportunities for 1, 5 and 952 scenes, plus deterministic append/resume. This correction changes only pending production recipe allocation; C24 used explicit validated pilot recipes and did not call this scheduler. No additional collection benchmark was run. Final source hashes include the corrected scheduler and its canonical allocation dependency.

The unchanged offline `eventwm_wm_train_index.balance_weights` policy is SAFE_SUCCESS 30%, SOCIAL_VIOLATION_SUCCESS 30%, HUMAN_COLLISION 40%. No class weighting or outcome-dependent rejection is applied during raw collection. Every valid canonical outcome is retained in role-separated root shards. The existing pilot indexing CLI remains pilot-schema-specific; a later offline production-index adapter must reuse this weighting function and preserve root/scene roles. This collection release does not claim to have built a production training index.

The scheduler traverses all compatible scenes before the next round, interleaving families. Each recipe has an independent deterministic seed; root category, human count and legal visible/lost substrata rotate. The existing wm_train scan/hold selector and natural static-motion sampler are called unchanged. No collision label, future trajectory, hidden human position or privileged PSC distance selects a root.

New initial episode recipes use the validated controlled generator’s connected-path and human-waypoint geometry constraints. Among 128 static navigable start candidates, the least-covered diagnostic cell is preferred. This changes episode realization/coverage only, not the native human policy, query execution or observation semantics. MP3D room annotations are preserved, but an unverified room-coordinate association is not fabricated; observed root coverage uses stable 3m × 2m × 3m cells.

Scene holdouts are a fixed family-stratified hash partition (~12.5%). Natural sampling occupies one in eight recipe opportunities, including holdout scenes; scene holdout takes precedence over training/reference role. All queries from a root share `split_group=root_id`. Targeted training, natural-reference and unseen-evaluation roles remain explicit in index shards and aggregates. The 300,000 target is total accepted primary queries across declared roles, not 300,000 targeted-training queries. Requested-count forecasts below describe the first 9,091 recipe opportunities; failures and unavailable strata mean accepted proportions will differ.

## Resume, resources and safety

A single runner holds dataset and private-catalog locks. Completed roots have per-file seals and completion markers and are deeply verified before being skipped. Failed attempts remain in separate paths. Only interruptions/timeouts can retry, at most twice; unavailable legal strata are recorded without substitution. New recipe sequence numbers and seeds extend the same manifest when the target increases. Duplicate primary query IDs are rejected.

Production startup requires a clean worktree, exact release commit, matching source/static-checker hashes, the frozen structured profile and benchmark-selected worker/GPU configuration. Every running loop checks free disk. It stops above the 150 GB hard floor with an additional in-flight reserve, and also stops on low available RAM, less than 2 GiB sampled free VRAM on either GPU, or global integrity failure. SIGTERM/SIGINT drain current bounded roots; no new roots launch. The separate monitor only reads state and resources. Existing datasets are never automatically deleted.

The no-image rule concerns retention: the unchanged legal depth observation pipeline and replay image fingerprint still operate in memory. No sensor or tracker semantics were altered. The existing fresh-process branch replay remains computationally expensive and was not replaced by an unsafe in-process shortcut.

## Measurements and limits

Role forecast over the first 9091 attempted recipes: {'train': 6951, 'natural_reference': 998, 'unseen_evaluation': 1142}. These are scheduling opportunities, not guaranteed accepted quotas; failures remain recorded and are never replaced based on outcome.

The live C1 footprint sample measured about 0.92 core equivalents (short-lived process transitions can undercount CPU), 2.06 GB peak aggregate collector RSS, 1,206 MiB peak collector-GPU VRAM, 1.625% average collector-GPU utilization, and 0.0109% IO wait. This supports a CPU/process-overhead bottleneck rather than GPU or disk saturation. C24 was estimated at ~22 core equivalents, ~49 GB RAM (~119 GB with 2x worker memory plus 20 GB host allowance), and ~14.5 GiB per GPU before desktop overhead. The actual high-wave measurements below supersede this linear sizing estimate.

Benchmark JSON files retain elapsed time, completed/failed roots, accepted query rate, CPU/load, sampled RAM/GPU usage, disk writes/IO wait, failure reasons, static violations and per-root byte counts. The user canceled C2/C4/C6. After a 49-second C1 footprint sample, exactly one C24 wave of 24 distinct validated Castle recipes was selected without consulting outcomes. Starts were staggered by three seconds; each worker used one internal CPU thread, alternating GPU0/GPU1. C1 contains three Castle roots and one much slower HM3D root, so the raw speedup confounds workload mix. A Castle-only serial comparison is also reported. No intermediate concurrency sweep or production run was launched. Measurements are small workload estimates, not a long-run saturation claim. Both GPUs were assigned explicit worker processes; GPU0 desktop activity is included in host telemetry. No same-concurrency single-GPU control was run, so the incremental benefit of GPU1 is not separately identified. Peak memory values are sampled, not continuous hardware maxima.

Storage uses current v3 structured-only completed roots, never legacy eventwm_prod_v1. Source recipes repeated across benchmarks contribute once to the storage distribution; high-wave recipes are distinct, but most measurements are Castle. The small HM3D sample is retained explicitly; scene-size extrapolation remains uncertain. Expected and p90 estimates are shown separately; an additional 10% provision is used for the conservative disk ceiling. Failed-attempt overhead and new-scene rollout durations remain uncertain, so the live disk guard is authoritative.

Measured root-size range: 24.319–327.688 MB; per-primary-query min/max: 0.737 / 9.930 MB. Measurement sample size: 25 distinct completed root recipes.

## Validation

The original worktree already contained prior diagnostic work. To preserve it, this release lives on `research/eventwm-v3-mass-collection` in the isolated checkout shown in every command. Dataset and scene assets are shared read-only for inputs; new production output is separate. Run the launch and status commands from this checkout, not the original branch.

Final C24 offline verification checked 8,400 sealed files and recomputed all 864 labels (792 primary plus 72 audit/truncation labels). Exact same-root fingerprints, repeated-query truth/controls/legal observations, query hashes, static v2.2 certificates, primary uniqueness and collector source hashes all passed. No worker corruption or visual archive was found. See `c24_final_integrity.json`.

Unit tests cover deterministic round robin and append/resume prefix stability, unique recipe IDs, role isolation, category/count validity, region metadata, hard benchmark bounds, disk stop before launch, protocol-drift rejection and monitor non-mutation. The bounded smoke and benchmark reports provide the simulator-level results. Script syntax and Python compilation are checked.

## Use

Choose either the foreground start command or the tmux command, not both. The tmux session continues after disconnect. Run the status command any time; it is read-only. The safe-stop command requests a drain and lets active bounded roots finish canonical writes. Run the same start command to resume the original target, or the larger-target resume command to extend the same manifest without repeating completed roots. Do not change source/scene inventories or profiles during a dataset run. No RGB/depth/video archives are stored.

**Storage capacity limits the requested scale:** 300k queries require about 1228 GB at the measured mean, while free disk is 992 GB before the guard/reserve. Conservative capacity is about 87,761 queries using p90 plus 10%. The launcher remains configured for the requested target but stops safely on actual disk availability. The listed collection times assume sufficient storage and the benchmark workload; most production scenes were not benchmarked end to end.

## Manual production summary

```text
MASS COLLECTION READY: YES

FROZEN COMMIT: resolved by tag eventwm-v3-mass-collection-ready
TAG: eventwm-v3-mass-collection-ready

================ SCENE LEVEL ================
EPISODE/CONFIG FILES DISCOVERED: 935
UNIQUE PHYSICAL SCENES DISCOVERED: 997
LOCAL SCENE ASSETS PRESENT: 993
COLLECTOR-COMPATIBLE SCENES: 952 (native preparation preflight; runtime acceptance remains per-root)
SELECTED TRAIN SCENES: 833
RESERVED UNSEEN-SCENE HOLDOUT: 119
EXCLUDED SCENES: 45
EXCLUSION REASONS: {'ERROR: No connected 4–10 m start/goal pair': 37, 'ERROR: No separated human point': 4, 'controlled scene asset missing locally': 4}

DATASET FAMILIES:
  MP3D: 76 compatible
  HM3D: 874 compatible
  ReplicaCAD: 0 compatible
  other: 2 compatible

================ REGION LEVEL ================
SEMANTIC ROOMS AVAILABLE: YES
ROOMS / REGIONS DISCOVERED: 2206 MP3D annotation IDs; 1864 in prepared scenes
SPATIAL REGION FALLBACK: deterministic 3m x 2m x 3m cells; diagnostic only
EXPECTED REGIONS COVERED: up to 9091 root cells; no guaranteed unique-region count before collection

================ ROOT LEVEL ==================
TARGET COMPLETED ROOTS: 9091
EXPECTED ROOTS / SCENE: 9.55 average across all roles
EXPECTED DISTINCT SEEDS: at least 9091; failed attempts increase required new recipes
EXPECTED DISTRIBUTION:
  zero: 796 requested opportunities
  unseen: 1192 requested opportunities
  visible: 2386 requested opportunities
  lost: 2387 requested opportunities
  visible_lost: 1194 requested opportunities
  natural unspecified: 1136

EXPECTED HUMAN COUNTS:
  h0: 1024
  h1: 1721
  h2: 2119
  h4: 2118
  h6: 2109

================ QUERY LEVEL =================
MEAN PRIMARY QUERIES / ROOT: 33.00
TARGET PRIMARY QUERIES: 300000 (all declared roles)

EXPECTED SPATIAL PATH GROUPS: 72728
EXPECTED IMMEDIATE_NORMAL: 72728
EXPECTED WAIT_0.5: 72728
EXPECTED WAIT_1.0: 72728
EXPECTED HALF_SPEED: 72728
EXPECTED WAIT: 9091

================ STORAGE =====================
MEASURED MB / ROOT:
  min: 24.319
  median: 128.306
  mean: 135.084
  p90: 271.417
  max: 327.688

MEASURED MB / PRIMARY QUERY:
  median: 3.888
  mean: 4.093
  p90: 8.225

ESTIMATED DATASET SIZE:
  100K queries: 409.3 GB expected; 822.5 GB p90; 904.7 GB p90 + 10% margin
  200K queries: 818.7 GB expected; 1645.0 GB p90; 1809.4 GB p90 + 10% margin
  300K queries: 1228.0 GB expected; 2467.4 GB p90; 2714.2 GB p90 + 10% margin
  400K queries: 1637.4 GB expected; 3289.9 GB p90; 3618.9 GB p90 + 10% margin
  500K queries: 2046.7 GB expected; 4112.4 GB p90; 4523.6 GB p90 + 10% margin

CURRENT FREE DISK: 992.00 GB
MINIMUM FREE DISK GUARD: 150 GB (+ 48 GB in-flight reserve)
SAFE MAXIMUM QUERY COUNT BEFORE GUARD: 87761 (p90 + 10% estimate; guard enforces actual disk availability)

================ PERFORMANCE =================
CONCURRENCY 1:
  queries/hour: 198.88
  CPU: mean 5.30%; peak 13.51%; load mean/peak 1.94/4.67
  RAM: sampled peak 17.01 GB
  GPU0: mean 28.37%; sampled peak 1066 MiB
  GPU1: mean 5.10%; sampled peak 1206 MiB
CONCURRENCY 2:
  queries/hour: NOT TESTED
  speedup: NOT MEASURED
CONCURRENCY 4:
  queries/hour: NOT TESTED
  speedup: NOT MEASURED
CONCURRENCY 6:
  tested: NO
  queries/hour: NOT TESTED
  speedup: NOT MEASURED
CONCURRENCY 24:
  tested: YES
  queries/hour: 3952.15
  speedup: 19.872x
  CPU: mean 55.98%; peak 71.78%; load mean/peak 19.82/25.64
  RAM: sampled peak 47.83 GB
  GPU0: mean 21.97%; sampled peak 8318 MiB
  GPU1: mean 18.78%; sampled peak 8289 MiB
CASTLE-MATCHED C1: 310.82 queries/hour; high-wave speedup 12.715x
CORRECTNESS: 24 complete roots; failures 0; static violations 0; integrity failures 0

SELECTED PRODUCTION CONCURRENCY: 24
SELECTED GPU ASSIGNMENT: [0, 1] round robin
SELECTED CPU THREAD SETTINGS: OMP=1, MKL=1, OpenBLAS=1, NumExpr=1; collector PyTorch=1; native Bullet settings unchanged

EXPECTED THROUGHPUT: 3952.15 queries/hour; 94852 queries/day (bounded benchmark estimate)
ESTIMATED TIME:
  100K: 25.3 hours / 1.05 days
  200K: 50.6 hours / 2.11 days
  300K: 75.9 hours / 3.16 days
  400K: 101.2 hours / 4.22 days
  500K: 126.5 hours / 5.27 days

DISK IO BOTTLENECK: NO (mean IO wait 0.017%; writes 14.17 MB/s)

================ DIVERSITY ===================
SCENE DIVERSITY ASSESSMENT: 952 prepared physical scenes, all included in role-separated round robin
REGION DIVERSITY ASSESSMENT: least-covered static start cells; semantic source IDs retained; root-cell coverage measured live
ROOT DIVERSITY ASSESSMENT: independent recipe seeds, start positions, goals/routes and legal prefixes
HUMAN DIVERSITY ASSESSMENT: h0/h1/h2/h4/h6 with unchanged native controllers
PARTIAL-OBSERVABILITY DIVERSITY ASSESSMENT: unchanged legal zero/unseen/visible/lost/visible_lost selectors; unavailable requests retained

KNOWN REMAINING DIVERSITY BOTTLENECKS:
Native episode preparation excludes 45 scenes; ReplicaCAD assets missing. Targeted scan/hold roots have limited translation; natural-reference motion is separate. Accepted-stratum proportions and unseen-scene throughput cannot be guaranteed from a small benchmark. Storage may stop the run below the requested target.

================ COMMANDS ====================
PRODUCTION OUTPUT: /home/haodong/workspace/datasets/eventwm-falcon/raw/eventwm_social_outcome_v3_train_scale_01

EXACT START COMMAND:
cd /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout && ./scripts/run_eventwm_v3_mass_collection.sh
EXACT TMUX BACKGROUND COMMAND:
tmux new-session -d -s eventwm-v3-mass 'cd /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout && ./scripts/run_eventwm_v3_mass_collection.sh'
EXACT STATUS COMMAND:
cd /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout && ./scripts/check_eventwm_v3_mass_collection.sh
EXACT RESUME COMMAND:
cd /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout && TARGET_PRIMARY_QUERIES=500000 ./scripts/run_eventwm_v3_mass_collection.sh
EXACT SAFE STOP COMMAND:
cd /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout && /home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_mass_collection.py stop --output /home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout/datasets/raw/eventwm_social_outcome_v3_train_scale_01
```
