> Source-publication note: this is a frozen local research report. Model weights, datasets and detailed generated evidence are deliberately excluded; reported results were not rerun for publication. Location references are redacted. See [SOURCE_REVIEW_GUIDE.md](../../../SOURCE_REVIEW_GUIDE.md).

# Final Stage 1 research release

**Stage 1 development is closed. This is a research checkpoint, not an officially accepted release. Stage 2 is the primary development objective.**

The collector is inactive, disabled and MANUALLY_STOPPED with zero workers. Its final complete snapshot contains 186,089 sealed independent Roots, 1,341,492 sibling Query outcomes and 879 scenes. All splits, source identities, logs and payloads are preserved; no further collection is planned.

The final compatible fitting union contains **105,838 Roots / 769,349 outcomes / 476 scenes**. It retains every previous fitting Root and adds all compatible eligible new Roots. Existing exclusions and the original 2,110-Root / 108-scene development boundary remain unchanged. Query siblings and causal anchors are dependent; confidence intervals resample scenes.

Dataset manifest: manifest.json (local artifact; excluded from source publication). Fitting manifest: manifest.json (local artifact; excluded from source publication). Independent full-byte audits: REVIEW_DATA_INTEGRITY.json (local artifact; excluded from source publication) and REVIEW_ALL_SPLITS_SNAPSHOT.json (local artifact; excluded from source publication).

## Checkpoint and exact fit

Loadable final checkpoint: checkpoint.pt (local artifact; excluded from source publication). SHA256 `99b73751bbf343946893b7f45ca869925fe46f8bccd7661b65845f148560b92f`.

The established history-conditioned O model was fitted once to all 105,838 Roots and 322,897 causal anchors. Fixed 8 epochs, seed413, batch48, AdamW lr0.0002/decay0.001, clip5, four iid worlds, history fraction0.5 and guided association. Initialization is the original O pilot, not the previous full-data checkpoint or a later calibration pilot. Final selection is last.pt. Completed 17,640 steps in 3386.1s after loading; peak CUDA allocation 42,359,808 bytes.

Exact configuration: config.json (local artifact; excluded from source publication). Training Git commit `ac0d391110a288cbc3f29c3d55e639472ff7c1a4`; immutable original-source overlay and every source hash are preserved in FINAL_SOURCE_IDENTITY.json (local artifact; excluded from source publication). Only its offline target-cache path changed. The model and population law remain unchanged.

## Forecasting and coverage

Values show mean [scene-bootstrap 95% interval] and independent support. Native target times and original masks are unchanged.

| Cohort | Trajectory Energy Score | Coverage90 |
|---|---|---|
| All tracked | 0.220588 [0.205129, 0.234415]; 809 Roots / 96 scenes | 0.901217 [0.884549, 0.917235]; 809 Roots / 96 scenes |
| Visible | 0.188347 [0.175831, 0.200928]; 767 Roots / 94 scenes | 0.896514 [0.878476, 0.913721]; 767 Roots / 94 scenes |
| Native-target practical gap | 0.426177 [0.382855, 0.468350]; 266 Roots / 65 scenes | 0.917458 [0.883395, 0.948501]; 266 Roots / 65 scenes |
| Reappearance | 0.187313 [0.169231, 0.205552]; 259 Roots / 72 scenes | 0.917363 [0.895061, 0.937951]; 259 Roots / 72 scenes |

| Horizon | Energy Score | Coverage90 |
|---|---|---|
| 0.5s | 0.177934 [0.165048, 0.189960]; 809 Roots / 96 scenes | 0.901129 [0.883701, 0.917862]; 809 Roots / 96 scenes |
| 1.0s | 0.270603 [0.251740, 0.289210]; 699 Roots / 85 scenes | 0.886947 [0.862846, 0.907800]; 699 Roots / 85 scenes |
| 2.0s | 0.389942 [0.349151, 0.433965]; 310 Roots / 62 scenes | 0.898266 [0.852818, 0.939159]; 310 Roots / 62 scenes |
| 3.0s | 0.594969 [0.512480, 0.680042]; 114 Roots / 38 scenes | 0.907696 [0.849480, 0.955113]; 114 Roots / 38 scenes |

True native-occlusion horizon results, all cohort/horizon cross-tabs, whole-population scores, status counts and all baseline parameters/comparisons are retained in forecast.json (local artifact; excluded from source publication).

| Matched baseline | Aggregate relative change, candidate minus baseline |
|---|---|
| cv | -0.289290 [-0.312174, -0.264724]; 767 Roots / 94 scenes |
| memory_cv | -0.282265 [-0.304574, -0.261643]; 809 Roots / 96 scenes |
| graph_ridge | -0.164803 [-0.193940, -0.135810]; 809 Roots / 96 scenes |
| pilot_k_bounded_unique | -0.039446 [-0.051712, -0.026785]; 809 Roots / 96 scenes |

Relative changes are fractions; negative favors the final model. Original primary improvement≥2%, positive lower confidence bound, native-gap noninferiority upper bound≤2%, support and coverage requirements remain unchanged.

## Population quality and limitations

| All-current metric | Value and independent support |
|---|---|
| current_set_nll | 6.857481 [6.274465, 7.511122]; 2110 Roots / 108 scenes |
| current_count_nll | 1.498463 [1.438554, 1.561042]; 2110 Roots / 108 scenes |
| count_rps_finite_bank | 0.715513 [0.676500, 0.756867]; 2110 Roots / 108 scenes |
| count_rps_law_unbiased | 0.701069 [0.661984, 0.742261]; 2110 Roots / 108 scenes |
| occupancy_brier_finite_bank | 0.007486 [0.006937, 0.008105]; 2110 Roots / 108 scenes |
| occupancy_brier_law_unbiased | 0.007281 [0.006734, 0.007898]; 2110 Roots / 108 scenes |

Exact H0 reliability retains 2,901 anchors / 1,059 Roots / 108 scenes and reports **8 supported pointwise deviations**. Finite40 reliability reports 130 supported pointwise deviations. Pointwise reliability diagnostics retain their original multiplicity and Monte Carlo caveats; they are not silently converted into new acceptance thresholds.

Exact H0, H+, occupancy reliability, raw probability banks, current-count NLL, proper-score intervals and estimator MCSE bounds remain in final_stage1_exact_h0_20261024 (local artifact; excluded from source publication), final_stage1_hpositive_reliability_20261024 (local artifact; excluded from source publication) and final_stage1_reliability_20261024 (local artifact; excluded from source publication).

Across 1,765 anchors /608 Roots /94 scenes, 141,200 sampled worlds had zero recorded reachability, native-motion or historical-lineage violations, with zero data/model failures. These finite checks retain the original uncertain-geometry limitations. Historical physical-feasibility and persistence rerun: persistence.json (local artifact; excluded from source publication). All failures, sampling caps and insufficient-support strata are retained. The historical development cohort is previously used research data; no fresh protected-confirmation claim is made. Population-count calibration still fails the original acceptance requirements. The checkpoint remains a research artifact. Remaining deficiencies do not trigger another Stage 1 research cycle.

## Complete CPU performance

The unchanged K40/64knots/3s complete-call protocol made 2,007 measured calls. Result: **PASS_EXISTING_INITIALIZED_P95_GATE**; maximum required-workload P95 33.099ms. The original Stage1 P95≤40ms gate and overall integrated Layer2 P95≤100ms target remain unchanged.

| Workload | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| representative_warm | 8.199 | 11.404 | 64.653 |
| representative_dynamic_cold | 8.175 | 11.765 | 13.666 |
| dense_warm | 11.708 | 13.366 | 14.155 |
| dense_dynamic_cold | 11.811 | 13.418 | 42.886 |
| stress_largest_positive30_warm | 13.064 | 15.332 | 17.775 |
| stress_largest_positive30_dynamic_cold | 12.771 | 15.448 | 41.547 |
| stress_longest_frames10_warm | 10.819 | 16.349 | 16.746 |
| stress_longest_frames10_dynamic_cold | 11.067 | 18.512 | 61.761 |
| stress_longest_frames10_seed_tail | 17.130 | 33.099 | 33.294 |
| stress_rare_old_K_acceptance_below_half_warm | 12.835 | 15.689 | 16.685 |
| stress_rare_old_K_acceptance_below_half_dynamic_cold | 12.840 | 15.638 | 58.556 |
| stress_rare_old_K_acceptance_below_half_seed_tail | 14.210 | 30.664 | 33.243 |
| stress_additional_e444_fixed20_extra_seed_tail | 17.167 | 18.441 | 18.452 |
| hard_discovery107 | 16.406 | 16.406 | 16.406 |
| hard_discovery41 | 15.015 | 15.015 | 15.015 |
| hard_dynamic_cold | 14.561 | 14.605 | 14.609 |
| hard_seed_tail | 15.085 | 16.017 | 16.323 |
| hard_warm | 15.634 | 16.012 | 16.054 |

Warm/cold, representative/dense, hard workloads, memory/RSS, full raw-history initialization, empty-JIT first startup, P99/max tails and all call rows are preserved in runtime.json (local artifact; excluded from source publication) and its referenced evidence. A component runtime pass does not establish integrated Layer2 latency.

First empty-JIT raw-history forecast: **6.306s**, after 0.902s of imports; checkpoint loading 6.439ms. With the compiled cache, raw-history initialization P95 was 242.435ms representative /268.102ms dense. Peak process RSS across the benchmark groups was 714,493,952 bytes. These startup costs are separate from initialized-call latency.

Reproduction and inference: [REPRODUCE.md](REPRODUCE.md). Independent final audit: STAGE1_REVIEW.json (local artifact; excluded from source publication). No further Stage1 development or collection is planned.
