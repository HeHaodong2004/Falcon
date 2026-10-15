# Precommitted reserved forecasting evaluation

All 30 model/bank panels completed. The 15 exact checkpoints were committed in
`3ed2d2e` before any reserved forecast scores were read. Selection receipt:
`confirmation_selection/committed.json`. Decision SHA256:
`2c5ccf8cdad87812dd632dcdf5c82858eb79bf3f72fd5d16d25fd8aabc7ae93b`.
Source identity was frozen at evaluator start and preserved through a collector
guard pause. No checkpoint, probability parameter or membership was changed.

The two banks each contain 128 fixed episodes. They exclude the old scalar
evaluation index, prior pilot/extension plans, and overlapping physical identities.
Other episodes in some reserved scenes have been inspected historically; these
are untouched forecast episodes, not a claim that every scene had never appeared
in any earlier diagnostic. Unseen scenes remain disjoint from training.

Primary model P14 and replication P15 use the initial-belief head and the same
per-agent persistent variance law as CV, filtered CV, B and C. The map model is
a single-seed comparison. Two seeds are not independent episode populations.

## Natural reference

| Horizon | Identity roots | Identity agents | Physical occupancy roots | Foreground AP roots |
|---|---:|---:|---:|---:|
| 1.0 s | 46 | 70 | 127 | 87 |
| 3.0 s | 45 | 69 | 125 | 86 |
| 5.0 s | 31 | 49 | 92 | 62 |

| Per-agent variance model | ES 1 s | ES 3 s | ES 5 s | 3 s NLL/coordinate | 3 s mean ADE (m) | 3 s foreground AP | 3 s joint coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV | 1.4997 | 2.0219 | 2.9038 | -0.4361 | 3.2561 | 0.0564 | 0.5333 |
| Filtered | 1.5385 | 2.0766 | 3.0349 | -0.4498 | 3.3121 | 0.0469 | 0.5333 |
| B | 1.4271 | 1.5767 | 1.9839 | -0.4034 | 2.6643 | 0.0421 | 0.4667 |
| C | 1.3778 | 1.4770 | 2.0052 | -0.4459 | 2.6347 | 0.0472 | 0.4667 |
| P14 | 0.8712 | 1.2686 | 1.8078 | -0.5966 | 2.2719 | 0.0477 | 0.6222 |
| P15 | 0.8441 | 1.2314 | 1.8022 | -0.6566 | 2.2523 | 0.0476 | 0.5556 |
| Map | 1.3669 | 1.4586 | 1.9185 | -0.4312 | 2.5529 | 0.0478 | 0.4889 |

| Model | 3 s mean FDE | Max-weight mode ADE / FDE | minADE@6 / minFDE@6 | Occupancy Brier | Occupancy log loss | Scalar 90% coverage / width (m) |
|---|---:|---:|---:|---:|---:|---:|
| CV | 4.6133 | 3.2561 / 4.6133 | 2.6407 / 3.3395 | 0.001615 | 0.011098 | 0.6894 / 4.7083 |
| Filtered | 4.6117 | 3.3121 / 4.6117 | 2.7159 / 3.5214 | 0.001619 | 0.011166 | 0.6951 / 4.7908 |
| B | 3.3110 | 2.6643 / 3.3110 | 2.1608 / 2.3855 | 0.001622 | 0.011181 | 0.7341 / 4.3930 |
| C | 3.2594 | 2.6988 / 3.4816 | 2.1418 / 2.3524 | 0.001619 | 0.011144 | 0.7796 / 5.0227 |
| P14 | 3.1825 | 2.2541 / 3.1766 | 1.7937 / 2.3285 | 0.001614 | 0.011105 | 0.8640 / 5.3987 |
| P15 | 3.1733 | 2.3235 / 3.3264 | 1.7740 / 2.3927 | 0.001614 | 0.011078 | 0.8715 / 5.5067 |
| Map | 3.1031 | 2.4917 / 2.9895 | 2.1693 / 2.2471 | 0.001618 | 0.011083 | 0.7685 / 4.7754 |

Gaussian OU is a separate matched probability-law control. Physical means are unchanged by this calibration.

| Gaussian OU model | ES 1 s | ES 3 s | ES 5 s | NLL/coordinate 3 s |
|---|---:|---:|---:|---:|
| CV | 1.4844 | 2.0235 | 3.0224 | -0.1737 |
| Filtered | 1.5277 | 2.0882 | 3.1688 | -0.1903 |
| B | 1.4237 | 1.6084 | 2.0239 | -0.1595 |
| C | 1.3897 | 1.5176 | 1.9595 | -0.1765 |
| C64 | 1.3860 | 1.5297 | 2.0061 | -0.1616 |
| P14 | 0.8721 | 1.2884 | 1.8199 | -0.4202 |
| P15 | 0.8533 | 1.2636 | 1.7484 | -0.4735 |
| Map | 1.3680 | 1.4914 | 1.8964 | -0.1547 |

P14 minus CV, paired scene-clustered 95% intervals:

- 3 s Energy Score: -0.7533 [-1.2323, -0.3112], 45 roots / 41 scenes.
- 3 s NLL/coordinate: -0.1605 [-0.2995, -0.0268], 45 roots / 41 scenes.
- 3 s mean ADE: -0.9842 [-1.6958, -0.3139], 45 roots / 41 scenes.

## Unseen scenes

| Horizon | Identity roots | Identity agents | Physical occupancy roots | Foreground AP roots |
|---|---:|---:|---:|---:|
| 1.0 s | 79 | 113 | 127 | 101 |
| 3.0 s | 73 | 105 | 120 | 94 |
| 5.0 s | 50 | 77 | 84 | 62 |

| Per-agent variance model | ES 1 s | ES 3 s | ES 5 s | 3 s NLL/coordinate | 3 s mean ADE (m) | 3 s foreground AP | 3 s joint coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.5007 | 0.9572 | 1.6121 | -0.8823 | 1.5274 | 0.1282 | 0.8767 |
| Filtered | 0.5252 | 0.9909 | 1.6241 | -0.8652 | 1.5767 | 0.1130 | 0.8630 |
| B | 0.4805 | 0.7707 | 1.1223 | -0.8283 | 1.2528 | 0.1081 | 0.8219 |
| C | 0.4615 | 0.7541 | 1.1061 | -0.8578 | 1.2451 | 0.1092 | 0.8356 |
| P14 | 0.3402 | 0.6811 | 1.0172 | -0.8645 | 1.1513 | 0.1059 | 0.8356 |
| P15 | 0.3464 | 0.6837 | 1.0262 | -0.8875 | 1.1601 | 0.1074 | 0.8356 |
| Map | 0.4561 | 0.7238 | 1.0827 | -0.8293 | 1.2098 | 0.1064 | 0.8356 |

| Model | 3 s mean FDE | Max-weight mode ADE / FDE | minADE@6 / minFDE@6 | Occupancy Brier | Occupancy log loss | Scalar 90% coverage / width (m) |
|---|---:|---:|---:|---:|---:|---:|
| CV | 2.6155 | 1.5274 / 2.6155 | 1.3546 / 2.0730 | 0.001996 | 0.012540 | 0.9229 / 4.7083 |
| Filtered | 2.6565 | 1.5767 / 2.6565 | 1.3886 / 2.1458 | 0.002006 | 0.012597 | 0.9285 / 4.7908 |
| B | 1.9254 | 1.2528 / 1.9254 | 1.1509 / 1.5586 | 0.002009 | 0.012529 | 0.9547 / 4.3930 |
| C | 1.9344 | 1.2687 / 2.0021 | 1.1296 / 1.5952 | 0.002008 | 0.012520 | 0.9594 / 4.5029 |
| P14 | 1.9224 | 1.2605 / 2.0865 | 1.0111 / 1.4693 | 0.002002 | 0.012428 | 0.9719 / 4.1064 |
| P15 | 1.9260 | 1.2258 / 2.0156 | 1.0347 / 1.4801 | 0.002001 | 0.012443 | 0.9753 / 4.2283 |
| Map | 1.8626 | 1.2126 / 1.8731 | 1.0576 / 1.3319 | 0.002009 | 0.012463 | 0.9649 / 4.3523 |

Gaussian OU is a separate matched probability-law control. Physical means are unchanged by this calibration.

| Gaussian OU model | ES 1 s | ES 3 s | ES 5 s | NLL/coordinate 3 s |
|---|---:|---:|---:|---:|
| CV | 0.5145 | 0.9839 | 1.5834 | -0.6350 |
| Filtered | 0.5380 | 1.0107 | 1.6038 | -0.6167 |
| B | 0.4905 | 0.7721 | 1.0882 | -0.6177 |
| C | 0.4787 | 0.7531 | 1.0594 | -0.6303 |
| C64 | 0.4754 | 0.7550 | 1.0725 | -0.6287 |
| P14 | 0.3462 | 0.6771 | 0.9991 | -0.7444 |
| P15 | 0.3504 | 0.6789 | 1.0066 | -0.7401 |
| Map | 0.4594 | 0.7249 | 1.0404 | -0.6285 |

P14 minus CV, paired scene-clustered 95% intervals:

- 3 s Energy Score: -0.2761 [-0.4475, -0.1277], 73 roots / 37 scenes.
- 3 s NLL/coordinate: +0.0178 [-0.0688, +0.1201], 73 roots / 37 scenes.
- 3 s mean ADE: -0.3761 [-0.6185, -0.1446], 73 roots / 37 scenes.

## Conclusions and failed claims

P14 improves 3 s Energy Score over CV by 37.3% on the natural bank and 28.8%
on unseen scenes. P15 shows similar improvements. P14 also improves over B on
both banks. Against original stochastic C, its natural-bank ES difference is
-0.2084 [-0.4175, -0.0333]; its unseen difference is -0.0730 [-0.1660, 0.0023],
which is inconclusive. P15 minus C on unseen scenes is -0.0704
[-0.1528, -0.0054]. These are paired episode/scene findings, not a universal
multimodality or capacity claim.

Likelihood improves over CV on the natural bank. The unseen likelihood
difference is inconclusive. Foreground occupancy AP is lower than CV on both
banks. Background-dominated Brier scores do not override that failure.

Joint uncertainty is not reliably calibrated. At finite-rank nominal coverage
89.23%, P14 covers 62.2% of natural-bank trajectories, scene interval
[48.9%, 75.0%]. Its unseen coverage is 83.6% [73.1%, 92.3%]. P15 covers
55.6% and 83.6%. This is a decisive failure of a broad reliable-uncertainty claim,
despite improved trajectory prediction and the training-only calibration fit.

The post-confirmation input census measures much older observed tracks in the
natural bank: supported-agent median age 0.920 s, versus 0.048 s in development
and unseen scenes. Root-mean age is 1.975 / 0.326 / 0.479 s respectively. At least
one track is aged ≥3 s in 16/45 natural roots, 4/147 development roots and 2/73
unseen roots. This supports a distribution-shift diagnosis; it does not prove
that age alone causes the calibration failure. No parameter was retuned after
this diagnosis. Artifact: `confirmation_missingness/result.json`.

Five-second scores are supported-subset extrapolation beyond three-second
training. Missing/censored trajectories never contribute zero error. AP excludes
empty foreground scenes; full-scene occupancy uses complete native exposure.
Max-weight mode ADE/FDE describe the largest categorical component, not the
global continuous-density mode. Fixed-K samples have independent residual paths
with persistent scene/variance identities.

The contribution supported here is a stronger recurrent probabilistic forecaster
for legally tracked humans, with tested composition and actual-observation
correction. This is not yet a validated general scene-belief backbone: changed-ID
association, existence/birth calibration, dynamic never-seen occupancy, geometry
compliance and reliable uncertainty under old observations remain open. No
navigation or selector result was used to reach that conclusion.

## Artifacts

All results are under `output/eventwm_forecast_full_20261014`:

- `confirmation_final/result.json`: all 30 fixed panels and hashes.
- `confirmation_final/<model>/<bank>.json`: every horizon, support mask, per-root record, occupancy score and joint diagnostic.
- `confirmation_comparison_{natural_reference,unseen_evaluation}_{agent,ou}/result.json`: matched baseline comparisons, root and scene intervals.
- `confirmation_learned_comparisons_<bank>/result.json`: P14/P15 versus B/C.
- `confirmation_missingness/result.json`: post-confirmation age/missingness census.

Primary selection and probability-law definitions are in [confirmation_decision.json](confirmation_decision.json) and [MODEL.md](MODEL.md).
