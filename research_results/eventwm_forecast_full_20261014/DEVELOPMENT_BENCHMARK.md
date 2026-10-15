# Matched development forecasting benchmark

These are train-internal development results. The precommitted reserved panel is reported separately. All models receive equivalent legal histories; each probability law is calibrated on fitting roots only. Static maps are known in this setting. No vision-only benchmark equivalence or navigation result is claimed.

## Persistent per-agent variance law

ES is the proper joint trajectory Energy Score, using 48 paths. NLL is exact joint trajectory log density divided by the number of supported spatial coordinates. ADE/FDE are metres. Negative continuous-density NLL is valid. All models use the same 256 roots and support masks.

### 1.0 seconds

Support: 156 identity roots, 253 identity agents, 250 complete scene-occupancy roots.

| Model | ES | NLL/coordinate | Mean ADE | Mean FDE | Max-weight mode ADE | Max-weight mode FDE | minADE@6 | minFDE@6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.4111 | -0.7367 | 0.6120 | 0.9022 | 0.6120 | 0.9022 | 0.6019 | 0.8293 |
| Filtered CV | 0.4232 | -0.6933 | 0.6568 | 0.9464 | 0.6568 | 0.9464 | 0.6261 | 0.8495 |
| Deterministic B | 0.3827 | -0.5816 | 0.6129 | 0.8435 | 0.6129 | 0.8435 | 0.5820 | 0.7487 |
| Original stochastic C | 0.3839 | -0.6113 | 0.6141 | 0.8506 | 0.6219 | 0.8620 | 0.5875 | 0.7252 |
| Improved P14 | 0.3476 | -0.6356 | 0.5843 | 0.8129 | 0.6142 | 0.8482 | 0.5167 | 0.6720 |
| Improved P15 | 0.3472 | -0.6535 | 0.5801 | 0.8111 | 0.6304 | 0.8772 | 0.5052 | 0.6736 |
| Map stochastic | 0.3795 | -0.6316 | 0.6011 | 0.8255 | 0.6094 | 0.8347 | 0.5761 | 0.7280 |

| Model | Occupancy Brier | Occupancy log loss | Foreground AP | Positive-cell log loss | Scalar 90% coverage | Mean interval width (m) |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.002136 | 0.012788 | 0.215445 | 4.422542 | 0.925975 | 2.154621 |
| Filtered CV | 0.002152 | 0.012884 | 0.202100 | 4.475356 | 0.922583 | 2.218144 |
| Deterministic B | 0.002156 | 0.012841 | 0.194762 | 4.435330 | 0.939129 | 2.207167 |
| Original stochastic C | 0.002155 | 0.012833 | 0.201618 | 4.434049 | 0.939463 | 2.240973 |
| Improved P14 | 0.002143 | 0.012752 | 0.189615 | 4.382884 | 0.946715 | 2.007534 |
| Improved P15 | 0.002143 | 0.012752 | 0.191191 | 4.385422 | 0.950962 | 2.059280 |
| Map stochastic | 0.002155 | 0.012811 | 0.199269 | 4.423769 | 0.938528 | 2.259876 |

### 3.0 seconds

Support: 147 identity roots, 240 identity agents, 240 complete scene-occupancy roots.

| Model | ES | NLL/coordinate | Mean ADE | Mean FDE | Max-weight mode ADE | Max-weight mode FDE | minADE@6 | minFDE@6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.8926 | -0.9453 | 1.4149 | 2.5329 | 1.4149 | 2.5329 | 1.2606 | 2.0291 |
| Filtered CV | 0.8991 | -0.9374 | 1.4422 | 2.5253 | 1.4422 | 2.5253 | 1.3239 | 2.1122 |
| Deterministic B | 0.7241 | -0.8909 | 1.1962 | 1.9511 | 1.1962 | 1.9511 | 1.1208 | 1.6771 |
| Original stochastic C | 0.7488 | -0.9242 | 1.2093 | 1.9708 | 1.2201 | 1.9910 | 1.0925 | 1.6725 |
| Improved P14 | 0.7032 | -0.9130 | 1.1682 | 1.9234 | 1.2356 | 2.0546 | 1.0750 | 1.5668 |
| Improved P15 | 0.7104 | -0.9253 | 1.1835 | 1.9652 | 1.2464 | 2.0580 | 1.0673 | 1.6142 |
| Map stochastic | 0.7046 | -0.9131 | 1.1398 | 1.8390 | 1.1441 | 1.8477 | 1.0583 | 1.5068 |

| Model | Occupancy Brier | Occupancy log loss | Foreground AP | Positive-cell log loss | Scalar 90% coverage | Mean interval width (m) |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.002191 | 0.013451 | 0.106467 | 4.917091 | 0.930726 | 4.708286 |
| Filtered CV | 0.002198 | 0.013493 | 0.099076 | 4.947611 | 0.927877 | 4.790834 |
| Deterministic B | 0.002200 | 0.013388 | 0.093798 | 4.864808 | 0.952674 | 4.392976 |
| Original stochastic C | 0.002199 | 0.013390 | 0.095786 | 4.869940 | 0.954304 | 4.452139 |
| Improved P14 | 0.002193 | 0.013314 | 0.093107 | 4.822849 | 0.948720 | 3.955365 |
| Improved P15 | 0.002194 | 0.013333 | 0.093133 | 4.834882 | 0.952513 | 4.079936 |
| Map stochastic | 0.002197 | 0.013302 | 0.098426 | 4.828689 | 0.957823 | 4.317751 |

### 5.0 seconds

Support: 99 identity roots, 153 identity agents, 167 complete scene-occupancy roots. This horizon extrapolates beyond the 3 s training horizon.

| Model | ES | NLL/coordinate | Mean ADE | Mean FDE | Max-weight mode ADE | Max-weight mode FDE | minADE@6 | minFDE@6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CV | 1.3646 | -1.0900 | 2.1605 | 3.9685 | 2.1605 | 3.9685 | 1.9813 | 3.3515 |
| Filtered CV | 1.2867 | -1.0960 | 2.0651 | 3.7749 | 2.0651 | 3.7749 | 1.8597 | 3.1937 |
| Deterministic B | 0.9778 | -1.0452 | 1.6353 | 2.7838 | 1.6353 | 2.7838 | 1.5465 | 2.3810 |
| Original stochastic C | 0.9891 | -1.0838 | 1.6494 | 2.8066 | 1.6892 | 3.0220 | 1.5110 | 2.3202 |
| Improved P14 | 0.9331 | -1.0525 | 1.5936 | 2.6712 | 1.6496 | 2.8007 | 1.4538 | 2.1657 |
| Improved P15 | 0.9719 | -1.0769 | 1.6469 | 2.7901 | 1.7231 | 2.9346 | 1.5167 | 2.2721 |
| Map stochastic | 0.9092 | -1.0696 | 1.5270 | 2.5687 | 1.5522 | 2.6113 | 1.4398 | 2.2002 |

| Model | Occupancy Brier | Occupancy log loss | Foreground AP | Positive-cell log loss | Scalar 90% coverage | Mean interval width (m) |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.002060 | 0.013008 | 0.075152 | 5.182040 | 0.938258 | 7.278712 |
| Filtered CV | 0.002064 | 0.013009 | 0.071328 | 5.187313 | 0.947433 | 7.351420 |
| Deterministic B | 0.002064 | 0.012891 | 0.065466 | 5.081438 | 0.965404 | 6.308338 |
| Original stochastic C | 0.002064 | 0.012903 | 0.066139 | 5.093732 | 0.969234 | 6.527442 |
| Improved P14 | 0.002060 | 0.012826 | 0.066179 | 5.040832 | 0.960332 | 5.627185 |
| Improved P15 | 0.002060 | 0.012844 | 0.065879 | 5.058435 | 0.961469 | 5.800549 |
| Map stochastic | 0.002061 | 0.012773 | 0.069939 | 5.031264 | 0.966309 | 6.046999 |

Occupancy is physical human-center cell occupancy on a 32×32 grid over ±8 m, with the same training-only anonymous prior for every model. Foreground AP/positive-cell loss are supported on 197 / 188 / 124 roots at 1 / 3 / 5 s; empty exposed scenes contribute Brier/log loss but no invented AP. Background prevalence makes Brier and aggregate calibration insufficient on their own.

The max-weight mode metric evaluates the mean of the scene component with largest prior categorical probability. It is not the global maximizer of the continuous mixture density. Fixed-K minADE/minFDE draw six independent scene scenarios with coherent residuals; they are diagnostics, not the optimization objective.

## Gaussian OU control

| Model | ES 1 s | ES 3 s | ES 5 s | NLL/coordinate 3 s |
|---|---:|---:|---:|---:|
| CV | 0.4311 | 0.8894 | 1.3548 | -0.7001 |
| Filtered | 0.4354 | 0.8873 | 1.2721 | -0.6977 |
| B | 0.3963 | 0.7078 | 0.9687 | -0.7123 |
| C | 0.3929 | 0.7179 | 0.9787 | -0.7226 |
| C64 | 0.3936 | 0.7099 | 0.9985 | -0.7187 |
| P14 | 0.3467 | 0.6876 | 0.9575 | -0.7656 |
| P15 | 0.3449 | 0.6908 | 0.9752 | -0.7653 |
| Map | 0.3853 | 0.6762 | 0.9227 | -0.7180 |

Changing the residual law leaves physical mean ADE/FDE unchanged. The selected per-agent law improves P14 NLL and joint-density calibration, but Gaussian OU has lower 3 s ES. Neither law dominates every metric.

## Paired evidence and limitations

P14 minus CV at 3 s under the per-agent law: ES -0.1894, scene-clustered 95% interval [-0.2637, -0.1230]; mean ADE -0.2467 m [-0.3664, -0.1352]. NLL difference +0.0323 [-0.0160, +0.0874] is inconclusive. Occupancy AP remains lower than CV. P15 independently improves ES over CV but does not establish a likelihood advantage.

Joint 3 s density-region coverage for P14/P15 is 85.0% / 84.4%, versus a 64-draw finite-rank nominal value of 89.23%. Their scalar 90% coverage is approximately 95%. These distinct diagnostics do not establish reliable uncertainty. The calibration plot includes full PIT and density-rank curves, rather than only one threshold.

Root- and scene-clustered intervals and lost/dense strata are in `comparison_full_completed_agent_scale/result.json` and `comparison_full_completed_ou/result.json`. Component and scaling contrasts are in [ABLATIONS.md](ABLATIONS.md). Later-origin windows use a different explicitly root-balanced population and are reported separately; they cannot be substituted for the primary root-origin results.

## Partial-observation populations

The following 3 s strata use the same per-agent variance law. Lost means at least
one previously observed track is not currently visible. Dense means at least
three recorded physical humans, used only to define an evaluation stratum.

| Stratum | Identity roots | CV ES | P14 ES | Occupancy roots | Foreground AP roots | CV AP | P14 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lost track present | 100 | 0.9147 | 0.6948 | 104 | 103 | 0.1325 | 0.1157 |
| Dense | 80 | 0.8685 | 0.6718 | 101 | 100 | 0.0968 | 0.0840 |
| No lost track | 47 | 0.8456 | 0.7210 | 136 | 85 | 0.0749 | 0.0658 |
| No human ever observed, but humans physically present | 0 | unsupported | unsupported | 43 | 39 | 0.0079 | 0.0079 |

The never-observed cases use the identical anonymous prior in both models:
Brier 0.002399 and log loss 0.016940. This weak foreground result is not evidence
of dynamic unseen-human forecasting. No identity trajectory metric is fabricated.
Groups overlap and must not be summed into independent populations.

Agent-level observation-age diagnostics for P14 have 3 s mean ADE 1.022 m for
age below 0.25 s (111 roots), 1.076 m for age 0.25–1 s (53), 1.658 m for age
1–3 s (20), and 4.139 m for age at least 3 s (only four roots). Continuously
visible recorded histories have 1.076 m ADE on 18 roots; stable-ID reacquisition
in recorded history has 1.038 m on 17. A group can include only some agents in a
root. Their ES uses separately seeded Monte Carlo draws and must not replace the
primary aggregate ES. See `observability_full_posterior_agent_scale/result.json`.

## Exact artifact directories

- CV: `benchmark_full_cv_agent_scale/result.json`; calibrated SHA256 `a0e5c101ac3469a994eddb683832282c731ff5e1a6841cc7c1634d1dea461709`.
- Filtered CV: `benchmark_full_filtered_agent_scale/result.json`; calibrated SHA256 `5a9f27a7907cecc86fea1bc12c61fe50ef832c11adcb419985115305be6e5824`.
- Deterministic B: `benchmark_full_deterministic_agent_scale/result.json`; calibrated SHA256 `3b35504e8022b76089b5918ca1a30aa1114f54321ccb8db17baa77619804825d`.
- Original stochastic C: `benchmark_full_correlated_agent_scale/result.json`; calibrated SHA256 `583a360925ae7658aeff09b353684b63d6b891a8c3c11beae15dfb6ea250bad4`.
- Improved P14: `benchmark_full_posterior_s20261014_agent_scale/result.json`; calibrated SHA256 `d092b5d05d38be4619251e009d2aa2dafc691f3f57440473d72e87acf7887970`.
- Improved P15: `benchmark_full_posterior_s20261015_agent_scale/result.json`; calibrated SHA256 `2faaeff8b19f7298b39eea506354d20e18247d4607ba2ff683c00758a38e6063`.
- Map stochastic: `benchmark_full_map_patch_agent_scale/result.json`; calibrated SHA256 `1615b9300810e47dfdad345efbd6e20d92de6bdfd386886383b01132b0d5d84e`.

All directories above are under `output/eventwm_forecast_full_20261014`. Each result also contains per-root records, probability-law configuration, provenance and source hashes. Complete calibration curves: `calibration_full_selected_figure/`.
