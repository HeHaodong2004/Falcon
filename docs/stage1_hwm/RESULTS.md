# Stage 1 human world model: results

Development split: 2,110 Roots / 108 scenes / 6,391 legacy anchors, scene-disjoint from all
476 training scenes. Two seed streams (41, 107), 40 worlds, every native time to 3 s. Means
are anchor→Root→scene; intervals are 2,000-draw scene bootstraps (legacy protocol). Paired
comparisons match rows by (Root, anchor, stream, person/horizon or probe) and report the
relative change of the matched aggregate (negative = better for every listed score).

## Selected model: `hwm_lite_e12`

Lite BEV backbone (64×64 at 0.4 m, ~20 M MACs, 0.57 M parameters in total), 6-mode trajectory
mixture, exact truncated population law, trained 12 epochs on all 105,838 training Roots (781k
anchors); best validation checkpoint at step 72,000. Checkpoint and identity: `SELECTED.json`
(SHA-256 `53d44d85…`).

**Selection rule.** The criterion was fixed before the final comparison: best validation NLL
among candidates ≥20% faster than legacy at P50 and at P95 on every latency cohort. It is
applied in every bracketed same-load benchmark in which the candidate was measured, against
the mean of the two bracketing legacy runs (`latency_table.py`). This multi-bracket reading
was specified before the final bracket; see §5. `hwm_lite2_e12` has the better validation NLL
(−16.85 vs −16.75) and passed under load, but missed in the quiet bracket (representative
P50 −19.7%), so the rule selects `hwm_lite_e12`. The two are statistically indistinguishable
on development (see below), so the switch costs no measurable quality.

Columns: legacy final; **selected**; `lite2` (the previous selection); full-architecture
model (`hwm_a`, best quality, ~3× slower on CPU); fast architecture, 12 epochs (best
validation NLL, fails the speed rule). Paired change = selected vs legacy, 95% scene-bootstrap
interval.

| Metric (dev) | Legacy | **Selected** | lite2 | Full arch | Fast arch | Paired change (selected vs legacy) |
|---|---|---|---|---|---|---|
| tracked path ES (all) | 0.2206 | **0.1610** | 0.1620 | 0.1599 | 0.1642 | −27.0% [−30.8, −23.3] |
|   visible | 0.1883 | **0.1341** | 0.1357 | 0.1330 | 0.1371 | −28.8% [−32.6, −25.0] |
|   practical gap (0.75–1.25 s unseen) | 0.4262 | **0.3535** | 0.3461 | 0.3344 | 0.3489 | −17.1% [−23.8, −9.8] |
|   reappearance | 0.1873 | **0.1391** | 0.1335 | 0.1388 | 0.1373 | −25.8% [−30.5, −20.5] |
|   horizon 0.5 s | 0.1779 | **0.1348** | 0.1354 | 0.1330 | 0.1376 | −24.3% [−28.6, −20.1] |
|   horizon 1 s | 0.2706 | **0.1909** | 0.1916 | 0.1906 | 0.1939 | −29.5% [−33.4, −25.5] |
|   horizon 2 s | 0.3899 | **0.2514** | 0.2575 | 0.2532 | 0.2570 | −35.5% [−40.6, −30.4] |
|   horizon 3 s | 0.5950 | **0.4116** | 0.4062 | 0.4099 | 0.4066 | −30.8% [−36.7, −24.5] |
| joint tracked ES | 0.2295 | **0.1688** | 0.1704 | 0.1679 | 0.1722 | −26.5% [−30.1, −22.9] |
| Coverage90 (legacy definition) | 0.901 | **0.890** | 0.887 | 0.892 | 0.882 |  |
| Coverage90 obs−self (0 = calibrated) | +0.031 | **+0.017** | +0.011 | +0.021 | +0.008 |  |
|   practical gap cohort | +0.050 | **−0.024** | −0.006 | | |  |
| joint Coverage90 obs−self | +0.040 | **+0.011** | −0.000 | +0.014 | −0.000 |  |
| count NLL (exact law / legacy unbiased estimate) | 1.5009 | **1.2764** | 1.2787 | 1.2908 | 1.2760 | −15.0% [−17.6, −12.3] |
| count RPS | 0.7126 | **0.6194** | 0.6198 | 0.6130 | 0.6171 | −13.1% [−15.6, −10.6] |
| count randomized-PIT KS (0 = calibrated) | 0.089 | **0.012** | 0.013 | 0.025 | 0.011 |  |
| P(count=0), histories without detections (observed 0.302) | 0.203 | **0.292** | 0.292 | 0.294 | 0.292 |  |
| 2 m occupancy Brier | 0.00853 | **0.00789** | 0.00788 | 0.00781 | 0.00786 | −7.5% [−8.7, −6.3] |
| 0.5 m near-robot occupancy Brier ×100 (1 s) | 0.3512 | **0.3153** | 0.3146 | 0.3153 | 0.3162 |  |
| joint future-set kernel score | 0.3147 | **0.2727** | 0.2736 | 0.2705 | 0.2726 | −13.3% [−15.3, −11.5] |
| executed path: P(human ≤1 m within 1 s) Brier | 0.0857 | **0.0704** | 0.0724 | 0.0708 | 0.0716 | −17.8% [−25.6, −11.2] |
|   same, ECE | 0.0569 | **0.0248** | 0.0255 | 0.0227 | 0.0266 |  |
| executed path: per-step 1 m social Brier | 0.0531 | **0.0450** | 0.0447 | 0.0442 | 0.0446 | −15.2% [−20.0, −11.0] |
| stationary robot: per-step 1 m social Brier | 0.0583 | **0.0460** | 0.0457 | 0.0447 | 0.0459 | −21.1% [−25.1, −17.4] |
| candidate routes: P(≤1 m) Brier | 0.0685 | **0.0562** | 0.0557 | 0.0553 | 0.0560 |  |
| candidate routes: P(≤1 m) ECE | 0.0233 | **0.0129** | 0.0145 | | |  |
| candidate routes: within-Root pairwise ranking accuracy (1 m) | 0.945 | **0.963** | 0.960 | 0.960 | 0.963 |  |
| candidate routes: within-Root Spearman of E[min distance] | 0.808 | **0.829** | 0.828 | 0.834 | 0.835 |  |

Every paired tracked-path and population score improves significantly, and so do both
per-step risk scores. Of the 18 per-horizon risk cells, 6 have intervals that include 0: the
executed-path 3 s cells (185 Roots) and three 2 m-radius cells. Their point estimates still
favour the selected model. No metric degrades significantly. The deployed CPU path (TorchScript engine + NumPy/numba
sampler) reproduces these metrics within Monte Carlo noise (path ES 0.16145 vs 0.16099;
identical exact count NLL 1.2764; candidate pairwise accuracy 0.960 vs 0.963).

**Calibration.** Legacy coverage is over-dispersed: obs−self +0.031 per person, +0.040 joint.
The selected model is closer to calibrated. Joint: +0.011, significantly closer to 0
(|selected| − |legacy| interval [−0.050, −0.001]). Per person: +0.017, interval
[−0.038, +0.011]. In the practical-gap cohort it is slightly under-dispersed (−0.024 vs legacy
+0.050), and |gap| is not significantly different. Count calibration is resolved: PIT KS 0.012
vs 0.089; P(count=0) for histories without detections 0.292 vs observed 0.302 (legacy 0.203).
Candidate-route ECE: 0.013 vs 0.023.

**Selected vs `lite2`** (paired, same protocol). The differences are path ES −0.6%
[−2.9, +1.7], gap ES +2.1% [−2.4, +7.4], reappearance +4.1% [−0.5, +9.3], count NLL −0.2%,
set kernel −0.3%. Coverage-gap differences are all within noise (joint +0.011 vs −0.000,
difference interval [−0.011, +0.033]). Of 33 paired score cells, 2 nominally favour the
selected model and none favour `lite2`. Per scene, the selected model wins 46–56% of scenes
per metric.

**Robustness across scenes.** Per-scene means, selected vs legacy (`report.py`):

| Metric | Scenes where selected is better | Median per-scene change |
|---|---|---|
| tracked path ES | 92% | −29% |
| count NLL | 89% | −15% |
| occupancy Brier | 91% | −8% |
| set kernel | 93% | −13% |
| per-step risk Brier | 92% | −23% |

The `lite2` figures were 93/92/89/94/91%. The few regressing scenes have 1–3 Roots each.

Qualitative samples: `figures/qualitative_hwm_lite_e12_vs_legacy_final_seed{3,11}.png` (top row
legacy, bottom row selected; true paths white; 40 sampled worlds, tracked keys orange, residual
people cyan; robot magenta).

### Latency (CPU, 1 thread, legacy runtime protocol)

Each online call ingests one new actual frame from the pre-arrival state and returns 40 worlds
× 64 times. Every comparison is a same-load bracket run back-to-back (legacy → candidates →
legacy; `scripts/stage1_latency_bracket.sh`, tables from `latency_table.py`), while Stage 2
training runs on GPU 0. P50 / P95 / P99 ms.

**Final bracket** (1-min load 4.0–6.9):

| Cohort | Legacy (before) | Legacy (after) | **Selected (`lite`)** | `lite2` |
|---|---|---|---|---|
| representative (first 200 Roots) | 8.93 / 12.53 / 109.8 | 8.70 / 12.22 / 107.5 | **6.87 / 7.70 / 8.0** | 7.41 / 8.30 / 8.5 |
| dense (30 most crowded) | 12.76 / 14.34 / 14.7 | 12.26 / 14.81 / 24.4 | **7.26 / 8.14 / 8.3** | 7.48 / 8.80 / 9.0 |
| longest history (30) | 9.71 / 12.85 / 13.8 | 10.74 / 13.82 / 16.0 | **7.08 / 7.95 / 8.0** | 7.51 / 8.46 / 8.9 |
| startup, full raw history (representative) P50 / P95 | 58 / 244 | 57 / 239 | **16 / 51** | 17 / 52 |

**Speed rule.** Selected vs the mean of the bracketing legacy runs, per cohort
(representative / dense / longest):

| Bracket | P50 | P95 |
|---|---|---|
| Final | −22% / −42% / −31% | −38% / −44% / −40% |
| Quiet (load 3.4–4.6) | −25% / −43% / −28% | −39% / −44% / −37% |

`lite2` failed in both brackets: representative P50 −16.0% (final) and −19.7% (quiet). It
passed only in the earlier loaded bracket (load 7–12, −22%). The fast architecture failed
there (−14%). All brackets: `output/stage1_hwm/benchmark/{final,quiet,loadedA}_*`.

Tails:
* P99 is 8.0–8.3 ms vs 14–110 ms for legacy (13× lower on the representative cohort).
* One call in 1,300 took 89 ms (attributed to a Python generation-2 garbage collection);
  legacy's maximum was 134–159 ms.
* Startup from a full raw history is 3.6–4.8× faster.

Where the time goes (representative median):
* legal state update 0.96 ms;
* network + sampling 5.09 ms;
* bank assembly 0.77 ms.

The tails are bounded by removing TorchScript shape re-specialisation (warm-up, non-profiling
executor) and by vectorising grid recentring.

**Downstream reuse.** Scoring a Root's real sibling candidate routes (3–24) with one shared
bank through the L2.5 `aggregate_joint` reduction takes 0.76 ms median.

**Batched GPU serving** of B concurrent robots (one network call), on a GPU shared with other
training:

| B | ms per call | ms per robot |
|---|---|---|
| 1 | 11.3 | 11.3 |
| 8 | 25.8 | 3.22 |
| 32 | 72.6 | 2.27 |
| 64 | 134.2 | 2.10 |

On one CPU thread, batching helps only modestly: 7.9 / 5.6 / 6.4 ms per robot at
B = 1 / 8 / 32. One call per environment worker is the simpler CPU deployment.

**L3 budget.** For an L3 run of 10⁶ decision steps, Stage 1 costs ≈2 core-hours on CPU or
≈0.6 GPU-hours batched. This is negligible against a 48–72 h budget.

## 1. Baseline reproduction (legacy final checkpoint `99b737…`)

* Re-running the legacy evaluator on the frozen legacy source reproduces **all 4,666 released
  metric values exactly** (`output/s1base/.../evaluation/repro_final_forecast`).
* The new model-agnostic harness (`metrics.py`) scoring dumped legacy banks reproduces the
  legacy path ES / Coverage90 / joint ES / count RPS / occupancy Brier / set kernel to float32
  precision (≤3e-10), so every comparison below uses identical scoring code for both models.

## 2. A measurement note on Coverage90

With 40 worlds, the legacy Coverage90 (truth inside the empirical 90% radius around the
sample mean) is **not 0.90 for a perfectly calibrated forecast**: simulation gives 0.87 (2-D)
down to 0.81 (60-D). We therefore also report a self-referenced check: hold out one world
as pseudo-truth against the other 39 (expected coverage of a calibrated model for that exact
case) and the same with the real truth; "obs−self" > 0 means over-dispersed. Legacy is
over-dispersed (+0.031 per person, +0.040 joint).

## 3. Ablations (small matched protocol: fast arch, 30% of Roots, 2 epochs, CPU)

Reference `abl_s0`: fast architecture, truncated law, first 30 training shards (220k anchors),
2 epochs, CPU-trained. Each row changes one component; paired relative change vs the
reference on development (negative = better), 95% scene-bootstrap interval.

| Variant | path ES | gap ES | count NLL | 2 m occupancy Brier | joint set kernel | executed-path step risk Brier | validation NLL |
|---|---|---|---|---|---|---|---|
| no map / visibility rasters | **+8.1%** [+6.2, +10.2] | **+5.9%** [+2.5, +9.6] | +0.4% [−0.4, +1.1] | **+1.6%** [+1.0, +2.2] | **+2.5%** [+1.6, +3.4] | **+3.1%** [+1.6, +4.7] | −13.99 (ref −15.24) |
| no visibility raster (map kept) | +0.9% [−0.3, +2.2] | +0.8% [−2.4, +3.8] | −0.1% [−0.5, +0.4] | +0.0% [−0.5, +0.5] | +0.3% [−0.5, +1.1] | −0.6% [−1.4, +0.4] | −15.06 |
| heuristic hard labels instead of the latent assignment marginal | **+1.9%** [+0.7, +3.4] | +1.5% [−1.0, +4.2] | **+2.7%** [+1.7, +3.8] | **+1.2%** [+0.6, +1.9] | **+3.3%** [+2.1, +4.6] | **+2.8%** [+1.0, +5.0] | −14.36 |
| single trajectory mode (M=1) | **−3.2%** [−5.9, −0.3] | −3.6% [−7.7, +0.2] | **−0.8%** [−1.4, −0.4] | **−0.6%** [−1.1, −0.1] | **−1.0%** [−1.6, −0.4] | **−3.0%** [−5.7, −0.5] | **−8.13** |

* Map context is the largest single contributor to tracked-path and downstream accuracy.
* The explicit visibility-age raster gives **no measurable gain** at this scale (negative result).
* Marginalising unresolved tracker keys matters most for the population law: hard heuristic
  labels worsen count NLL by 2.7% and count PIT KS from 0.013 to 0.048.
* A single Gaussian mode scores slightly better on sample-based metrics at this small scale
  but its likelihood is far worse (−8.1 vs −15.2 nats): it assigns near-zero density to real
  multimodal futures (turns, start/stop), which matters for rare-event risk. Post-hoc
  sharpening of the 6-mode sampler (temperature/spread grid on the validation split) did not
  reduce key-path ES either (identity was best), so the M=1 advantage is not simple
  over-dispersion.
* **Full-scale control** (`hwm_lite2_m1_e12`): identical to `hwm_lite2_e12` (all data, 12 epochs,
  same seed) except M=1. At full scale the small-protocol M=1 edge **reverses**; M=1 is worse
  on every headline score. Paired vs M=6:

  | Score | M=1 vs M=6 |
  |---|---|
  | path ES | +6.5% [+4.0, +9.5] |
  | gap | +7.1% [+1.0, +13.9] |
  | reappearance | +14.7% [+8.0, +23.1] |
  | joint ES | +5.5% [+3.1, +8.3] |
  | count NLL | +1.4% [+0.5, +2.5] |
  | set kernel | +1.2% [+0.3, +2.2] |
  | per-step risk Brier (executed) | +5.0% [+2.3, +8.1] |
  | per-step risk Brier (stationary) | +4.9% [+2.3, +8.4] |
  | count PIT KS | 0.022 vs 0.013 |

  Validation NLL: −8.54 vs −16.85. The 6-mode mixture is kept.

## 4. Consistency guarantees

* Total count ≥ maximum number of humans ever detected simultaneously: enforced exactly
  (truncated law; exact conditional sampling). Violations: legacy 0, new 0 (first new pilot
  without the constraint: 2.0% of worlds).
* Certified 3 m/s horizontal speed bound: legacy 0, new 0 (pilot without projection 0.38% of
  segments).

## 5. Negative results and limitations

* **Visibility-age raster**: no measurable benefit in the matched ablation; it remains in the
  selected model (trained with it) and costs ≈0.3 ms per call.
* **Single-mode / sharpened sampling**: see §3. The small-scale sample-score edge of M=1 did
  not hold at full scale (M=1 is 6.5% worse on path ES), and post-hoc sharpening gave no gain.
  A small-protocol ablation was therefore misleading for this design choice.
* **Speed vs likelihood**: the model with the best validation NLL among the 12-epoch runs (fast
  architecture) and the previous selection (`lite2`) both fail the pre-registered speed rule
  in at least one bracket. The selected `lite` has a slightly worse validation NLL (−16.75 vs
  −16.85) with no measurable development difference. The full architecture remains the most
  accurate on several cohorts (practical gap 0.334) but is ~3× slower on CPU.
* **Practical-gap calibration**: the selected model is slightly under-dispersed for people
  unseen 0.75–1.25 s (obs−self −0.024, interval of the difference to `lite2` includes 0).
* **Batched CPU serving** gains at most ~30% per robot (B=8, one thread). Large batches belong
  on the GPU.
* **Selection-rule precision**: the original rule did not specify how to combine several
  brackets. "Every bracket, against the mean of the two bracketing legacy runs" was written
  down after the quiet bracket put `lite2` at the margin (−19.7%), but before the final
  bracket. The final bracket independently confirms the outcome (`lite2` −16.0%, selected
  −22.1%).
* **Data limits**: privileged truth starts at the Root, so pre-Root tracker identities are
  latent (handled by marginalisation, 13.5% of development anchors). Long horizons are sparse
  (114 development Roots reach 3 s), so 3 s executed-path risk intervals are wide.
* **Simulator assumptions** the model relies on and that must be revisited for real robots:
  exact detections, exogenous humans (no reaction to the robot), the 3 m/s speed bound and
  persistent populations (used for the legal count bound).
* **Evaluation split**: the development split was also used by the legacy work; the
  architecture comparisons in this report used development metrics alongside validation NLL
  (checkpoint selection used validation only). No protected/holdout data were opened.
* **Legacy gates**: the legacy acceptance gates were not re-run as gates; the legacy H0
  P(count=0) failure (0.203 forecast vs 0.302 observed) is resolved (0.292), but the full
  legacy acceptance protocol (Hpositive reliability, persistence audit) was not repeated.
