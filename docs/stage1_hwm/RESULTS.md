# Stage 1 human world model: results

Development split: 2,110 Roots / 108 scenes / 6,391 legacy anchors, scene-disjoint from all
476 training scenes. Two seed streams (41, 107), 40 worlds, every native time to 3 s. Means
are anchor→Root→scene; intervals are 2,000-draw scene bootstraps (legacy protocol). Paired
comparisons match rows by (Root, anchor, stream, person/horizon or probe) and report the
relative change of the matched aggregate (negative = better for every listed score).

## Selected model: `hwm_lite2_e12`

Lite BEV backbone (64×64 at 0.4 m, ~36 M MACs), 6-mode trajectory mixture, exact truncated
population law, trained 12 epochs on all 105,838 training Roots (781k anchors). Checkpoint
and identity: `SELECTED.json` (SHA-256 `401c55ac…`). Selection rule fixed before the final
comparison: best validation NLL among candidates ≥20% faster than legacy at P50 and P95 on
every latency cohort. Columns: legacy final; **selected**; full-architecture model (`hwm_a`,
best quality, ~3× slower on CPU); fast-architecture 12 epochs (best validation NLL, failed the
speed rule). Paired change = selected vs legacy, 95% scene-bootstrap interval.

| Metric (dev) | Legacy | **Selected** | Full arch | Fast arch | Paired change (selected vs legacy) |
|---|---|---|---|---|---|
| tracked path ES (all) | 0.2206 | 0.1620 | 0.1599 | 0.1642 | -26.6% [-30.3, -22.7] |
|   visible | 0.1883 | 0.1357 | 0.1330 | 0.1371 | -28.0% [-32.0, -23.9] |
|   practical gap (0.75–1.25 s unseen) | 0.4262 | 0.3461 | 0.3344 | 0.3489 | -18.8% [-24.8, -12.5] |
|   reappearance | 0.1873 | 0.1335 | 0.1388 | 0.1373 | -28.7% [-33.2, -23.8] |
|   horizon 0.5 s | 0.1779 | 0.1354 | 0.1330 | 0.1376 | -23.9% [-28.2, -19.5] |
|   horizon 1 s | 0.2706 | 0.1916 | 0.1906 | 0.1939 | -29.2% [-33.4, -25.0] |
|   horizon 2 s | 0.3899 | 0.2575 | 0.2532 | 0.2570 | -34.0% [-39.4, -28.5] |
|   horizon 3 s | 0.5950 | 0.4062 | 0.4099 | 0.4066 | -31.7% [-38.0, -25.7] |
| joint tracked ES | 0.2295 | 0.1704 | 0.1679 | 0.1722 | -25.7% [-29.4, -22.0] |
| Coverage90 (legacy definition) | 0.901 | 0.887 | 0.892 | 0.882 |  |
| Coverage90 obs−self (0 = calibrated) | +0.031 | +0.011 | +0.021 | +0.008 |  |
| joint Coverage90 obs−self | +0.040 | -0.000 | +0.014 | -0.000 |  |
| count NLL (exact law / legacy unbiased estimate) | 1.5009 | 1.2787 | 1.2908 | 1.2760 | -14.8% [-17.3, -12.4] |
| count RPS | 0.7126 | 0.6198 | 0.6130 | 0.6171 | -13.0% [-15.3, -10.8] |
| count randomized-PIT KS (0 = calibrated) | 0.089 | 0.013 | 0.025 | 0.011 |  |
| P(count=0), histories without detections (observed 0.302) | 0.203 | 0.292 | 0.294 | 0.292 |  |
| 2 m occupancy Brier | 0.00853 | 0.00788 | 0.00781 | 0.00786 | -7.7% [-8.9, -6.3] |
| 0.5 m near-robot occupancy Brier ×100 (1 s) | 0.3512 | 0.3146 | 0.3153 | 0.3162 |  |
| joint future-set kernel score | 0.3147 | 0.2736 | 0.2705 | 0.2726 | -13.1% [-15.0, -11.3] |
| executed path: P(human ≤1 m within 1 s) Brier | 0.0857 | 0.0724 | 0.0708 | 0.0716 | -15.5% [-23.2, -8.2] |
|   same, ECE | 0.0569 | 0.0255 | 0.0227 | 0.0266 |  |
| executed path: per-step 1 m social Brier | 0.0531 | 0.0447 | 0.0442 | 0.0446 | -16.0% [-20.0, -12.3] |
| stationary robot: per-step 1 m social Brier | 0.0583 | 0.0457 | 0.0447 | 0.0459 | -21.5% [-25.8, -17.8] |
| candidate routes: P(≤1 m) Brier | 0.0685 | 0.0557 | 0.0553 | 0.0560 |  |
| candidate routes: within-Root pairwise ranking accuracy (1 m) | 0.945 | 0.960 | 0.960 | 0.963 |  |
| candidate routes: within-Root Spearman of E[min distance] | 0.808 | 0.828 | 0.834 | 0.835 |  |

Every tracked-path, population and candidate metric improves; no metric degrades
significantly. Executed-path risk cells at 3 s have wide intervals (185 Roots) that include 0.
The selected model's coverage is close to the calibrated reference (legacy over-dispersed).
The deployed CPU path (TorchScript engine + NumPy/numba sampler) reproduces these metrics
within Monte Carlo noise (path ES 0.16206 vs 0.16199; identical exact count NLL).

### Latency (CPU, 1 thread, legacy runtime protocol)

Bracketed back-to-back run on the same machine state (legacy → selected → fast → legacy;
load average 6.8–11.7, Stage 2 training running on GPU 0). Each online call ingests one new
actual frame from the pre-arrival state and returns 40 worlds × 64 times. P50 / P95 / P99 ms.

| Cohort | Legacy (before) | Legacy (after) | **Selected** | Fast arch |
|---|---|---|---|---|
| representative (first 200 Roots) | 9.53 / 14.77 / 117.8 | 10.04 / 14.92 / 120.1 | **7.60 / 8.67 / 10.3** | 8.41 / 9.65 / 10.8 |
| dense (30 most crowded) | 13.35 / 15.10 / 16.5 | 14.22 / 16.56 / 17.4 | **7.76 / 10.10 / 15.0** | 8.95 / 10.07 / 10.7 |
| longest history (30) | 10.13 / 13.48 / 26.0 | 10.70 / 13.22 / 17.7 | **7.81 / 9.00 / 12.1** | 8.81 / 9.90 / 11.9 |
| startup, full raw history (representative) P50 / P95 | 60 / 251 | 63 / 258 | **18 / 56** | 18 / 57 |

Selected vs legacy (before): P50 −20% / −42% / −23% and P95 −41% / −33% / −33% for the three
cohorts; P99 10–15 ms vs 16–118 ms (similar on the dense cohort, 10× lower on representative);
startup 3–4× faster. Tails of the new
path are bounded by removing TorchScript shape re-specialisation (warm-up, non-profiling
executor) and vectorising grid recentring; a rare generation-2 garbage collection remains.

Downstream reuse: scoring 3–24 candidate robot routes with one shared bank via the L2.5
joint reduction takes ≈0.9 ms. Batched GPU serving of B concurrent robots (one network call):
B=1 16.3 ms/call (16.33 ms per robot), B=8 29.4 ms/call (3.68 ms per robot), B=32 73.9 ms/call (2.31 ms per robot), B=64 133.2 ms/call (2.08 ms per robot) on a
GPU shared with other training. For an L3 run of 10⁶ decision steps, Stage 1 costs ≈2 core-hours
on CPU (one call per environment worker) or ≈0.6 GPU-hours batched.


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
  over-dispersion; the final model keeps M=6. Full-scale M=1 was not trained.

## 4. Consistency guarantees

* Total count ≥ maximum number of humans ever detected simultaneously: enforced exactly
  (truncated law; exact conditional sampling). Violations: legacy 0, new 0 (first new pilot
  without the constraint: 2.0% of worlds).
* Certified 3 m/s horizontal speed bound: legacy 0, new 0 (pilot without projection 0.38% of
  segments).

## 5. Negative results and limitations

* **Visibility-age raster**: no measurable benefit in the matched ablation; it remains in the
  selected model (trained with it) and costs ≈0.3 ms per call.
* **Single-mode / sharpened sampling**: see §3; likelihood strongly prefers 6 modes; the
  small-scale sample-score edge of M=1 was not reproduced by post-hoc sharpening. A matched
  full-scale M=1 control was queued (see the appendix once complete).
* **CPU latency margin** at the representative median is ≈20%; the network engine is ~half of
  the call. The faster 12-epoch fast-architecture model has the best validation NLL but only
  −12% median latency, so it was not selected.
* **Batched CPU serving** is slower per robot than independent single calls (use one call per
  environment worker on CPU; batch on GPU).
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
