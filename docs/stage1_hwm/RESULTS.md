# Stage 1 human world model: results

Development split: 2,110 Roots / 108 scenes / 6,391 legacy anchors, scene-disjoint from all
476 training scenes. Two seed streams (41, 107), 40 worlds, every native time to 3 s. Means
are anchor→Root→scene; intervals are 2,000-draw scene bootstraps (legacy protocol). Paired
comparisons match rows by (Root, anchor, stream, person/horizon or probe) and report the
relative change of the matched aggregate (negative = better for every listed score).

FINAL_MODEL_SECTION

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

NEGATIVE_SECTION
