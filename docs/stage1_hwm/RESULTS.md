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

ABLATION_SECTION

## 4. Consistency guarantees

* Total count ≥ maximum number of humans ever detected simultaneously: enforced exactly
  (truncated law; exact conditional sampling). Violations: legacy 0, new 0 (first new pilot
  without the constraint: 2.0% of worlds).
* Certified 3 m/s horizontal speed bound: legacy 0, new 0 (pilot without projection 0.38% of
  segments).

## 5. Negative results and limitations

NEGATIVE_SECTION
