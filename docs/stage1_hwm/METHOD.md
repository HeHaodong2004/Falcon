# Stage 1 human world model (HWM): method

This document describes the redesigned Stage 1 predictor in `eventwm_graph/stage1_hwm/`.
Results, comparisons and benchmarks are in `RESULTS.md`; reproduction commands in `REPRODUCE.md`.

## Problem and legal information boundary

At each decision time the predictor receives only arrived observations: robot base pose,
camera pose, tracker positives (exact positions of currently detected humans with tracker
keys) and static C-space certificates (FREE/BLOCKED cells). It must output a joint
distribution over the future positions (0–3 s) of **all** humans in the episode: tracked,
recently lost, and never seen. Tracker keys are bookkeeping, not identities: a person who
reappears after a gap may get a new key, and the old key then refers to nobody.

Privileged simulator data (all humans' native positions) is read only to build offline
labels after the legal features of an anchor are frozen (`extract.py`), and for scoring.

## Legal state (online and offline share one code path)

`legal_state.LegalState` replays arrived frames incrementally:

* exact static-certificate semantics of `PartialGraph.update` (BLOCKED wins within a frame;
  later FREE clears BLOCKED; ego support skipped), rasterized to a rolling 0.2 m grid;
* a **visibility-age raster**: per frame, 91 rays over the 90° camera FOV to 10 m, stopped at
  certified BLOCKED cells; the raster stores the last time each cell was visible. This is
  the negative information "we looked here recently and saw nobody" — an approximation of
  true line of sight (human occlusion and vertical FOV are ignored);
* per tracker key: up to 16 most recent positives, finite-difference / tracker velocity,
  observation count and first-seen time; legacy key order is preserved.

The anchor input is a camera-centred canonical frame (origin robot base, +x = camera
forward): map and visibility rasters (resampled on the device), key histories, robot motion
history and summary counts. `tests/stage1_hwm/test_online_equivalence.py` checks that the
online API reproduces the offline features exactly; the extractor reproduces the legacy
keys, physical labels, ages and positions on all 6,391 development anchors.

## Joint probability law

For one anchor the model defines a complete law over the human population:

1. **Keys.** Every key exists with probability `p_k` (1 for keys visible now). An existing key
   follows a mixture of M=6 Gaussian displacement sequences: segment 0 spans last
   observation → now (stale keys), segments 1..12 are 0.25 s steps to 3 s. Within a mode
   `d_s = μ_s + Δ_s L z + ε_s` with a shared velocity offset `z ~ N(0, I₂)` (correlates all
   segments) and independent `ε_s ~ N(0, σ_s² I)`. Paths are piecewise linear between knots,
   so any time in [0, 3.028] s is evaluated exactly.
2. **Residual humans** (people not represented by an existing key: never seen, on other
   floors, or reappearing under a new key) form an IID cluster process: count ~ Categorical
   (0..10), positions ~ softmax over a 64×64 grid of 0.4 m cells in the canonical frame plus
   one "outside 12.8 m" bin, futures from the same mixture family conditioned on the sampled
   position's map features.
3. **Legal population constraint.** Humans persist in this simulator, so the total count is at
   least the maximum number ever detected simultaneously. The law is conditioned exactly on
   this event: the training likelihood subtracts `log P(total ≥ m)`; sampling draws the total
   from the truncated count law and then key existences sequentially from suffix
   Poisson-binomial ⊗ residual distributions (exact, never fails, unlike rejection when the
   event is improbable); the reported count law is truncated and renormalized.
4. **Speed bound.** The collector certifies a 3 m/s horizontal human speed bound; sampled
   constant-velocity segments are projected onto it.

Worlds are IID samples of this law (uniform weights); key existence, modes and residual
draws are independent across worlds, so every world is a coherent joint population.

## Training objective: exact joint likelihood

The loss is the exact negative log-likelihood of the complete observed future population
under the law above. Keys whose identity the labels resolve contribute their own existence
and trajectory terms. **Unresolved keys** (e.g. last seen before the Root, where privileged
truth begins) are not dropped or heuristically labelled: their assignment to humans is a
latent variable, summed over every injective partial assignment of up to 4 unresolved keys
to the remaining humans, with each unmatched human explained by the residual process
(Janossy density with the `n!` factor). This keeps count, existence and residual-density
supervision consistent with deployment semantics.

Network (`model.HWM`): BEV CNN over map+visibility rasters (full: 128×128 at 0.2 m;
fast/lite/lite2: 64×64 at 0.4 m; the selected `lite2` uses one conv per scale, widths 16/32/48,
32-channel field), key encoder (history GRU/MLP + map features sampled along the
expected heading), 2-layer transformer over keys and a global token, heads for existence,
key trajectories, residual count, residual density (FiLM-conditioned) and residual
trajectories. AdamW, cosine schedule, random reflection augmentation.

Model selection uses a scene-disjoint ~6% validation subset of the training split; the
development split is used only for reporting.

## Evaluation (`metrics.py`)

All models are scored from saved world banks by identical code:

* **Legacy-compatible**: tracked path Energy Score, Coverage90, joint ES/coverage by cohort
  (visible, practical gap, reappearance) and horizon; count NLL, count RPS, 2 m occupancy
  Brier, joint future-set kernel score; Root→scene means with scene bootstrap. Scoring the
  legacy banks reproduces the released legacy metrics to float32 precision.
* **Count calibration**: randomized PIT (KS distance), P(count = 0) and count-CDF reliability,
  overall and for histories without any detection (the legacy H0 failure).
* **Downstream risk**: for the executed robot path and a stationary robot, P(any human within
  r ∈ {0.5, 1, 2} m during the next H ∈ {1, 2, 3} s) → Brier, log loss, reliability (ECE);
  per-step 1 m social probability; expected social time.
* **Candidate ranking**: at each Root, all sibling Query branches (3–24 executed robot
  candidates against the same exogenous humans) are scored with one shared bank: Brier,
  within-Root pairwise ranking accuracy and Spearman correlation of expected minimum
  distance.
* **Consistency**: fraction of worlds violating the historical count bound or the speed bound.

## Deployment inference path (CPU, 1 thread)

`api.HumanWorldModel` keeps one incremental `LegalState` per robot stream and, on CPU with the
fast/lite architectures, runs `infer.FastSampler`:

* one TorchScript module (raster resampling, BEV, key/global encoders, transformer, all heads)
  bound to the non-profiling executor and warmed up at construction (no shape
  re-specialisation stalls; the global executor mode is restored afterwards);
* NumPy/numba sampling: fused path drawing (`kernels.draw_paths`), exact conditional
  population sampling, residual cell sampling, exact count law;
* numba visibility raycast and vectorised crops/recentring; a bitwise-identical
  single-anchor feature builder (`data.collate_one`).

Equivalence evidence: online features equal offline extraction (79 anchors); scripted engine
equals the eager network to 4e-6; fused kernel equals NumPy to 5e-7; fast-path dev banks
reproduce the batched torch sampler's metrics within Monte Carlo noise with identical exact
count NLL. `predict_batch` serves many streams with one network call (GPU or CPU).

## Sandbox interface

`HumanWorlds` subclasses the legacy `JointHumanWorlds` (positions [W,T,N,2], present [W,N],
times, uniform weights, track keys, residual flags) and adds the exact count law, per-key
existence probabilities and the 0.25 s knots (`positions_at` is exact at any time).
`evaluate_candidates(robot_paths)` scores K candidate robot trajectories with the one shared
bank through the L2.5 `aggregate_joint` reduction (collision envelope, 1 m social events,
first-event masses, minimum distances); humans are exogenous, so the bank is valid for every
candidate.
