# Stage 2 query-ranking benchmark (diagnostic) — do Stage 2 predictions distinguish the future-observation consequences of alternative Queries?

Scope: offline, development split only, using the frozen dataset's recorded Query Outcomes, executed trajectories,
legal partial maps / sensor histories and observed future-map labels. No Sandbox, Layer 2.5, Layer 3, GNN/RL,
planner rollouts or new trajectories. This benchmark is diagnostic: it does not replace the original
full-development label-aware IoU or the latency gate, and it does not change Stage 2 prediction semantics. Training
continued independently while it ran.

Code: `eventwm_graph/stage2_claude/query_audit.py` (step 1), `query_bench.py` (labels, predictions, legal
heuristics), `query_report.py` (metrics, scene bootstrap, tables). Outputs: `output/claude_stage2/query_bench/`
(`audit.json`, `predictions/<method>.json.gz`, `report.json`, `report_per_root.json`).

## 0. Answers (primary set: common 1.0 s horizon — 885 Roots, 93 scenes, 3,077 distinct outcomes, 3,658 decisive pairs)

All numbers are from `output/claude_stage2/query_bench/report.json`; every table is in `QUERY_BENCH_TABLES.md`.
Brackets are 95 % scene-bootstrap intervals. v16_b is the latest verified checkpoint (full-dev IoU 0.4809), v15 the
previous one (0.4784).

**Is the current Stage 2 useful for distinguishing the future-observation consequences of alternative trajectories,
although its map IoU is below 0.50?** Yes, for the recorded Query Outcomes evaluated here, and with a large margin over
every deterministic baseline:

* For a decisive pair of alternative outcomes from the same Root, v16_b orders their information gain correctly
  **92.8 % [91.0, 94.3]** of the time (Root-macro). Its predicted-best outcome is the actual best **81.2 % [78.3, 84.0]**
  of the time, and within 10 % or 3 cells of it 92.5 % of the time.
* Selecting by its prediction loses on average **17.9 [12.1, 23.9] cells** of information gain. That is 3.4 % of the
  mean gain of 526 cells, or 0.10 of each Root's gain range; the median loss is 0 and the P90 33 cells.
* The best legal-map heuristic (legal visibility count) reaches 80.2 % / 68.5 % / 49.3 cells, and trajectory
  length/duration heuristics are at chance at a fixed horizon. Paired, v16_b minus the legal visibility heuristic is
  +12.6 points of pairwise accuracy [10.1, 15.1], −31.3 cells of regret [−41.6, −21.5] and +12.7 points of top-1 hit
  [9.2, 16.1].
* The same holds at 0.5 s (93.0 % vs 77.1 %), 2.0 s (92.3 % vs 82.9 %) and for full outcomes on non-nested pairs
  (95.1 % vs 84.0 %).
* Why the ranking works despite the low cell IoU: it depends on aggregate gains. Within a Root, errors are shared by
  the alternatives (over-prediction in the same unseen space), so they largely cancel in the comparison; within-Root
  Spearman correlation between predicted and actual gain is 0.79.
* What this does not show: hypothetical Sandbox trajectories, gains far beyond the recorded executions, or navigation
  utility (section 4).

| question | answer (v16_b unless noted; primary set) |
| --- | --- |
| 1. Accuracy of the predicted information gain | Raw expected gain (sum of FREE+BLOCKED probabilities) correlates well with the actual gain (Pearson 0.884, Spearman 0.937, within-Root Spearman 0.792), but it **over-predicts**: mean 626 vs 526 cells, bias +99.5 (+19 %), MAE 140.8 [125.2, 156.3] cells. FREE gain MAE 84.0 (bias +37.5), BLOCKED gain MAE 80.5 (bias +62.0). Counting hard decisions at the frozen operating point gives MAE 111.1 (bias −19.9). A binned recalibration fitted only on TRAIN validation reduces the expected-gain MAE to **108.3** (bias +26.3, +5 %) and slightly improves ranking (93.5 %). Per-trajectory prediction quality: per-outcome label-aware IoU mean 0.557, median 0.569 (IQR 0.426-0.697); 11 % of outcomes are below 0.3 and 43 % above 0.6 |
| 2. Within-Root ranking reliability | Root-macro pairwise accuracy over decisive pairs **92.8 % [91.0, 94.3]** (pooled 93.8 %), scene-macro similar. It grows with the actual difference: 89.3 % (gain differs by 10-25 %), 94.0 % (25-50 %), 98.0 % (>= 50 %). Near ties (1,213 pairs) are ordered correctly 70.2 % of the time and the 60 exact ties are excluded. FREE-gain ranking 88.2 % [85.9, 90.3], BLOCKED-gain ranking 92.2 % [90.5, 93.9]. Kinematically matched pairs (distance within 0.2 m, heading change within 0.3 rad), which length heuristics cannot separate: 92.3 % (legal visibility 78.5 %) |
| 3. Information lost when selecting the predicted best Query | Top-1 hit 81.2 % [78.3, 84.0] (near hit 92.5 %). Mean regret 17.9 [12.1, 23.9] cells, normalized regret 0.102 [0.082, 0.125] (Roots with a gain range >= 3 cells: all but 11), median 0, P90 33. Random selection: hit 34.7 %, regret 185.4 cells; legal visibility: 68.5 %, 49.3 cells. Secondary sets: 0.5 s regret 6.5 cells (hit 84.3 %), 2.0 s 30.1 cells (80.3 %), full outcomes 19.5 cells (84.9 %) |
| 4. How often newly observed FREE space is predicted wrongly | Of the 511,418 cells predicted FREE (frozen operating point, initially UNKNOWN cells): **6.1 % were observed BLOCKED** (a state error) and **28.5 % were still UNKNOWN after execution** (a future-observation error, not evidence of an obstacle). FREE precision 0.655, recall 0.684. In never-observed cells FREE precision drops to 0.512 and 37.0 % stay UNKNOWN. The FREE→BLOCKED share grows with the horizon (2.3 % at 0.5 s, 6.1 % at 1.0 s, 8.8 % at 2.0 s), while the FREE→still-UNKNOWN share stays ~28-31 % |
| 5. Trajectory conditions with the largest errors | Map accuracy (pooled outcome IoU) is lowest for **turn-only Queries** (0.400, gain MAE 213), **Queries that turn into previously unseen directions** (0.442 vs 0.607 for seen directions; gain MAE 181 vs 108) and heading changes of 1-2 rad (0.466). It is highest for translations > 1.5 m (0.608). At cell level, previously observed cells reach 0.719 but never-observed cells 0.389, and the latter hold 56 % of the targets. History length matters little (0.515 short vs 0.531 long). Decision accuracy is much more robust: 86.5 % for turn-only pairs, 93.0 % when both outcomes look into unseen directions, 89.6-97.5 % elsewhere |
| 6. Better than simple deterministic baselines? | Yes, significantly, against every baseline and in every comparison set. Pairwise accuracy at 1.0 s: random 50 %, distance 46.6 %, duration 51.5 % (durations are equal at a fixed horizon), heading change 53.3 %, known-visible count 60.9 %, frustum count 72.6 %, legal visibility count 80.2 %, **v16_b 92.8 %**; the paired differences against each baseline exclude zero |
| 7. Does better map IoU translate into better ranking / regret? | **No measurable effect on ranking or regret within the available range.** v7 (full-dev IoU 0.4563) → v11 → v12 → v15 → v16_b (0.4809): pairwise 92.6 / 93.0 / 93.3 / 92.9 / 92.8 %; paired v16_b − v7 +0.2 points [−1.0, +1.4], regret +0.1 cells [−2.2, +2.9]. Full outcomes show a small significant ranking gain (+0.7 points [0.3, 1.0]). What improves strongly with the newer models is the **magnitude** of the predicted gain (expected-gain MAE 272 → 141 cells; v7 bias +49 % vs +19 %), FREE precision (0.610 → 0.655), the share of predicted FREE cells that stay UNKNOWN (32.6 % → 28.5 %) and the per-outcome IoU (0.525 → 0.557) |

Calibration (question E): the probabilities **discriminate well but are overconfident**. Among cells with predicted
revelation >= 0.05, v16_b predicts 0.497 on average while 0.398 are revealed (ECE 0.109 there; all cells 0.026, Brier
0.0059). For example, predicted 0.50-0.55 → observed 0.32 and 0.85-0.90 → 0.55; FREE and BLOCKED behave the same
(ECE 0.13 / 0.10). This is expected from the positive class weighting and the soft-IoU term of the training loss, and
it explains the gain over-prediction. A monotone recalibration fitted on TRAIN validation removes most of the bias
(see 1). The binned AUC (0.95-0.98) only indicates good discrimination. With 99 % of cells in the lowest of 20 bins it
is too coarse to compare checkpoints; a per-cell AUC within the hardest stratum is 0.92 (main report, section 3).



## 1. Dataset audit (step 1)

How the data are represented: every development Root (`legal_root.json.gz`) holds the decision-time partial graph,
the legal observation history and sensor frames; `queries.json.gz` holds the Queries (ordered waypoints + terminal
yaw, compiled into discrete actions by the known-space route follower; `static_traversability: known_free_reachable`,
i.e. planned on the legal map), `outcomes.json.gz` one recorded Outcome per Query (executed node path, termination,
safety summary) and `future_observations.json.gz` the executed nodes as a prefix DAG (Queries with identical action
prefixes share nodes). The original Stage 2 windows (`stage2_segment.data.windows`, strategy `mixed`) are
deduplicated (start node, end node) segments: per Query the full executed path plus 0.5 / 1 / 2 s sub-windows at
random start steps.

| audit question | answer (development split: 2,110 Roots, 108 scenes, 14,546 Queries) |
| --- | --- |
| 1. Roots with >= 2 distinct comparable Outcomes | full outcomes: 2,089 Roots (108 scenes, 13,481 outcomes, 49,840 pairs); common horizon 1.0 s: 885 Roots (93 scenes, 3,077 outcomes, 4,931 pairs); 0.5 s: 1,380 Roots (103 scenes, 4,415 outcomes, 6,222 pairs); 2.0 s: 249 Roots (56 scenes, 691 outcomes, 761 pairs). Queries per Root: median 6 (3-24) |
| 2. Same initial observation, history and partial map? | yes: every Query of a Root is decided at the Root's decision time (checked for all 14,546) and every Outcome starts at the Root state; the benchmark feeds each Root's comparable outcomes to the predictor with the Root's partial graph, legal range frames and track history (identical for all of them) |
| 3. Executed trajectory and observed increment recoverable? | yes for all: the executed path is the Root pose plus the node poses; the observed increment is replayed with the original legal DAG replay (`replay` of `stage2_segment.data`, recorded graph deltas verified) as the initially UNKNOWN cells that are FREE or BLOCKED at the outcome's end node |
| 4. Common time horizon? | yes, by truncation at saved nodes (no interpolation): each Query is cut at its node closest to t0 + H if that node lies within 0.05 s (one action is ~0.048 s) of t0 + H; Queries that ended earlier have no outcome at H and are excluded at that horizon. Full outcomes have different durations (median 0.73 s, P90 1.85 s; 12,157 completed, 2,382 ended by a human collision, 7 timeouts) |
| 5. Are existing windows independent outcomes? | no: the 0.5 / 1 / 2 s windows of a Query are subdivisions of the same executed outcome and are never compared with each other. Different Queries that share the action prefix up to the horizon end at the same node and are merged into one outcome (1,311 Query outcomes merged at 1.0 s; 4,404 at 0.5 s); in the full set, 9,981 outcome pairs are prefix-nested (one path is the start of the other) and are excluded from pairwise scores (reported separately) |

Primary comparison set (fixed before results): **common horizon 1.0 s** (885 Roots, 93 scenes). The 0.5 s, 2.0 s
and full-outcome sets are reported as secondary. Roots with fewer than two distinct outcomes are excluded (1,225 at
1.0 s); no counterfactual outcome is created.

## 2. Definitions

* **Actual information gain** G of an outcome: number of initially UNKNOWN cells of the original candidate domain
  (cells within 11 m of the supplied positions) that are observed FREE or BLOCKED at the outcome's end (FREE gain /
  BLOCKED gain separately). No target fell outside the domain.
* **Predicted gain**: expected gain = sum over the same domain of the model's FREE + BLOCKED probabilities
  (`model`); hard gain = cells predicted FREE or BLOCKED at the frozen operating point (`model:hard`);
  recalibrated expected gain = the same sum after a binned (20 revelation x 10 BLOCKED-given-revelation)
  recalibration fitted only on the model's 400-Root TRAIN-validation selection histogram (`model:recal`).
* **Pairs**: all pairs of distinct outcomes of a Root. Decisive if |G_i - G_j| >= max(3 cells, 10 % of the larger);
  near ties (smaller non-zero difference) and exact ties are counted and reported separately; a predicted tie scores
  0.5. **Principal pairwise accuracy = Root-macro mean over decisive pairs** (each Root weighs equally).
* **Top-1 regret** = G(best outcome) - G(outcome with the highest prediction) (prediction ties: mean over the tied
  set); normalized by the Root's gain range only when that range is >= 3 cells; hit = the selected outcome has the
  maximal actual gain; near hit = within max(3 cells, 10 %).
* **False FREE** (frozen operating point, initially UNKNOWN cells): predicted FREE but observed BLOCKED (a state
  error) vs predicted FREE but still UNKNOWN after execution (a future-observation error, not evidence of an obstacle).
* **Calibration**: reliability of the revelation, FREE and BLOCKED probabilities over all candidate cells (20 bins),
  Brier scores, ECE (all cells and cells with p >= 0.05) and binned AUC (discrimination).
* **Regimes**: duration, path distance, total heading change, history length (Root, split at the 0.632 s median),
  unseen direction (some future heading > 45 deg from every legal history heading), previously observed vs
  never-observed cells (no observed voxel in the floor/low/mid height bands of the legal map).
* **Uncertainty**: 1,000 scene-level bootstrap resamples (seed 0) of Root-level values; method differences are
  paired (same resampled scenes).

## 3. Methods compared (identical groups and labels)

Stage 2 checkpoints, each with its own frozen operating point and the sensor resolution of its official evaluation:
v7 (128 px; full-dev IoU 0.4563), v11 (0.4688), v12 (0.4716), v15 (0.4784), v16_b (latest verified, 0.4809).
Deterministic baselines from legal inputs only: random (exact expectation), trajectory distance, duration and total
heading change, and three legal-map counts computed with the same legal voxel simulation the model uses (no
learning): `frustum` (unknown candidate cells whose centre lies inside a future camera frustum within 10 m, occlusion
ignored), `legal_visible` (cells whose 63 FREE probes have no KNOWN blocker from some future pose, unknown space
treated as passable) and `known_visible` (probes fully visible through known free space). Recorded future
observations are used only for labels.

## 4. Limitations and uncertainty

* **Recorded outcomes, not hypothetical trajectories.** The alternatives are Queries that were actually executed: a
  known-space route follower on routes planned in known-free space, with real human interactions; 2,382 of the 14,546
  ended in a human collision. Sandbox proposals may differ (more aggressive motion into unknown space, longer
  horizons), and nothing here measures that shift or downstream navigation.
* **Information gain is only one consequence.** It ignores safety, progress to the goal and the value of where the
  gained cells lie. Top-1 regret therefore measures information-gain selection only.
* **Horizon truncation.** The common 1.0 s horizon excludes Queries that ended earlier: 1,225 Roots have fewer than
  two distinct 1.0 s outcomes. Full-outcome comparisons are confounded by duration, and 9,981 prefix-nested pairs are
  excluded there.
* **Near ties are hard.** Gain differences below max(3 cells, 10 %) are ordered correctly only ~70 % of the time.
  Decisive-pair accuracy also depends on the tie threshold, which was fixed in advance; the near-tie and difference-bin
  tables show the sensitivity.
* **Absolute gains are biased** (+19 % for v16_b, +26 % for v15) unless recalibrated, and the probabilities are
  overconfident.
* **Baseline strength.** The strongest baseline reuses the same legal visibility simulation that feeds Stage 2, with
  unknown space treated as passable. A more elaborate hand-made predictor of unknown space might narrow the 12.6-point
  gap.
* **Sample size.** The primary set has 93 scenes, so pairwise accuracy is known to about ±1.5-2 points and top-1 hit
  to about ±3 points. Pairs within a Root are dependent, which the scene bootstrap accounts for.
* **Engineering note.** One prediction re-run (v16_b, adding histograms) stopped with a CUDA illegal-memory-access
  error on the GPU that also drives the display; its output was discarded. Rerunning the same 2,110 Roots completed and
  reproduced the first pass exactly (0 of 22,093 outcome records differ; the v15 re-run also matched exactly). The
  cause was not reproduced and is recorded here.

## Reproduction

```bash
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python; R="$PY scripts/stage2_segment_run.py"
export CUDA_VISIBLE_DEVICES=0,1
$R eventwm_graph.stage2_claude.query_audit --workers 6                   # step 1 -> audit.json
$R eventwm_graph.stage2_claude.query_bench prepare --workers 6           # labels per comparable outcome (cache/)
for m in v7:full_range_branches v11 v12 v15 v16b; do n=${m%%:*}; s=${m#*:}; [ "$s" = "$m" ] && s=full_range_branches_256
  $R eventwm_graph.stage2_claude.query_bench predict --checkpoint output/claude_stage2/final/$n.pt --name $n \
     --device cuda:0 --workers 2 --sensor-directory $s; done
$R eventwm_graph.stage2_claude.query_bench heuristic --checkpoint output/claude_stage2/final/v15.pt --device cuda:0
$R eventwm_graph.stage2_claude.query_report                              # report.json, report_per_root.json
```
