# Layer 3 shared Query Selector: progress report

Branch `research/l3-shared-policy-20261029`. Design: [DESIGN.md](DESIGN.md). Protocol: [PROTOCOL.md](PROTOCOL.md).

## 0. Declared configuration split (2026-10-29, before the first training run)

One shared policy is trained on the SEEN configurations from the start; HELD-OUT configurations are never trained on
and measure zero-shot transfer.  "Official" = the native discrete Falcon action set; every other dynamics family uses
the continuous base-velocity action (modified robot dynamics, outside the official benchmark).

| configuration (planner-dynamics-candidates) | official | role | what is new at test time |
|---|---|---|---|
| orca-native-k10 | yes | seen | -- |
| dwa-native-k10 | yes | seen | -- |
| orca-native-k6 | yes | seen | -- |
| orca-inertia-k10 | no | seen | -- |
| dwa-slowturn-k10 | no | seen | -- |
| dwa-slowturn_inertia-k10 | no | seen | -- |
| orca-slowturn_inertia-k6 | no | seen | -- |
| dwa-native-k6 | yes | held out | planner x candidate-set combination |
| sfm-native-k10 | yes | held out | planner (social force model never seen) |
| dwa-inertia-k10 | no | held out | planner x dynamics combination |
| orca-slowturn-k10 | no | held out | planner x dynamics combination |
| orca-heavy_brake-k10 | no | held out | dynamics family (asymmetric braking) |
| dwa-slowturn_heavy-k10 | no | held out | dynamics family |
| sfm-inertia-k10 | no | held out | planner and modified dynamics |

## 1. Status (2026-10-29 13:25)

* Foundation built and tested (DESIGN.md): Graph Compiler v1, consequence-graph network, off-policy learner, actor
  fleet, official replay verification, Oracle-guided warm-up data (380 Oracle solutions replayed through the full loop;
  counterfactual labels accumulating), frozen suites.
* Training runs (`scripts/l3_runs.sh`; every run directory holds its code commit and arguments):

| run | what | status |
|---|---|---|
| R1 | shared policy, full world model, warm-up, 7 seen configurations | stopped after 40 min / 9.8k decisions: critic over-optimistic (below) |
| B1 | no-imagination ablation, R1 learner | stopped with R1 (28 episodes) |
| R2 | R1 + Monte Carlo target mixing (0.5 -> 0.1 over 300k decisions) + replay ratio 64; seeded with R1's episodes | running since 13:20 |
| B2 | no-imagination ablation with the R2 learner, seeded with B1's episodes | running since 13:20 |
| R2 | (see above) | learner aborted after 5 min with a native heap corruption (`malloc(): invalid size`) while its ingestion thread wrote the ring; fleet stopped (no saved state yet) |
| R3 | R2 settings, thread-free learner (ingestion on the main loop), state saved every 10 min, auto-restart; seeded with all R1 + R2 episodes | running since 13:37 (main run) |

**R1 diagnosis.** After warm-up on the Oracle demonstrations (all successful episodes) the success head predicted
P(success) = 0.98 for the chosen candidate against an actual 0.36 (per decision, versions 0-2), still 0.86 vs 0.34 at
versions 6-8, correlation 0.27.  3-step bootstrapping propagates the optimism to every decision more than three
decisions before a failure, and the replay-ratio cap (32) allowed only ~2k updates per hour.  R2 mixes the Monte Carlo
return of the episode into the value targets (weight decaying from 0.5) and doubles the replay ratio; R1's episodes
remain in the buffer (the learner is off-policy).

## 2. Baselines on dev (no learning)

Default policy ("floor": keep the running query, else the goal query), no imagination needed; first 21-22 dev
episodes per configuration (`output/eval/floor_dev24`).

| configuration | SR | SPL | H-Coll | timeout |
|---|---|---|---|---|
| orca-native-k10 | 0.59 | 0.38 | 0.32 | 0.09 |
| dwa-native-k10 | 0.64 | 0.50 | 0.27 | 0.05 |
| orca-native-k6 | 0.57 | 0.38 | 0.29 | 0.14 |
| dwa-native-k6 | 0.50 | 0.44 | 0.23 | 0.27 |
| sfm-native-k10 | 0.62 | 0.39 | 0.33 | 0.05 |
| orca-slowturn-k10 | 0.48 | 0.33 | 0.33 | 0.19 |
| dwa-slowturn-k10 | 0.48 | 0.36 | 0.38 | 0.14 |
| orca-inertia-k10 | 0.32 | 0.23 | 0.23 | 0.45 |
| dwa-inertia-k10 | 0.24 | 0.19 | 0.43 | 0.33 |
| sfm-inertia-k10 | 0.24 | 0.17 | 0.29 | 0.48 |
| orca-heavy_brake-k10 | 0.29 | 0.19 | 0.38 | 0.33 |
| orca-slowturn_inertia-k10 | 0.14 | 0.11 | 0.43 | 0.43 |
| dwa-slowturn_inertia-k10 | 0.24 | 0.18 | 0.29 | 0.48 |
| dwa-slowturn_heavy-k10 | 0.14 | 0.12 | 0.48 | 0.38 |

Reference points from the literature (official val splits, end-to-end RGB-D policies; our robots observe through
the ideal legal geometry sensor, so the comparison is indicative only): Falcon Social-HM3D SR 55.2 / SPL 55.2 /
PSC 89.6 / H-Coll 43.0, Social-MP3D SR 55.1 (arXiv 2409.13244); NavThinker Social-HM3D SR 59.5 / SPL 55.0 / PSC 89.9 /
H-Coll 39.1 (arXiv 2603.15359).

## 3. Evaluation plan

**Pre-registered checkpoint selection (2026-10-29 13:45, before any dev evaluation of a learned policy).**
Milestone dev evaluations (all 72 dev episodes x the 14 configurations, greedy policy) of the main run's checkpoints.
The final shared policy is the checkpoint with the highest mean dev SR over the SEVEN SEEN configurations (ties: mean
SPL); the held-out configurations never influence selection.  The conservative-deviation margin (0, 0.25, 0.5, 1.0)
is chosen the same way on dev for that checkpoint.  The same rule selects the checkpoints of the ablation runs.

* E1 main: the shared policy vs the floor, the hand-written imagination heuristic (`heuristics.greedy_index`), and
  the Oracle where available, on the official configurations, test split (after the freeze; official replay).
* E2 transfer: seen vs held-out configurations, zero-shot, same policy.
* E3 ablations: no imagination (B2), specialist (S1), no warm-up (N1); representation ablations with the data held
  fixed (critics trained offline on the main run's replay with feature groups removed -- Stage 2 maps, Stage 1
  human futures, waypoint/interaction nodes (flat), imagination entirely -- each then evaluated online on dev).
* E4 throughput and resources (measured).
* E5 the decomposed heads (calibration of P(success) and P(collision)).
