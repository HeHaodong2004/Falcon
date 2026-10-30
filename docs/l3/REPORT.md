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
| R3 | R2 settings, thread-free learner (ingestion on the main loop), state saved every 10 min, auto-restart; seeded with all R1 + R2 episodes | 13:37-13:58; calibration fixed (P(success) 0.47 vs actual 0.33, corr 0.59, was 0.86 vs 0.34 / 0.27), but momentum robots thrashed among manoeuvres (ORCA-inertia 6/40, DWA-slowturn-inertia 1/27 successes, ~85% non-continue choices) |
| R4 | R3 + Graph Compiler v1b (the 3 s risk extension starts with the Sandbox's imagined emergency stop: momentum robots cannot freeze at the end of an interval; inertia 0.52 m / heavy-brake 1.15 m stop distance) + default-anchored exploration (the actor takes the default candidate with probability 0.3 -> 0.05 over 300k decisions; Q-learning is off-policy, so only the data distribution changes); seeded with earlier main-run episodes of the configurations v1b leaves unchanged | running since 14:00 (main run) |
| B3 | no-imagination ablation with R4's learner and exploration, seeded with B1 + B2 | 14:00-14:33, paused (SIGSTOP) to give R5's labellers the CPU; the world-model ablation moves to the fixed-data protocol (E3) |
| R4 | (see above) | 14:00-14:33; calibration good (P(success) 0.49 vs 0.44), training SR at the floor on native robots |
| R5 | continued from R4's learner state + 10 on-policy counterfactual labellers (`cfonline.py`: every candidate at the last 4 decisions of failed episodes / spread over timeouts / one random decision of 25% of successes, default continuation, capped at 160 steps; replay verified action by action) + 25% of each batch from labelled states + hinge floor 0.5; 16 training actors | running since 14:34 (main run) |

**Updated goals (user, 2026-10-29 ~14:30):** SR is the primary metric -- best configuration >= 85% SR, ideally 90%+;
every configuration (incl. unseen combinations) pushed toward SOTA with one shared policy; speed (training and
evaluation throughput) second; SPL / PSC / H-Coll tracked.

**Why on-policy counterfactual labels.** An RL decision costs ~3 CPU-s and reveals one candidate's outcome; most
decisions do not matter and the few critical ones must be found by chance.  The branch simulator can go back to the
critical states of the policy's own failures and execute every candidate (exact, privileged supervision only).
Example (R4 policy, ORCA native): at decision 1 of a collision episode the policy chose `st_goal_th` (collision)
while 7 of the 10 alternatives succeed with the default continuation; at decisions 2-3 nearly every candidate
collides (doomed).  First 18 labelled failures: 40-50% of the native-robot failures have such a one-deviation fix
within the last 4 decisions; for momentum robots the default continuation itself mostly fails (few positive labels).

**R5b (15:12)**: R5 restarted in place (spool, labels, learner state carry over) with dense counterfactual
supervision (the value heads regress on every labelled candidate's exact outcome, weight 0.5, next to TD), a
6-decision critical window and 14 labellers / 14 actors.  Label statistics after 314 labelled failure states: in 55
the chosen candidate fails while another succeeds; the failing choice is mostly `continue` (16) or a fresh `goal`
(8), the fixes mostly OTHER goal variants (goal_tc 24, goal 21, st_goal_tc 13, st_goal 12, goal_th 12) or a short
wait (11) -- different execution styles of "go to the goal" near people, separable only through their imagined
consequences.  Fixable failures (one deviation + default, within the labelled window): native 47-57%, slowturn ~50%,
momentum 19-45%.

**Speed (exact optimisations).** The legal sensor's per-cell support certificates are pure functions of the scene
geometry under `pure_caches` but were recomputed in every episode (~100 us per cell through ~10 native pathfinder
calls).  `sensorcache.py` shares them across episodes and processes (disk, ~12-24 MB per scene): sensor 61 -> 40-45
ms/step on revisited cells; actions, legal map and tracks identical (checked).  Remaining sensor cost: native
support-probe raycasts (18 ms/step, the next candidate for GPU emulation).

**Measured actor cost per decision (R5, loaded machine):** Sandbox round trip with imagination 1.33 s, legal sensor
0.59 s, range frames (Stage 2 input only) 0.44 s, real executor 0.23 s, simulator 0.22 s, compiler + generator +
policy 0.10 s; total ~2.95 s.  ~3-3.5 decisions/s for 16 actors next to 10 labellers.

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

## 4. Diagnostics: what limits a legal high-level selector (2026-10-29 afternoon)

**Paired dev evaluation, learned selector R5 (version ~160, greedy) vs the floor (64 episodes per configuration):**
DWA-native 48.4% vs 50.0%, ORCA-native 46.9% vs 46.9%, ORCA-native-K6 32.8% vs 40.6%, SFM-native (held-out) 51.6% vs
43.8%; policy-only / floor-only successes 6-11 each -- the policy reshuffles outcomes rather than improving them.

**Reference points recomputed.** The querygen suites behind "Oracle ~100%, floor 78%" contain 28-40% episodes WITHOUT
people (unseen6: 17 of 60).  The official val split (our test) and the L3 dev/train suites contain only 2-6-person
episodes.  Privileged v14 Oracle on the episodes WITH people: 94-100% of the episodes not proven infeasible (held-out
unseen6: 42/42 for ORCA and DWA K=10), 88-98% when proven-infeasible episodes count as failures (official SR; unseen2
has 3 among its 43 people episodes).  Floor on the same people episodes: 58-67% (unseen6); 47-57% on L3 dev.  On L3
dev, 7 of 72 episodes were never solved by any run (3 collide within 5 steps in all 30 runs: spawn collisions), and the
Oracle with the expert budget solves 82-84% of the dev episodes searched so far.  An 85% target therefore requires a
legal policy to close ~75-85% of the gap between the floor and an Oracle that knows every person's future.

**Collisions.** (306 collision episodes of the policy) at the last decision before contact 32% had no tracked person
and only 50% a tracked person within 2 m; privileged check (25 collisions): the colliding person was never inside
the jaw camera frustum during the last 2 s in 28% (briefly in 12%); bearing at contact front 48% / side 32% / behind
20%.  The predicted state-level danger is informative (AUC 0.79 for a collision in the current interval; P(social)
0.81) but it appears only 2-3 decisions (1.5-2.5 s) before contact: from T-4 back it equals that of successful
episodes.

**Critical decisions are dominated by timing chaos.**  At labelled critical states (some candidates succeed with the
default continuation, others fail), scoring every candidate's imagined trajectory (interval + 1.2 s continuation)
against the TRUE human future (privileged) does not separate them: within-state AUC 0.475 (predicted clearance
0.494, Stage 1 risk ~0.50), and picking the candidate with the largest true clearance succeeds in 36% (random 40%).
Candidate pairs with near-identical imagined motion (< 6 cm, continuation included) still differ in outcome 15% of
the time (33% for arbitrary pairs).  The counterfactual "one-deviation fixes" (47-57% of native failures) are mostly
other goal-execution variants -- i.e. largely luck that the privileged Oracle can exploit and a legal selector cannot.
The only consistently informative candidate features at these states are progress-type ones (continuation route
gain 0.62, route gain 0.59, goal query 0.60, moving share 0.59).

**What moves SR (72 paired dev episodes):**

| configuration | floor 90 deg | floor 360 deg | progress heuristic 90 deg | progress heuristic 360 deg |
|---|---|---|---|---|
| DWA-native-K10 | 48.6% | 50.0% | 47.2% | 59.7% |
| ORCA-native-K10 | 47.2% | 52.8% | 54.2% | 52.8% |
| SFM-native-K10 | 46.5% | 53.5% | -- | 56% |

360-degree people detection (option `QG_HUMAN_360`: people detected all around with the same occlusion test; map
certification unchanged) cuts the floor's collisions by ~40% relative (ORCA 39% -> 22%) and turns them into
timeouts (14% -> 25%), which are deadlocks (in the last 150 steps the robot moves forward ~15% of the time; ~63% fully
stuck) -- systematic failures a selector can address.  The hand-written safe-greedy heuristic is worse than the floor
(DWA 38.6%).

**Consequences for the method.** (1) With jaw-camera-only perception, a legal selector over this query space is
expected to top out near the floor plus ~10 points; (2) the decisions that matter are systematic deadlock escapes
and exposure (progress), not the timing lottery at the last seconds before contact; (3) labels at collision moments
are mostly noise -- their regression weight is reduced in R6.  Experimental run R6: 360-degree detection + Compiler
v2 (continuation imagination) + warm start from R5 (pending the user's decision on the sensing).

**Further dev tests (72 paired episodes, progress heuristic, 360-degree sensing):** adaptive decision frequency
(6-step intervals while a tracked person is within 3 m) DWA 51.4% / ORCA 55.6% vs fixed intervals 59.7% / 52.8% (more
decisions per episode: ORCA 30 vs 18) -- no consistent gain, not used.  Deadlock categories (44 timeouts, 3
configurations): candidates move but make no route progress 24, no candidate moves (nobody near) 10, a standing person
within 2.5 m 9, no legal route 1 -- the candidate set offers no escape in these states.  R6 (360 + Compiler v2
continuation imagination) cost 3.56 s of Sandbox time per decision (R5: 1.33 s) and was stopped; R7 = 360-degree
sensing + v1b features + warm start from R5, 20 actors, 4 labellers.

**Stall-breaking heuristic (360-degree sensing, dev):** "progress, but when stalled with no progress on offer take the
candidate that moves the most" -- DWA 47.0%, ORCA 52.8%, SFM 50.0% (floor 45.5 / 52.8 / 54.2%, progress 59.1 / 52.8 /
55.6%): fewer timeouts (ORCA 25% -> 19%) but more collisions; naive deadlock escapes do not pay.

**Identifiability at the Oracle's critical decisions (offline, episode-level split; seed 0, 19 held-out critical
states so far, labelling continues):** where the default fails and an alternative succeeds, a critic trained on the
legal consequence graph picks a succeeding candidate in 37% (random alternative 54%, the Oracle's own choice with
the default continuation 58%), AUC 0.52; at mixed states the critic (78%) is below simply keeping the default (87%).

**Assessment (2026-10-29 ~18:00, given to the user):** the gap to the Oracle is mainly an information limit (unseen
people, danger visible only 1.5-2.5 s ahead, timing chaos at critical moments; ~75-80% confidence), with a smaller
fixable design component (the learned policy deviates at ~60% of decisions and breaks as many episodes as it fixes).
Expected Layer 3 ceiling with the current inputs: floor + 10-15 points; 85% SR judged ~5% likely (~10% with
360-degree sensing and query-space escape options).

## 5. Revised direction (2026-10-29 evening, after the user's review)

**User feedback.** The 18:00 assessment over-read the evidence: a critic that is *worse* than random at critical
states is a learning/representation failure, not proof of missing information (an uninformed critic is ~random); the
privileged-clearance AUC 0.48 scored ~2 s of motion against outcomes decided up to 8 s later (horizon mismatch);
near-identical first intervals continue as different queries (the default continuation keeps the candidate's query
running), so their outcomes legitimately differ; RL had not converged, and matching the floor is not acceptable.
Primary benchmark: legal 90-degree jaw camera only (no 360-degree sensing, no privileged future, no hidden geometry);
Oracle information for training supervision only.  The Oracle never needs to look around, so its behaviour carries no
information gathering -- under partial observability the robot must learn when and where to look.  Three directions:
active perception, query-centric features, better training (focus compute on informative decisions).

**Identifiability with query semantics (offline, R5 on-policy + Oracle-state counterfactual labels, 1,794 labelled
states / 186 scenes, scene-level split, 5 seeds, 316 held-out critical states).**  Critic top-1 success at critical
states: v1 graph 31.5%, + label-derived query semantics 37.6%, random alternative 38.1%; within-state AUC 0.53 ->
0.55; mixed states 68% -> 71% (keeping the default 72%).  Up-weighting critical states (25% of batches + ranking)
did not help (32.8%).  Across states the critic does know WHEN the default is failing (AUC 0.80 for the default's
outcome over 454 held-out labelled states), but deviating where it predicts failure gains nothing (51-52% vs 51%):
most of those states are already lost -- only ~15% of labelled states are fixable by one deviation.  Reading: the
decisions that matter come earlier than the labelled window (the last 4-6 decisions before a failure); the policy
must avoid getting into these states (information, exposure, route choice), which only whole-episode RL with an
adequate representation can learn.

**Graph Compiler v3 (`L3_LAYOUT=v3`, `l3/infofeat.py`).**
* Query-centric candidate descriptor from the Query object itself: kind, label family (goal / st_goal / pass / yield /
  sidestep / retreat / alt_route / frontier / turn / rot / scan / ...), the generator's execution-style flags
  (tight, close, vclose, thru, commit, short, unknown_cost, ...), duration, speed, relative heading; the 'continue'
  candidate is described by the RUNNING query.  Goal-execution variants are no longer collapsed into one intent.
* Information state (ego): camera view age per 8 robot-frame sectors x 2 rings (stale free area and mean age), Stage 1
  hidden-person mass per sector (Stage 1's residual slots are a learned prior conditioned on the visibility history),
  lost tracks (count, nearest extrapolated position, age), time since anyone was visible, people seen so far, recent
  sweep coverage.  Pitfall found: the legal map's `last_seen` is the FIRST certification time (static cells are
  certified once), so the view age is recomputed from the robot's own pose history with legal-map occlusion.
* What each candidate would see (cand): yaw sweep left/right, stale free area its imagined camera uncovers (occlusion
  on the legal map), probability over the Stage 1 worlds that a hidden person / a lost tracked person enters its
  imagined frustum with a clear line of sight, hidden-person mass in view at its end pose.
* Compile time 92 -> 120 ms per decision (median, 25 captured decisions); all compiler / ring tests pass under v3.

**Layer 3 scan queries (`env.scan_queries`, `--scan`).**  In-place rotations of 80 degrees to either side (~76 degrees
in one native interval; slower robots continue them) and a rotation toward the most recently lost person when the
generator did not offer one.  The counterfactual labellers rebuild the same candidate sets, so scans are labelled.

**Training changes (`R8b`).**  Prioritised replay of online decisions (priority = success + collision prediction
error, alpha 0.6, IS beta 0.4 -> 1); within-state ranking loss on counterfactual labels; risk-directed exploration
(the default's predicted failure risk sets the Boltzmann temperature x(0.5 + 1.5 r) and the default anchor x(1.5 - r));
episode curriculum (`l3/curriculum.py`: weight 0.25 + 3 s(1-s) from the episode's smoothed success rate across
earlier runs and the run's own outcomes, 0.8 unseen, 0.03 for spawn collisions); partial warm start now zero-inits
new input columns (the v3 network starts exactly as R5).

**Runs.**  R8 = v3 + scans + warm start from R5, legal 90-degree sensing, 7 seen configurations, 18 actors + 2 dev
evaluators + 6 labellers (5.4-7.2 decisions/s); restarted in place as R8b with the training changes after 15 min.

**Graph Compiler v4 (`L3_LAYOUT=v4`) and R9 (19:00).**  The counterfactual label of a candidate is "this query, then
the default policy" -- and the default keeps the candidate's query running -- so the outcome depends on what the
query does AFTER the 16-step interval.  v4 adds each candidate's intention: the running query's remaining executor
route from the imagined end pose (`SandboxConfig.route`, client-side, no extra simulation), else the default
continuation's legal goal route, else standing; it is followed at the candidate's own speed for the rest of Stage 1's
3 s horizon and scored against all 40 worlds (P(< 0.55 m), P(< 1 m), 10% distance quantile, expected first-conflict
time; hidden people separately) plus unknown / stale / clearance / turning exposure along it (11 columns).  On a live
smoke episode the default's route showed P(contact) 0.22 / P(social) 0.88 two decisions before the actual collision
(goal_th and yield routes 0.02), one decision earlier than any v1-v3 feature.  The information features were
vectorised (bit-identical): compile 237 -> 90-130 ms per decision with v4.  Curriculum: episodes whose human tape is
not cached yet are weighted x0.3 (recording a tape costs about one episode: R8b's per-episode setup rose 7 -> 24 s
with the uncertainty-only curriculum), 6 episodes per scene, spawn collisions (collision within 3 steps, never solved)
dropped after one occurrence.  Labellers now label failures over the whole episode (last 2 + 4 earlier decisions,
continuations to the episode end).  R9 = v4 + scans + prioritised replay + ranking + risk-directed exploration +
curriculum, warm start from R8b's network, 18 actors + 2 dev evaluators + 6 labellers.

**Collision context under the legal 90-degree floor (privileged replay, analysis only; `l3/collctx.py`; 60 dev
collisions of ORCA / DWA / SFM native K=10).**  6 spawn collisions (contact at step 0).  Of the other 54: 32 while the
robot was moving -- 84% of the colliding people had been a visible legal track within the last 3 s (94% at some time),
typically ~4 m away 2 s before contact, contact front 13 / side 16 / behind 3; 22 while rotating in place -- 73% seen
within 3 s, ~2 m away 2 s before contact, contact side 12 / front 8 / behind 2; none while standing.  24 of 54 in the
first 80 steps (leaving the spawn).  So ~80% of collisions are with people the robot had seen 2-3 decisions earlier at
2-4 m: decision failures with the information in hand (route / timing choice against seen people's predicted motion),
not an information limit; active perception addresses the remaining ~20% (mostly during in-place rotations).

**Route-safe heuristic (v4 features, no learning; first 81 paired dev episodes):** keeping the default unless its
interval or intention risk exceeds 0.15 and switching to a much safer candidate cuts collisions (38% -> 32%) but
doubles timeouts (13% -> 27%): SR 41% vs the floor's 48%.  It deviates to turns, yields and alternative routes that
are safe but make no progress -- the same safety/progress trap as 360-degree sensing.  Consequence for the learner:
R9's value scalarisation (success 10, collision 5, path 0.3/m) paid +5 for turning a collision into a timeout, which
is the wrong trade with SR first; from 19:25 R9's learner uses collision 2 and path 0.15/m (hot override
`learner_config.json`; the decomposed heads are unchanged, the target policy and the actors' choices follow).

**R9, first hour.**  The greedy dev policy of versions 0-19 deviated from the default at 90% of decisions (single-step
rotations, scans, turns, waits; median claimed gain +0.5 reward units, top quarter +2 to +4 -- maximisation bias over
~13 candidates with new actions and inputs) and was well below the floor (ORCA 29%, DWA 25%, SFM 23%, n = 12-14).
From 19:45 the learner adds a conservative-Q regulariser (`--cql 0.05`: logsumexp over candidates minus the taken
candidate's value, online data).  First on-policy counterfactual labels with whole-episode targets (104 states): at
the 88 states where the default fails, a scan / look candidate saves the episode in 10.3% of cases (21/204), other
alternatives in 6.9% (49/712).

**Prediction vs decision (paired dev, ORCA / DWA native; 46-59 episodes each, 20:05).**  The progress-preserving
route-safe rule (deviate only to a candidate with < half the default's interval + intention contact risk and no loss
of route progress): SR DWA 56.5% / ORCA 42.4% vs floor 56.5% / 45.8%, collisions 28 -> 24% / 39 -> 32%.  The same rule
on the TRUE 3 s futures of the tracked people (analysis-only bank, `l3/oraclebank.py`): 50.0% / 45.8%.  Better
predictions of seen people do not help a one-step rule; safety is converted into stalls.  The missing piece is
multi-step behaviour (wait / yield / pass at the right time while keeping progress) -- the learned policy's job.
Throughput profile (in-process, per decision): imagination of ~12 candidates 420 ms, Stage 1 90 ms, Stage 2 70 ms,
compile 120 ms, generator 45 ms; sensing + range frames ~1 s.  Only 8% of decisions are trivially safe (no person
within 6 m, no hidden mass, default continuing), so decision skipping is not worth it.  From 20:00 R9's learner also
ingests the Oracle solutions replayed under v4 (supervision data, off-policy, 380 episodes being generated).

**What the counterfactual labels can and cannot teach (R9, 159 failure-bound labelled states, whole-episode targets).**
Where the default is heading to failure, one deviation followed by the default saves the episode in only ~5-14% of
cases for any family (early and late decisions alike; social manoeuvres ~0%).  The labels measure Q under the DEFAULT
continuation, which undervalues alternatives whose benefit needs good follow-up decisions -- the multi-step behaviour
we need.  From 20:10 R9 keeps them as lower bounds (hinge) but regresses on them at 0.05 (was 0.25) and ranks at 0.2
(was 0.5); the on-policy returns drive the values.  Final paired dev result of the progress-preserving route-safe rule
(72 episodes per configuration): DWA 48.6% / ORCA 45.8% / SFM 47.2% vs floor 48.6 / 47.2 / 47.2%; collisions 36 -> 35%,
39 -> 31%, 39 -> 36%.

**Gate outlook (2026-10-29 20:15, given to the user).**  Privileged Oracle ceiling on dev with the expert budget
(24 episodes per configuration): ORCA-native-K10 21/24 (88%), DWA-slowturn 21/24 (88%), ORCA-inertia 15/24 (63%).
Estimated confidence: 85% SR on the best configuration (legal 90-degree sensing) ~5% within days, ~10-15% eventually
(85% is ~97% of the privileged ceiling; floor 47-49%; no legal method has yet moved SR by > 5 points); 70% on the
other native configurations ~10-15% each; 70% on modified-dynamics robots < 5% (the privileged search itself reaches
63% on ORCA-inertia); all thirteen non-best settings >= 70% ~1-2%.  A realistic good outcome: best configuration
60-65%, other native 55-60%, momentum robots 25-35% with one shared policy.  What would change this: a learned or
search-based policy beating the floor by 10+ points on dev.

**Correction (user, 20:25).**  The dev "ceiling" above used the expert search with a 1,500-edge budget: FAIL_SEARCH
means the budget ran out, not that the episode is infeasible -- 21/24 (ORCA-native) and 15/24 (ORCA-inertia) are
lower bounds.  With full budgets the privileged Oracle solved 94-100% of feasible people episodes (88-98% counting
proven-infeasible ones); the only known-infeasible dev episodes are the 3 spawn collisions (4%).  85% is therefore
~88-90% of the reachable rate, not ~97%.  The schedule also allows extended exploration before a final 2-3 day
training run.  Revised confidence: 85% on the best configuration ~20%; 70% on the other native configurations ~30%;
70% on modified-dynamics robots ~5-10% (their full-budget ceilings are unmeasured).

**Why choosing 1 of ~10 is hard (R9 labels, 204 failure-bound states).**  Where the default is heading to failure, 31%
(early decisions) / 34% (last two) of the states have at least one alternative that saves the episode (followed by
the default), and those that do typically have ONE (0.7-0.9 of ~10 alternatives).  The one-step rule on the TRUE 3 s
future of the tracked people (final, 72 + 71 paired episodes): ORCA 47.2% vs floor 47.2%, DWA 43.7% vs 47.9%.  So
the needle is real (an exact evaluator applied at every decision would get several chances per failing episode) but
3 s of perfect prediction does not find it; next test: the same rule with the true future of everyone over 9 s.

**Execution loop: interrupting a query had a hidden cost (user's question, 20:30).**  Controlled test on dev (blind,
ORCA + DWA native): the floor's intent, but a FRESH goal query at every decision instead of continuing the running one
-> SR 41.4% vs 48-52% for the floor on the same episodes, timeouts 13% -> 21% (140 episodes).  Each restart rebuilt
the executor: the stall streak and the 'unstick' escalation (tight-map plan following), its kept plan and the
stuck-STOP memory were lost and re-detected every interval.  Our learned policies interrupted the running query at
60-90% of decisions (R9 greedy 65%), paying this tax constantly; every deviating heuristic paid it too, so the earlier
"one-step rules do not help even with perfect prediction" results are confounded.  Fix (`QueryExecution.INHERIT`,
L3_INHERIT=1, default off): a goal query issued while a goal query is running inherits that history, identically in
the branch runtime, official replay, sandbox runner and Sandbox imagination.  With it the same restart control
recovers completely (final, 72 paired episodes per configuration): DWA floor 48.6% / restart 43.1% / restart with
inheritance 48.6%; ORCA 47.2% / 43.1% / 50.0%.  No bug in
the loop mechanics was found (continue / decision-boundary handling is identical in reality, imagination and replay;
intervals end early when a query finishes).  R10 (20:55) = R9's method + the fix, warm start from R9's network.

**First learnable signal for candidate evaluation (v4 features, 21:05).**  Identifiability on R9's on-policy labelled
states (317 states / 61 scenes, 92 critical, 4 seeds, scene split): at failure-bound states the critic's top pick
succeeds 35.0% vs 30.2% for a random alternative (v1-v3 features: at or below random); at mixed states 46.8% vs 31.1%
for keeping the default; gated deviation 22% vs 16% (labels: failure-heavy states, default success 16%).  With the
rollout-policy reference (exact one-step evaluation over the default; analysis only) at 4/4 vs the floor's 2/4 on the
first paired dev episodes, the lever is an accurate per-candidate evaluator.  R10 restarted in place as R10b (21:09):
8 labellers (failures over the whole episode, successes with p 0.5), 16 actors, cf_reg 0.25 / cf_frac 0.25 / cf_rank 0.3.
Route-safe (progress-preserving) re-tested with inheritance: identical to the run without it on ORCA / DWA (it
deviates at ~6% of decisions, mostly to non-goal queries), so its floor-level result was not caused by the reset cost.

**What saves a failing episode (R9 + R10 labels, 119 critical states, 376 fixing vs 1,052 failing alternatives;
standardised mean differences).**  Fixing candidates are more often in-place rotations (rotate kind d = 0.46; scans
27% of fixes vs 13% of failing alternatives), have a later predicted conflict along their intention (r_tcol d = 0.61)
and a larger 10% distance quantile (d = 0.37), reveal more of the map in Stage 2's prediction (m_cells d = 0.69,
m_blk 0.51, m_free 0.39 -- rotations expose new cells), run at full speed (slow variants rarely fix) and are less
often via-point manoeuvres (goto d = -0.33: pass / yield / sidestep / alternative routes).  At the decisive moments,
stopping to look around is often what saves the episode -- active perception -- and the v4 inputs carry it.  The
Stage 2 map-gain columns carry signal here, so dropping Stage 2 for speed is not advisable.
Resume inheritance (21:25): a goal query interrupted by non-goal queries (scans, waits, rotations) is remembered and its
history resumes when a goal query is issued again; the Sandbox imagination receives the running execution in any
status so imagined goal candidates inherit exactly as reality does.  Official replay with L3_INHERIT=1 reproduces the
branch runtime's actions exactly (2/2 smoke episodes).  R10 restarted in place with it (21:26).

**Rollout-policy reference, first 8 paired dev episodes (ORCA-native, analysis only):** exact one-step evaluation over
the default succeeds in 7/8 vs the floor's 4/8 on the same episodes (12 episodes: 10/12 vs 7/12; 18 episodes: 15/18 vs
9/18 -- all three failures timeouts, NO collisions: perfect one-step evaluation removes collisions entirely here); its deviations are almost all goal-execution
variants (goal_th 5, st_goal_tc 5, st_goal 4, goal 3, st_goal_th 3, goal_tc 2, turn_left 2).  The right one or two
choices from the pool do exist; the hard part is evaluating near-identical variants legally -- the label critic's
job.  Labellers now skip the families that almost never save an episode (pass / yield / sidestep / frontier /
alternative routes / retreat / approach: placeholders keep candidate indices), roughly halving the cost per state.

## 6. Night of 2026-10-29: approaches to a legal evaluator (the gate depends on it)

Rollout-policy reference (exact one-step evaluation over the default, analysis only): 13/15 vs floor 9/15 on paired
dev ORCA episodes -- choosing well among the ~10 queries suffices; the open problem is a LEGAL evaluator.  R10's RL
learner at checkpoint v100 (paired dev, ~100 of 216 episodes): ORCA 47.1% vs 44.1%, SFM 31.4% vs 40.0% -- not above the
floor.  Three attacks run in parallel overnight, each judged on the same paired dev episodes:
1. Label critic (approximate policy iteration, `l3/apicritic.py`): R10 restarted as a label factory (6 actors, 18
   labellers, rarely-saving families skipped); v0 (587 labelled states, fine-tuned from R10's network, best holdout
   at step 200: 1.37 -> 1.14) is evaluated as a pipeline check; v1 / v2 follow with ~3k / ~6k labels.
2. Legal imagined rollout (`heuristics.imagined_rollout_index`, selector `imagroll`): every candidate's imagined
   interval + imagined default continuation (104 steps, executor and planner in the loop against the nominal Stage 1
   people) scored against the 40 Stage 1 worlds (constant velocity beyond 3 s) for contact and against the legal map
   for progress -- the closest legal copy of the exact reference, no learning.  Sandbox time per decision grows to
   0.3-8 s (evaluation only).
3. RL (R10's learner) continues on the same labels; milestone checks.

**Milestone: R10 checkpoint v100, paired dev (72 episodes per configuration, with the loop fix).**  ORCA 55.6% vs
floor 47.2% (SPL 0.442 vs 0.351, collisions 26.4% vs 38.9%); DWA 50.0% vs 48.6%; SFM 45.8% vs 47.2%.  Discordant
pairs: ORCA 10 policy-only vs 4 floor-only successes (sign test p ~ 0.18), DWA 10 vs 9, SFM 6 vs 7 -- the first learned
shared policy with a gain on the main official configuration, suggestive but not significant yet.

**Metric audit (user question, 22:30).**  SR (STOP within 0.2 m) and human collision (official physics-contact measure,
episode ends) are the official definitions in the branch runtime, and official replays verify identical actions and
success.  PSC was NOT: the benchmark config composes `psc` with `use_geo_distance: True` (navmesh geodesic robot-human
distance < 1 m, 3D Euclidean when no path), while the branch runtime counted planar Euclidean centre distance --
stricter, so every PSC reported before 22:40 is biased LOW.  SPL used the official start geodesic but a planar path
length (official: 3D Euclidean steps).  Fixed (`BranchState.psc_official_steps`, `path_length_3d`; L3 metrics `psc` /
`spl` now official, `psc_euclid` / `spl_planar` kept).  Verified against official replays (unchanged `Env.step`, Falcon's
measures) on 10 dev episodes: actions identical 10/10; success, SPL and the corrected PSC equal the official values in
10/10; the old Euclidean PSC matched in 9/10 (0.8686 vs official 0.8723 once) -- earlier PSC values were biased low
only slightly.  Numbers of record remain the official replays.

**Label critic v0 (API step on 516 labelled states, fine-tuned from R10's network; paired dev, 72 per configuration).**
DWA 55.6% vs floor 48.6% (12 wins / 7 losses, p = 0.36), ORCA 51.4% vs 47.2% (11 / 8, p = 0.65); pooled 23 / 15
(p = 0.26).  R10 v100 pooled over three configurations 26 / 20 (p = 0.46).  Both learned policies sit modestly above the
floor (+2 to +8 points), not yet significant, far from the gate.  The rollout reference's remaining failures are
deadlocks: in its 3 timeouts, NO candidate followed by the default succeeds at any of 32-35 decisions -- one-step
improvement cannot fix them; they need multi-step escapes (or are infeasible).
Legal imagined rollout (`imagroll`, partial paired dev 120 episodes): ORCA 48.6% vs floor 47.1% (7 / 6), DWA 50.0% vs
56.0% (4 / 7) -- no gain: imagining 6 s of default continuation against Stage 1 (constant velocity beyond 3 s) does
not reproduce the exact reference's choices.  Direction dropped.

**Rollout reference complete (24 paired dev ORCA episodes, analysis only):** 21/24 (88%) vs floor 14/24 (58%); its 3
failures are deadlock timeouts (no single deviation + default succeeds anywhere in them); the floor's 10 failures on
the same episodes were 8 collisions + 2 timeouts.  Its deviations are mostly the contact-tolerant ('_th') and
space-time goal variants: st_goal_th 15, goal_th 15, st_goal_tc 12, goal 6, st_goal 5.  Test launched: fixed policies
that always run st_goal_th / goal_th instead of the plain goal query (blind, ORCA + DWA dev).
Fixed variant defaults (blind, ~60 paired dev episodes each): always st_goal_th 7 wins / 13 losses vs the floor, always
goal_th 5 / 10 -- worse; the exact evaluator's use of these variants is situational, not a better default.
R10 v125 (paired dev, 72 per configuration): ORCA 52.8% vs 47.2% (10 / 6), DWA 44.4% vs 48.6% (6 / 9), SFM 48.6% vs
47.2% (8 / 7); pooled 24 / 22 (p = 0.88).  Across v100 and v125 the ORCA gain recurs (10 / 4, 10 / 6), DWA / SFM are flat.
Imagined rollout final ~floor (pooled 14 / 13).  Deadlock probe (privileged, floor timeouts): mid-episode (step 240)
1-4 single and ~40 two-step escapes exist (explicit turns toward the route, a short wait, st_goal_th); none late (step
384) -- deadlocks are escapable if acted on by mid-episode.

**Learning curve of the label critic (no warm start -- warm starting from R10's learner leaks the held-out labels, an
invalid 82% was discarded; scene split, 4 seeds, critical states of R9 + R10 labels).**  Training states 250 / 500 /
1,000: critic top-1 at critical states 35.2% / 32.1% / 33.8% (random alternative ~29.5%), within-state AUC 0.565 /
0.566 / 0.569, default-outcome AUC 0.61.  Flat: more labels of this kind, with the legal v4 inputs and this critic, do
not improve candidate evaluation.  Next diagnostic: the same critic and labels with the TRUE future of every person in
the inputs (`l3/privfeat.py`, analysis only) -- predictable with perfect foresight (a prediction problem) or not
(outcomes not determined by anything knowable at decision time)?
Timing of labelled outcomes (R9 + R10, 10,287 colliding / 3,889 successful candidate continuations): collisions occur a
median 40 steps (1.9 s) after the decision -- 65% within Stage 1's 3 s horizon, 86% within 6 s; successes take a median
146 steps.  Most outcomes are decided within the prediction horizon, yet a one-step rule with the TRUE 3 s future did
not pick fixes: at critical states hit vs miss is decided by fine timing (how the robot's planner reacts to the people
it actually perceives), which the imagination -- planner reacting to the NOMINAL Stage 1 world -- only approximates.

**R10 latest (v175) after the label-factory period, partial paired dev (107 episodes): significantly WORSE than the
floor** (5 wins / 15 losses, p = 0.04; ORCA 41.7% vs 47.2%, SFM 31.4% vs 45.7%).  Heavy regression of the policy's
values onto default-continuation labels (cf_reg 0.25 / cf_frac 0.25 / rank 0.3) with few actors damaged the greedy
policy; with the flat learning curve, the label-factory direction is stopped.  R10 restarted in RL mode (01:21) from
checkpoint v100 (the best so far) with the light counterfactual supervision v100 was trained with.

**Decisive diagnostic: privileged critic (02:00).**  Same critic architecture and the same kind of counterfactual labels,
on the same recompiled episodes, with legal inputs vs the TRUE future of every person (analysis-only bank, route /
pair / interaction features against true positions over 9 s): critic top-1 at critical states legal 29.6% vs random
29.7% (261 states, 77 critical), privileged 30.9% vs random 32.4% (342 states, 110 critical); within-state AUC 0.55 vs
0.54; default-outcome AUC 0.61 vs 0.63.  Perfect knowledge of the people's futures does NOT make the labels predictable
from compiled consequence features.  The exact reference's +30 points come from exactly simulating the robot's planner
reacting to the people step by step (hit vs miss at fine timing), which the compiled features -- a 16-step imagined
interval against a nominal world plus a route-following proxy -- do not capture.  Consequence: a longer-horizon people
predictor would not help; what could is high-fidelity imagination of each candidate's execution with the planner
reacting to each plausible future -- the multi-world rollout (`mwroll`) now under test.
Partial (02:05): label critic v1 (3-member ensemble on R9 + R10 labels, built on R10's late network) 11 wins / 17
losses vs the floor (122 paired episodes); legal multi-world imagined rollout (4 worlds, 48-step continuation; the
planner reacting to each sampled world) ORCA 3 / 4 (35 episodes).  Running: the same multi-world rollout with the
TRUE future (analysis only; 1 world = truth, 100-step continuation) -- does high-fidelity imagination with perfect
prediction reproduce the exact reference?
True-future multi-world rollout (analysis only; 1 world = truth, 100-step imagined continuation; 35 paired ORCA dev
episodes): SR 54.3% vs floor 54.3% (3 / 3) -- collisions 20% (floor ~36-39%) but timeouts 26%: the same safety /
progress trap; a ~5 s look-ahead rates "wait safely" well, while the exact reference simulates to the episode end and
sees the resulting timeout.  Legal multi-world rollout stopped at 54 episodes (5 / 7).  Running: true-future
imagination over 300 steps (~to the episode end) on 36 ORCA dev episodes -- if it reproduces the exact reference, the
legal path is a world model whose people prediction reaches the whole episode, used through imagination.
Label critic v1 (paired dev, 205 episodes): 23 wins / 27 losses vs the floor (ORCA 48.6 vs 47.2%, DWA 49.2 vs 50.8%,
SFM 41.7 vs 47.2%) -- floor level, as the flat learning curve predicted.
Full-horizon true-future imagination (analysis only; 300-step continuation, 19 of 36 paired ORCA dev episodes so far):
SR 63.2% vs floor 57.9% (2 / 1); outcomes 12 success / 2 timeout / 5 collision; deviations mostly wait_short, st_goal,
goal_th, st_goal_th.  The timeout trap disappears with a full horizon, but candidates imagined safe -- against the TRUE
future -- still collide in reality in 5 / 19 episodes (the exact reference: none): the imagined closed loop diverges
from the real one at the timing that decides hits.  In imagination the planner perceives every tracked person's
future perfectly; in reality it perceives people through the 90-degree camera (occlusion, track loss) and reacts
differently.  Reproducing the exact evaluator would require simulating the robot's own perception inside the loop.

## 7. Early morning 2026-10-30: closed-loop fidelity

Completed overnight (paired dev vs the floor):
- Label critic v1 (3-member ensemble, ~2.2k labels), final: DWA 47.9% vs 47.9% (10 / 10), ORCA 48.6% vs 47.2% (8 / 7),
  SFM 41.7% vs 47.2% (7 / 11); pooled 25 / 28 -- floor level.
- Legal multi-world imagined rollout (6 Stage 1 worlds, 60-step continuation), stopped at 54 ORCA episodes: 44.4% vs
  48.1% (5 / 7).
- True-future multi-world rollout, 100-step continuation (analysis only), 63 ORCA episodes: 49.2% vs 50.8% (5 / 6).
- True-future rollout over 300 steps (analysis only), 29 of 36 ORCA episodes: 50.0% vs 50.0% (3 / 3); outcomes 14
  success / 9 timeout / 6 collision.  The early lead (63% at 19 episodes) came from the short episodes finishing first.
- Scan rollout reference (exact one-step evaluation incl. the scan queries), 14 of 24 episodes: 12 / 14, identical to the
  plain reference on the same episodes (scans are chosen 5 times, rotations 3).  Active looking adds no successes
  to exact one-step evaluation.
- Deadlock probe, complete (6 floor timeouts, ORCA + DWA, 5 distinct episodes): only ONE episode (in both
  configurations) has escapes at mid-episode (step 240: 3-4 single deviations, ~40 two-step sequences); the other
  four have none at step 240 or step 384.  Correction to section 6: most floor deadlocks are not escapable by 1-2 query
  choices from mid-episode on.
- Collision context of the floor (54 collisions, 3 native configurations): 22 happen while the robot turns in place
  (median 84 deg in the last second), 61% of contacts are outside the camera's +-45 deg; the colliding person was seen
  within 1 s in 61%, within 3 s in 80%, ever in 87%.
