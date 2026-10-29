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
