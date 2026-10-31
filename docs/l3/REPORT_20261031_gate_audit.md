# Social navigation on the Falcon benchmark under strictly legal observations: audit and experiment report

Date: 31 Oct 2026.  This document is self-contained.  It is written for a reader with no prior context (for example,
another AI model asked to brainstorm).  It covers the problem, the constraints, the system, every experiment of the
last days with numbers, what we concluded, and the questions on which we want advice (section 12).

---

## 1. The task and the benchmark

* **Benchmark:** Falcon social navigation (Habitat 3.0), datasets **Social-HM3D** and Social-MP3D.
* **What the robot must do:** reach a PointGoal in a photo-realistic indoor scene while simulated humans walk around.
* **Success:** the robot issues STOP within 0.2 m (geodesic) of the goal, with **no physical contact with any human**,
  within **500 steps**.
* **Metrics:** SR (success rate, primary), SPL, PSC (personal space compliance), H-Coll (episodes with a human
  collision).
* **Humans:**
  * They follow recorded trajectories (a deterministic "tape" per episode).
  * They **do not react to the robot**.
  * They walk stop-and-go: bursts of about 1.7 m/s with many short pauses (median about 0.4 s, about 6 pauses per
    person), often turning when they restart.
  * After 100-400 steps they stop for good.
  * Typically 2-6 humans per episode.
* **Official test split:** the official val split.  We have **never run it**; it is reserved for a final freeze.
* **Model selection uses TRAIN episodes:**
  * "tuning set": 112 TRAIN episodes, used for screening;
  * "tuning set B": 112 other TRAIN episodes in 112 scenes, built today;
  * "held-out set": 143 TRAIN episodes, never probed and never trained on, used for selection.
  * Episodes on the tuning set run about 3-5 SR points easier than on the held-out set.

### Published results on Social-HM3D (SR / SPL / PSC / H-Coll)

| Method | SR | SPL | PSC | H-Coll | Notes |
|---|---|---|---|---|---|
| A* | 44.81 | 43.99 | 90.38 | 54.80 | |
| ORCA | 37.44 | 32.91 | 92.23 | 39.77 | |
| Falcon (RL, depth) | 56.26 | 52.05 | 89.76 | 41.22 | the benchmark's own method |
| NavThinker (world model) | 59.46 | 55.00 | 89.91 | 39.09 | best result without privileged sensing |
| HUMA (VLM + RL) | 62.07 | 54.54 | 92.32 | 34.48 | see the two caveats below |

HUMA caveats:
* It observes the **true distance to the closest human** (omnidirectional, through walls).
* Its headline number **freezes the environment while the VLM thinks**.  Its own "realistic timing" ablation gives
  **56.90 SR**.

## 2. Binding constraints (set by the project owner)

1. **Legal observations only at decision time.**  Allowed:
   * the robot's 90-degree forward ("jaw") depth camera;
   * people detections inside that camera frustum with line of sight.  These are idealised detections; a track is
     dropped 2 s after the last detection, and velocity comes from consecutive detections;
   * a legal occupancy map built from the camera;
   * odometry;
   * the goal vector.

   Not allowed: privileged future information, hidden geometry, or unrestricted 360-degree sensing.  Privileged
   information (the human tape, true futures, 360-degree sensing) may be used **only as training labels or for
   analysis**.
2. **One shared policy** for heterogeneous robots.  Settings are planner x dynamics:
   * native robot with the ORCA, DWA or SFM planner;
   * slowturn: half the turn rate;
   * quick_inertia: 12 speed levels, accelerates and brakes by 6 levels per step, must stop to turn.
3. **It is a world-model paper.**  The core is a closed-loop "world-model Sandbox" that imagines candidate futures.
4. **GATE (a must):** at least **70% SR in the best robot setting and at least 60% in every other setting** on the
   official benchmark, under the legal rule.  The project owner has explicitly refused weakening it: no extra sensing,
   no dropped settings, no swapped planners.

## 3. The robot

* **Native action set:** forward 0.208 m per step (= 4.34 m/s), turn 4.77 degrees per step (90 degrees = 19 steps =
  0.9 s), STOP.  One step = 0.048 s.
* **The robot cannot stand still.**  A "hold" is implemented as alternating left/right turns.
* **Body:** a 1.23 m x 0.48 m rectangle; the front corners sweep out to about 0.69 m when turning in place.
* **Speed and turning:** in translation the robot is 2.5 times faster than a walking human; turning in place is slow.

## 4. The current system

* **Query generator.**  Every 16 steps (0.77 s) it proposes K = 10 candidate "queries", for example:
  * goal-following;
  * a space-time-planned goal path (`st_goal`);
  * close-pass variants with smaller human margins (`_tc` = 0.27 m, `_th` = 0.20 m contact level);
  * rotations, waits, wait-until-clear, creeps, sidesteps.
* **Executor.**  Runs the chosen query with a classic planner (ORCA, DWA or SFM) on top of a body-aware lattice plan or
  a 40-step space-time A* plan.  Shared safety layers apply.
* **World-model Sandbox.**  For every candidate it imagines the robot's motion with the exact robot model against
  predicted people.  The legal constant-velocity "bank" of tracked people works best; learned Stage-1 people worlds
  and a map-aware forecaster exist but were not better.  The Sandbox also models a camera, so imagined people are only
  "seen" when inside the imagined view.
* **Layer-3 ranker.**  A LightGBM model scores each candidate from about 180 per-candidate features (imagined outcomes,
  progress, clearances, map and hazard context).
  * It is trained on **exact outcome labels**: every candidate of a probed decision is executed for real in the
    simulator with privileged replay, over 64 steps.
  * It deviates from the default ("goal") only when the predicted gain exceeds a tuned threshold.
  * Current model: `model_multi2`, about 8,400 labelled states across all five settings.
* **Privileged "Oracle".**  It picks among the same candidates using exact futures (perfect foresight).  It reaches
  **86-89% SR** on dev with the same planners, so **the action space is sufficient**.

## 5. Results before today (held-out set, 143 episodes; SR / collision / timeout %)

| Setting | Floor (default policy) | Shared ranker v1 (4.3k states) | Ranker v2 `model_multi2` (8.4k states) |
|---|---|---|---|
| ORCA | 40.6 / 45.5 / 13.3 | 49.7 / 42.7 / 7.0 | 52.4 / 44.1 / 3.5 |
| DWA | 42.0 / 44.8 / 12.6 | 51.7 / 42.0 / 6.3 | 54.7 / 40.9 / 4.4 (137 episodes) |
| SFM | 41.3 / 46.9 / 11.2 | 50.3 / 44.1 / 4.9 | 52.4 / 43.4 / 4.2 |
| slowturn | 31.5 / 46.2 / 22.4 | 43.4 / 46.2 / 10.5 | 42.0 / 49.0 / 9.1 |
| quick_inertia | 31.5 / 44.1 / 24.5 | 35.7 / 46.2 / 18.2 | 39.7 / 45.6 / 14.7 (136 episodes) |

* **World-model ablation:** the same ranker without Sandbox features gains only -2 to +3 over the floor.  On DWA it
  loses 9 vs 26 episodes against the full ranker (p = 0.006).
* **Data scaling has flattened:** going from 4.3k to 8.4k states added +1.4 SR on average, not significant.
* **The ranker's gains are almost all from fewer timeouts.**  Collisions are unchanged at about 41-49%.
* **The ranker cannot predict contacts:** its ability to rank candidates by contact risk within one decision is at
  chance (AUC 0.50-0.58).

## 6. Today's executor work (all default-off options; legal at decision time)

| Option | What it does |
|---|---|
| `L3_EVADE2` | Plan-aware 1.5 s escape reflex.  Every step it rolls out the robot's committed motion (this primitive, then the plan, then holding) against predicted people.  If the robot rectangle would come within 0.45 m, it picks the best of about 60 native sequences (hold, advance n, turn k then advance, hold then resume) by capped clearance, then progress.  Standing people count only within 0.20 m. |
| `L3_EVADE_REAL` | Runs the reflex in the real robot only, not inside the Sandbox's imagination, so the ranker's learned features are unchanged. |
| `L3_EVADE_CAP=0.10` | More progress-seeking choice among safe escapes. |
| `L3_BUMP_ALL` | Momentum robots record "bump memory" (unseen obstacles) like the native robot.  This fixed quick_inertia pushing into obstacles for hundreds of steps. |
| `L3_UNREACH_FALLBACK` | An unreachable goal query retries on a relaxed map instead of failing every step. |
| Others | Turn guard, track memory, restart hypotheses, hold sweep and gaze (active looking), corner caution, plan commitment, a STOP guard for off-mesh goals, near-contact reflex fixes, route-fix probes, setting-specific fixes.  Results in section 8. |
| Analysis-only | `L3_EVADE_TRUTH=all` or `tracked`: the reflex sees the true future of everyone, or of tracked people.  These are diagnostics, never a result. |

**Infrastructure:** ranker evaluation and data probes now run on CPU only, with actions identical to the GPU-served
runs.

### Best legal system today = ranker `model_multi2` + bundle F1

F1 = reflex in the real robot only + CAP 0.10 + bump-all + unreachable fallback.

| Setting | Tuning set (112) | **Held-out (143)** | Held-out wins / losses vs ranker alone |
|---|---|---|---|
| ORCA | 55.9 | **53.8** | 11 / 9 |
| DWA | 53.7 | **56.2** (137 episodes) | 8 / 6 |
| SFM | 55.4 | **52.4** | 6 / 6 |
| slowturn | 49.1 | **43.4** | 15 / 13 |
| quick_inertia | 57.0 | **46.3** (136 episodes) | 15 / 6 (p = 0.08) |

So the system is about **52-56% on native, 43-46% on slowturn and quick_inertia**.  The gate needs 70 / 60.

## 7. Privileged diagnostics: what information would buy (tuning set, ORCA, ranker + reflex)

| Stack | SR | Collision | Timeout |
|---|---|---|---|
| Ranker only | 52.7 | 39.3 | 8.0 |
| Ranker + legal reflex (real robot only) | 54.5 | 36.6 | 8.9 |
| + true futures of **tracked** people only (analysis) | 57.1 | 32.1 | 10.7 |
| + **360-degree** people sensing (analysis) | 61.0 | 27.6 | 11.4 |
| + true futures of **everyone** (analysis, "ceiling stack") | 65.2-67.0 | 18.8 | 14.3 |
| slowturn ceiling stack (analysis) | 58.6 | 18.9 | 22.5 |

Floor-level versions show the same pattern: floor + true-future reflex 58.9 (collisions 12.5, timeouts 28.6), against
the floor's 45.5.  Even with perfect knowledge of every person's next 1.5 s, this architecture reaches only about 67% on
ORCA and 59% on slowturn.

## 8. Levers tried today (tuning set, 112 episodes)

One 112-episode paired run has noise of about **±4.5 SR points**: any change that perturbs every episode flips 23-26
outcomes.

| Lever | Result |
|---|---|
| Reflex (legal), floor | +3.6 (collisions -9, timeouts +5) |
| Reflex everywhere incl. imagination, + ranker | -1.8 (the ranker's features shift) |
| Reflex in the real robot only, + ranker | +1.8 |
| Bump-all + unreachable fallback | quick_inertia +14 on tuning; others ~0 |
| Turn guard + 0.40 m margin for moving people | +0.9 |
| Map-aware learned people forecaster in the reflex | ~0 |
| Learned threat model: LightGBM on legal features, privileged labels (true clearance of each candidate) | no better than constant-velocity clearance on held-out scenes; closed loop ~0 |
| Track memory (people dropped by the tracker) | 0 (identical outcomes) |
| Restart hypotheses for paused walkers | 0 of 29 collisions converted, 4 successes lost |
| Hold camera sweep / gaze (active looking while holding) | 0 (the ranker stack rarely holds) |
| Corner caution | negative (and buggy: see the pitfall below) |
| Masking contact-level close-pass queries | ~0 |
| Space-time plan commitment (less replanning) | legal -5 (7 wins / 13 losses); ceiling flat |
| Near-contact reflex fixes (own-motion slide model, rotation model, contact-range choice) | legal ~0 (3 / 5); ceiling ~0 at 100 episodes |
| Route fixes (glance toward goal, thru-map probe), smoke test on 8 episodes | 0 of 3 target timeouts converted; 1 success lost |
| Setting-specific fixes for quick_inertia / slowturn | ~0 or slightly negative |

**Pitfall found:** the legal map's `last_seen` field is the **first**-seen time, because a free cell is certified only
once.  So there is no "how recently did I look there" signal today.  The corner-caution test and the staleness features
of the learned threat model read it wrongly.

## 9. Analyses (five parallel analysts, exact replays of every relevant episode)

1. **Spawn collisions are unwinnable.**  About 7% of episodes (8 of 112) collide at step 1 because a person spawns
   inside the robot's footprint; every first action collides.  So the cap is 93% for any policy.
2. **Collisions of the best legal stack** (41 per 112 episodes), by what the robot knew 1 s before contact:

   | Category | Count |
   |---|---|
   | spawn | 10 |
   | never seen | 13 |
   | track lost 1-3 s earlier | 4 |
   | tracked but out of view | 8 |
   | tracked and visible | 6 |

   * The robot was **turning in place in 26 of 31 non-spawn contacts**.
   * Contacts were on its **side in 22 of 31**.
   * The robot stayed within 1 m of the person for a median of 0.5-1.4 s before contact.
3. **Information is the limit at the decision level.**  59% of the people who come within 2 m in the next 3 s are
   untracked or never seen at decision time.  Even true futures of tracked people catch only 35% of contacts in an
   offline decision benchmark; true futures of everyone catch 93%.
4. **Encounters vs contact probability.**
   * About 1.4 "encounters" (a person within 1 m) per episode in every stack.
   * Contact probability per encounter: 0.20 legal vs 0.05 with true futures.
   * For 70% SR the contact probability per encounter must drop to about 0.12.
5. **Exposure and time.**
   * Successful episodes take 1.85 times the geometric minimum time; half of the turning is required by the geometry.
   * The excess is mostly turning that is later undone, coming from space-time replanning and reflex escapes.
   * But turning is not over-represented in contacts per step; holding near people is (confounded).
6. **Timeouts of the best legal stack** (10 per 112):
   * wrong legal route, 5: the conservative map closes the real exit, or the route takes a shortcut through unknown
     space that does not exist;
   * standing people block the route, 2;
   * off-mesh goal with no STOP, 1;
   * slow, 1;
   * impossible, 1.
7. **Even the true-future reflex collides**, in 12 of 112 episodes after spawn.  Causes:
   * the reflex mispredicts its own motion when sliding along walls, 7;
   * the standing-person margin is too small, 6-7;
   * a rotate query is modelled as one turn, 3;
   * escape choice when already very close, 6.

   Fixing these converted 3 to success and 5 to standstill timeouts next to stopped people.
8. **slowturn and quick_inertia** trail native by 9-10 points, mostly on long episodes (12 m or more).
   * slowturn turns as many degrees as native at half the rate.
   * quick_inertia's extra timeouts were a bug (fixed by bump-all).
   * Their 60% needs native at about 62-67% with legal inputs, which is about the native ceiling.

## 10. Our current conclusion on the bottleneck

* The bottleneck is **observability of people, amplified by the robot's embodiment**:
  * the people who cause contacts are mostly outside the 90-degree view;
  * the people we do see are unpredictable at 1-3 s because they walk stop-and-go;
  * the robot is most vulnerable exactly when blind: slow in-place turns, and holds that are wiggles.
* The action space and the planners are not the limit (the Oracle reaches 86-89% with them).
* Data volume is not the limit (scaling has flattened).
* Every legal way we found to add information gives about 0.
* Under the legal rule the gate looks about **1-3%** likely with this architecture.  The most likely held-out end
  state: native 55-60, slowturn and quick_inertia 48-53.

## 11. What is running and what is planned

* **Running:**
  * an on-policy re-probe (exact labels of every candidate while the current ranker drives, 960 new TRAIN episodes,
    all settings) under the new executor, then a ranker refit, a tuning evaluation and a held-out evaluation.
    Expected +0-3;
  * a design study of a from-scratch redesign.
* **Plan (with kill criteria):**
  * standstill escape next to stopped people;
  * route fixes;
  * learned "look" candidates for the ranker (the only untested strong form of active perception);
  * a second on-policy round.
* **Redesign candidate we are considering:**
  * Replace the 16-step query ranker and the hand-built reflex with **one learned, recurrent, step-level policy**,
    trained by **privileged-teacher distillation**.
  * Teacher: privileged state, trained by RL and/or imitating the exact reference.
  * Student: legal inputs (map crop, a correct visibility memory, tracks with history, own action history, goal),
    trained by DAgger, then RL fine-tuning.
  * Auxiliary world-model targets: predict people occupancy, including unseen regions.
  * Goal: learn active perception and exposure-aware motion end-to-end.
  * Honest expectation: 60-65% native; still short of 70/60.

## 12. Questions we would like advice on

1. **Is 70% (best setting) / 60% (others) achievable at all** under these legal observations, given that about 7% of
   episodes are unwinnable at spawn and that perfect knowledge of everyone's 1.5 s future gives only about 67% in our
   architecture?  Is there a reason the privileged Oracle's 86-89% could be approached without that knowledge?
2. **Unseen people:** how can a robot with a 90-degree camera reduce collisions with people it has never seen, who
   walk into its side while it turns in place?
   * Our active-looking heuristics (gaze, sweep, look at the least recently seen sector) gave 0.
   * Are there better strategies: belief-space planning over unseen people, learned priors of where people come from,
     "turn only where it is safe", motion patterns that avoid exposure?
3. **Stop-and-go humans:** what prediction approaches work for pedestrians who pause and restart unpredictably?
   Constant velocity, a map-aware learned forecaster and restart hypotheses all gave about 0 in closed loop.
4. **Architecture:** would an end-to-end learned step-level policy with privileged-teacher distillation plausibly beat a
   modular query-ranking system here?  What observation design would you use (raw depth history vs tracks + map)?
   What training tricks matter most (asymmetric critic, DAgger, auxiliary prediction, curriculum)?
5. **Decision frequency:** our ranker decides every 0.77 s while contacts unfold in 0.5-1 s.  How would you combine
   long-horizon route choice with step-level reactions?  Our reflex helped little with legal predictions.
6. **Evaluation noise:** with about ±4.5 SR per 112-episode run, how would you screen many small levers efficiently?
   We now pool 224 episodes and use exact affected-episode replays.
7. **Anything we are missing:** benchmark-specific properties of Falcon's humans or scenes that a legal policy could
   exploit, for example deterministic tapes, humans stopping for good after 100-400 steps, or the PSC definition.  Or
   alternative strategies (timing, waiting positions, routes) that could cut contacts with unseen people.

## Appendix A: key definitions

* **Floor:** the default policy, which always issues the goal query.
* **Ceiling stack:** the same system, with the reflex given true futures (analysis only).
* **Exact labels:** for a probed decision, every candidate query is executed for real (privileged replay) for its
  interval plus the default continuation, up to 64 steps.  Each label is the real outcome.
* **Tuning set / held-out set:** disjoint TRAIN episode lists.  The official val split is never touched before a
  freeze.

## Appendix B: where things are (repository `HeHaodong2004/Falcon`, branch `research/l3-shared-policy-20261029`)

* `eventwm_graph/querygen/executor.py`: executor and all `L3_*` executor options (comments explain each).
* `eventwm_graph/l3/riskrank.py`, `riskfit.py`: ranker features and fit.  `eventwm_graph/l3/reflex.py`: learned
  threat-model attempt.
* `eventwm_graph/sandbox/imagine.py`: world-model Sandbox imagination.
* `docs/l3/REPORT_20261031_overnight.md`: the earlier overnight report.
