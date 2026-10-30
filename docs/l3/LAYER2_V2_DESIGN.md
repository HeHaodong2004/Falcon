# Layer 2 v2: what a good Layer 2 is for the closed-loop Sandbox (design, 2026-10-30)

Status: design, not built.  It comes from a five-angle design panel with two judges, a synthesis and two adversarial
reviews (raw output: `output/design_panel/layer2_panel_20261030.json`), corrected against the episode logs.  All
runtime inputs are legal.  Privileged information (the recorded people tape, the exact branch simulator) is used only
for labels, evaluation and the analysis twins.  The released Layer 2, Oracle, Query Generator and Sandbox are
unchanged; everything below is a default-off option.

## 1. Where the failures are (ORCA dev, 72 paired episodes, 1.5 s window, outcome rule)

| arm | SR | collisions | timeouts |
|---|---|---|---|
| floor (default policy) | 47.2% | 28 | 10 |
| Sandbox, legal constant-velocity tracks, no hidden people ("CV-null", `S_cv32`) | 54.2% | 19 | 14 |
| same Sandbox, TRUE future of every person (`S_true32f`) | 59.7% | 18 | 11 |
| exact branch simulator, same window (`rp_short/cap32_field`) | 76.4% | 6 | 11 |
| exact branch simulator, 3.1 s window (`cap64_field`, 43 episodes so far) | 86.0% | 2 | 4 |

* Perfect people knowledge does not reduce collisions inside today's Sandbox.  CV-null and TRUE share 13 collision
  episodes; TRUE fixes 5 and creates 4.  Its +5.5 points are timeouts (14 -> 11).
* With the same TRUE people, the exact simulator turns 13 of those 18 collisions into successes (2 become timeouts,
  3 remain).  At 1.5 s the collision gap is therefore the Sandbox's own robot loop (map discovery, blocked steps,
  tracker, contact geometry) and the selection rule.  It is not the people model.  Attribution of the 18 is running
  (`output/eval/A_true32f_attr`, section 9).
* Other arms on the same episodes (none differs significantly from CV-null; sign tests p = 0.25-0.73):

  | arm | SR | collisions | timeouts |
  |---|---|---|---|
  | 6 Stage 1 worlds, legal | 50.0% | 21 | 13 |
  | CV + 5 s lost-track memory in the loop | 50.0% | 22 | 14 |
  | TRUE tracked people only (see note) | 51.4% | 25 | 10 |
  | TRUE + floor-run final map in the loop | 52.8% | 24 | 10 |
  | TRUE + legal Stage 2 map loop (44 episodes) | 50.0% | 10 | 12 |

  Note on the tracked-only arm: it is confounded.  `OracleBank` matches tracks by last-seen position within 0.75 m,
  so people lost more than ~0.6 s ago vanish from it.

### 1.1 Attribution of the 18 TRUE-people imagination collisions (`output/shortprobe/attr_true32f`)

Each collision episode of `S_true32f` was replayed with its recorded choices (`L3_PROBE_ACT=replay`).  At each of its
last 10 decisions, every candidate was executed exactly (candidate, then default, 64 steps) and the selector was
re-run (its choices reproduce the recording).

| class | episodes |
|---|---|
| false safe: the imagination saw no contact for the chosen candidate; an exactly-safe alternative existed | 12 |
| false alarm: every exactly-safe alternative was imagined as colliding | 2 |
| unavoidable (collision at step 1) | 3 |
| no exact escape within 1.5 s of the collision | 1 |

So 14 of the 15 avoidable collisions are errors of the Sandbox's own robot simulation, with the people exact.  In 8 of
the 12 false-safe cases the chosen or still-running query is a `_th` ("thru") query.

### 1.2 Fidelity ladder: where the imagined robot leaves the real one (`output/shortprobe/ladder_true32f`)

Same decisions, TRUE people in every rung; privileged replacements one at a time
(`perceive.ReplayTracks`, `perceive.TimedReveal`).  Divergence of the imagined robot from the real one at 32 steps
(1.5 s), median / p90, 17 of 18 episodes:

| imagination gets | non-thru candidates | thru candidates |
|---|---|---|
| camera model, frozen map (today) | 0.42 / 3.12 m | 1.68 / 3.67 m |
| the real tracker's tracks | 0.25 / 3.12 m | 1.53 / 3.67 m |
| the real map stream (cells as the real robot certified them, by time) | 0.00 / 1.66 m | 1.51 / 3.65 m |
| both | 0.00 / 1.36 m | 1.12 / 3.65 m |

With imagined core physics added (`ladder2_true32f`; 942 paired candidates, 1.5 s, median / p90):

| imagination | non-thru | thru |
|---|---|---|
| today | 0.42 / 3.00 m | 1.69 / 3.57 m |
| + core physics | 0.42 / 2.81 m | 0.81 / 2.25 m |
| real tracker + real map stream | 0.00 / 0.83 m | 1.13 / 3.36 m |
| real tracker + real map stream + core physics | 0.00 / 0.64 m | 0.42 / 2.17 m |

* The map the robot discovers while moving is the dominant gap: the real planner replans on it, the imagined one
  plans on the map frozen at t.
* The tracker matters little on average, but in 2-3 critical encounters only the real tracker's output reproduces the
  contact (imagined distance 0.89 -> 0.11 m).
* `thru` queries diverge even with both, because the imagination had no obstacle physics.  `thru` queries treat
  observed walls as soft and learn real walls by bumping.  They make up 40-45% of all deviations of every Sandbox
  selector (CV-null 293 of 713, TRUE 248 of 556).  New default-off option `SANDBOX_PHYSICS=core|band`
  (`sandbox/imagine.py`): a forward step into a legal wall core or a bump-memory disc does not translate the base, and
  a native forward that does not move records a bump, as the real legal state does.  First check (one episode): core
  physics brings the thru drift from 2.45 to 0.16 m without changing other candidates; 'band' over-blocks.
* The official human collision is a physics contact between the articulated robot and the human mesh, not a distance.
  At the last logged step before a real contact, the person's centre is a median 0.60 m from the robot's centre
  (p90 0.78 m, max 0.99 m).  The real branch's legal state is not updated at the terminal step, so these are one step
  early.  The imagination therefore needs a calibrated contact model in place of a 0.75 m centre rule.
* Imagined camera: the real jaw camera is 0.52 m ahead of the base centre and yawed about 4.5 deg right, but the
  imagination placed it at the base centre.  New default-off option `L3_CAM_MOUNT=1` (`perceive.camera_pose`).
  Correction (train state bank, 53 episodes): the arm, and with it the camera, is rigid relative to the base within an
  episode but differs between episodes (0.61-0.86 m above the base pose, yaw offset -15..+17 deg, pitch -12..+32 deg;
  medians 0.52 m ahead, 0.73 m up, yaw 0, pitch +9 deg), so no fixed mount is right.  `L3_CAM_TRACKER=1` (below)
  reads it from the legal row at t.

Tracker emulation (`L3_CAM_TRACKER=1`, default off; `perceive.VisibleWorld(..., tracker=True)`; legal: the robot's own
software and sensor spec).  The imagined camera + tracker run the robot's perception code on imagined detections:
* camera rig from the decision's legal row (`perceive.CameraRig`); composed with a base pose it reproduces the real
  camera pose to 6e-6 m over 6,098 train frames;
* the real 3-D frustum test (`partial.observer.in_frustum`: 90 x 90 deg, 10 m, torso = base + 1 m).  People stand on
  the robot's floor: real person base y minus robot base y is median 0.00 m, p95 0.19 m.  Line of sight is checked on
  the legal map;
* the real `LegalHumanTracker`, seeded with the legal tracks at t and the next id from the legal track log.  Fed the
  real detections at the real timestamps, it reproduces the real track log exactly (0 mismatches in 10,816 train
  steps), and at 29 of 29 probed dev decisions the seed equals the runtime's own tracker object.  The imagination's
  0.048 s clock (the real clock is floor(n * 73 / 12) * 0.008 s) changes track ids at 0.2% of steps; median velocity
  change is 0.

Open loop (perception replayed along the real robot's poses with the true people, compared with the real track log,
1.5 s):

| state bank | model | exact track set | tracks P / R | near (2.5 m) P / R | velocity-valid agreement | velocity error |
|---|---|---|---|---|---|---|
| train, 192 states | camera model (today) | 0.758 | 0.73 / 0.86 | 0.87 / 0.90 | 0.944 | 0.13 m/s |
| train | emulated | 0.836 | 0.78 / 0.90 | 0.96 / 0.97 | 0.982 | 0.06 m/s |
| dev, 232 states | camera model (today) | 0.780 | 0.73 / 0.83 | 0.85 / 0.83 | 0.975 | 0.10 m/s |
| dev | emulated | 0.840 | 0.79 / 0.87 | 0.92 / 0.91 | 0.982 | 0.05 m/s |

* Within 2 m of the robot, emulated detections match the real sensor's (train: precision 0.99, recall 0.98).  The
  base-centre 2-D frustum disagrees with the real sensor on 13% of near person-steps; the rig frustum disagrees on
  0.2%.
* The remaining disagreement comes from the frozen map.  93% of false detections are people behind walls the robot
  certifies only later (precision 0.30 beyond 4 m).  Inflated legal walls lower recall at 2-4 m to 0.85.
* The vertical FOV almost never binds: 9 of 4,916 person-steps inside the horizontal FOV, none detected by the real
  sensor.

Closed loop, fidelity ladder on 4 of the 18 collision episodes (`output/agents/tracker/ladder_tk`): 9K1WbyTZ456/9294,
Uxmj2M2itWa/38344, XVSZJAtHKdi/8177 and nJTPfwbAj4S/7302.  29 decisions, 290 candidates, TRUE people, core physics.
Divergence is for non-thru candidates.

| imagination | exact track set, 1.5 s | near exact | divergence at 16 steps, median / p90 | divergence at 32 steps, median / p90 | contact P / R within 0.77 s | contact P / R within 1.5 s | decisive cases foreseen |
|---|---|---|---|---|---|---|---|
| camera model (`camTPc`) | 0.62 | 0.76 | 0.00 / 0.62 m | 0.21 / 1.91 m | 0.38 / 0.50 | 0.44 / 0.56 | 1 / 4 |
| emulated camera + tracker (`camTkPc`) | 0.81 | 0.94 | 0.00 / 0.21 m | 0.00 / 1.85 m | 0.53 / 0.72 | 0.57 / 0.72 | 2 / 4 |
| real tracker output (`camRPc`) | 1 | 0.98 | 0.00 / 0.21 m | 0.00 / 1.88 m | 0.53 / 0.72 | 0.58 / 0.72 | 2 / 4 |
| emulated tracker + real map stream (`camTkMPc`) | 0.87 | 0.96 | 0.00 / 0.21 m | 0.00 / 0.46 m | 0.52 / 0.72 | 0.57 / 0.75 | 2 / 4 |
| real tracker + real map stream (`camRMPc`) | 1 | 1.00 | 0.00 / 0.00 m | 0.00 / 0.24 m | 0.52 / 0.72 | 0.59 / 0.75 | 2 / 4 |

* The emulated tracker drives the imagined planner as the real tracker's output does.  Imagined runs are identical
  for 65% of candidates (camera model: 42%).  The 1.5 s divergence differs from the real-tracker run by more than
  0.1 m for 10% of candidates (camera model: 22%).
* Uxmj2M2itWa: the imagined centre distance falls from 0.89 m (camera model) to 0.08 m (real tracker: 0.11 m).
* What remains is the map (frozen walls), not the tracker.
* Regenerate with `python scripts/l3_run.py eventwm_graph.l3.trackcheck seed|openloop --bank <state bank>` and
  `... trackcheck ladder --probe output/agents/tracker/ladder_tk`.  The probe used the section 1.1 replay recipe with
  `L3_PROBE_VARIANTS=camTPc,camTkPc,camRMPc,camRPc,camTkMPc L3_PROBE_TRACKS=1`.  Of its 580 `camTPc` / `camRMPc`
  runs, 572 equal the earlier `ladder2_true32f` runs.  The 8 others are the 4 `pass_left` candidates whose exact real
  branch also differs from the earlier run.

Which part of the map stream matters (`output/shortprobe/ladder3_true32f`; 832 paired candidates; non-thru
divergence at 1.5 s, median / p90):

| rung | divergence |
|---|---|
| frozen map | 0.41 / 2.29 m |
| real FREE cells only | 0.29 / 2.29 m |
| real BLOCKED cells only | 0.05 / 0.80 m |
| full stream | 0.00 / 0.77 m |

The planner reacts to walls it discovers in currently unknown space.  The map prediction Layer 2 owes the Sandbox is
therefore narrow: will the camera reveal a wall in the unknown cells along each imagined route within 1.5-3 s?

How much of the wall stream today's legal Stage 2 recovers (`ladder5_true32f`; Stage 2's predicted increments for
every candidate's imagined path, one union map revealed through the imagined camera; 16 of 18 episodes,
902 candidates, non-thru divergence at 1.5 s, median / p90, all with core physics):

| map | divergence |
|---|---|
| frozen | 0.38 / 2.85 m |
| Stage 2 predicted (legal) | 0.23 / 2.12 m |
| real wall stream (privileged) | 0.07 / 1.41 m |
| real tracker + real map stream (privileged) | 0.00 / 0.64 m |

Stage 2 recovers about half of the wall-stream gain.  It already outputs per-cell probabilities of what the camera
will certify along a path (stays unknown / free / blocked); only the argmax is used today.

Stage 2 map-stream options (default off; `docs/l3/PROTOCOL.md`; probe `output/agents/stage2/s2prob`).  Reveal model:
replayed along the robot's own past poses on 53 TRAIN legal-map histories (`l3/revealfit.py`), the 5 m `Discovery`
frustum reveals 45% of the wall cells on the band's free side within 0.5 s of their real certificate (20% never) and
56% of the FREE cells (18% never); `perceive.ObserverReveal` (the legal sensor's certificates: FREE halo in the
frustum at 1.0-9.6 m, 35 wall rays to 10 m with the 0.25 m band, the real mount) 57% (7%) and 75% (10%).  Fidelity
probe on TRAIN (24 floor episodes, 57 decisions, TRUE people, core physics; non-thru divergence median / p90 (mean)):

| map stream | 16 steps | 32 steps |
|---|---|---|
| frozen | 0.00 / 0.83 (0.24) | 0.62 / 2.45 (0.96) |
| Stage 2, release (union, `Discovery`, class rule) | 0.00 / 0.63 (0.20) | 0.52 / 2.21 (0.83) |
| Stage 2, BLOCKED p > 0.3 | 0.00 / 0.62 (0.21) | 0.62 / 2.03 (0.82) |
| Stage 2, observer reveal | 0.00 / 0.44 (0.16) | 0.43 / 2.25 (0.78) |
| Stage 2, observer reveal, BLOCKED p > 0.45 | 0.00 / 0.44 (0.17) | 0.48 / 2.08 (0.78) |
| **Stage 2, observer reveal, no predicted FREE cell (`L3_S2_PF=1.01`)** | 0.00 / 0.47 (0.17) | 0.51 / 1.89 (0.76) |
| real map at the window's end, observer reveal (privileged) | 0.00 / 0.41 (0.11) | 0.05 / 1.32 (0.39) |
| real wall stream (privileged) | 0.00 / 0.36 (0.11) | 0.06 / 1.27 (0.41) |

On 3 of the 18 dev collision episodes (ladder recipe, 13 decisions) the bold setting gives 0.03 / 1.93 (0.51) and
0.63 / 4.31 (1.29) against the release's 0.06 / 1.87 (0.57) and 0.69 / 4.38 (1.46), the frozen map's 0.05 / 1.87 (0.60)
and 1.25 / 4.24 (1.82) and the wall stream's 0.00 / 1.87 (0.48) and 0.47 / 3.85 (1.32).  With the real end-of-window
map the observer reveal matches the real stream, so the remaining gap is Stage 2's map itself: its union map certifies
42% of the real new wall cells (precision 0.90) and predicts FREE where the real robot certified walls for 16% of them.
Candidates that drift more than 1 m beyond the wall stream have more missed walls, not more false ones (TRAIN batch 1);
lower BLOCKED thresholds trade median for tail.  A second Stage 2 round on the re-imagined paths (wall recall 0.42 ->
0.49) gave no gain at 32 steps (0.66 / 2.05 (0.82) vs 0.65 / 1.88 (0.80), 20 decisions), and Stage 2 conditioned on the
REAL paths (privileged; recall 0.57) moved the divergence little (0.43 / 2.20 (0.75)).

Contact foresight at the critical decisions (the same 832 candidates; imagined centre distance < 0.75 m vs official
contact within 1.5 s):

| imagination | precision | recall |
|---|---|---|
| TRUE people | 0.75 | 0.80 |
| legal CV people | 0.50 | 0.14 |
| legal CV + physics | 0.50 | 0.14 |

At critical moments the legal people model is the dominant gap.

### 1.3 Imagined physics in the selector (paired, 72 ORCA dev episodes, 1.5 s, outcome rule)

| arm | SR | collisions | timeouts | deviations to `_th` |
|---|---|---|---|---|
| legal CV-null (`S_cv32`) | 54.2% | 19 | 14 | 293 |
| legal CV + continuation fidelity + core physics (`S_cv32fP`) | 54.2% | 28 | 5 | 162 |
| TRUE people + fidelity (`S_true32f`) | 59.7% | 18 | 11 | 248 |
| TRUE people + fidelity + core physics (`S_true32fP`) | 66.7% | 15 | 9 | 178 |

With TRUE people, physics helps (10 wins / 5 losses, p = 0.30).  With legal people, the more faithful robot loop moves
more (timeouts 14 -> 5) and meets people the constant-velocity model does not know (collisions 19 -> 28).  Once the
robot is simulated well, the people model becomes the binding constraint.

Value of a perfect hidden-people model: the same legal physics loop with the TRUE untracked people added to contact
scoring only (analysis, `L3_MW_SCORE_TRUTH=1`, `S_cv32fPst`) gives 58.3% (19 collisions / 11 timeouts).  That is 7
wins / 4 losses against 54.2% (28 / 5).  Collisions fall by 9, but timeouts rise by 6: with a 1.5 s window the
selector turns avoided collisions into deadlocks.

Decomposition on the same 72 episodes (1.5 s unless stated):

| configuration | SR | collisions | timeouts |
|---|---|---|---|
| floor | 47.2% | 28 | 10 |
| legal CV + physics | 54.2% | 28 | 5 |
| legal CV + physics + real camera mount (`S_cv32fPM`; 4 wins / 2 losses vs physics, n.s.) | 56.9% | 24 | 7 |
| legal CV + physics + Stage 2 map discovery (`S_cv32fPS2`, LEARNED map world model; 10 wins / 5 losses vs physics; 15 / 5 vs floor, p = 0.041; best legal) | 61.1% | 23 | 4 |
| legal CV + physics + camera mount, 3.1 s window, frozen map (`S_cv64fPM`; 5 wins / 10 losses vs 1.5 s) | 50.0% | 25 | 11 |
| legal CV + physics + Stage 2 + camera mount (`S_cv32fPS2M`; 3 wins / 9 losses vs without mount, p = 0.15) | 52.8% | 27 | 6 |
| legal CV + physics + Stage 2 + camera mount, 3.1 s (`S_cv64fPS2M`; 6 wins / 11 losses vs the best legal) | 54.2% | 26 | 7 |
| best legal + tracker emulation (`S_cv32fPS2K2`; 0 wins / 2 losses) | 58.3% | 26 | 3 |
| best legal with Stage 1 mean-world people instead of CV (`S_mean32fPS2`; 3 / 4) | 59.7% | 25 | 4 |
| same, 3.1 s window (`S_mean64fPS2`; 7 / 12) | 54.2% | 24 | 8 |
| best legal + contact radius 0.60 m + STOP radius 0.18 m (`S_cv32fPS2fix`; 5 / 6) | 59.7% | 27 | 2 |

The camera mount hurts once Stage 2 discovery is on (it also moves the point from which predicted cells are revealed);
it is dropped from the legal configuration.
| + TRUE untracked people in scoring | 58.3% | 19 | 11 |
| TRUE people everywhere + physics | 66.7% | 15 | 9 |
| TRUE people everywhere + physics, 3.1 s window, frozen map (`S_true64fP`; 9 wins / 6 losses vs 1.5 s; 22 / 5 vs floor, p = 0.0015) | 70.8% | 14 | 7 |
| TRUE people + physics + privileged navmesh wall discovery, 1.5 s (`S_true32fPN`; 7 wins / 9 losses vs without) | 63.9% | 18 | 8 |
| TRUE people + physics + privileged navmesh wall discovery, 3.1 s (`S_true64fPN`; 6 wins / 15 losses vs without, p = 0.078) | 58.3% | 21 | 8 |
| exact simulator, 1.5 s | 76.4% | 6 | 11 |
| exact simulator, 3.1 s (71 episodes) | 87.3% | 3 | 6 |

Revealing the TRUE geometry (navmesh truth maps, `l3/navtruth.py`) through the imagined camera hurts.  The imagined
planner then knows cells that the real camera would not yet have certified.  The map world model must reproduce the
camera's certification stream (what the robot WILL know), not the true geometry: the privileged real map stream by
time (ladder rung camM) reproduces the real robot, while navmesh truth does not.

The longer window is what removes deadlocks: in the exact reference, collisions fall 6 -> 3 and timeouts 11 -> 4 from
1.5 to 3.1 s.  The Sandbox can only use a 3.1 s window if its robot loop stays faithful that long.  Map discovery
dominates that (section 1.2), so the SR ceiling of perfect wall discovery at 3.1 s is the next measurement:
`S_true64fP` (frozen map) vs `S_true64fPN` (navmesh truth maps, `l3/navtruth.py`).

### 1.3b Across robot settings (best legal Sandbox: CV people + continuation fidelity + core physics + Stage 2 discovery, 1.5 s; 72 dev episodes each, paired with the floor)

| setting | Sandbox | floor | wins / losses |
|---|---|---|---|
| ORCA-native | 61.1% | 47.2% | 15 / 5 (p = 0.041) |
| SFM-native | 56.9% | 47.2% | 13 / 6 |
| DWA-native | 54.2% | 48.6% | 13 / 9 |
| ORCA-slowturn | 38.9% | 41.7% | 9 / 11 |
| ORCA-inertia | 25.0% | 29.2% | 8 / 11 |

TRUE people + physics (analysis) per dynamics, 72 episodes each:

| setting | 1.5 s | 3.1 s | floor |
|---|---|---|---|
| ORCA-native | 66.7% | 70.8% | 47.2% |
| ORCA-slowturn | 48.6% (12 / 7) | 52.8% (14 / 6) | 41.7% |
| ORCA-inertia | 25.0% (8 / 11) | 31.9% (11 / 9; 21 timeouts) | 29.2% |

Exact 3.1 s references (early): inertia 78.9% vs 36.8% (19 episodes), DWA 100% vs 61.5% (13), SFM 82.4% vs 52.9% (17).
Inertia's failure is a decision problem.  With TRUE people the selector loops on `wait_short` (up to 71 times in one
episode; timeouts) and chooses explicit STOPs imagined within 0.3 m of the goal that fail officially (< 0.2 m needed;
10 stop_fail episodes).  Options `L3_MW_RADIUS` and `L3_MW_GOAL_R` are being tested.

The robot self-model holds up beyond ORCA-native.  On floor collision episodes with TRUE people, divergence at 1.5 s
(p90) is 1.7 m (DWA), 2.7 m (slow-turn) and 0.9 m (inertia), and contact recall is 0.87-0.90.  With TRUE people the
Sandbox helps slow-turn (interim 40.9% vs 29.5%, 8 wins / 3 losses), so the legal people model is its limit, as on
native.  Inertia does not gain even with TRUE people at 1.5 s (interim 17.9% vs 25.0%; deviations dominated by
wait / creep); a 3.1 s window is being tested.

Bank statistics (`output/agents/bankimag`, train): 57% of real contacts within 1.5 s involve a person untracked at t;
the legal CV imagination's contact AUC is 0.66 (0.50 when nobody is tracked), against 0.97 with TRUE people.  With TRUE
people a 0.60 m centre-distance rule gives precision / recall 0.79 / 0.77, against 0.50 / 0.83 for 0.75 m.

## 1.4 The WM Sandbox this calls for

Purpose: every 0.77 s, for about 10 candidate queries, predict what happens in the next 1.5-3 s if the robot executes
each candidate and then its default policy.  The bar: choosing by the exact simulator's outcome gives 76.4% (1.5 s)
and 86% (3.1 s) against the floor's 47.2% on the same episodes.

**Capabilities.**
* Robot self-simulation (Layer 2.5):
  - real executor, planner and continuation (done);
  - map discovery while moving: the imagined camera reveals predicted cells and the planner replans (missing; the
    largest gap);
  - contact physics (`SANDBOX_PHYSICS=core`, built);
  - the real camera mount and the robot's own tracker code on imagined detections (built: `L3_CAM_TRACKER=1`, the
    per-episode camera rig from the legal row + the real `LegalHumanTracker`; section 1.2);
  - a calibrated official-contact model.
* World prediction (Layer 2, 0-3 s, legal):
  - tracked people with stop / turn modes;
  - untracked people as a calibrated hazard scored over each candidate's path (never phantoms);
  - near-field map completion: free / wall for unknown cells the camera will see within 1.5-3 s.
* Outputs per candidate: P(contact) at 0.77 / 1.5 / 3.1 s with uncertainty, P(stall / deadlock), progress, the
  paired difference from the default with a standard error, and a reliability flag.
* Real time: 0.77 s or less per decision with parallel workers (about 2 s today).

**Metrics.**

| level | metric | today | target |
|---|---|---|---|
| 1. robot fidelity (TRUE people) | imagined-vs-real divergence at 1.5 s, median / p90 | 0.42 / 3.1 m | <= 0.05 / 1.0 m |
| | false-safe decisive cases (avoidable collisions imagined safe) | 12 of 15 | <= 3 of 15 |
| 2. per-candidate prediction (dev bank, exact labels) | collision AUC within 1.5 s, nobody / someone tracked | legal 0.82 / 0.69; TRUE 0.98 / 0.91 | legal >= 0.90 / 0.80 |
| | critical states fixed / broken (frozen rule) | legal 0 / 0 of 22; TRUE 8 / 1 | legal >= 6, <= 1 break |
| | calibration error per regime | - | <= 0.03 |
| 3. decisions (paired SR, collisions and timeouts separate) | Sandbox with TRUE people, 1.5 s | 59.7% -> 67.7% (physics) | >= 72% (exact 76.4%) |
| | legal Sandbox, 1.5 s | 54.2% | >= 62-65% |
| | legal Sandbox, 3.1 s (needs map discovery) | - | >= 75%, toward 85% |
| 4. cost | per decision, p95 | ~2 s | <= 0.77 s |

**How the Sandbox helps query selection.**
* It turns collisions into avoided collisions: the selector deviates when the default's imagined contact is real and an
  alternative's absence of contact is real.  False-safe predictions caused 12 of the 15 avoidable collisions with
  TRUE people.
* It prevents illusory progress: imagined thru queries drove through walls and drew 40-45% of deviations; physics
  halves them.
* It gives the deviation rule paired, calibrated evidence, so the default is kept unless an alternative is clearly
  better (today's selectors deviate on 55-64% of decisions).
* Consequences must be simulated: critics without simulation are near chance at critical states (35% vs 32%).

## 2. Definition

Layer 2 is a calibrated model of the world outside the robot, built from legal inputs and covering only the window
that decides a query (0-1.5 s for decisions; out to 3.1 s once the Sandbox's robot loop is accurate that far).  The
Sandbox (Layer 2.5) simulates the robot itself: real executor and planner, an imagined 90 deg camera with 2 s tracker
memory, and continuation fidelity.  It does this against Layer 2's prediction of
(a) where people will be, split by what the robot can know, and
(b) what geometry the camera has not certified will do to the robot.

Layer 2 is judged by whether its outputs change decisions correctly, against a re-measured CV-null under a frozen
rule.  Trajectory error, pooled AUC and the 50% Stage 1 loop are not the bar.

## 3. Capabilities

1. **Tracked people.**
   - One nominal future that the imagined camera perceives: constant velocity, or Stage 1's MEAN world, to be tested.
     At 1.5 s the mean beats constant velocity: median / p90 error 0.77 / 1.86 m vs 0.93 / 2.39 m.
   - Calibrated spread as correlated sample paths (keep walking / stop / turn at a waypoint), not a per-step Gaussian.
     Constant-velocity residuals are heavy-tailed (p90 / median = 4.1 at 0.77 s).
   - Combined per joint sample with the robot's own imagined-vs-real deviation (median 0.22 / 0.58 m at 0.77 / 1.5 s).
2. **Untracked people, scored and never inserted as phantoms.**
   - Remembered people (lost 2-30 s ago): weighted path particles with an existence probability, confined to non-BLOCKED
     legal cells.
   - Removal by negative information: a particle is cleared where the robot's camera looked and saw nobody.
   - People seen standing stay put until their spot is seen empty.
   - New tracker ids are matched back to remembered people.
   - Never-seen people: births at openings, frontiers and stale or unknown cells.  Their count prior is fitted on train
     labels and never uses the episode's `human_num`.
   - Untracked people are scored only.  The two variants that fed them to the imagined planner both scored 50.0% vs
     54.2% (n.s.).  Detection-event branching (the planner reacts to people it would see) is enabled only if an oracle
     shows it matters.
3. **Uncertified geometry.**
   - P(forward step blocked) for UNKNOWN cells.  6.8% of real forward steps are blocked, 29 of 31 at UNKNOWN cells
     (8 floor episodes; evaluate deduplicated first contacts).
   - A gated legal map-completion option for the imagined planner.  The exact 3.1 s window is the only measured route
     to 85% or more, and the frozen map diverges by 1.23 m at 3 s.  It is accepted only if it lowers divergence with
     TRUE people on the bank.
4. **Blockage persistence.**  P(a route chokepoint stays blocked for more than T s), for example by a person who stopped
   for good (Falcon people stop after 100-400 steps), feeding progress and timeout risk.  Timeouts are half of the
   failures of every legal selector, and a 1.5 s window cannot see them.
5. **Decision-grade probabilities.**
   - Calibrated and never exactly zero (Stage 1 gives probability 0 to 37% of occupied points).
   - Paired against the default with the same random numbers for every candidate.
   - Standard errors that include the imagination error, estimated from TRUE-people imagination residuals.
6. **Legal by construction, with a guard.**  Every output carries a typed provenance record.  The official evaluator
   refuses to start if any analysis switch is set (`L3_ORACLE_BANK`, `L3_ORACLE_NOISE`, `L3_MW_TRUTHMAPS`,
   `QG_HUMAN_360`).  The current `layer2/state.py` checks would not reject a forecaster fed the tape.
7. **Real time as a reported constraint, not a veto:** about 30-60 ms for Layer 2 per decision, with Stage 1 (if kept)
   run alongside imagination.

## 4. Output contract

Once per decision (shared by all K candidates):

| output | form | meaning |
|---|---|---|
| `tracked_world` | float32 [N, 65, 2] + track ids | nominal future of each legal track (0.048 s steps to 3.07 s); what the imagined camera can see |
| `tracked_samples` | float32 [S, N, 65, 2] + weights | correlated error paths (walking / stop / turn) for scoring |
| `untracked_particles` | float32 [P, 65, 2] + weight, existence, class (remembered / never seen) | path particles for people not tracked now |
| `blocked_prob` | float16 grid on the legal map | P(forward step blocked) at UNKNOWN cells |
| `blockage` | per route chokepoint | P(blocked for more than T s) |
| `exposure` | 8 sectors x 2 horizons + scalars | untracked hazard by direction, view age per sector, time since anyone was seen (neutral, for Layer 3) |
| `provenance` | dict | legal flag, model hashes, calibration version, latency, out-of-distribution score |

Per candidate, after imagination (Layer 2 owns the scoring; Layer 2.5 supplies the imagined trajectories):

| output | form | meaning |
|---|---|---|
| `p_contact` | [K, 3 horizons, 4 sources] | P(official human contact within 0.77 / 1.5 / 3.1 s) if candidate k runs and then the default policy; sources are tracked, remembered, never seen, combined.  Untracked contact is split into blind and perceivable (seen at least T_react before contact), computed per path particle |
| `p_stall`, `progress`, `p_blockage` | [K, 2] | stall risk from blocked steps, legal cost-to-go decrease, chance of running into a lasting blockage |
| `delta_vs_default` | [K, 3] + SE | paired evidence that k beats the default (contact, stall, progress), with the same predictions for every candidate |
| `regime`, `valid_3s` | flags | someone tracked within 3 m; whether the 3.1 s values are usable (analysis-only until imagined-vs-real divergence at 3.1 s is 1.0 m or less) |

## 5. Not building

* Forecasts of 10-15 s, future partial graphs, long-horizon leaf values: no gain measured (exact 15 s vs 3.1 s window;
  0.77 s plus a learned value 47.2% vs 44.4%).
* Stage 2 map increments in the planner's loop: the legal map loop scored 46.3% (unfinished) / 50.0% (44 episodes).
  Stage 2 stays a separate result.
* Stage 1's 40 uniformly weighted worlds as the risk distribution, and per-world re-imagination of every candidate.
  The weights are all 1/40, the cost is 8-15 s per decision, and there was no SR gain.
* Critics or values instead of simulated consequences: 35% vs 32% random at critical states; imagined-leaf values 32-38%.
* High-capacity calibrators on about 250 exact-label states: they overfit.
* Neural surrogates of the robot, 360 deg inpainting, pixel or range generation, people reacting to the robot.

These rest on paired differences that are mostly not significant; they stay cheap ablations, not settled facts.

## 6. Training data

* People ignore the robot.  Every step of every recorded episode is therefore a labelled sample: legal inputs at t,
  and the recorded people over the next 3.1 s.  Pipelines: `l3/peopledata.py`, `l3/peoplefast.py`.  These are free
  labels for tracked spread, belief parameters and untracked contact on arbitrary trajectories (blind contact is an
  exact function of the trajectory).
* Ignorability rule: train only on histories from legal policies.  Exclude Oracle replays, `S_true*`,
  `H_mwroll_true*` and exact-reference runs.  Correct for outcome-based episode and state selection (R10 curriculum,
  `cfonline`) with inverse inclusion weights, or use uniformly sampled episodes covering every planner family
  (the R10 spool has no SFM).
* Exact counterfactual labels (about 120 s per state) are used for evaluation, rule thresholds and calibration only.
  Calibration is hierarchical: shared slopes, per-family intercepts.

## 7. Gates and statistics

* The bar is CV-null re-measured under the same frozen rule, scorer and contact definition, plus a no-people Sandbox
  arm that separates self-simulation from people modelling.  Thresholds are frozen on train.
* Collisions and timeouts are co-primary endpoints; SR is reported with them.
* Bank metrics: within-state AUC on mixed states, and fixes / breaks against a deviation-rate-matched random null.
  Critical states are defined at the decision horizon; confidence intervals are episode-clustered.  The bank is from
  51 episodes.
* Dev SR is a futility check.  216 paired episodes have a paired SE of about 3 points; the decisive comparison is
  pre-registered for after the freeze.

## 8. Honest expectation

* Better people prediction alone: about 0-3 points over CV-null.  Layer 2 cannot reach 85% alone.
* The route to 85%:
  - Layer 2.5 robot-loop fidelity: 59.7% -> 76.4% with perfect people;
  - the 3.1 s window, which needs map-discovery fidelity (exact 86.0% so far);
  - timeout handling: blockage persistence and Layer 3 deadlock escape.

## 9. Order of work

1. Attribute the 18 TRUE-imagination collisions (running).  At each of the last 10 decisions: the selector's own
   per-candidate prediction (re-run, checked against the recorded choice) vs the exact outcome of every candidate.
   Classes: imagined safe but really collides / no exact escape at this decision / the rule declined a safe
   alternative / new downstream encounter / unavoidable.
2. Fidelity ladder on the bank and the replayed states: which robot-side gap decides (map discovery, blocked steps,
   tracker, contact geometry).
3. Freeze the rule on train; re-measure CV-null and the no-people arm under it.
4. Episode-level oracle: TRUE untracked people in scoring only vs CV-null under the frozen rule.  This decides how
   much Layer 2 people work can pay.
5. Layer 2 v2, version 0 (no learning): path particles, blocked-step risk, blockage persistence, paired outputs,
   provenance guard.
6. Learned components only where version 0 leaves a measured gap, each with a learning-curve stop rule.
