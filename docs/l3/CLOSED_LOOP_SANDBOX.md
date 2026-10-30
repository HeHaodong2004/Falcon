# Closed-loop World Model Sandbox: what it computes and how reliable it is (2026-10-30)

Status document for the closed-loop Sandbox (`eventwm_graph/sandbox/closedloop.py`, option; the released core
`sandbox/core.py` is unchanged).  All Sandbox inputs are legal; privileged information (the recorded people, the
exact branch simulator) is used only for evaluation and training labels.

## 1. What changed relative to the released Sandbox

| | released (`wm_sandbox_core_1`) | closed loop |
|---|---|---|
| people the imagined planner perceives | every tracked person of the Stage 1 MEAN future, as freshly seen every step | each sampled world's people (tracked and Stage 1's hidden slots) through an imagined 90 deg camera from the imagined pose, line of sight, 2 s tracker memory |
| robot trajectory per sampled world | one trajectory per candidate (mean world), scored against the 40 worlds afterwards | each candidate re-imagined per world; worlds clustered by the tracked people's futures, one representative each; contacts scored against EVERY world's people |
| world weights | Stage 1's (uniform: every world 1/40) | optional learned re-weighting from the tracked people's predicted paths (`l3/worldweights.py`) |
| window | the 0.77 s interval | interval + default continuation (1.5-3.1 s) with the real decision boundaries and goal inheritance |
| map | frozen at t; Stage 2 after the trajectory | frozen (default) or a Stage 2 map loop (predicted local map discovered through the imagined camera) |
| outputs | trajectory, Layer 2 risk | per candidate: P(contact within 0.77 / 1.5 / 3.1 s) over all worlds, clearance distribution, legal progress, STOP risk, leaf states |

## 2. Reliability benchmark

`l3/statebank.py` stores the legal inputs of 283 dev and 245 train floor decisions (ORCA-native) together with the
EXACT outcome of every candidate over 16 / 32 / 64 steps (`shortprobe.py`: the candidate's interval and the default
policy executed in querygen's branch simulator).  `l3/sbeval.py` re-runs any Sandbox variant on the banked inputs and
scores it against those outcomes.

## 3. Candidate collision prediction (dev bank, AUC for a real human collision within the window)

| regime (at the decision) | released people | closed loop, legal | closed loop, TRUE people (reference) |
|---|---|---|---|
| nobody tracked within 3 m (2,070 candidates) | 0.40 / 0.45 / 0.49 | 0.84-0.91 / 0.79-0.82 / 0.67-0.68 | 0.999 / 0.98 / 0.89 |
| someone tracked within 3 m (760 candidates) | 0.72 / 0.69 / 0.69 | 0.70-0.75 / 0.63-0.68 / 0.66-0.71 | 0.94 / 0.91 / 0.87 |

(0.77 / 1.5 / 3.1 s.)  With the right people the closed loop predicts collisions almost perfectly at short
horizons: its simulation of the robot (executor, planner, perception, continuation) is reliable.  Where nobody is
tracked, the closed loop's hidden people make it predictive (the released Sandbox is blind there).  The weak regime
is close interaction with tracked people.

## 4. Why: the people layer (Layer 2, Stage 1)

* Stage 1's 40 worlds are uniformly weighted; the "6 heaviest" worlds of the earlier selectors were arbitrary (the
  world nearest the truth was among them 19% of the time).  Fixed by clustering + all-world scoring.
* Tracked people (331 instances): constant velocity 0.31 / 1.28 m (median / p90) at 0.77 s, Stage 1 mean of worlds
  0.37 / 1.15, average single world 0.61 / 1.35, best world 0.07 / 0.33; 1.5 s: 0.93 / 2.39, 0.77 / 1.86, 1.21 /
  2.28, 0.17 / 0.64.  Heavy tails come from people turning or stopping at waypoints; the right manoeuvre is among the
  worlds but nothing observed identifies it.  Learned world re-weighting: weighted error 0.58 -> 0.39 m (0.77 s),
  1.07 -> 0.86 m (1.5 s).
* Untracked people: of the people who come within 2 m of the robot within 1.5 s, 50% are tracked at the decision,
  35% are untracked behind / beside the robot with a clear line of sight (outside the 90 deg camera), 11% occluded,
  2% inside the frustum (tracker latency).  Stage 1's near-field occupancy discriminates them weakly (AUC 0.63); a
  learned occupancy model with every legal cue (map structure, view age, lost tracks, Stage 1) does not do better.
* The map loop does not help: with the true people, SR 58.9% without it, 54.0% with a privileged map, 46.3% with
  the legal Stage 2 map (its predicted geometry misleads the imagined planner into timeouts).

## 5. Cost

Per decision (10 candidates, under load): frozen map 1.5-4 s for 6 worlds x 32 steps, ~8-12 s for 8 clustered
worlds x 64 steps; the Stage 2 map loop adds ~3x.  The released Sandbox: 0.51 s p50.  Real time needs the worlds in
parallel processes and a faster planner step (2.6 ms per imagined step in Python).

## 6. Open items

* SR of the legal closed-loop selector (clustered, all-world scoring) -- running (`output/eval/CL_c8_32_dev`).
* Learned world re-weighting inside the closed loop -- running on the bank.
* Calibration of the per-candidate risk (more training states being collected).
