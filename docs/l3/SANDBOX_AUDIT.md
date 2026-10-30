# World Model Sandbox: code audit (2026-10-30)

Scope: what the released Sandbox (`eventwm_graph/sandbox/`, `wm_sandbox_core_1`) actually computes for one Layer 3
decision, read from the code, plus the Layer 3 options added on 2026-10-29/30 (all default off). Line numbers refer
to commit `8c625fa2`.

## 0. Execution cadence (measured on the floor's dev episodes, 72 per configuration)

- A movement or rotation query runs for one decision interval of **16 native steps = 0.768 s** (`sandbox/core.py:35`,
  `querygen/oracle.py:249-256`; 1 step = 0.048 s world time, `querygen/dynamics.py:22`). Wait / creep / k-turn
  queries run for their own duration, and STOP for 8 steps.
- Executed steps per decision: **15.4–15.5 on average (0.74 s) for every momentum family and for the k6 sets**
  (94–95% of decisions run the full 16 steps). The native k10 configurations average 10.8–12.7 steps (0.52–0.61 s),
  but only because of two pathological timeout episodes per planner. In those, the goal query ends after 1 step
  with `done:unreachable` and is reissued every step, up to 342 times in a row. Excluding them, native k10 decisions
  also run 15.6 steps. **Your figure (~0.7 s executed per query, a new decision after at most 0.768 s) is
  correct.**
- One floor episode lasts 188 steps (9.0 s) on average on ORCA-native: about 12 decisions. The 10–15 s forecasts I
  asked for overnight therefore cover nearly the whole episode.

## 1. How a candidate trajectory is generated

- **The real executor and the real planner run step by step.** Each candidate gets a querygen `QueryExecution` with
  the configuration's own planner object (`make_planner`) and dynamics family (`sandbox/imagine.py:433-434`,
  `sandbox/core.py:254-274`). The imagination loop calls `execution.step(legal_view, level)` once per imagined step
  (`imagine.py:314-348`). This is the same call that drives the real robot (`sandbox/episode.py:176`,
  `querygen/episode.py:162`). A `continue` candidate is a deep copy of the running execution with
  `decision_boundary()` applied, exactly as the real runtime does (`core.py:259-264` vs `querygen/episode.py:144-148`).
- **The robot's motion is NOT the simulator's.** Each command is integrated by nominal family kinematics: a native
  forward step is 0.208 m straight ahead, a turn is ±1/12 rad (`imagine.py:186-209`). There is **no navmesh**: the
  imagined robot can drive into walls (only counted in `blocked_steps`, `imagine.py:341-342`) and never slides or
  gets stuck. Open-loop replay of the real commands with these kinematics diverges from reality by 0 m (median) but
  0.42 m p90 after 32 steps and 1.02 m p90 after 48 steps (fidelity probe, `kin` row).
- Horizon: one interval (16 steps; `core.py:87-98`). Candidates are imagined one after another in a Python loop
  (`core.py:285-297`).

## 2. One nominal trajectory per candidate, scored afterwards

- **Released Sandbox: one trajectory per candidate, computed against one nominal people future.** That future is the
  presence-weighted **mean** of the 40 Stage 1 worlds for each tracked person (`imagine.py:82-83`, mode `'mean'`,
  `core.py:74`, `core.py:325-326`). Layer 2 then scores that fixed trajectory against all 40 worlds for collision /
  social risk (`core.py:332-340`). The robot's reaction to each sampled world is not simulated, and the mean of
  multimodal futures is not itself a plausible world.
- Layer 3 option (overnight, default off): `heuristics.multiworld_index` re-imagines every candidate once per sampled
  world (6 worlds) and checks contacts against that world.

## 3. Stage 2 is outside the loop

- **The legal map is frozen at the decision time for the whole imagination** (`imagine.py:6-8`, `imagine.py:162-165`,
  `core.py:281`). Stage 2 is called once, after all trajectories are complete, in one batched call that predicts
  each trajectory's map increment and future partial graph (`core.py:339-340`). Those predictions are flagged
  hypothetical and are **never fed back to the imagined planner** (`core.py:17-19`). Map discovery along the
  imagined path cannot change the imagined route.
- Layer 3 option (`L3_MW_DISCOVER`, default off): two passes. Pass 1 imagines on the frozen map. Stage 2 predicts the
  cells revealed along those paths. Pass 2 re-imagines while those cells are certified as they enter the imagined
  camera (`heuristics._discovery_rollouts`, `l3/perceive.py: Discovery`). This is one round, not a closed loop.

## 4. No imagined camera, tracker or map update (released)

- Every person tracked at t is handed to the imagined planner **at every step as a freshly observed track**:
  `visible_now=True`, `observation_age=0`, exact finite-difference velocity. This happens whatever the imagined
  heading, field of view or occlusion (`imagine.py:128-139`, `imagine.py:315`). People who are not tracked at t never
  appear, including Stage 1's predicted unseen (residual) slots, which the planner never perceives.
- The real legal state, by contrast, updates the map from every sensor row, replaces the tracks with the 90° tracker's
  output, and records a bump disc whenever a forward step does not translate (`querygen/legal.py:195-213`). In
  imagination the map is frozen, the tracks are synthetic, and **no new bumps are ever recorded**.
- Layer 3 option (`L3_MW_PERCEIVE`, default off): `perceive.VisibleWorld`. People are seen only inside the 90°/10 m
  frustum from the imagined pose, with line of sight on the legal map; a lost track is kept at constant velocity for
  2 s; tracked and hidden slots of a Stage 1 world are included. Real tracker velocity noise and lag are not modelled.

## 5. Do new observations change the planner's later actions?

- **The mechanism exists:** the executor re-reads `legal.tracks` / `legal.map` every step (`querygen/executor.py:465-473`),
  so imagined people do steer the imagined planner.
- **But the inputs are not what the robot would see (released).** The people are the nominal mean and always fully
  visible. No new person or wall is ever discovered. So the imagined planner reacts to a different world from the
  one the real planner will perceive. Information gain (looking, discovering geometry) has no effect inside the
  imagination unless the two options above are on.

## 6. Executor state, recovery, turning, `continue`

| Element | Reproduced? | Note |
|---|---|---|
| Deep copy of the running query (`continue`) | yes | `core.py:259-264` |
| Goal-history inheritance (`L3_INHERIT`) | yes, when the option is on | `core.py:272-273`, `executor.py:386-414` |
| Unstick escalation (stillness streak) | logic yes, trigger no | It counts imagined stillness (`executor.py:319-355`), but imagined poses are never blocked by walls, so a real wall-push deadlock is not reproduced |
| Bump memory (`bumpf`) | frozen at t | No contacts in imagination (`executor.py:260-274`) |
| Evade reflex | yes | Driven by the synthetic tracks (`executor.py:276-317`) |
| Stuck-STOP rule, speed debt, momentum realiser (levels, braking) | yes | Inside `execution.step` (`executor.py:475-524`); level carried by `imagine.py:317` |
| In-place turning | yes | Native turn rate per step; turning while moving per family |
| Decision boundary every 16 steps in the continuation, failed-goal re-issue | Layer 3 fidelity option only | `SANDBOX_CONT_BOUNDARY` / `SANDBOX_CONT_INHERIT`, `imagine.py:369-415`, default off |
| Navmesh contact / sliding | no | Nominal kinematics |

## 7. Information classes

| Class | Released Sandbox | Layer 3 options |
|---|---|---|
| Legally observed at t | legal map (certified cells), legal tracks (≤2 s old), pose history, bump memory, executor state, goal | same |
| Predicted | Stage 1: 40 worlds × 3 s of tracked and unseen people (planner sees only the tracked mean). Stage 2: map increments after the fact | per-world people incl. unseen slots (`VisibleWorld`); Stage 2 cells in a second pass (`L3_MW_DISCOVER`) |
| Assumed | nominal kinematics (no navmesh); people always visible; map frozen; constant velocity beyond 3 s | camera model replaces "always visible" |
| Privileged | **none at runtime** (`core.py:16-19`) | **analysis only:** `L3_ORACLE_BANK` (true people future), `L3_MW_TRUTHMAPS` (privileged maps), `fidelity.py` / `rolloutpolicy.py` (exact branch simulator) |

## 8. Latency (isolated, single process, RTX 3090, ORCA-native, K = 10, 30 decisions)

| Stage | p50 | p95 |
|---|---|---|
| Generator (K candidates) | 38 ms | 52 ms |
| Stage 1 bank (40 worlds × 3 s) | 76 ms | 107 ms |
| Imagination (10 candidates × ~15 steps, sequential Python) | 385 ms | 598 ms |
| Stage 2 + Layer 2 risk | 50 ms | 59 ms |
| **Sandbox total** | **511 ms** | **719 ms** |
| Graph Compiler | 53 ms | 60 ms |
| **Decision total** | **~0.60 s** | **~0.83 s** |
| One extra world, 48 steps, release people | 1,044 ms | 2,161 ms |
| One extra world, 48 steps, camera model | 1,184 ms | 2,276 ms |

- **Bottleneck: the Python executor and planner stepping**, at 2.6 ms per imagined step (3.0 ms with the camera
  model). That is 75% of the Sandbox time. Stage 1 and Stage 2 together are about 25%.
- The released pipeline barely fits the 0.768 s interval. Last night's multi-world selector (6 worlds × 76 steps,
  ~4,500 imagined steps) took 20 s p50 per decision under fleet load: research-only speed. Real-time multi-world
  imagination needs parallel workers or a faster planner.

## 9. What this means

The released Sandbox runs the real executor and planner, but in a **frozen, fully observed, averaged world**.
The map never updates, people never leave or enter view, there is one nominal future, and there is no wall contact.
The released Sandbox is a capable **open-loop consequence estimator**. It is **not yet a closed-loop simulator of the
robot's own observation → map → planner loop**, and the gap matters most when the robot turns, meets unseen people,
discovers walls, or deadlocks. The overnight Layer 3 options close parts of this (camera model, per-world
re-imagination, discovery), but none is in the release.
