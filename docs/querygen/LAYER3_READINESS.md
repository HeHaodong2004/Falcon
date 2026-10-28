# Foundation for Layer 2.5 / Layer 3 (official Falcon rules) -- status and interfaces

User priorities (2026-10-28): the Oracle is an offline diagnostic of the capability ceiling of Planner + Dynamics +
Executor + Query Generator; the research system is the shared World-Model Sandbox (Layer 2.5) and a learned Layer 3
policy over the query space.  Rules: `OFFICIAL_RULES.md` (official success / collision / PSC; denominator = every
episode not proven natively infeasible).

## 1. Capability ceiling (Oracle, official rules)
Development set = DEV 28 + first unseen 40 + second unseen 60 episodes (nothing excluded).  Natively proven
infeasible: 3 (unseen2 5Poh4Qz68hd, sKLMLpTHeUy, AMEM2eWycTq); confirmed feasible 124; unresolved 1 (nMeXfQU4PMS).
Configurations (official discrete actions): C1 ORCA x native K=10, C1 K=6 (compact), C2 DWA x native K=10; Oracle =
v10 two-member portfolio, 12,000 edges.  Results: `output/runs/official/base/` (TBD when complete).

## 2. No-selection floor (the default policy: keep the running query, else the goal query)
`output/runs/official/exec*/`.  Failure modes (first 83 episodes): human collision 34, timeout 34, success 14.
* 13 of 16 analysed collisions hit a robot that held or turned in place while a human it had tracked (in the
  camera field of view) for 10-20 steps walked into it (Falcon's humans do not yield).
* Timeouts: static deadlocks -- the planners treat the legal map's inflated wall band as a hard constraint
  (DWA cut every advance on auFeVz9Go4m; narrow dead ends could not be turned in), and obstacles the camera does
  not see (8B43pG641ff: 290 steps pushing into an unseen navmesh edge).
Foundation options under A/B (all legal): `goal_unstick` (static deadlock -> tight map + plan following, rotate
toward the route when no body-feasible plan exists -- in-place rotation is always executable in the official
simulator), `bump_memory_m` (a forward step that does not move marks the front contact as BLOCKED --
proprioception), `evade` (human-threat reflex from constant-velocity predictions of tracked humans).

## 3. Learnability
* Critical decisions per episode -- minimum-intervention Oracle (`OracleConfig.intervention_cost`, lambda = 300
  steps per non-default decision): the solution with the fewest decisions that differ from the default policy.
  First checks KsD3yx9nZCv 1 of 7 decisions, HMkoS756sz6 1 of 5, LEFTm3JecaC 1 of 17; 8B43pG641ff 11 of 17.
* Readiness study (`tests/querygen/l3_readiness.py`; labels privileged and offline, features legal).  At up to 4
  decision states along each replay-verified Oracle solution (its critical states first, then evenly spaced
  others), every candidate the generator offers is executed for one decision interval and then handed to the
  default policy until the official episode end.  Per state:
  - *good candidates* = candidates whose default continuation succeeds (0 = only a longer sequence of
    non-default decisions saves the episode from here; 1 = brittle; >= 2 = several reasonable choices);
  - *top-1 success* of selection rules: random, default, the Oracle's own choice, single legal features
    (sandbox progress, predicted overlap probability, ...), and a leave-one-episode-out logistic regression on the
    legal consequence features (+ their within-state deviations) -- a first identifiability probe of what a
    learned Layer 3 could achieve from legal information, against the upper bound "some candidate is good";
  - *within-state AUC*: P(a good candidate outranks a bad one at the same state);
  - *one-decision recovery* (`--recovery N`): for up to N bad candidates per state (the default first), can ONE
    later decision still save the episode -- at the next decision state, does any candidate followed by the
    default policy succeed?
  The default policy uses the foundation of the Oracle run (`--default-meta auto`: v11 unstick / evade; bump
  memory is process-wide via the generator configuration).
* K=6 vs K=10 coverage: Oracle capability per configuration (section 1) and good-candidate counts per state.

## 4. Interfaces for integration
* `eventwm_graph/querygen/l3_env.py` -- `QueryEnv`: decision-level MDP.  `reset()` -> legal observation (pose,
  goal, legal map, legal tracks) + K candidate queries (+ optional legal consequence features); `step(i)` executes
  candidate i for one decision interval in the exact branch simulator (Env.step mirror; humans from the recorded
  tape); reward = progress - time + success / - collision (training signal only); `official_replay()` = evaluation
  of record with the unchanged Env.step.
* Layer 2.5 hook: `sandbox.rollout(legal, query, planner, dyn, F, steps)` + `sandbox.score_vec(traj, F)` take any
  human-future tensor F [M, T+1, N, 3]; `sandbox.cv_futures` is the constant-velocity stand-in -- the Stage 1 world
  model's samples conditioned on the legal tracks replace it without other changes.  `l3_interface.consequence_
  features` builds the per-candidate feature rows (progress, effort, social risk over futures, static risk).
* Speed (loaded 36-core machine, per process): ~1 decision/s, ~15 native steps/s, generator ~25 ms per decision,
  sandbox features ~1.1 s per decision (K rollouts x M futures).  Episode setup 24 s -> 11 s with the exact human-tape
  cache (`QG_TAPE_CACHE`); scene reuse across resets is the next saving.
