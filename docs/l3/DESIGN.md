# Layer 2.5 Graph Compiler + shared Layer 3 Query Selector: design

Branch `research/l3-shared-policy-20261029` (base: World Model Sandbox `research/wm-sandbox-20261028` @ 4af2e730).
Package `eventwm_graph/l3/`. Protocol and data separation: [PROTOCOL.md](PROTOCOL.md). Results: [REPORT.md](REPORT.md).

## 1. Decision loop

```
legal state at t ──> querygen generator (frozen v14 K=10 / K=6 sets, the robot's own dynamics) ──> K candidate Queries
                 └─> World Model Sandbox: Stage 1 joint human bank (40 worlds x 3 s, shared)
                                          each candidate imagined by the robot's OWN planner + executor + dynamics
                                          Stage 2 hypothetical map increment per imagined trajectory, joint risk
                 └─> Graph Compiler ──> consequence graph (robot frame; planner / dynamics / identity free)
                 └─> shared selector (consequence-graph network, decomposed values) ──> one Query
                 └─> fixed low-level stack executes it for one interval (16 steps; waits their duration)
```

The selector never sees a planner name, dynamics family, robot identity or execution-style flag. Two robots that
execute the same Query differently produce different imagined consequences, which is the only way the difference
reaches the policy.

## 2. Graph Compiler (`compiler.py`, schema `l3_graph_v1`)

Dense heterogeneous spatio-temporal graph (one decision; K candidates, P = 8 waypoints, N tracked humans):

| node / edge set | content |
|---|---|
| `ego` [69] | goal in the robot frame, legal route cost-to-go (current legal map, 9 m window), time used / left, recent displacement / progress / turning / stall, local legal map in 8 sectors x 2 rings (blocked, unknown), bump memory, tracked-human situation, Stage 1 unseen-person mass near the robot and the risk of holding the current pose (3 s), generator diagnostics (stuck, manoeuvre mode, conflicts, route length) |
| `cand` [K, 63] | coarse intent (goal / st-goal / goto / wait / rotate / manoeuvre / stop / continue), continued flag, via-point geometry, duration; imagined motion (end displacement, heading change, path, speed, move / turn / hold fractions, termination); progress (legal route cost reduction, end distance, at-goal); static (blocked / unknown cells along the path, clearance, bump proximity); Stage 1 risk over the interval (collision / social probability, social duration / severity, minimum-distance quantiles, first-collision time) and over a 3 s hold extension; Stage 2 increment (expected revealed free / blocked cells, goal-ward and straight-ahead blocked mass, hard counts) |
| `traj` [K, P, 17] | waypoint time, pose and heading in the robot frame, speed, legal map state and clearance at the pose, route-cost gain, P(social) / P(collision) and minimum-distance quantiles across the 40 Stage 1 worlds at that time, clearance to the nominal humans the imagined planner perceived |
| `hum` [N, 18] | tracked person: position / velocity in the robot frame, distance, bearing, age, visibility, closing speed, Stage 1 mean position at 1 s and 3 s, spread, presence |
| `hfut` [N, P, 5] | the person's Stage 1 future at the waypoint times: mean, spread, presence |
| `pair` [K, P, N, 6] | same-time interaction traj(k, p) -- hfut(n, p): relative position in the waypoint frame, distance, P(< 1 m), P(contact envelope), q10 distance over the worlds |

Relations are implied by tensor axes (membership, temporal order, same-time interaction, all-pairs candidate
comparison).

**v1b (main run from R4).** Momentum is already inside each imagined interval (the robot's own executor, planner and
dynamics produce the trajectory, and Layer 2's risk is computed on it).  The compiler's 3 s risk extension beyond the
interval, however, assumed the robot freezes at the end pose -- impossible for a robot still moving with speed state.
The Sandbox now also imagines the minimal-motion emergency stop with the robot's own braking limits (`imagine.brake_tail`,
an additive output; Layer 2 inputs and all 18 sandbox tests unchanged) and the extension starts from where the robot
can actually come to rest.  Identical to v1 for robots without speed state (native, slowturn).  `GROUPS` names the feature groups (intent, motion, progress, static, human, map) for ablations;
`compile_blind` builds the no-imagination input (intent + via geometry + current observation only).
Compile cost ~27 ms per K=10 decision (P50).

**Why a graph.** The decision is relational: a candidate is good or bad because of where its imagined body is
relative to each predicted person at the same time, relative to the map along its path, and relative to the other
candidates.  A flat per-candidate summary loses the person-time alignment and the comparisons; the dense graph keeps
them at modest cost (all tensors fixed-size per decision, no sparse ops).  The flat set model is kept as an ablation.

## 3. Shared selector (`model.py`)

Consequence-graph network (1.24 M parameters, ~5 ms single-decision CPU inference): input MLPs per node type; L = 2
rounds of (i) human-future -> waypoint attention at the same time with pair features and a null slot, (ii) temporal
self-attention along each imagined trajectory, (iii) attention pooling waypoints -> candidate, (iv) human futures ->
humans -> ego, (v) a transformer layer over {ego, candidates}.  Readout per candidate: five general value functions
of the greedy policy -- P(success), P(human collision), expected remaining steps, path length, social steps.  The
action value is a fixed linear scalarisation (success +10, collision -5, -0.002 / step, -0.3 / m, -0.02 / social
step), aligned with the official metrics (SR first, SPL through path length, PSC through social steps).

## 4. Learning (`learner.py`, `actor.py`, `fleet.py`)

* **Algorithm**: decomposed n-step (n = 3) double Q-learning over candidate sets (undiscounted episodic GVFs; every
  episode ends at an official terminal and the remaining time is a state feature), target network (Polyak 0.005),
  replay ring of packed graphs (1.5 M decisions, ~12 KB each), Boltzmann exploration in the scalar value with
  epsilon mixing, temperature annealed.  Off-policy because environment samples are expensive (~3.5 CPU-s each):
  every transition is reused (replay ratio 32) and Oracle demonstrations share the buffer.
* **Oracle-guided warm-up**: (a) the v14 Oracle solutions of training episodes replayed through the full loop
  (graphs in the policy's input space, real returns, `source` 1); (b) counterfactual labels (`cflabel.py`): every
  candidate at every Oracle state executed for one interval and continued by the default policy to the official end
  (exact).  Losses: counterfactual regression (warm-up phase only) and a hinge lower bound Q(s, c) >= Q_cf(s, c)
  (any continuation can only improve on the default); a margin Q(s, a_oracle) >= Q(s, c) + 1 only against
  candidates whose counterfactual FAILS while the Oracle succeeds (other viable candidates are never pushed down);
  TD on the demonstrations' real returns.  All decay as online data accumulates.  Oracle information is supervision
  only, never input.
* **Value targets** mix the episode's Monte Carlo return into the n-step target with a weight decaying 0.5 -> 0.1 over
  300k decisions (R1 showed that 3-step bootstrapping alone left the success head at ~0.9 vs ~0.3 actual after the
  all-success demonstrations); replay ratio 64.
* **Default-anchored exploration**: during training the actor takes the default (no-selection) candidate with a
  probability decaying 0.3 -> 0.05, otherwise Boltzmann(Q / tau) + epsilon.  Off-policy learning is unaffected; the
  data stay near the floor's competent behaviour instead of thrashing among near-equivalent candidates (R3 on
  momentum robots).  Greedy evaluation: argmax Q, optionally with a conservative deviation margin chosen on dev.
* **Heterogeneous training from the start**: every actor episode samples a planner x dynamics x candidate-set
  configuration from the curriculum; the same network and buffer serve all of them.
* **Throughput**: split-process Sandbox (Layer 2 servers on the GPUs, 20+ CPU actors with the torch-free range
  backend), CPU policy inference in the actors, learner on GPU1; no Oracle search or world-model recomputation in
  gradient updates (graphs are compiled once, at acting time).
