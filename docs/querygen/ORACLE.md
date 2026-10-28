# Privileged Oracle (frozen as `querygen-freeze-v1`)

Code: `eventwm_graph/querygen/oracle.py`, configuration `OracleConfig()` (recorded in every result).

- **Selection is privileged, generation and execution are not.** Every edge is one generator query
  (proposed by `QueryGenerator.propose` from the branch's own legal state) executed by the fixed
  executor/planner/dynamics in the native branch simulator (bitwise equal to `Env.step`), with
  canonical contacts and the declared collision rules; a branch dies at its first human collision,
  body-proxy overlap or non-exempt static contact. Branch states (simulator, legal sensor, legal
  map, executor) are isolated per branch.
- **Search:** lazy weighted best-first over edges (an edge is simulated when popped),
  f = steps + 2 h + penalties, with h = max(geodesic / 0.17 m/step, space-time arrival time).
  Space-time arrival: backward DP over 0.2 m navmesh cells x episode steps using the true human tape
  (cells within 0.6 m of a same-floor human are unsafe), unit moves to 8 neighbours or waiting; it
  encodes when passages are blocked, so waiting on the feasible schedule lowers f.
  Penalties: rectangle clearance < 0.6 m to humans along the branch (30/m) and danger of the held
  pose (true human future within 0.8 m over the next 60 steps, 60/m).
- **Priors / manoeuvres:** small per-category priors; exact manoeuvres cost +10 and are padded to a
  full 16-step interval; entering manoeuvre mode costs a further +120 unless every ordinary movement
  query tried from the parent failed natively (privileged gate).
- **De-duplication:** 0.3 m cells x 8 heading bins (x 10-step time bins within 4 m of a human);
  states reached by a manoeuvre edge use 5 cm x 75 heading bins.
- **Budget:** 6000 simulated edges (deterministic; 9000 s wall-clock safety cap).
- **Acceptance:** the first branch that STOPs within the success radius is replayed continuously
  from reset with the unchanged `habitat.Env.step` (see FINAL_PROTOCOL.md); only a replay that
  reproduces the actions and passes every check counts.
