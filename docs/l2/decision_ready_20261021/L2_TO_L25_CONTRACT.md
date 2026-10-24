# Layer 2 → Layer 2.5 decision contract

Contract frozen before implementation and before opening development outcomes.
This revision continues release `d66e6691`; it does not replace its filter or
learned models. The downstream planner needs a supported motion interval,
geometry, endpoint/yaw, separate progress/time/contact/PSC quantities,
conditional observation information, and explicit missing probability/support.
Layer 3 remains a recurrent graph RL planner; no RL or full L2.5 is built here.

The actual continuous belief is retained separately from finite imagined
successors. Real arrived centers update that actual belief. A finite imagined
successor supports its stored endpoints, coarse event continuation and terminal
absorption. Arbitrary exact continuous refinement and off-tape interpolation are
unsupported. GH integration and Gaussian projection are declared approximations;
omitted-mass intervals do not certify their error or native calibration.

The packet references the immutable observed graph and one shared exogenous
human forecast. UNKNOWN is not executable. Static configuration-space support
is not human-empty or optical visibility evidence. Humans, including hidden
people, are not confined to graph nodes or known free cells. Imagined branches
retain the current graph; static revelation is unsupported. Legal frontier
information comes from Stage 0, and unknown continuation belongs to learned
long-horizon value. No complete-map, simulator or future-delta input is allowed.

Queries are `(one or two ordered waypoints, final_yaw)`, compiled by the Stage 1
fixed follower against arrived geometry. Its executor/version and nominal
0.048 s clock travel with predictions. There is no STOP; query completion is
not episode success. Actual arrival time is authoritative. A future social L4
requires a new execution/consequence validation, not a renamed follower.

Each candidate preserves support status, executed absolute interval, planned
endpoint/yaw and duration, remaining cursor, contact and social risk intervals,
represented stopping-time/progress mass and unresolved ranges, termination
outcomes, coarse physical branches and successor capabilities. Physical branch
lower masses are not normalized. Their upper endpoints share one residual
allocation simplex. The epistemic unresolved bucket is not an observation.
An unfinished prefix permits continuation; it does not establish Stage 1
interruption or a new high-level decision point. Terminal states absorb.

Computation is scheduled externally: one shared forecast, a cheap batch, and
branches for at most four caller-selected rows. Status distinguishes complete,
unsupported geometry, physical horizon censoring, unrequested branch work and
atomic timeout. An online timeout publishes no posterior/packet and leaves the
actual observation pending. Neither conservative risk endpoints nor pessimistic
decision scores are advertised as calibrated point probabilities.

The implementation/API details and measured applicability will be appended
after validation without changing the predeclared evaluation below.

## Implemented packet version

`eventwm_graph.l2_partial.planning`, schema `l2-planning-packet-1`, retains the
existing structural model version and all eight public operations. The physical
law/checkpoint was not retrained. Source hashes additionally bind this revision's
validation, ownership and interface changes. The packet is an in-process object;
durable belief/carrier serialization remains schema 2.

| Packet field | Meaning and ownership |
|---|---|
| `belief_reference` | Fingerprint, absolute clock, model version and ancestry role. Actual initialization tags `actual_continuous`; generated analytic/finite successors tag `imagined_analytic`/`imagined_finite`. Raw externally constructed states remain a trust boundary. |
| `graph_reference`, `graph_version` | Shared immutable observed configuration-space support and `(namespace, revision)`. UNKNOWN defaults to unknown. This support object is not Stage 0's sparse topology. |
| `stage0_context` | Optional arrived topology/global nodes/edges and PointGoal, bound after clock/support equality checks. L2.5 may retain this small context once per graph version. Future deltas are rejected. |
| `shared_future` | One immutable set of absolute times, continuous XZ position/velocity arrays, absolute lower world weights, seed, unresolved mass and versioned exogeneity reference. Every candidate uses these aligned worlds. |
| `candidates` | Ordered typed query consequences, retaining repeated row identities when a systems workload repeats rows. Actual utility evaluation uses unique proposals. |
| `capabilities` | Supported continuous filtering/finite successors and explicit false flags for exact imagined refinement, static revelation, general empty certificates, delay/recovery, social L4, midpoint interruption and native-event-noise coupling. |
| `uncertainty_diagnostics` | Actual filter versus forecast unresolved mass, incremental forecast omission, likelihood/pruning envelope diagnostics and approximation scope. |
| `computation_status`, `timings_ms` | Published packet is COMPLETE only. Online wrapper returns a separate pending-observation result with no packet on timeout. Component timings are nested, not additive quantiles. |

`SharedHumanFuture.reference` binds belief/model fingerprint, seed, world count
and forecast clock. Arrays are the exact sealed forecast buffers. The reference
preserves shared *human* uncertainty; the native event surrogate does not define
a joint random coupling of contacts across mutually exclusive robot queries.
Per-world query responses allow paired comparisons without asserting that
additional coupling. Current continuous belief moments, association histories
and population support remain available through the separately retained actual
belief; the packet does not duplicate the full filter state.

## Candidate inputs and support

Supply a legal current Stage 0 snapshot and one/two ordered waypoint proposals
with final yaw. `compile_observed_queries` uses the existing Stage 1 compiler.
The runtime checks current start pose/cursor, known swept body support, finite
poses, action/pose cardinality, allowed forward/left/right primitives, and a
digest binding requested waypoint/yaw, poses/actions, executor and graph
revision. STOP/teleport are rejected. Full route-follower kinematic consistency
and truthful graph certificates remain the trusted compiler/Stage 0 boundary;
manual mathematical fixtures are not certified native executions.

Up to 24 rows, 1–256 worlds and physical horizon up to 30 s are domain limits,
not a latency or statistical-accuracy guarantee. Default forecast clock retains
every nominal 0.048 s endpoint. A query cannot silently use an earlier sampled
endpoint. Actual input timestamps remain exact. A saved query's safe/intrusion
provenance is optional explanatory metadata; it supplies no outcome label.

| Query consequence | Required interpretation |
|---|---|
| `geometry` | Requested ordered waypoints/final yaw, owned readonly planned poses, starting/evaluated action indices, evaluated **survival** endpoint/yaw, full nominal plan duration and proposal provenance. Contact may stop earlier. |
| `evaluated_interval` | Absolute executed prediction support, separate from full plan and computation budget. |
| `status`, `nominal_plan_covered`, `remaining_cursor` | Nominal complete coverage, physical horizon censoring, 120-action executor limit, unsupported geometry or absorbed terminal. Nominal coverage is not probability of actual completion. |
| `collision_risk_interval` | Native-label contact-surrogate retained contribution plus possible residual allocation. Not disk overlap or calibrated point risk. |
| `social_risk_interval` | First/any official-PSC-label surrogate over the evaluated prefix, root excluded and contact endpoint included. Planar proximity features are proxies. |
| `duration`, `route_progress`, `goal_progress` | Lower stopping/completion submeasure plus unresolved mass/range; seconds, route arclength meters, signed Euclidean PointGoal-distance reduction respectively. Signed progress can be negative. |
| `termination_lower_masses`, `termination_intervals` | Physical causes only. Contact and the one structurally supported survivor cause can receive residual mass. Impossible causes remain zero. |
| `first_event` | Retained first-contact/first-PSC/auxiliary-appearance masses at relative native times, plus unresolved first-event mass. Absolute support comes from `evaluated_interval`. Appearance is an auxiliary label proxy, not the spatial observation law. |
| `observation_branches`, `branch_status` | Supported physical categories and successor handles, AVAILABLE/NOT_REQUESTED/no-new-exposure status. An empty branch list does not imply no future detections. |
| `represented_mass`, `unresolved_mass` | Absolute probability accounting, never represented-only renormalization. Discrete/quadrature/global projection limitations remain explicit. |
| `per_world_collision`, `per_world_social` | Readonly conditional responses aligned to the shared tape and weights. Preserve correlations useful for counterfactual ranking. |
| `executor_version`, `approximation_status` | Fixed follower identity and the local-surrogate probability scope. A changed social L4 cannot reuse the consequence claim by relabeling it. |

Official intrusion duration, severity, persistent human blockage, execution
delay/recovery and arbitrary episode-limit hazards are not modeled by this
packet. Social output is first/any intrusion risk and timing, not a calibrated
full PSC exposure-duration distribution. No unsupported quantity is filled with
a fabricated point probability. Goal progress is not geodesic progress through
unknown space; final PointGoal success requires a separate episode/STOP law.

## Observation events and successors

The declared endpoint optical mixture induces `survived_no_detection`,
`survived_detection` (at least one detection over the whole executed prefix,
excluding root), and `terminal_contact`. Empty/invalid/absorbed zero-exposure
cases are explicit. No-detection is a probabilistic event, not an optical or
human-empty certificate. Static support and human occupancy remain separate.

For physical lower branch mass a and residual u, probability interval is
`[a, min(1, a+u)]` under the retained optical/event surrogate. Physical branch
lower masses sum to represented mass for ordinary executed prefixes. Their
upper endpoints share the same unknown allocation; do not normalize branch
lower masses into a complete distribution. `unresolved_physical_outcome` from
the legacy runtime is exposed only as epistemic packet mass, never as a physical
observation/successor option. Deterministic absorbed/invalid cases may have unit
physical outcome mass despite unresolved human state.

A successor handle borrows a legacy belief under a no-mutation contract; its
graph/tape storage is immutable and shared. Finite conditional worlds retain
joint modes/counts/path/optical/RNG state. A completed nonterminal query or
executor-limit end can select another legally compiled query. A horizon-censored
prefix supports continuation of the same reference; Stage 1 mid-query
interruption is not declared. An invalid reference cannot continue, but its
unchanged nonterminal state can try a different legal action. Terminal branches
absorb and cannot choose another action.

`predict_shared`, `successor_at` and `continue_rollout` retain supported finite
recursion and analytical unconditioned propagation. Arbitrary exact continuous
imaginary detections/certificates, changing finite world count, and missing tape
times needing a stochastic bridge stay unsupported. No particle smoothing or
fabricated exact measurements were introduced. After execution, use the
separately retained actual continuous prior and the actual arrived observation.

## Computation and approximation contract

`evaluate_queries(..., refined_queries=4)` remains compatible. New
`refined_query_indices=(...)` permits any at-most-four distinct rows, including
the empty tuple for base-only evaluation. Forecasts can be reused between base
and selected passes. The selected pass currently recomputes base hazards; the
model does not choose a value-of-computation policy. That belongs to L2.5/L3.

`online_planning_call` includes lock wait, actual observation/optional execution
likelihood, forecast, selected branches and packet creation under the earliest
deadline. Success returns actual belief and packet atomically; timeout returns
the original actual belief, no packet, pending observation and stage/elapsed
status. Native kernels are cooperative, so hard preemption is not guaranteed.

Gaussian/GH observation projection is approximate and not globally dominated
by the residual bound. Interval scope is local omitted-mass accounting under
the retained surrogate, excluding quadrature/projection/native-calibration
error. Continuous visible centers are exact observations in this model. The
real sensor is idealized structured geometry/centers with articulated camera
and dynamic occlusion; the optical mixture is a predictive surrogate.

Final evidence: packet/API tests pass; 25/60 actual-history roots are nearly
vacuous; any/no categories lose measured finite-oracle decision information;
warm/cold P95 is 137.11/153.16 ms. This is a stable restricted integration
contract with **DECISION INTERFACE REVISE** and **SYSTEMS REVISE** blockers, not
a claim that all outputs are already sufficient for the future RL planner.
