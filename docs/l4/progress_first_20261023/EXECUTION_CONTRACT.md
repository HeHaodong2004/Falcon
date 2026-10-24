# Progress-first execution contract

The Python interface remains `eventwm.l25l4.contract.v1`; the control semantics
are identified by `l4.progress_first.*.v2` executor profiles. The reference
compiler remains `l4.reference.1`. This document supersedes the conservative
runtime semantics in the 2026-10-22 release. Historical measurements remain
unchanged in that release directory.

`compile_reference(QuerySpec, legal_graph, ActionSpec)` preserves the immutable
selected ordered XYZ route, initial actual XYZ/yaw, terminal yaw, tolerances,
timestamp, graph version, deterministic nominal schedule, and reference hash.
Compilation checks certified route connectivity. The static support flag
records what the supplied geometry supports; it does not certify complete body
clearance. Unsupported references are explicit pre-execution failures. The
compiler neither selects a new goal nor repairs graph connectivity.

`Executor.start(query, reference, RuntimeContext(geometry))` requires matching
query identity and graph version. `act(handle, live_state, actual_observations,
forecast_view=None)` returns one native `forward`, `left`, or `right` action
while RUNNING. Forward is 25/120 m; turns are ±10/120 rad. STOP terminates the
episode and is never an executor action. There is no backward or pause action.
Repeated `act` before acknowledgment returns the same pending decision.

The executor computes the nominal action first. Only credible near-term risk
from visible or recently observed tracks can activate pedestrian avoidance.
Six fixed side maneuvers are compared using cheap geometry and measured-motion
predictions. A maneuver can be accepted when it improves worst predicted
clearance by at least 0.15 m over nominal execution even if the conservative
human envelope still overlaps. No such score is a collision-free certificate.
A selected maneuver is committed for at most 40 native actions,
including forward translation. Its next step is compared with nominal against
current human observations; materially worse immediate risk resumes nominal
tracking. A distance-based cooldown
requires 0.8 m translation before another avoidance attempt. Failed screening
executes nominal immediately. Two six-action static nudges may be checked at
supported grid boundaries. There is no online beam search or mesh/FK search.

Route advancement is monotone in waypoint index, while actual projected
arclength can retreat. Advancement accepts an arrival within the existing
position tolerance, passing the waypoint plane within a 0.25 m route corridor
and 1.2 m beyond, or rejoining the immediately adjacent outgoing leg within
0.20 m and at most 1.2 m beyond a non-hairpin corner. These local gates prevent
returning to a passed waypoint. They do not permit jumping to an unrelated
later leg. Final position and orientation tolerances are unchanged.

`on_transition` acknowledges exactly the pending action with a strictly later
finite native timestamp, actual pose, observation reference, measured contacts,
termination, and truncation. It records sliding, prediction error, contact and
clock discrepancies. Static contact alone does not interrupt execution. Native
episode termination or truncation is authoritative. A human collision on the
arrival transition cannot be reported as query completion. Completing a query
does not end the episode. L3 owns its query schedule, interruption and any
decision to STOP at an episode goal.

Legacy `QuerySpec.execution_limit`, `duration_limit`, `RuntimeContext.horizon`,
`beam_width` and `no_progress_seconds` remain accepted for caller compatibility.
The compiler uses execution_limit to bound nominal compilation. The runtime
does not enforce a local lifetime or no-progress watchdog. L3 can explicitly
call `interrupt` at its ordinary schedule after resolving any in-flight action.
Native limits remain in force. Legacy BLOCKED and EXECUTOR_LIMIT enum values
are retained but are never emitted by this controller. An avoidance failure
increments diagnostics; it does not trigger handoff, STOP, or query reselection.

Feedback retains actual action/time/pose/observation ledgers, route progress,
deviation, path length, detours/rejoins, status, native episode flags and mode
history. It adds per-transition contact/motion/clock events and avoidance
attempt/failure counts. The legacy `blocked_reason` field carries an execution
reason for compatibility, not a BLOCKED control mechanism. Predicted risk and
grid support are recorded in decisions, separately from native outcomes.

L2 is optional. The actual-observation owner initializes and updates the frozen
belief from each real packet, preserving lineage across queries. L4 does not
create a second belief or observe imagined packets. `ForecastCache` retains
actual-belief fingerprints, model version, absolute forecast times, expiry,
visibility changes and observed disagreement checks. The native B2 adapter
requests a bank only at an eligible near-term avoidance attempt. Forecasts
contribute a soft maneuver ranking term within their time support. Missing or
expired forecasts immediately fall back to measured-motion execution.

The native benchmark's full certified static map privilege is unchanged.
`StaticGrid.from_partial` still accepts supplied arrived geometry; neither
adapter adds graph edges or reads hidden humans. Unknown occupancy remains
unknown. Unsupported body clearance is uncertainty, not a reason to suppress
all task actions. Real contacts and success metrics come from native Habitat.

When a template lacks full-body grid support, the already supplied native
center-segment interface can establish best-effort center traversal. This
fallback never claims body clearance, adds graph connectivity, or loads a mesh.
Native prediction must still translate on template forward steps. Repeated
online geometry only considers the fixed local templates and the actual next
motion; it does not search the graph.
