# eventwm.l25l4.contract.v1

> Historical runtime semantics, superseded by the
> [progress-first contract](../progress_first_20261023/EXECUTION_CONTRACT.md).
> The Python API remains compatible; ordinary BLOCKED/search handoffs are removed.

Canonical implementation: `eventwm_graph.l4.reference`. L2.5 and L3 import this
compiler; they must not implement an alternative nominal route schedule. No
L2.5 service is required for an L4 action.

`compile_reference(QuerySpec, legal_graph, ActionSpec) -> CompiledReference`
requires a stable query ID, ordered world-XYZ reference route including the
actual start, actual start pose `[X,Y,Z,yaw]`, terminal yaw, start timestamp,
completion tolerances, action budget and elapsed-time budget. The route and
query are immutable tuples. Yaw zero faces +X; positive yaw faces toward -Z.
Angles are radians; positions are metres; times are absolute native seconds.

The compiler never recomputes a supplied route or adds graph connectivity.
`from_stage1` converts the existing Stage 1 compiled route and final yaw.
Nominal schedules use forward 25/120 m and turns ±10/120 rad. Their timestamps
use 48 ms nominal steps. Actual 48–56 ms transitions supersede that schedule.
The supplied static graph must expose segment and body-motion checks; the
native profile also exposes the installed kinematic predictor. A nominal
schedule may need avoidance: `nominal_static_supported=False` is explicit and
does not authorize executing it. Every actual L4 candidate is screened anew.

The compiled record contains the original QuerySpec, nominal waypoints, native
action schedule, timestamped poses, graph version, compiler version and a hash
binding all those values, the action specification and static-support flag.
The QuerySpec's own hash binds only its immutable high-level semantics.
`CompiledReference.l2_segment(first,count)` deterministically exposes a bounded
fixed-query segment compatible with L2's Stage 1 observation-prediction API.
Its parent hash and absolute nominal times preserve provenance. Such a segment
is a nominal query, never an actual observation packet.

Compilation failure raises before any execution handle exists. The native
benchmark reports this pre-execution outcome as `REFERENCE_UNSUPPORTED`, retains
the reason and counts the episode as failed. It is not an additional Executor
status, query completion, a wait, or permission to repair graph connectivity.

`Executor.start(query, compiled_reference, RuntimeContext)` returns a handle.
`act(handle, live_state, actual_observations, forecast_view)` returns a native
action or an explicit terminal/handoff decision. Repeating `act` before an
actual transition returns the same pending decision. `on_transition` must
acknowledge that action with a strictly later timestamp, actual pose, durable
observation reference, collisions and native termination/truncation flags.
Unsolicited, mismatched and zero-time transitions are rejected. A query can
complete without ending the episode. L3 alone decides whether an episode goal
has been reached and STOP is appropriate. The benchmark driver explicitly
uses STOP at completion or control handoff and records it outside query feedback.

Execution feedback reports actual actions, timestamps (including the initial
state), poses, duration, projected route progress, route deviation, status,
blocked reason, observation references, episode flags, path length, detours,
rejoins and mode transitions. Progress is recomputed from actual poses and can
decrease. Route identity and total execution budgets never reset on a mode
change. Turning toward a required terminal yaw is meaningful progress.

Modes: FOLLOW / YIELD / AVOID / REJOIN. Statuses: RUNNING / QUERY_COMPLETED /
BLOCKED / EXECUTOR_LIMIT / INTERRUPTED. BLOCKED means bounded search or progress
monitor failure, not proof that no physical solution exists. `None` advances
neither simulator time nor belief. It requests L3 control handoff.

`BeliefSnapshot` binds belief/model/graph versions, timestamp, an actual belief
reference and support metadata. The owning observation path calls L2 `observe`
once for each actual packet. L4 creates no competing persistent belief.
`ForecastCache.get(actual_belief, actual_packet)` shares one forecast across all
candidate evaluations, refreshing on visibility change, >0.35 m observed
disagreement or its 0.24 s validity deadline. Exact current measurements always
override the planning role of predictions. Prediction failure leaves the
owner's actual belief untouched and invokes observation-based checks.

`ForecastView` binds source belief fingerprint, model version, prediction start,
absolute future times, the held joint future, validity deadline and support
metadata. Actual-source lineage and time support are validated. No probability
is extrapolated beyond the held tape. L2 sampled worlds provide a soft proximity
cost only; zero occupancy cannot clear an observation-based collision rejection.
Unknown humans remain unknown. Missing tracks in a finite sampled forecast are
not evidence that the environment is empty.

The native runner uses an explicitly declared full static-map profile for
B0/B1/B2, with legal frustum human measurements. `StaticGrid.from_partial` also
supports supplied observed certificates and rejects UNKNOWN cells. Neither
profile permits online hidden human states, IDs, goals or future tapes. Falcon
B3 uses its depth/PointGoal observation interface and does not share the
explicit map advantage; benchmark tables retain this qualification.
