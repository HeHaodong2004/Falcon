# L2.5 ↔ L3/L4 integration contract

Conceptual shared contract: `eventwm.l25l4.contract.v1`.
Local packet schema/compiler: `l25-query-outcomes-1`.
This branch does not change any shared file in the concurrent L4 worktree.

## Immediate callable interface

```python
from eventwm_graph.l25 import QueryOutcomeCompiler, CompiledReference, ComputeBudget

# actual_belief comes from frozen L2 initialize/observe, never an imagined branch.
compiler = QueryOutcomeCompiler(world_model)
context = compiler.build_context(
    actual_belief, actual_belief.graph, actual_belief.robot_pose, public_goal_xyz)

# L3 proposes references. L4's canonical compiler will ultimately own these.
# Existing Stage 0/1 native compiled query dictionaries work today.
references = [CompiledReference.from_native(q, context.timestamp) for q in queries]
packet = compiler.evaluate_candidates(context, references, ComputeBudget())
query_nodes = packet.graph['query_node_indices']
legal_candidates = packet.graph['candidate_validity']
```

No compiled reference or future bank is needed for Stage A. L3 may propose a
different legal candidate set on every invocation. The package has no permanent
action catalog, query generator or selector. The integration consumer only
demonstrates one logit per current candidate.

## Field mapping and ownership

| Shared concept | Current mapping / owner |
|---|---|
| `QuerySpec.query_id` | Stable unique string; supplied by L3/L4; never positional identity |
| `reference_route`, `target`, terminal orientation | Explicit query fields, native `compiled.route`, final waypoint and `final_yaw`; absent data stays null |
| Nominal timing | Native 48 ms/action; optional `compiled.nominal_seconds` must agree; actual timestamps remain authoritative |
| Completion rule | Explicit rule or native compiler position/yaw tolerances, labeled as compiler tolerances, not an L4 success model |
| Execution limit | Explicit input if supplied, otherwise null; no invented L4 timeout |
| Reference hash | Supplied canonical hash if present, otherwise adapter content digest; independent `content_hash` always binds consumed geometry/actions/tolerances/revision |
| `CompiledReference` | Existing Stage 0/1 poses/actions viewed by the adapter; no competing permanent compiler |
| `BeliefSnapshot` | Immutable actual L2 `Belief`; fingerprint/model/graph namespace/timestamp and support metadata in `ContextGraph` |
| `ForecastView` | L2 shared bank retained internally; packet exports source fingerprint, model, absolute forecast times, bank identity and diagnostics |
| `ExecutionFeedback` | Produced/owned by L4; not a forecast label and not consumed as fabricated belief evidence here |

Robot poses are XYZ meters plus yaw radians; humans are horizontal XZ meters.
Reference timestamps, forecast timestamps and actual observation times share the
same absolute simulation clock. Yaw forward is `[cos(yaw), -sin(yaw)]` in XZ.
The local adapter accepts `known_space_discrete_route_follower_1`; other compiler
versions currently raise an error instead of pretending to have native timing.

L3/L4 must retain the original compiled reference by ID/hash. A DecisionPacket
is an outcome representation, not a self-contained executable action plan.
Reference-route points and ordered graph occurrences are exported; the native
action schedule/nominal pose buffer lives in the separately held reference.

## L4 integration requirements

1. Supply a canonical reference with matching start ego, actual timestamp,
   arrived graph revision, ordered route, target and final yaw. Preserve the
   canonical hash and compiler version.
2. Implement a small explicit adapter from the final L4 reference to L2.5's
   consumed pose/time schedule. Native `.048` timing is a temporary boundary,
   not permission to round arbitrary L4 timestamps onto this clock. The frozen
   L2 optical API also requires exact stored native endpoints.
3. Continue executing collision avoidance independently of L2.5. No real-time
   L4 safety dependency on this compiler is introduced.
4. Return actual actions/timestamps/poses/duration, route progress, executor
   profile, query status, blocked reason, actual observation references and
   episode termination/truncation. Actual observations update the retained actual
   L2 belief with `observe`; an imagined branch never replaces that belief.
5. Leave `p_l4_query_completion` unsupported until L4-consistent execution
   modeling or matched data exist. Old Stage 1 follower labels do not establish
   this probability for the new executor.

Observed native tapes sometimes include a 56 ms interval after nominal 48 ms
steps. The frozen reference compiler already states that actual timestamps are
authoritative. The exact-clock scientific audit censors misaligned labels;
auxiliary interpolated targets are separately named. Final L4 timing/action
semantics need resolution rather than assuming exact native-clock execution.

## L3 integration requirements

- Generate candidates using the cheap arrived context; compile them upstream.
  Respect the 24-candidate explicit batch limit or deliberately schedule batches.
- Read support masks and `candidate_validity`; unsupported placeholders are
  finite zeros in graph tensors, not zero probabilities. Raw unsupported values
  are null. Preserve query-node/ID mapping and batch pointers.
- Interpret collision as the declared enclosing-body native-endpoint model
  statistic over `covered_duration`. Unsupported tails remain censored. Official
  contact and geodesic PSC are unsupported. Separate nominal risk, L4 completion
  and full episode success.
- Treat conditional nodes as imagined features, not executable root candidates
  or automatic Bellman targets. Read represented/residual model mass separately
  from unknown support. A horizon boundary is not a new execution decision point.
- Rebuild context after every actual observation update. A compiler instance
  rejects stale active contexts. Serialize access to a single stream's compiler;
  use separate instances for independent actual streams.

## Unresolved cross-branch decisions

The final L4 compiler type/import path, canonical hash algorithm, timing adapter,
executor profile ID, completion/status enumeration and `ExecutionFeedback`
serialization remain owned by L4 and require a later integration change. Static
sensor occlusion, identity-specific reobservation, conditional legal exact
measurements, new static revelation and calibrated completion models remain
unsupported by this L2 release. No existing shared contract was silently changed.

`DecisionPacket.save` serializes portable finite summaries/tensors, with exact
model/belief/graph/reference identities. Finite successor carriers remain opaque
in-process handles. Saved JSON cannot be restored as an operable actual belief
or executable successor; durable actual belief uses the existing L2 serializer.
