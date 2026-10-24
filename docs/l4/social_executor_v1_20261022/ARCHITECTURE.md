# Query-conditioned native execution

> Historical conservative architecture. The current executor is described in
> the [progress-first contract](../progress_first_20261023/EXECUTION_CONTRACT.md).

The implementation separates immutable reference compilation, actual-pose
route tracking, candidate generation, collision screening, action selection,
and actual feedback. `Executor` imports no live simulator or hidden-human API.
The native adapter is the only environment boundary. L3 supplies the selected
query and retains ownership of high-level goals; L4 never substitutes one.

The normal control path first tests a complete turn-then-forward route/rejoin
proposal. If it is unsafe, a 16-wide beam searches up to 32 native actions.
First-action diversity prevents immediate forward-only pruning. An additional
36 fixed native turn/advance/rejoin templates retain sustained lateral maneuvers
that short-term beam costs can otherwise remove. Only the first action executes.
Every proposal uses the same swept static and time-aligned human checks. A
shorter plan can end on arrival; it does not pad with fictional waiting.

Known immediate physical overlap is rejected before progress, social proximity,
time, deviation and oscillation costs are considered. The human checker uses
oriented robot rectangle versus the audited human disk at time-aligned
subintervals, with an enclosing interpolation margin. Exact currently measured
centers, estimated velocities and last-seen ages remain separate. Uncertainty
growth and constant-velocity assumptions are engineering risk controls, not
calibrated probabilities or guarantees for arbitrary human behavior.

The optional frozen L2 provides a shared 1.92 s, 16-world forecast. It ranks
candidates through a soft proximity term and cannot override current-observation
collision rejection. Forecast reuse has explicit source lineage, absolute time
support, 0.24 s validity, visibility-change and observed-disagreement triggers.
Initialization, actual belief updates and refresh latency are reported separately.
The actual-observation owner retains belief across prediction failures.

The fast static checker consumes a supplied configuration grid. The benchmark
profile prepares it from full certified static geometry; the partial-graph
adapter consumes only arrived observed certificates. These profiles are not
represented as equivalent privileges. An isolated native mesh/FK fallback
handles support-square rejection and height changes using bounded native
proposals. Its latency and residual false blocks are reported separately.
The mesh fallback retains 19 fixed proposals, including translating before a
turn and a four-turn escape. These repair a constrained native starting pose
without changing the reference. The restored 25 mm margin rejects a retained
native arm-contact witness that a 5 mm development margin missed.

Route progress is a projection onto visited ordered legs and the current leg,
so a nearby later hairpin cannot create false progress. Actual path length is
distinct from projected progress; retreat can reduce progress. Completion
requires ordered arrival and orientation. No-progress monitoring recognizes
new best route progress, new best endpoint distance during rejoining, and
required terminal turns. It does not renew on repeated back-and-forth motion.
The five-second progress bound is independent of total query action/time caps.

FOLLOW/YIELD/AVOID/REJOIN are control modes, separate from termination states.
Avoidance side preference and inverse-turn cost reduce oscillation. Yielding
consists only of screened native turns. If no action is found, BLOCKED requests
control handoff with no implied clock advance. An interrupted or budget-limited
query cannot report completion.

The mesh visibility, link collision and legal ego-FK helpers are reused from
the read-only 2026-10-20 L4 research worktree (`dd071107`). They are copied into
this branch, with the visibility import localized; the gait archive retains
SHA256 `d3ea9f1bb0dde811a217484453d5e525fcf6a3fe9cc26488249e1319217f7142`.
No old L4 human policy or old L2 model is used. The authoritative starting point
for this branch remains cf197122 and the frozen 500k L2 checkpoint is unchanged.
