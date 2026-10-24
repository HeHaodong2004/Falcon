# Safety and progress validation

The executor is a runnable research implementation. Its broader navigation and
visibility failures prevent an unconditional safety or RL-readiness claim.
Zero observed collisions in a small suite is reported with its denominator.

## Protocol and reference checks

Nineteen automated tests pass on the frozen release source. They cover immutable
and deterministic reference compilation, actual transition accounting, ordered
route completion, terminal orientation, negative projected progress, hairpin
ordering, action acknowledgment, native clock bounds, stale-current measurement
rejection, collision precedence over success, bounded execution, temporal
crossings, static swept motion, the enclosing-region optimization, and actual
frozen-L2 forecast lineage/cache behavior. These are new L4 tests; the source
release's 399 historical regressions are not presented as newly rerun tests.

The separate offline trace auditor checks legal actions, strictly advancing
48–56 ms timestamps, STOP placement, literal feedback/trace equality, current
measurement timestamps, ordered actual waypoint visits and terminal pose. It
checks successful native episodes against the unchanged STOP/0.2 m rule.
`artifacts/trace_audit.json` records the completed audit counts.

The native arm-contact regression retains its actual contact witness. The
original 5 mm development margin accepted the action. The release's restored
25 mm margin rejects it. Native reruns with a checked forward-before-turn
escape complete the same static-detour query without contact. This repair does
not establish that the mesh approximation covers all possible contact geometry.

## Fixed and randomized fixtures

The fixed analytic suite was declared before its acceptance runs and retains
all twelve cases: straight, corner, multiple waypoints, orientation, head-on,
crossing, slow leader, temporary block, narrow yielding, limited visibility,
reobservation and multiple humans. All 12/12 complete; human collisions are
0/12. These fixtures use verified native ego increments but an analytic human
disk of 0.30 m and idealized static support. They are not Habitat episodes or
native collision-rate evidence. The planner itself retains its larger 0.616 m
human envelope. Offline interpolation checks human collision truth separately.

The 64 speed-varied analytic trials complete 61/64, with 0/64 human collisions.
The three failures are slow-leader seeds 9010, 9026 and 9050, at speed factors
0.819733, 0.781553 and 0.766340. Each returns BLOCKED for lack of meaningful
progress. They remain in the denominator. Only human speed is randomized here;
this is not a comprehensive density/occlusion/scene robustness study. Repeated
trials of eight geometries are not 64 independent scenes.

## Frozen native scenarios

The eight inherited native cases and seeds are unchanged in
`native_scenarios.json`. Both B1 and B2 have 7/8 query completions and native
successes, 0/8 human collisions, and 0/8 static contacts. All seven cases other
than the declared permanently blocked passage complete. B0 has 3/8 native
successes, 5/8 human collisions and 1/8 static-contact episodes. The B0 static
contact does not invalidate the native evaluator's success flag; it invalidates
a collision-free-navigation claim.

| Native case | B1 | B2 | B1 native actions, including driver STOP |
|---|---|---|---:|
| Empty | Completed | Completed | 32 |
| Head-on | Completed | Completed | 66 |
| Crossing | Completed | Completed | 50 |
| Stationary blocker with passing room | Completed | Completed | 90 |
| Multiple humans | Completed | Completed | 89 |
| Static detour | Completed | Completed | 81 |
| Permanently blocked passage | BLOCKED | BLOCKED | 26 |
| Occlusion/reappearance intention | Completed | Completed | 32 |

The blocked case places a stationary human in a narrow route. No independent
collision-free native witness was established for it. Bounded search failure
does not prove physical infeasibility. It remains a failed query and a failed
navigation episode. It is not counted as successful yielding.

The named occlusion/reappearance case did **not** realize a confirmed same-track
0.5–2 s disappearance/reobservation in the retained observations. It also has
PSC 0.75. Its name is an intention, not evidence of coverage. A controlled
0.85 s reobservation gap is exercised by the analytic fixture and shown in its
explicitly labeled figure. Native slow-leader, temporary-block and calibrated
occlusion-duration coverage remain insufficient for the complete requested
basic acceptance claim.

## Rejoining, budgets and failure handling

On the native scenarios, B1 and B2 each record four detours and four rejoins
(head-on, stationary human, multiple humans and static detour). Their completed
queries add 0.334 m / 0.393 m mean distance and 1.382 s / 1.341 s mean query time
relative to the nominal schedule. Turning and yielding time is included. The
original route and goal are never replaced.

Both use a 480-action, 26.88 s query cap within the unchanged 500-action episode
limit, plus a five-second meaningful-progress watchdog. The analytic cap is
300 actions / 16.8 s. Modes do not reset these budgets. Actual ordered route
projection can decrease during escape. New best endpoint distance and required
terminal orientation count as progress; repeated reversals do not.

The native suite still has 60 adjacent turn reversals for B1 and 46 for B2;
both have a maximum 37-action turn run. Some are legitimate orientation work,
but remaining repeated-turn behavior is a real limitation. The public panel
contains stronger deadlocks and a collision without visible human tracks.
The progress watchdog reports failure; it does not solve these failures.

There is no native WAIT or BACKWARD in this configuration. Checked turns are
time-advancing motions with changing visibility, not intrinsically safe waits.
No admissible action returns BLOCKED/None and hands control back; it does not
advance a fictional clock. The experiment driver then explicitly STOPs the
episode, outside the query's action ledger. Failed queries never count as
successful navigation merely because no collision occurred.

## Scope of collision checks

Static disk-support checks are deterministic only under the supplied certified
surface, footprint, floor-height and swept-cover assumptions. Native mesh/FK
checks additionally depend on asset triangles, observed arm posture, gait phase,
support exclusions and contact margin. They do not certify arbitrary slopes,
overlapping floors, noisy arm motion or all native collision approximations.
No graph edge is fabricated to escape a rejection.

Human checking uses exact legal measured centers, short-term measured velocity,
track age, relative swept motion and explicit uncertainty margins. Those margins
are engineering heuristics. Arbitrary acceleration, unknown humans and visibility
gaps remain uncertain. L2 supplies a soft forecast cost and cannot clear an
immediate observation-based rejection. Small or zero sampled occupancy is not
a safety guarantee. Reported PSC is the native evaluator, whereas the search's
Euclidean distance penalty is a documented planning surrogate.

`BASIC_AVOIDANCE_PASS` is **not established for the full requested native
coverage**. Fixed analytic correctness and seven positive native demonstrations
are narrower, useful results. Broader robustness and progress require revision.
