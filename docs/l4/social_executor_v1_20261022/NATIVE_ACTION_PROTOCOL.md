# Verified native action protocol

Authority is the code at cf197122, not a paper's nominal speed. The configured
robot actions are `agent_0_discrete_move_forward`, `agent_0_discrete_turn_left`,
`agent_0_discrete_turn_right`, and `agent_0_discrete_stop`. Their public adapter
names are forward/left/right/stop. Pause and backward implementations exist but
are commented out in the frozen configuration. L4 cannot emit either action.

Forward integrates 25/120 m (0.208333 m) along local +X. Left/right integrate
±10/120 rad (±4.77465 degrees) about +Y. World pose is XYZ plus signed yaw;
positive yaw points from +X toward -Z. Native static sliding and height
projection can alter translation. The executor interrupts on an unpredicted
horizontal discrepancy over 4 cm; the precise fallback uses the existing
native KinematicModel. It never changes the configured motion law.

An action advances 48 or 56 ms in the observed build. Robot control integration
uses 120 Hz, simulator stepping uses four substeps, and the task additionally
calls `step_physics(1/60)`. Physics accumulation makes the 48 ms nominal clock
inexact. The smoke test observed both durations. L4 screens with 56 ms action
timing and reports only actual native timestamps. Human-motion screening is
conditional on its motion model and timing approximation, not an unconditional
guarantee under arbitrary acceleration.

Update order: robot task action, active human task actions, task physics step,
simulator step/articulation update, sensor observations, task active-state
evaluation, environment measurement updates and step-limit accounting. The
L4 runner then acquires the legal structured packet, checks native outcome
labels and supplies the actual transition. Optional L2 observation update
occurs on that actual packet before obtaining/reusing a forecast for the next
decision. No imagined packet enters this path.

STOP sets `is_stop_called` and `should_end`, and the environment still performs
the transition before ending. STOP is never a wait. The native episode cap is
500 actions; human collision ends the episode, while static contact alone does
not. Query completion is ordered route arrival within 0.155 m plus terminal yaw
within half a native turn. Official navigation success additionally requires
native STOP inside the unchanged 0.2 m goal threshold. A blocked query remains
failed even when an experiment driver ends the episode to hand control back.

Robot geometry follows the established audited 756-frame gait envelope:
local X [-0.646495, 0.584202] m and local Z [-0.242846, 0.238225] m. The fast
checker covers this rectangle with supplied .25 m configuration-space disks,
using ≤.08 m longitudinal spacing and ≤.005 rad rotational samples. UNKNOWN
support is rejected. These are conditional 2.5D certificates and can be
conservative near boundaries. Precise fallback uses native static triangles,
individual link boxes, actual robot joint measurements, known current/next
gait phases, and 25 mm margin. A 5 mm development margin missed a measured native
arm contact and was rejected; the retained replay verifies this regression.
This does not alter Stage 0 certification or add
graph edges. Floor-support exclusion follows the prior audited mesh checker.

Legal human measurements are exact world base centers gated by the articulated
jaw camera's frustum, range, static occlusion and current human-mesh occlusion.
No artificial noise is added. The existing anonymous tracker estimates velocity
from actual measurements and preserves last-seen positions for up to two
seconds. Stale positions are labeled and aged; physical human IDs and goals are
not provided to L4. The human .616 m enclosing disk is the existing audited
all-asset envelope, separate from the benchmark's center-distance PSC metric.

PSC is the unchanged native `PersonalSpaceCompliance`: fraction of post-action
steps for which every human center is at least 1 m away by native geodesic
distance, with the installed Euclidean fallback when disconnected. L4's
Euclidean proximity cost is explicitly a conservative planning surrogate;
reported PSC comes from the exact evaluator. SPL and STL also use the original
measure classes. In particular STL's historical distance/(0.25/10) reference
is retained even though it differs from measured execution speed.
