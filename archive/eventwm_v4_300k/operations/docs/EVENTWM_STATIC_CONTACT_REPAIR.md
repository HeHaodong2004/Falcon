# Static contact certification repair, 11 October 2026

Scientific release: `research/eventwm-v4-static-contact-repair-20261011`,
commit `eac1c27813a753d26fc1756671d8d0bd198a65c6`, based on frozen
`2e792aa26f677e4d6b83bbbefebb733c0efa8e63`. The original scientific checkout
and commit remain unchanged. Operational supervision remains based on `b266128`.

## Confirmed invariant violation

Root `v4g2_2cc61907376608bec99f`, branch `b007`, query `q_4c518149379d`
was statically certified yet encountered a classified static obstacle contact
at relative native time 3.960 s, control 81. Both isolated fresh and COW execution
reproduce the production failure. The contact is robot object 1, Spot foot
link 19, against stage object 0/link -1. Its normal is
`[-0.4026223123, 0.7997871041, 0.4452369511]`, distance `0.0074324477`,
normal force zero, and native contact active. Normal Y falls below the unchanged
0.8 support-contact cutoff. No contacts are discarded or reclassified.

Certified and executed base matrices, positions, yaw, every joint position,
joint velocity and root velocity match exactly at that control. Independent
checkpoint probes demonstrate that one Bullet collision-detection pass misses
the contact; the third pass at the same pose reports the exact native contact.
Physics stepping also reports it, without any articulated posture change.
The failure is history-dependent persistent Bullet contact-manifold refresh,
not a different planned path, footprint omission, changed controller or OOM.
The violated invariant is: an accepted static certificate must exclude every
static obstacle contact observable during native execution of those controls.

## Minimal repair

Only `collector/static_habitat.py` and the checker version in
`collector/static_feasibility.py` change scientific execution. The native-float
static replay now checks each repeated collision-detection pass across the
control-pose refresh cadence: `ceil(control_dt / physics_time_step) + 1 +
ac_freq_ratio`. It advances no physics, clock, RNG, sensor or human policy,
restores the original matrix, and retains the existing contact predicate.
Checker version is `eventwm-static-swept-v2.4-native-contact-refresh`.
SR definitions, legal observations, query controls and role splits are unchanged.
A native-binding-free unit regression verifies later-refresh rejection, valid
support acceptance, propagated detector errors and restored poses.

The original query is rejected at segment 81 / collision pass 2. The original
root cannot produce eight statically valid paths and returns the existing
`InsufficientStaticQueries` rejection. The unchanged pipeline maps this to
`insufficient_static_feasible_queries` and continues other roots. Branch-level
unexpected integrity violations remain collector-fatal and guard-blocking.

## Durable epoch transition

`recover_static_epoch.py` accepts only the diagnosed previous science commit,
these two changed scientific files and the one added regression test. It backs
up SQLite and the manifest before a stopped, locked transaction. It publishes
already validated old-epoch work with its original implementation/provenance.
Completed root records remain byte-identical. Every interrupted, unsealed root
retains all prior attempt paths and receipts; its old attempt is marked failed
with `scientific_epoch_transition` and it is explicitly scheduled once as
pending in the new epoch. The normal maximum attempt count remains two.
This also exercises the offending root even though its two old attempts were
exhausted. No sealed root is replayed, replaced or regenerated.

Production uses the same SQLite checkpoint/output/archive and C24 native limit.
The guard is restarted with the new source pin, remains bounded to three
restarts/hour, and blocks any later scientific/data integrity failure.

## Evidence

Diagnostic artifacts: `/tmp/eventwm_static_contact_repair_20261011/`.
`original/{cow,fresh}_result.json` reproduces the static collision;
`probe/{single,repeated,flush,physics}.json` isolates detector history;
`probe/probe.json` contains the repaired rejection and exact matrix comparison.
`native_gate/VERDICT.json` records bounded native validation;
`sealed_before.json` records all 3866 original completed roots and IDs;
`production_observation/RESULT.json` records actual publication observation.

The completed release gate and production verification are recorded in
`docs/release_evidence/static_contact_repair_gate.json` and the production
observation artifacts. Production restoration requires five minutes of native
execution with increasing unique retained publication, not merely live helpers.
