# EventWM human-event collection protocol v2

The model target is `P(human_event | legal_history, fixed_query)`. A human event is native human contact OR the existing research human-disc contact. Static feasibility is a deterministic prerequisite, never a model target. Deployment order is candidate generation → static filter → human-event prediction → planner ranking. Optional planner `SR_human = 1 − P(human_event)` needs no separate collision model.

## Versions and storage

Protocol: `eventwm-human-event-v2`. Checker: `eventwm-static-swept-v2.2`. `scripts/eventwm_v2.py` is the bounded v2 entry point; `eventwm_collect.py collect` defaults to v2. The old overnight/scale production CLIs are disabled. V1 reader/label functions remain available for explicit regression use.

New output must be empty, have its own manifest, and cannot overlap `datasets/raw/eventwm_prod_v1`, including symlink aliases, descendants or ancestors. The bounded runner has no resume/append mode. V1 is read-only legacy/debug/regression data, checkpointed at `4a72eaa` before refactoring.

The manifest records protocol/checker version, query/static configuration, source hashes, commit, root scene/count/profile, completion state and seals. Root query files contain immutable fixed controls, timed trajectory, path, variant and static certificate. The reference-only `primary_index.json` records legal history and query references, root/query/branch IDs, scene/count/profile, visible/lost tracks, history category, timing/speed, durations, human/native/disc/static evidence, event times, validity and canonical binary label. Truth and sensor images stay in the existing root layout; no duplicated raw training dataset is created. `reverse_0`, `reverse_1`, and `truncated_wait` remain audit-only branches.

## Static feasibility: one implementation

`collector.static_feasibility.is_static_feasible(root, query, static_map, geometry, config)` is the sole geometric checker for collectors and future planners. `collector.static_habitat.HabitatStaticWorld` supplies the current simulator adapter; `collector.v2.generate_queries_v2` calls the checker before accepting any fixed variant.

The existing navmesh is built with radius **0.25 m**, height **1.41 m**, including static objects. The human research-disc radius remains **0.25 m** (human radius **0.30 m**). Neither value is silently changed. Actual Spot geometry is articulated and noncircular: folded arm, body and four legs extend beyond the research disc. V2 measures a conservative horizontal circle centered on `base_pos` from transformed bounding-box corners of all robot links at the root posture. The installed Habitat API exposes link scene-node bounds; the supplied Spot URDF uses matching visual and collision meshes/origins. The measured radius and derivation are persisted per root/certificate. No arm, joint, sensor, human-policy or physics parameters change.

Default safety margin: **0.03 m**. Linear sampling distance: at most **0.02 m**; angular increment: at most **0.04 rad**. Required clearance to the already-eroded navmesh edge is `max(0, measured_radius − 0.25) + margin + 0.01 m`; the last term covers half the maximum sample interval. At each interpolated pose the checker also tests the native robot geometry against the known static stage/STATIC objects, with the exact collector ground-support exclusion. The root pose, turns and WAIT are included. The adapter uses `habitat_sim.stage_id` (0 on this build), never an assumed stage ID. Native collision queries restore the robot transform in `finally`, do not step time or sensors, do not draw RNG and ignore human objects. Translation segments must agree with the same static navmesh step-filter used by the controller.

The checker reconstructs the timed trajectory from the fixed controls and rejects any disagreement before geometry checks. A free target alone never suffices. Every compiled temporal variant is checked independently: finite-step turning/translation counts differ at half speed, so no spatial-equivalence shortcut is assumed. Results include pass/fail, rejection reason, minimum navmesh-edge clearance, failed pose/segment, checked pose count, geometry/config and checker version. Missing/incompatible certificates prevent branch execution; immutable query hashes cover certificates.

This is conservative, discretized checking, not a formal continuous-time physics proof. It assumes the supplied static scene/navmesh and root robot posture; moving obstacles, changed robot articulation, different collision assets or changed controllers require a new compatibility assessment. Native static contact after acceptance is a hard failure, not a reason to shrink the footprint or relabel the sample.

## Candidates, N, variants and WAIT

Default **N=8 movement spatial paths**. `QueryConfig` explicitly configures N, maximum 128 candidate attempts, radii `(1.5, 1.0, 0.65)` m, endpoint separation **0.30 m**, and explicit WAIT inclusion. The search starts with N evenly spaced headings, then interleaves rotated headings and configured radii deterministically. It uses only the legal robot pose plus static world, never human state to select targets. Existing shortest-path construction has a 0.25 m projection bound and 4 m path-length bound.

Every candidate compiles all four existing variants: immediate normal, wait 0.5 normal, wait 1.0 normal, immediate half speed. Accept a spatial path only when all variants pass. Rejections and near duplicates consume the candidate-attempt budget but never become branches. Search continues until N diverse accepted endpoints exist. Exhaustion raises `insufficient_static_feasible_queries`; partial candidate diagnostics persist, but no movement/WAIT training branches execute. There is no WAIT-only fallback.

The separate WAIT query lasts as long as the longest accepted movement variant and undergoes the same checker. The non-Castle smoke fixture is an explicit recorded prefix selected from legal robot pose and the known static map; it turns toward +z and advances 0.8 m into open space after the original prefix. The existing prefix replay API and history semantics are unchanged. The original close-to-wall root remains an archived rejected fixture.

With WAIT enabled, a completed root has `4*N+1` primary branches, plus three existing audit/repeat branches. A bounded smoke may explicitly use a smaller N; this does not change the default contract.

## Human-event labels

`collector.v2.human_event_labels` uses unchanged collector collision declarations. Precedence:

1. Any static collision, including mixed static/human: `human_event_valid=false`, `human_event_label=null`, `static_feasibility_violation=true`. Quarantine and stop collection.
2. Missing accepted certificate, zero-human positive inconsistency or collection error: invalid/null.
3. Native-human OR research-disc event with no static violation: valid label **1**, including early human-event termination.
4. No human/static event: valid label **0** only if execution completed with `query_completed`, every fixed control was issued, and observed horizon agrees with planned horizon within the unchanged native-clock tolerance **0.008001 s**.
5. Any remaining incomplete/ambiguous/nonhuman execution termination: invalid/null.

Invalid is a validity mask, not a third model class. Legacy `completed`, `censored`, and `full_query_safe` remain diagnostic fields; they do not define the v2 target. A zero-human accepted branch cannot be positive. Zero-human accepted queries must have zero static contacts.

## Legal boundary and unchanged components

Future model inputs still use `core.load_wm_input`, which whitelists arrived legal history, nested legal tracks, and the fixed query. Static certificates and target labels are not forwarded by that loader. Track IDs remain bookkeeping. Hidden human positions, future state, goals, GT velocities and controller internals stay in privileged audit/target files. Known static geometry is allowable deterministic context, not a prediction target.

The human observer/tracker, visibility and lost memory, prefix selection, native timing, seeded branch replay, external human dynamics, controller, simulator, sensor model and declared collision semantics remain unchanged. Root completion, seals and source provenance are retained. Unexpected static contact halts further smoke/scale work and requires diagnosis; post-hoc filtering is not a remedy.
