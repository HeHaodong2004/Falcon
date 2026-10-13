# EventWM social-outcome protocol v3

Protocol `eventwm-social-outcome-v3` predicts the categorical distribution
`P(SAFE_SUCCESS, SOCIAL_VIOLATION_SUCCESS, HUMAN_COLLISION | legal_history, fixed_query)`.
There is no binary SR target, fourth failure class, training change, or planner change.
Possible future planner quantities are `P(any social risk)=P(1)+P(2)`, `P(physical human collision)=P(2)`, and `P(fully safe success)=P(0)`.

## Classes and precedence

`collector.v3.SocialOutcome` defines the enum; `social_outcome_labels` is the only canonical label function. Every row stores validity, integer class and enum name.

| Class | Name | Requirement |
|---|---|---|
| 0 | SAFE_SUCCESS | Full fixed query completed normally, no native human contact, no PSC violation, no static contact |
| 1 | SOCIAL_VIOLATION_SUCCESS | Full fixed query completed normally, at least one PSC violation, no native human contact, no static contact |
| 2 | HUMAN_COLLISION | Native articulated robot–active-human contact, no static contact; early contact termination is a determined outcome |

Static contact overrides **everything**: invalid/null plus `static_feasibility_violation=true`, quarantine and stop. Otherwise physical human contact overrides social violation, which overrides safe success. A premature nonhuman termination remains invalid even if personal space was entered. Collection errors and missing/ambiguous social evidence cannot become successful labels. A native contact can determine class 2 without a complete subsequent social future. Missing accepted precheck or wrong protocol is invalid. Zero-human class 1/2 evidence is an integrity error.

Completion reuses the existing v2 horizon checks: every requested control issued in order, `completed=true`, `termination_reason=query_completed`, and observed duration within the unchanged query clock tolerance (0.008001 s). Query completion is not benchmark navigation-goal success.

## Physical contact and research discs

Physical detector: `collector.runtime.Recorder.capture`, consistent with FALCON `DidMultiAgentsCollide` / `HumanCollision` in `falcon/additional_metric.py`. Discrete native contact points connect Spot's articulated object to an **active** humanoid object. This uses actual collision meshes and native contact semantics, not an invented radius or contact-force threshold. The inherited recorder samples every native physics callback. The benchmark human-collision measure is episode-cumulative and ends the task; the collector retains the first native contact time relative to the query root.

`research_disc_collision` is retained separately: horizontal base-center distance minus robot radius **0.25 m** minus human radius **0.30 m** is `<=0`. This legacy proxy is neither canonical PSC nor demonstrated physical mesh contact. **It does not independently cause class 2 in v3.** A full query with disc overlap but no native contact is classified by PSC; an incomplete such query without native contact is invalid. The v2 binary definition remains unchanged in v2 data/readers.

## Canonical social-distance definition

Source: `falcon/additional_metric.py::PersonalSpaceCompliance.update_metric`, with `PersonalSpaceComplianceMeasurementConfig.use_geo_distance=True` (the configured FALCON default; no override in the smoke configuration).

For each active human, use `sim.get_agent_state(i).position`. In this simulator (`falcon/rearrange_sim_v2.py::get_agent_state`) this is the articulated agent's `base_pos`.

1. Ask the **live simulator pathfinder** whether a shortest path exists between robot and human centers.
2. If found, use `sim.geodesic_distance(robot_position, human_position)`.
3. Otherwise, use the full 3D Euclidean center distance, exactly as FALCON does.
4. Violation iff **distance < 1.0 m**. Equality is compliant.

No robot/human radii are subtracted. It is a center-distance definition with navmesh topology, not a surface-clearance definition. The navmesh is built for a 0.25 m agent; that erosion influences geodesics but is not an extra personal-space radius. Around walls the boundary need not be circular. Disconnected/no-path pairs use the benchmark's Euclidean fallback, even across obstacles; this limitation is preserved rather than silently corrected.

`collector.social_habitat.capture_social_metric` records these exact operations as privileged evidence. `collector.v3.social_violation` applies the strict predicate. The benchmark reports compliant steps divided by total steps. **V3 instead takes any violation over the query**, sampled at the existing native recorder timestamps, including the root. Thus v3 reuses the benchmark's instantaneous geometry/threshold but is not the episode PSC score or its control-step aggregation. Finite native sampling is not a continuous-time guarantee. Metric version: `falcon-psc-geodesic-fallback-lt1-native-samples-v1`.

## Continuous/raw diagnostics

Each native truth row retains contacts, robot/human state and `social_metric={version, distances:[{human_slot,distance,mode}]}`. Mode is `geodesic` or `euclidean_no_path`. Only privileged truth/audits contain these fields.

Labels retain native/direct-human collision, research-disc collision, social violation, static collision, termination, query completion, observed/planned duration, first physical/social event times, minimum 3D human center distance, minimum research-disc clearance, minimum social distance and social clearance (`PSC distance − 1.0m`). `min_human_clearance` is explicitly documented as the legacy horizontal **disc proxy**, not an exact articulated mesh surface distance. Exact mesh separation is unavailable; native contact points remain the physical evidence.

Social duration uses the existing **right-endpoint piecewise-constant native-sample convention**: sum each elapsed interval when any human violates at its right endpoint. Violation count sums outside→inside entries separately per stable active-human slot, counting an initially inside human once. Minimums/times are null for no humans; duration/count are zero. Missing social evidence is marked unknown, never silently safe.

## Unchanged static contract and fixed queries

Checker remains **eventwm-static-swept-v2.2**, byte-identical to validated v2. Measured conservative Spot footprint (~0.600 m), safety margin 0.03 m, 0.02 m linear / 0.04 rad angular sampling, half-spacing allowance, native stage/STATIC geometry tests, and controller/navmesh agreement are unchanged. Static contact is not any supervised class. No accepted mixed human/static branch is retained as valid.

`generate_queries_v3` calls `generate_queries_v2` and only versions the query envelope/hash/IDs. It does not repeat candidate generation or feasibility logic. Default N=8 feasible movement paths; four original wait/speed variants per path; separate WAIT; bounded resampling/diversity unchanged. Smoke explicitly requests N=3. Human truth never influences query choice.

## Legal boundary and storage

Legal input remains arrived structured history, legally observed visible/lost tracks and fixed controls/query through `collector.core.load_wm_input`. Its nested allowlists exclude social metrics/classes, future states, hidden humans, GT goals/velocity, controller state and certificates. No neural model is built.

Root layout/seals/replays remain unchanged. New manifest and query envelopes use schema version 3 / v3 protocol. Labels use `social_outcome_valid`, `social_outcome_class`, `social_outcome_name`; no v2 binary target is published on v3 label rows. General legacy diagnostics (`completed`, `censored`, `human_collision_event` etc.) remain documented raw/history fields, not v3 targets. `primary_index.json` references completed sealed roots and excludes reverse/truncated audits. Saved labels are recomputed from truth before indexing.

V3 does not persist RGB/depth archives: the structured loader uses no pixels. It returns `history_images=null` when explicit retention metadata says images were omitted. Sensor acquisition, legal history, observer behavior and image-based replay fingerprints still run unchanged. Removing sensor acquisition or changing replay hashes is **not** authorized here. Native truth, controls, legal structured metadata, seeds/configs and image hashes support audits and later deterministic replay; exact pixel archives require a new explicit retention option, not a hidden production default.

Visual generation defaults **OFF**. `--debug-visuals` opts into at most four native static-ray references and one composite PNG per selected root (up to six branch examples in that composite). No per-step PNGs, branch images, videos or matplotlib imports in the collection path. Debug assets are not model input. Offline outcome-based example selection is permitted for display only, never collection/sampling.

## Versioning and migration

Validated v2 is preserved at commit `36e2872d09cced67050be9776770550906aeaf65`, branch `research/eventwm-human-event-v2`. V1 and all v2 datasets are read-only regression data; their labels are never migrated in place. V3 output guards reject legacy directory names, resolved symlink aliases, ancestor/descendant overlaps, and foreign protocol manifests/recipes. New smoke path: `datasets/raw/eventwm_social_outcome_v3_smoke_01`.

V3 is a new supervised contract, not an interpretation change applied to v2. V2 native truth alone lacks the saved online PSC geodesic/fallback evidence required by this v3 label path. No legacy relabeling or production collection is performed.
