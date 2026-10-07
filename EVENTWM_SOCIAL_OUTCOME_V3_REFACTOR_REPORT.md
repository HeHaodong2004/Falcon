# EventWM social-outcome v3 refactor report

This is a new label protocol, not a change to v2 data. Validated v2 remains recoverable at `36e2872d09cced67050be9776770550906aeaf65` on `research/eventwm-human-event-v2`. Development is isolated on `research/eventwm-social-outcome-v3`. The working tree was clean before branching; all 48 baseline tests passed. The interrupted prior N=8 request had not created its dataset, and no collector process was active. No parallel production runner was implemented or launched.

## Source audit: distances, collisions and success

All sources below are repository implementation evidence. No benchmark threshold was chosen to balance labels.

| Concept / exact source | Geometry and condition | Temporal meaning / official status | Availability and supervised-label decision |
|---|---|---|---|
| `falcon/additional_metric.py::DidMultiAgentsCollide.update_metric` | Native discrete contacts between robot articulated object and one of the active humanoid object IDs. Actual meshes; no circular radii or distance cutoff. | Instantaneous benchmark measure; updates discrete collision detection. | Privileged simulator contact; appropriate physical label evidence. |
| Same file, `HumanCollision.update_metric` | Depends on `DidMultiAgentsCollide`; latches any contact and sets `task.should_end`. | Official benchmark episode HC indicator, early termination. | Privileged; class 2 uses the same native-contact declaration sampled by the collector. |
| `scripts/collector/runtime.py::Recorder.capture` | Native human contacts using object IDs; static contacts exclude nearly vertical ground support below base+0.12m. | Raw native physics timestamps, retained contact witnesses. | Privileged; unchanged physical/static evidence and timing. |
| `scripts/collector/core.py::build_labels` research disc | Horizontal x/z center distance `d`; clearance `d−0.25−0.30`; disc contact iff `<=0`. Neither radius is the measured articulated Spot envelope. | Any native recorded sample; research collector proxy, **not** official HC or PSC. | GT centers; retain raw diagnostic, **do not equate with physical contact or social space**. Not independently class 2. |
| `falcon/additional_metric.py::PersonalSpaceCompliance.update_metric` | Robot/human base centers, geodesic distance if path exists, else 3D Euclidean; violation strictly `<1.0m`. No radius subtraction. Config `PersonalSpaceComplianceMeasurementConfig.use_geo_distance=True`. | Official PSC = compliant control steps / total steps; zero-human PSC=1. | Privileged full crowd/map; canonical instantaneous supervised social predicate. V3 query-any/native-sample aggregation is explicitly different from benchmark episode aggregation. |
| `falcon/rearrange_sim_v2.py::get_agent_state`; `habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py::geodesic_distance` | AgentState position is `articulated_agent.base_pos`; native MultiGoalShortestPath on simulator pathfinder. | Query-time geometry; not itself a score. | GT; provides exact PSC centers/distance implementation. |
| `scripts/collector/core.py::build_labels` minimum/near exposure | Minimum horizontal research-disc clearance; near threshold `clearance<0.5m`, equivalent to horizontal centers `<1.05m`. Right-endpoint exposure duration. | Collector-only continuous diagnostics; **not PSC**. | GT; retained legacy diagnostics only. Not the v3 class-1 predicate. |
| `habitat-lab/habitat/tasks/rearrange/social_nav/social_nav_sensors.py::SocialNavStats` | Stores 3D L2 robot-human distance for averages; `_check_human_dis` uses geodesic inclusive `[min_dis_human,max_dis_human]`, defaults `[1,2]m`. Default backup/yield distance 1.5m also exists. | Habitat social-following statistics, not selected FALCON PSC metric. | GT plus optional visual checks; not the v3 social outcome. |
| Same file, `SocialNavSeekSuccess` | Safe-follow range `[safe_dis_min,safe_dis_max]`, defaults `[1,2]m`, plus task visibility/follow conditions. | Different social-follow task's success measure, not fixed-query completion. | Not the v3 label contract. |
| `falcon/additional_metric.py::MultiAgentNavReward` and its config | Goal-distance gating `allow_distance=0.5m`; facing-human gate `facing_human_dis=1.0m`; exponential close-to-human reward, scene/human collision penalties, future-path penalty. | Training reward shaping, not PSC or a physical contact detector. | Includes privileged futures; deliberately not reused for labels or legal inputs. |
| `habitat-lab/habitat/tasks/nav/nav.py::Success` | STOP called and goal distance strictly below configured success distance (base config 0.2m); no human radii. | Official navigation success, not fixed-horizon success. | Goal/task GT; not substituted for query completion. |
| Same file, `SPL`, `Collisions` | SPL combines navigation success with shortest/actual path length. `Collisions` counts `previous_step_collided`, without human/static separation. | Official navigation metrics. | Inappropriate substitutes for the three-way label; static remains outside it. |
| `habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py` | Reports benchmark `PSC`, `human_collision`, `success` and aggregate scoring. | Confirms PSC/HC are actual evaluation outputs. | Audit context only; no evaluation/model changes. |

The canonical rule is in `EVENTWM_SOCIAL_OUTCOME_V3_PROTOCOL.md`. It deliberately uses the existing **geodesic/fallback <1m center-distance predicate**, not a new surface-clearance radius. V3 samples that predicate at every inherited native timestamp (including root) and takes any violation across the fixed query. This can identify violations between benchmark control-step observations; it is not claimed to reproduce the episode PSC percentage.

## Implementation scope

- New `collector/v3.py`: version/output guards, enum, canonical label derivation and envelope-only wrapper around v2 query generation.
- New `collector/social_habitat.py`: passive online privileged distance evidence; a v3 recorder subclass preserves the original `Recorder` implementation exactly.
- Minimal protocol dispatch in `eventwm_collect.py` and `collector/runtime.py`; existing replay/controller/physics/prefix/observer/tracker paths unchanged. V3 gets the same fresh-process counterfactuals, seals and integrity assertions.
- New bounded `eventwm_v3.py`: four declared smoke fixtures, reference-only primary index, offline label recomputation, seal validation, fail-stop behavior. No scale/resume/training path.
- New opt-in `collector/debug_static.py` and offline `eventwm_v3_visualize.py`: native static-ray reference plus bounded root composites. No plotting library dependency in collection.
- Explicit image retention metadata: v3 omits pixel archives, keeps image hashes and unchanged sensor acquisition/replay checks. Loader returns null for omitted images; existing v1/v2 archives remain untouched. Storage audit reports actual retention.

The original v2 module, checker, Habitat static adapter and bounded runner remain byte-identical to the validated checkpoint. Tests enforce this. Common protected phase-1 recorder/controller helpers are also compared against git by the original regression suite.

## Storage dependency audit

`collector.core.load_wm_input` reads structured history and fixed query; previously it returned an image filename without opening it. Structured tracking uses current sensor/world observations online; saving NPZ does not feed it. Runtime replay fingerprints hash the acquired images in memory, independently of persistence. `eventwm_collect.validate` compares histories and fingerprints, never opens image archives. The legacy `storage_smoke.py` and v1 audit explicitly examine v1/v2 pixel archives and remain unchanged; they are not v3 validation paths. The old movie renderer needs branch archives and is explicitly disallowed for v3.

Therefore v3 can omit `history_images.npz`, all branch image archives and videos without changing structured inputs, queries, labels, RNG, sensors or exact replay checks. Root/control/state/config/provenance and image hashes remain. Sensors are **not disabled**: that would require a separate dependency/replay investigation. Native truth is still verbose; no unvalidated truth compaction was attempted. Debug assets have separate opt-in retention and never enter model loading.

## Validation and smoke results

Final measured results are recorded below after native completion. No production/training run is authorized by this smoke. The smoke deliberately uses N=3 (13 primary + 3 audit branches per root), preserving the prior legal sampling matrix rather than selecting queries by future outcomes.

| Scene | Humans | Profile | Seed |
|---|---:|---|---:|
| Castle | 0 | wait | 8301 |
| Castle | 1 | seek_visible | 8401 |
| Castle | 2 | seek_lost | 8502 |
| HM3D NBg5UqG3di3 | 0 | explicit existing static open-space prefix | 13301 |

The non-Castle fixture, seed, controls and source hash are reused unchanged from `configs/eventwm_v2_noncastle_smoke_prefix.json`. It was selected using legal robot pose/static geometry in v2, never future human labels. No controlled class-forcing fixture is mixed into this natural smoke.

Native command:

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3.py --output datasets/raw/eventwm_social_outcome_v3_smoke_01 --movement-queries 3 --debug-visuals --debug-visual-roots 4 --debug-visual-branches 6
```

Unit/regression command:

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python -m unittest discover -s tests/collector -p 'test*.py'
```

Additional read-only native audit:

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python tests/collector/v3_native_audit.py --dataset datasets/raw/eventwm_social_outcome_v3_smoke_01
```

## Limits

A geodesic PSC boundary can be asymmetric around obstacles; the benchmark's Euclidean no-path fallback can flag nearby humans across disconnected geometry. Native contacts and social violations are sampled, not guaranteed between all samples. Research-disc clearance is not an articulated mesh separation. Static checking remains conservative and discretized. A tiny N=3 smoke establishes integration, not large-dataset distribution, model quality, or production readiness. No N=8 production approval is claimed. Missing historical scene assets/configurations remain untouched.

The debug native height reference is first downward collision-ray hit from root+1.2m within 1.6m, restricted to the stage and STATIC objects. It cannot depict walls entirely above/below that slice or infer complete 3D occupancy. Unknown hits stay blank; no furniture outlines are invented. Contact witnesses display actual physical events; no circular physical boundary is claimed for articulated humans/Spot.


## Final measured result — PASS

All **77 collector tests** pass (48 existing + 29 v3 tests), and all **6 controlled-environment regressions** pass. The latter were run with the existing conda `ffmpeg` on PATH and outside the sandbox for their local multiprocessing socket; no dependencies or environment behavior were modified. The first sandbox invocation's PATH/socket limitations were resolved by using the installed environment, not by skipping or weakening tests.

Main smoke: **4/4 roots completed, 52 primary branches, 12 audit/repeats**.

| Scene / humans | Legal tracks visible / lost | Class 0 | Class 1 | Class 2 | Invalid primary | Unexpected static |
|---|---|---:|---:|---:|---:|---:|
| Castle / 0 | 0 / 0 | 13 | 0 | 0 | 0 | 0 |
| Castle / 1 | 1 / 0 | 0 | 9 | 4 | 0 | 0 |
| Castle / 2 | 1 / 1 | 0 | 9 | 4 | 0 | 0 |
| HM3D / 0 | 0 / 0 | 13 | 0 | 0 | 0 | 0 |
| **Total** | | **26** | **18** | **8** | **0** | **0** |

All 26 human-present branches recorded a social violation; eight subsequently had native human contact and correctly became class 2. Only **one** primary branch met the research-disc criterion, while eight had native contact: this smoke directly demonstrates why the proxy cannot replace the articulated physical detector. All class-0 examples in this small matrix are zero-human; no claim of balanced human-present safe coverage is made. No future-label selection was used to alter this distribution.

Candidate diagnostics match v2: 26 proposals, 14 static rejections, 12 accepted movement paths (three per root), four separate WAIT queries. Root runtimes were 161.00, 181.88, 191.43 and 560.98 seconds. There were zero static violations across **all** primary and audit branches. Four deliberately truncated WAIT audits are invalid/censored as designed and excluded from the primary index. The inherited `human_reactivity` scientific diagnostic remains FAIL (native humans do not react in this fixture), as in v2; it is not an integrity failure and was not suppressed.

Read-only post-smoke audit verified all **680 sealed root files**, every saved label via canonical offline recomputation, and **144 comparisons to the actual FALCON PersonalSpaceCompliance implementation**. For all 64 branches including repeats/truncations, native truth (excluding only the added social evidence), controls and legal observations exactly equal the corresponding v2 smoke; legal root histories also match. Eleven legacy manifests/completion markers match their pre-task hashes. Every collector/controlled source hash recorded by all four smoke roots still matches the final code. The bounded v3 entry point additionally gained argument-bound checks and benchmark/entry-point provenance hashes before the separate no-visual smoke; these change neither collector nor labels.

The separate storage-only integration check at `datasets/raw/eventwm_social_outcome_v3_no_visuals_check` used one zero-human Castle root, N=3, **without** `--debug-visuals`. It passed with 13 class-0 primaries, zero invalid/static outcomes and **zero** PNG/JPEG/MP4/NPZ/ZIP/debug-reference files. Its full branch records are exactly equal to the debug-enabled root. Its 13 primaries are **not added** to the main 52-sample result above.

All four final composites were inspected during development, including the non-Castle footprint rejection geometry and the lost-track inset. Delivered visual storage: **2,372,463 bytes** in four PNGs; **1,068,328 bytes** in four native reference JSONs; **1,349 bytes** in the visual manifest; **3,442,140 bytes total**. No videos/storyboards or per-branch images were generated. Production visualization remains OFF. See `EVENTWM_V3_VISUALIZATION_REPORT.md` for examples and limitations.

Machine-readable evidence is in `output/eventwm_v3_validation/`: `native_audit.json`, `source_check.json`, `composition.json`, `no_visuals_check.json`, `visual_storage.json`, `unit_tests.txt`, and `controlled_tests.txt`; raw smoke also has its own manifests, summary, primary index, labels and sealed audits. Generated outputs are ignored; only source/tests/docs are committed.

No EventWM training, micro-overfit, architecture change, large-scale production, old-root repair, v1/v2 relabeling, or checker/physics/behavior change occurred. This is a bounded v3 integration PASS, not authorization for production scale.
