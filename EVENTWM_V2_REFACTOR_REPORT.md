# EventWM v2 operational refactor

## Freeze and checkpoint

V1 source/audit checkpoint: `4a72eaa`, on `research/structured-human-observation`. V2 branch: `research/eventwm-human-event-v2`. Validated v2 implementation commit: `cbfe0a907af6af0bb5f265e524647d5aa8f174c1`. The report is committed separately on the same branch. Smoke provenance records the checkpoint base plus exact working-source hashes; the tested source is committed after validation.

V1 raw data remain at `datasets/raw/eventwm_prod_v1`, read-only legacy/debug/regression. Their manifest still matches the SHA256 from the passing SR audit. No raw data, labels, images, or root seals were moved, rewritten or added to git. The reason for retirement is excessive statically invalid query motion, not integrity corruption.

Retained: all existing collector/observation/storage/replay tests and source; phase-1, structured-observation, scale and SR documentation; `scripts/audit_eventwm_sr.py`. Archived: unfinished training-preparation utilities under `scripts/legacy/eventwm_v1/`, and generated 174-sample event index/micro16/integrity report under ignored `output/legacy/eventwm_tiny_overfit/`. No training ran. Disposable bytecode/logs remain ignored; the empty former `scripts/training/` directory was removed. No uncertain artifacts were deleted. See `legacy/eventwm_v1/EVENTWM_V1_LEGACY_STATUS.md`.

The old overnight launcher now exits before creating logs or directories. Old production scale/overnight CLI launches are disabled. Existing readers and explicit v1 regression library functions remain available. New v2 writers resolve symlinks and reject legacy-path overlaps.

## Pipeline and implementation

The pre-edit source/function trace is `EVENTWM_V2_PIPELINE_TRACE.md`. V1 checked snapped target/static shortest-path availability, then compiled directional motions for an articulated robot without checking the full robot footprint and executed trajectory. Several non-Castle v1 roots even had static contact at decision time. V2 rejects unsafe root/query geometry before training branches exist.

Implementation:

- `collector/static_feasibility.py`: one reusable swept-query checker, explicit geometry/config/version and failure diagnostics.
- `collector/static_habitat.py`: immutable pathfinder + native static collision adapter; no environment stepping or human-state inputs.
- `collector/v2.py`: bounded candidate resampling, version/path guard, canonical human-event validity/labels.
- `collector/runtime.py`: v2 generation and certificate enforcement; existing controller, native clocks, prefix and replay unchanged.
- `eventwm_collect.py`: v2 default, target construction, static-violation quarantine/stop and integrity gates.
- `eventwm_v2.py`: intentionally bounded four-root smoke driver, isolated manifest/seals, reference-only primary index and aggregate report; no scale/resume mode.
- `eventwm_v2_visualize.py`: lightweight static-map/path/proposal overlays.

Default N is eight accepted movement spatial paths, each with the original four wait/speed variants, plus one separate horizon-matched WAIT. Search keeps proposing after rejections, uses deterministic rotated headings and radii 1.5/1.0/0.65 m, enforces ≥0.30 m endpoint separation, and stops at 128 proposals. Insufficient feasible paths fail the root explicitly; no invalid branch or WAIT-only fallback is fabricated. Every temporal variant is checked because finite-step geometric discretization differs with speed.

Static representation: independent existing navmesh (0.25 m radius, 1.41 m height, includes static objects), dense full-trajectory clearance, exact static navmesh step-filter agreement, and native static geometry overlap at each sampled pose. The conservative circular radius is measured from the articulated Spot link bounds about `base_pos`; no simulator radius is changed. Spot's 21 URDF links were inspected: collision/visual geometry and origins match. Safety margin is 0.03 m; max spatial/angular sample intervals are 0.02 m / 0.04 rad, plus 0.01 m half-spacing clearance allowance. The checker covers root, turns, translations and WAIT. Measured values/results appear in smoke diagnostics below.

The complete frozen contract, output schema, validity precedence and assumptions are in `EVENTWM_HUMAN_EVENT_V2_PROTOCOL.md`. Human-only event positives remain valid on early collision termination. Clean negatives require every control/full horizon. Any static contact, including mixed human/static, invalidates the branch and stops collection. Other incomplete/error outcomes without a determined human event are invalid/null, not a third class. No static-collision supervision or any-collision model is introduced.

The existing legal loader and nested track whitelist remain unchanged. It exposes arrived history and fixed query, excluding labels/certificates/privileged fields. No tracker, visibility, human behavior, sensor, controller or simulator redesign was made.

## Tests and native smoke

Full unit-test output is saved at `output/eventwm_v2_validation/unit_tests.txt`. The initial simulator-free suite passed all 43 tests; the final suite passes **48 tests**, including 22 focused v2 tests for full-path obstacle rejection, footprint/margin, resampling/N/diversity/WAIT, labels, zero-human invariants and privileged/output-path boundaries.

Native smoke attempt `eventwm_human_event_v2_smoke_01` is retained as a failed integration artifact: four attempted roots, zero executed query branches. This installed Habitat build has no articulated-object `aabb` attribute. The adapter was corrected to use supported per-link scene-node bounds; no radius, physics or controller change was used to bypass the issue. The tests passed again before the fresh smoke attempt.

Smoke 02 is also retained for debugging, not promoted as final validation. Its three Castle roots completed (39 primary branches; zero static contacts); the original non-Castle root correctly exhausted 128 proposals with zero accepted paths because root clearance was 0.093722 m versus a required 0.390200 m. Independent inspection also caught an incorrect stage-ID assumption: the installed Habitat stage ID is 0, not -1. Checker v2.2 now uses the native API constant, logs the static IDs, and verifies controls/trajectory consistency. No footprint reduction or label filtering was used.

The final smoke uses a distinct non-Castle fixture recorded in `configs/eventwm_v2_noncastle_smoke_prefix.json`. It replays the existing 138-command prefix, then turns toward +z in 32 fixed steps and moves 0.8 m in 80 fixed steps. This open-space location was selected from the prior smoke's **legal robot pose and static map**, not human GT. Every intermediate observation is still acquired normally; controller, simulator, root-history definition and prefix implementation are unchanged. The old rejected root remains preserved.

The final matrix in `datasets/raw/eventwm_human_event_v2_smoke_03` explicitly requests N=3 to bound runtime:

| Scene | Humans | Profile | Seed |
|---|---:|---|---:|
| castle | 0 | wait | 8301 |
| castle | 1 | seek_visible | 8401 |
| castle | 2 | seek_lost | 8502 |
| hm3d_NBg5UqG3di3 | 0 | explicit static open-space prefix | 13301 |

Each successful root has 13 primary queries + 3 audit/repeats. No old failed-root repair or production collection is attempted. A zero-human root tests the no-positive/zero-static invariant, and historical legal profiles seek visible/lost histories without consulting hidden humans for candidate selection.

## Limitations and next run

The checker is discretized and posture-specific, conservative rather than a mathematical proof over all possible articulated physics. Tighter passages may be rejected despite some orientation-specific feasible motion. Static map/asset/robot/controller changes require revalidation. The small smoke does not establish coverage or generalization; no EventWM training is authorized by it. Missing CODA/MP3D and ReplicaCAD problems are unchanged separate tasks.

Final smoke results and the next bounded command follow. No training or larger collection was run.

## Final validation result — PASS

Final dataset: `datasets/raw/eventwm_human_event_v2_smoke_03`.
Protocol `eventwm-human-event-v2`; checker `eventwm-static-swept-v2.2`.

| Metric | Result |
|---|---:|
| Roots attempted / completed | 4 / 4 |
| Candidate proposals | 26 |
| Static candidate rejections | 14 |
| Accepted movement spatial paths | 12 (3 per root) |
| Mean / maximum proposals to obtain N | 6.5 / 14 |
| Explicit WAIT queries | 4 |
| Primary branches | 52 |
| Audit/repeat branches (excluded from model index) | 12 |
| Human-event positives / negatives | 8 / 44 |
| Invalid primary branches | 0 |
| Unexpected static collisions, including audits | **0** |
| Termination: query completed / native human contact | 44 / 8 |

| Scene / humans | Profile | Proposals | Static rejections | Movement / WAIT | Positive | Negative | Invalid | Visible / lost tracks at root |
|---|---|---:|---:|---|---:|---:|---:|---|
| Castle / 0 | wait | 4 | 1 | 3 / 1 | 0 | 13 | 0 | 0 / 0 |
| Castle / 1 | seek_visible | 4 | 1 | 3 / 1 | 4 | 9 | 0 | 1 / 0 |
| Castle / 2 | seek_lost | 4 | 1 | 3 / 1 | 4 | 9 | 0 | 1 / 1 |
| HM3D NBg5UqG3di3 / 0 | explicit_static_open_space | 14 | 11 | 3 / 1 | 0 | 13 | 0 | 0 / 0 |

Measured circular radii were 0.599654675, 0.599655151, 0.599655271 and 0.600200176 m, respectively. The first three rejections were controller/navmesh clipping disagreement; the remaining eleven were insufficient swept-footprint clearance. The non-Castle root moved from z=10.708728 to z=11.508746 as specified by the explicit prefix. Its accepted endpoints have measured execution errors of order 10⁻⁵ m.

All 48 automated tests pass, including 22 v2 tests. Final post-run audit: all four completion markers and **676 sealed files** verified; every saved label recomputed from truth through the canonical v2 label function; all 52 primary query IDs unique; all inputs loaded through the unchanged legal loader; every certificate is accepted checker v2.2; all zero-human primary labels are 0. The native stage ID 0 is recorded in every root's `static_world` diagnostics. All collector integrity assertions pass; the inherited human-reactivity scientific diagnostic is not an integrity gate. Four deliberately truncated WAIT audits remain censored/invalid as intended and are excluded from primary statistics.

The accepted Castle h0/h1 traces were compared across smoke02 and smoke03: native truth, controls and legal observations were byte-canonically identical for every branch. The final smoke source hashes showed no drift. The v1 manifest still exactly matches its prior passing audit SHA256. No completed-v1 integrity contradiction was found.

Post-run evidence:

- Dataset `manifest.json`, `smoke_summary.json`, `primary_index.json`, and sealed per-root audits.
- `output/eventwm_v2_validation/unit_tests.txt`.
- `output/eventwm_v2_validation/post_smoke_integrity.json`.
- `output/eventwm_v2_validation/source_check.json`.
- `output/eventwm_v2_validation/native_regression.json`.

## Visual/manual check

Rendered and inspected the Castle zero-human, Castle lost-track and non-Castle static-map overlays. White is the navigable center region, gray is non-navigable, red crosses are rejected proposals, colored lines/endpoints are accepted spatial paths, and the dashed circle is measured footprint + margin. Accepted motions visibly occupy open space; the non-Castle search selects shorter feasible paths rather than substituting WAIT for blocked movement.

Final images: `output/eventwm_v2_visuals_smoke03/`:

- `v2_ab5b6c39df7734e5.png` (Castle zero-human).
- `v2_0d3f98e4dd99ee1d.png` (Castle visible-human).
- `v2_c242d6e78dee754f.png` (Castle visible + lost tracks).
- `v2_32d1e34de12b5cba.png` (non-Castle open-space fixture).

The earlier rejected non-Castle root image remains in `output/eventwm_v2_visuals_smoke02/`, showing why the original root cannot admit the conservative footprint. No static-collision result was relabeled or filtered to manufacture a pass.

## Next bounded command — prepared, NOT executed

The smoke explicitly used N=3; the production-default N=8 has simulator-free resampling coverage but has not yet received a native smoke check. The next step is therefore **one zero-human Castle root at N=8**, not production scale:

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v2.py --output datasets/raw/eventwm_human_event_v2_default8_check --movement-queries 8 --first-root-only
```

This creates a new isolated dataset with 33 primary queries and three audits. Stop on any unexpected static contact. No full-scale collection, EventWM training, micro-16 or tiny-128 was started.

```text
V1 CHECKPOINT / ARCHIVE: PASS
V2 STATIC FEASIBILITY TESTS: PASS
V2 HUMAN-EVENT LABEL TESTS: PASS
V2 LEGAL-INPUT BOUNDARY: PASS
V2 SMOKE COLLECTION: PASS
```
