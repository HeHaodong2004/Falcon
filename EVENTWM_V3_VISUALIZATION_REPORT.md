# EventWM v3 debug visualization

Visualization is **OFF by default**. `eventwm_v3.py` imports no plotting library. V3 writes no RGB/depth image archives, per-step frames, per-branch PNGs or videos. It still acquires the original sensors and hashes image histories for exact replay; sensor/controller behavior is unchanged.

Explicit debug mode captures at most four compact native static references, then invokes an independent offline plotting process. The output is **one 3600 × 2160 composite PNG per root**, displaying at most six branch examples inside it. There is no video path or frame-dump option in the v3 tool. Plotting imports are confined to the offline script. Generated assets and raw datasets are ignored by git.

## Reading the composite

**A — Legal/planner view.** White is accurately labeled “navmesh navigable center region (agent r=0.25m)”. It is not full-body free space. Robot pose/heading, accepted paths/endpoints and rejected proposal endpoints are overlaid. Visible tracks are filled blue circles; lost tracks are hollow blue diamonds, at their legal retained estimates. No GT humans appear in this panel.

**B — Static feasibility.** A measured conservative Spot circle and separate 0.03m margin are shown. Green means sampled pose-center navmesh clearance meets `radius−0.25+margin+0.01`; tan fails this clearance, gray is non-navigable. Green alone is **not** a full checker certificate: native geometry and controller clipping must also agree. Rejected endpoints are small crosses, while the recorded first failed trajectory poses are large dark crosses. Compatible rejected fixed trajectories are reconstructed offline from the unchanged compiler and drawn dashed only when their failed segment/time/pose agrees with the recorded witness. Candidate endpoints are never substituted for the actual failure pose. Text lists rejection reason, failed segment and observed/required clearance.

**C — Native privileged reference.** The panel visibly states `DEBUG GT / PRIVILEGED`, `NOT MODEL INPUT`, `NOT QUERY SELECTION`. Its background is actual Habitat collision-ray data restricted to the stage and STATIC object IDs. Rays start at root height+1.2m and point down up to 1.6m. Color is the first hit's height relative to the root; missing hits remain unknown. This is a native surface slice, not invented furniture polygons or a claim of complete 3D occupancy. Paths, rejected endpoints and failed poses use the same world coordinates as panels A/B.

Privileged humans at a selected social-event time are magenta stars, distinct from panel A's legal tracks. Magenta contours use the exact canonical **geodesic/fallback 1m predicate**, evaluated through Habitat pathfinding on the unchanged scene navmesh; they are not arbitrary personal-space circles. A contour may be circular in open space. Red X markers are actual native physical-contact witnesses from a selected collision branch and are annotated with branch/time. Because articulated physical collision is not a circular center threshold, no false circular “physical boundary” is drawn. The native contact witness and the distinct social contour communicate the two definitions; exact body meshes are not reconstructed by this lightweight tool.

**D — Outcome evidence.** Selected branch timelines show the exact saved PSC distances, the strict 1m threshold, collision times and categorical outcome. Zero-human roots show an explicit safe interval, not fabricated distances. An adjacent diagnostics block states class counts, wait/speed variants, first social/collision times and research-disc clearance. Research-disc clearance is explicitly distinguished from real mesh separation.

## Selection and limitations

Offline display takes one available example per class, then fills the requested limit with existing primary branches. This may use labels **only after collection for illustration**; it never changes query generation, root selection or the natural class distribution. No controlled class-forcing fixture was needed in the main smoke. Audit/repeat branches are excluded from examples and primary counts.

The static surface slice can miss occluders outside its vertical interval; blank is unknown. The clearance background is a sampled visualization, not a substitute checker. Legal lost tracks remain estimates; privileged humans may be shown at a later event time and must not be interpreted as the root's legal observation. Social contours use the stored static navmesh, consistent with this smoke's unchanged simulator map. For changed dynamic maps they would require a separately audited snapshot. Native contact points establish physical events but do not reconstruct exact human body silhouettes. These limits are labeled rather than hidden behind plausible invented geometry.

No storyboard or MP4 was generated. The single-root composites provide the requested compact debugging artifact without retaining simulator frame streams.

## Commands

Opt-in collection/debug:

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3.py --output datasets/raw/eventwm_social_outcome_v3_smoke_01 --movement-queries 3 --debug-visuals --debug-visual-roots 4 --debug-visual-branches 6
```

Offline regeneration (requires explicitly captured native references):

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3_visualize.py --dataset datasets/raw/eventwm_social_outcome_v3_smoke_01 --output output/eventwm_v3_visuals --max-roots 4 --max-branches 6
```

Default collection omits `--debug-visuals`; it does not invoke the plotting tool or capture the debug static-reference grid. No existing smoke artifacts are deleted.

## Measured outputs

Final exact byte counts and manually inspected examples follow after the native smoke. PNGs are outside sealed root data; opt-in raw static reference grids are sealed with their root audits. Any temporary development previews are distinct from the delivered smoke assets and are not training data.


## Final artifacts and inspection

Directory: `datasets/raw/eventwm_social_outcome_v3_smoke_01/debug_visuals/`.

| Root / view | Composite | Exact PNG bytes |
|---|---|---:|
| v3_184fe8cee976be3e | [v3_184fe8cee976be3e.png](datasets/raw/eventwm_social_outcome_v3_smoke_01/debug_visuals/v3_184fe8cee976be3e.png) | 513,260 |
| v3_da43794d71d48ca5 | [v3_da43794d71d48ca5.png](datasets/raw/eventwm_social_outcome_v3_smoke_01/debug_visuals/v3_da43794d71d48ca5.png) | 649,426 |
| v3_1570b1acb615302b | [v3_1570b1acb615302b.png](datasets/raw/eventwm_social_outcome_v3_smoke_01/debug_visuals/v3_1570b1acb615302b.png) | 644,510 |
| v3_7722be655f39b1fb | [v3_7722be655f39b1fb.png](datasets/raw/eventwm_social_outcome_v3_smoke_01/debug_visuals/v3_7722be655f39b1fb.png) | 565,267 |

Four composites: **2,372,463 bytes** total. Four sealed native reference grids: **1,068,328 bytes**. Visual manifest: **1,349 bytes**. Complete delivered debug payload: **3,442,140 bytes** (3.283 MiB). There are no delivered videos, storyboards, sensor-image archives or per-branch PNGs. Temporary reproducible previews under ignored `output/eventwm_v3_preview/` are development artifacts, outside this delivered payload and outside all training indices.

Manual inspection confirmed that accepted paths occupy the conservative clearance region, non-Castle rejected endpoints and first failed poses are distinct, actual native contact points differ from social-boundary entries, and legal tracks are separated from privileged humans. The Castle lost-track estimate lies outside the main 5.4m crop; a wider **legal-only** inset shows its true saved location rather than placing it misleadingly on the crop edge. Visible/lost counts are annotated.

Main smoke naturally yielded all three classes: 26 safe, 18 social-success, 8 physical-human collision, zero invalid primaries, zero static violations. Example `v3_da43794d71d48ca5/b000` completes with social violation; `b004` first enters social space at 1.192s then physically collides at 2.456s and is class 2. The lost-track root demonstrates the analogous precedence. These are post-collection display choices, not targeted generation.

The additional native **default-OFF** check at `datasets/raw/eventwm_social_outcome_v3_no_visuals_check` generated zero visualization/image archive files and identical raw branch truth/controls/legal observations to the debug-on root. Evidence: `output/eventwm_v3_validation/no_visuals_check.json`. Production visualization default: **OFF**.
