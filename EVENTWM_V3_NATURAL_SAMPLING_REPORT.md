# EventWM v3 natural root-sampling report

**Stopped by user: 42 completed / 64 attempted roots. Natural sampling is evaluation/reference only. The targeted training policy is documented in [EVENTWM_WM_TRAIN_SAMPLING_REPORT.md](EVENTWM_WM_TRAIN_SAMPLING_REPORT.md). No further natural, supplemental, or seek-comparison collection is pending.**

Baseline: `82097ac974cba4f8aa604d3319238f7a2fe75a50`, preserved on `research/eventwm-social-outcome-v3`. Work is isolated on `research/eventwm-v3-natural-sampling`. The pre-edit source audit is [EVENTWM_V3_ROOT_SAMPLING_TRACE.md](EVENTWM_V3_ROOT_SAMPLING_TRACE.md).

## Frozen contract and initial diagnosis

Labels remain `eventwm-social-outcome-v3` with native physical human contact as class 2, canonical FALCON geodesic/fallback distance strictly below 1m for social violation, and static contact invalid. Checker `eventwm-static-swept-v2.2`, footprint/margins, human spawn/goals/speed policy, native controller/simulator, observation/tracker, legal loader and label implementation are unchanged. Regression tests compare these source files byte-for-byte to the validated commit.

The previous smoke is **coverage-conditioned**, not a natural distribution estimate: seek profiles rotate until a legal visible track appears (or 126 turns expire); `seek_visible` then waits 12 steps, `seek_lost` rotates 24 more. The latter does not explicitly test lost status. They inspect neither GT distance nor future outcomes, but visibility conditions root time and orientation. Their two human-present roots contributed 26 correlated branches, with zero class 0; that cannot establish the environment's general safe-query prevalence.

The old roots were **not already inside PSC**: h1 at t=4.032s had nearest PSC distance 2.572014m; h2 at t=3.744s had 2.740424m. Their first social violations occurred later. Thus “already too close at root” is disproven for those fixtures; the sampling comparison must assess subsequent interactions and prefix choices.

## Natural sampling rule

`collector.natural_sampling.natural_selection` uses a private `random.Random` seeded by SHA256 of `{episode_seed, sampler_seed, sampler_version}`. It draws an integer uniformly over the declared inclusive step window, then a movement-path index uniformly over the declared prefix candidate count. It neither consumes the simulator's Python/NumPy/Torch RNGs nor receives observations, humans or labels. Version: `eventwm-natural-static-prefix-v1`.

Castle window: **steps 20–180 inclusive** from reset (~0.97–8.76 native seconds). HM3D uses the unchanged 250-command static-open-space prelude and window **steps 270–430 inclusive** (~13.1–20.9s). Exact actual timestamps are saved; nominal clock multiplication is not substituted for native time. The scene-specific offset is declared from static fixture geometry, not social outcomes. Every job saves sampler seed, episode seed, window, chosen step, chosen path index and eventual prefix length/root timestamp.

The robot uses the existing canonical static-feasible query generator/compiler to obtain **three** feasible movement paths from its current static-world pose. The preselected path index chooses its immediate normal-speed controls. It executes only until the predetermined root step; if the path finishes sooner, it waits. This is a human-independent static navigation prefix, not a learned policy, human-seeking behavior or stationary-only bottleneck fixture. The full chosen motion is checked by the unchanged canonical checker. Existing WAIT/seek profiles and explicit prefix replay remain unchanged.

This explicitly samples a **static-navigation-prefix distribution over existing episode fixtures**. It is not claimed to match a deployed navigation policy, randomize every possible route, or independently sample new human spawn geometries. Human episode positions/waypoints are unchanged; varying explicit episode seeds changes the inherited stochastic speed draws, not their distribution or human controller. Humans remain ego-independent with respect to robot motion; human-human avoidance is inherited, so they are not independent of other humans. The disabled stop-and-walk-to-robot configuration stays -1.0. No human reactivity is added.

## Selection, failure and legal boundaries

Natural command generation never reads history; an adversarial history object that raises on access still produces the same commands. Injecting GT distances, visibility, lost flags, future trajectories or labels cannot change selected steps/controls. The motion callback receives robot geometry and a static map only. Root-time diagnostics are computed **after** capture in `collector.sampling_audit.root_diagnostics`; they cannot feed back into selection or query generation.

Privileged diagnostics record active count, nearest canonical PSC and 3D Euclidean distances, initial PSC violation, legal visible/lost counts, ever-observed history, root time and prefix length. They are written to audits and never loaded by the unchanged legal-input whitelist. Branch labels retain first-social-violation time relative to the root unchanged.

Only existing validity/static/integrity gates exclude unusable roots. If the native episode terminates before the predetermined root time, the selected time is **not moved or retried**. The attempt persists with its original specification and prefix-failure evidence; human-contact prefix censoring is reported explicitly. Complete roots therefore describe surviving executable prefixes, not an unconditional distribution including nonexistent post-termination states. No future label determines keeping, discarding, replacing or re-querying a root. Static contact in an accepted prefix or training query is a hard stop/quarantine condition.

## Declared bounded workload

Configuration: `configs/eventwm_v3_natural_pilot.json`.

- 21 Castle repeats each for human counts 0, 1, 2, 4, 6: **105 attempts**.
- 5 known-working HM3D NBg5UqG3di3 zero-human attempts using the existing static prelude.
- **110 natural attempts total**, fixed before collection, with no outcome-dependent replenishment; target is approximately 100 completed roots.
- Master seed 241000. Castle episode seed = master + global job index. Sampler seed = master + 10000 + index. HM3D retains fixture episode seed 13301 and varies sampler seed.
- Separate comparison: first two declared Castle repeats at h1/h2/h4/h6, each with `seek_visible` and `seek_lost`: **16 attempts**, matched episode seeds, separate manifest/dataset/index.
- Natural pilot N=1: **five primary branches per completed root** (four temporal/speed variants plus WAIT), plus three existing audit/repeats. This is a distribution diagnostic; it does not replace production N=8.
- Two isolated root subprocesses keep the bounded test manageable; each calls the unmodified one-root collection/replay implementation. No branches are distributed across root owners, no concurrent shared manifests, no production/resume/retry scheduler is added. Physical GPUs 0/1 are assigned through per-child CUDA visibility, retaining Habitat logical device 0.
- No new scene asset/config repair, training, model construction or overnight/production launch.

The unchanged N=8 native gate already passed at `datasets/raw/eventwm_social_outcome_v3_default8_check`: **33 class-0 primary branches + 3 audits, zero invalid primary outcomes, zero static violations**, runtime 348.46s. At this measured cost, 110 N=8 roots would be unnecessary for root-distribution diagnosis. N=1 is explicitly declared for the pilot; query-generation validation and sampling-distribution estimation are separate.

Native preflight at `datasets/raw/eventwm_social_outcome_v3_natural_preflight_01`: two roots complete, ten primaries, zero invalid/static, all replay/integrity gates pass. It is separate from the main pilot and its samples are not added to the reported pilot totals.

## Historical planned commands (collection stopped; do not resume)

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3_sampling_pilot.py --output datasets/raw/eventwm_social_outcome_v3_natural_pilot_01
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3_sampling_pilot.py --output datasets/raw/eventwm_social_outcome_v3_seek_comparison_01 --comparison
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_v3_sampling_report.py --dataset datasets/raw/eventwm_social_outcome_v3_natural_pilot_01 --comparison datasets/raw/eventwm_social_outcome_v3_seek_comparison_01 --output output/eventwm_v3_natural_validation/distribution.json
```

Visuals remain OFF for every pilot/comparison root. No debug subset is required to determine prevalence; any later offline examples must remain separate from selection. Every root persists `sampling_profile` and `sampling_profile_family` in the job, recipe and manifest. Seek profiles are explicitly `coverage`; the natural primary index never contains the separate comparison roots.

The stopped-at results below supersede the originally planned full workload and comparison. No comparison result is claimed.


## Superseded, unexecuted sample-count contingency

A separate 35-attempt contingency was declared in `configs/eventwm_v3_natural_pilot_supplement.json` before the user stopped the investigation: seven Castle human-count repeats, master seed 242000, identical sampler/window/prefix/N. **It was never executed and is superseded.** The file is retained only as historical sampling provenance; it is not pending work.

## User-requested stop and reference-only archive

The user stopped this investigation before the declared 110 attempts finished. The scheduler was paused before it could dispatch more jobs; its two active root processes exited cleanly and were sealed using the existing sealing function. The paused scheduler was then terminated. `natural_stop_manifest_before_finalization.json` preserves the pre-finalization manifest; the original 110-row `jobs.json` is unchanged. Final manifest state is `stopped_by_user`: **64 attempted, 42 completed, 22 failed, 46 unattempted**. Neither the supplemental block nor seek comparison was run; those declarations are superseded, not pending work.

**Natural sampling is retained as an evaluation/reference policy, not as the preferred training-data policy.** It will not be outcome-balanced or mixed into targeted training. No further natural sampling or improvement is performed in this task.

Completed roots: Castle only; h0=13, h1=8, h2=8, h4=7, h6=6. The non-Castle jobs were never reached. Legal root state: visible=0, lost=1, no current track=41. Among 29 human-present roots, 28 had no current track; 25 had never observed a human. Two roots were already inside canonical PSC <1m.

Completed primary branches: **210** = 132 SAFE_SUCCESS, 41 SOCIAL_VIOLATION_SUCCESS, 37 HUMAN_COLLISION, 0 invalid. Human-present: **145** = 67 safe (46.21%), 41 social success (28.28%), 37 collision (25.52%). Unexpected static violations: **0**. All completed seals, recomputed labels and legal-loader checks pass. Existing generic smoke summary says FAIL because it requires every attempted root to complete; this is not a completed-data integrity failure. Failed attempts remain separate from branch outcomes.

Machine-readable stopped distribution: `output/eventwm_v3_natural_validation/stopped_distribution.json`. No visuals generated. Collector tests: 100 PASS (77 baseline plus natural/report tests); controlled regressions: 6 PASS. The N=8 native gate remains PASS. The follow-up targeted sampling work is documented separately in `EVENTWM_WM_TRAIN_SAMPLING_REPORT.md`.

Failed-attempt detail: 21 native-human-contact prefix terminations and **one root replay fingerprint mismatch** (`v3n_4348f706905b4622`, h6, first branch b000). The failed root is excluded; it is not a completed-root integrity contradiction. Its original worker logs/fingerprints remain preserved. No replay rule was relaxed and no failed attempt was rerun.
