# EventWM collection infrastructure

Starting checkpoint: `64dcf29`, branch `research/structured-human-observation`.
The final checkpoint is `eventwm-scale-pilot-v1`. No training or overnight job
was launched. Existing Phase-1/structured datasets were not deleted or modified.

## Results and limits

- Multi-human association audit: **PASS**. Native articulated human pose fixtures
  use the actual jaw camera, native raycasts, simulator clock and frozen tracker.
  Separated two-person, close crossing and four-person stress cases have zero
  physical identity switches. An ambiguous bounce has **two switches in 38
  existing-track associations**, honestly retained as an observability limitation.
  One-to-one association, deterministic ordering, identity metadata poisoning,
  legal velocity, hidden memory, reacquisition and expiration pass. The loss case
  reacquires after 0.904 s and expires on the first sample above 2 s (2.096 s).
  These are controlled sensor fixtures in a clear elevated native scene volume,
  not natural controller trajectories. Actual live two-human collection is also
  exercised by the Castle smoke. The observer/tracker source remains unchanged.
- Reports: `output/scale_validation/multi_human/audit.json`, `trajectories.json`,
  and the useful privileged diagnostic plot `trajectories.png`.
- Scale smoke: three completed roots, 99 primary queries, 108 grouped branches,
  one isolated failed root. Castle h0/h2 passed the existing collector integrity
  audits. A normal stop skipped the completed h0 root on resume. SIGTERM after
  the first h2 branch retained a failed partial attempt; resume used a separate
  attempt with an exactly identical physical root fingerprint. No completed root
  was overwritten. Evidence: `output/scale_smoke_v1/manifest.json`, logs and seals.
  The four-root smoke harness correctly exited nonzero because of the remaining
  Apartment failure; **the full four-configuration smoke is not a PASS**.
- Apartment-1 h2 (`seek_lost`, seed 3501) terminated before reaching its root.
  Apartment-1 moving h0 (seed 3301) had static-contact termination in all 33 primary
  branches. These configurations are **not validated for useful collection**.
  They remain reported honestly; no tuning or retry experiment was performed.
- Smoke coverage: one root with nonempty legal human history (13 detections),
  one with visible tracks at root, zero with lost tracks at root, one moving root.
  Counts: 44 primary completions, 22 human-contact terminations, 33 static-contact
  terminations. Root times: 400.9, 426.8 and 702.2 s. Old retained size: 3.96 GB.
- Original structured-root equivalence: fingerprint, full prefix truth, history,
  images and all frozen queries match exactly. Fresh populated branch b023 also
  matches all physical truth, controls, legal observations and labels. Evidence:
  `output/scale_validation/equivalence/audits/physical_equivalence.json`.
  Native wall/FOV/memory regressions pass; observer off/on prefix remains physically
  identical with 41 visible detections. No full historical branch matrix was rerun.

## Production storage: measured PASS

Explicit `--storage-profile production`; default `validated` retains debug
archives. Production still renders/acquires the same sensors, computes the same
fingerprint and structured observations, and executes the same simulation.
It retains one root history image archive, legal structured history/tracks,
actual timestamps, frozen queries, canonical branch states/contacts/events,
termination/censoring, labels, configs, replay recipe, fingerprints, structured
worker diagnostics, execution logs and provenance. Branch RGB/depth archives,
duplicate worker images, comparison PNG/MP4 are not written.

Worker branch JSON is removed only **after** canonical branches, labels and the
existing collector audits succeed. Every duplicate is compared to its canonical
record before any removal. A hash/size inventory is written first. Failed roots
retain diagnostics. No sole canonical truth is deleted.

Smallest native comparison: one fresh original root prefix and one complete
populated-observation branch in each profile. **110,191,779 bytes validated vs
20,065,203 bytes production (81.79% reduction)**. Structured WM input, all frozen
queries, root fingerprint/images, complete physical branch and labels are exact.
WM image paths are normalized to the same relative path for comparison.
This size is a root-plus-one-branch fixture, **not a measured full production
33-query root**. No production scale matrix was run after the user's scope cut.

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python tests/collector/storage_smoke.py \
  --output output/scale_validation/storage_v1
```

Evidence: `output/scale_validation/storage_v1/audit.json`. Existing output is
protected: rerunning that smoke requires a new directory.

## Architecture, manifest and resume

`eventwm_scale.py` calls the existing one-root collector as its atomic unit.
Each branch still restores through a fresh process. No query/compiler/planner,
label, tracker, visibility or controller semantics changed. The collector only
adds recipe/seed/root-ID parameters, retention options, branch bounds and
scene/count-aware validation. The old one-human renderer is disabled for scale.

Manifest schema 1 records selection, storage profile, source/commit hashes,
observation protocol version, explicit limits and per-root scene, episode/config,
human count, seed, deterministic ID, scene/episode/root-family grouping,
episode/navmesh hashes, state, attempts, resolved recipe/configs and statistics.
Completed roots publish an attempt path and hashes of retained artifacts.
Unavailable configs on pre-initialization failures remain unresolved; generated
configs/logs stay with their failed attempt. Root/query provenance remains grouped.

`roots/<root_id>/attempt_NN/` preserves the existing raw root organization.
`recipes/`, `logs/`, `failures/`, `manifest.json` and `aggregate.json` connect roots.
An exclusive lock prevents concurrent writers. Completion requires collector
integrity checks, sealed artifacts, a completion marker, then atomic manifest
publication. Directory existence alone never means complete. Orphaned/running
attempts are retained and retried separately. Completed roots are skipped after
seal verification. Source/commit drift and corrupted completed data stop resume.

Each attempt reserves 36 branches; reservations are never refunded. The launcher
allows 50 unique roots, two attempts each, and at most 3,600 reserved branch
executions. Retries happen on later invocations. Root-specific failures are logged
and collection continues. Replay/collector integrity failures and disk I/O
corruption stop globally. SIGINT/SIGTERM terminate the current process group and
persist state. Each worker retains its 180 s bound, each root has a 1,800 s bound.
Before each new attempt, less than **150 decimal GB** free stops cleanly without
starting that root. No automatic data deletion. Concurrency is exactly one.

## Legal root sampling and monitoring

Deterministic profiles: stationary WAIT; bounded visible-person search followed
by WAIT; the same search followed by 0.12 m nominal forward motion; or search
followed by a turn that can retain lost memory. Search uses only current legal
visible tracks and at most 126 yaw actions. Prefix controls are frozen before
branch replay. No future collision or hidden GT chooses roots. Prefix termination
is an explicit collection failure. Profiles do not promise category balance.

The aggregate reports states/failed attempts, primary and total branches, human
counts, zero-detection roots, visible history, visible/lost tracks at root, any
lost prefix memory, final nonzero robot commands, primary collision rate,
primary completion/censoring/termination, timing/speed distributions, time/size.
Audit repeats and truncated WAIT remain distinguishable from primary samples.
The old first-human reactivity diagnostic remains separate scientific FAIL;
zero-human reactivity is effectively not applicable. It is not an integrity gate.

## Manual broad production launcher

`run_eventwm_overnight.sh` invokes `eventwm_scale.py` through a small preflight
helper; it duplicates no scale logic. Output: `datasets/raw/eventwm_prod_v1`.
All **10 configured scenes × human counts {0,1,2,4,6} × one root = 50 roots**.
Approximately 1,650 primary queries and 1,800 branches including per-root audits
before failures/target rejection. All scenes remain in the matrix, including
Apartment; there is no silent substitution or exclusion. This is a broad
configured run that can be resumed over multiple days, not a claim all scenes
are validated. Seed 8301, concurrency 1, production retention.

Preflight checks repository/tag/clean source, environment, visible CUDA GPU,
manifest consistency, disk, scene enumeration and configuration count without
running Habitat. Missing assets are printed and their configurations are left to
fail in isolation. At preparation time the configured MP3D GLB and CODA GLB are
missing at their declared paths; they are not silently replaced. Preflight
checks completion markers cheaply; the scale runner verifies full seals.
Persistent launcher stdout/stderr lives under the production `logs/` directory.
Progress includes scene/count/root and worker completion at least every 30 s.

```bash
# Read-only preflight, no collection or output directory creation:
scripts/run_eventwm_overnight.sh --dry-run
# Start manually (prefer a persistent terminal):
scripts/run_eventwm_overnight.sh
# Read-only status now or tomorrow:
scripts/check_eventwm_overnight.sh
# Resume the same manifest with the identical start command:
scripts/run_eventwm_overnight.sh
```

Status shows process state if discoverable, current/latest root, complete/pending/
failed/running counts, retries, primary and total completed branches, legal-human
coverage, dataset size, free disk and latest log tail. It creates no files.
For strict complete-dataset validation later, use `eventwm_scale.py validate`
and `tests/collector/check_saved_observations.py --output <production-output>`;
strict validation fails if any intended roots remain failed/incomplete.

## Final fast gates and files

Collector 6 tests, observation 11, scale 6 and storage 3: **PASS**. Scale tests
cover bounded plans, legal-only prefixes, isolated failures/retry/skipping,
partial completion, corruption, free-space stop, and global integrity stop.
Shell syntax and `git diff --check` pass. Controlled matrix previously passed
with its documented sandbox socket skip; no further matrix run was made.

Modified: `scripts/eventwm_collect.py`, `scripts/collector/runtime.py`.
Added: `scripts/eventwm_scale.py`, `scripts/eventwm_overnight.py`,
`scripts/run_eventwm_overnight.sh`, `scripts/check_eventwm_overnight.sh`,
`scripts/collector/{root_sampling,multi_human_audit,storage}.py`,
`tests/collector/{test_scale,test_storage,storage_smoke,run_resume_smoke,check_saved_observations}.py`,
and this document. Raw audit artifacts remain in ignored `output/` directories.
