# EventWM 300K production handoff

Recovery update, 10 October 2026: this operational recovery worktree supersedes
the original launcher after the b027 static-certification crash. Read
[EVENTWM_STATIC_RECOVERY.md](EVENTWM_STATIC_RECOVERY.md) and the committed
`docs/release_evidence/static_recovery_gate.json` for diagnosis and native gates.
The local production configuration pins the isolated repaired scientific epoch;
the original scientific checkouts, existing dataset, completed roots and saved
C24 settings remain preserved. Use this worktree's `run_eventwm_300k.sh` for
subsequent check, resume, status and stop commands.

## Release and scientific contract

Operational release: **eventwm-v4-300k-production-freeze-1**, branch
`research/eventwm-v4-300k-production`. Its full commit is obtained with
`git -C /home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-300k rev-parse HEAD`.
The source of scientific results is pinned to
**58b4b0a17c90c5c7de420834b8ef62c44194eee7**, release
`eventwm-300k-r4-train-geometry-v2-shadow-reference`.

The live collector was independently inspected through its durable SQLite
metadata, PID command, actual process working directory, Git HEAD/branch,
478 implementation hashes, and 29 protected reference hashes. It runs from
`output/eventwm_72h_v4_checkout`, branch
`research/eventwm-72h-geometry-production`, at the scientific commit above.
It was **not restarted, drained, stopped, or replaced during this handoff**.
The automatic exploratory slot responder was stopped; no collector or worker
was signalled. Existing production monitors continue in their existing tmux
sessions. No further optimization is enabled by the release utilities.

For future recovery the launcher executes the same scientific commit from the
stable detached worktree `production/eventwm-v4-science`. The operational
release is at `production/eventwm-v4-300k`. Separating these worktrees lets the
launcher improve without changing the scientific release digest stored in
existing checkpoints. **No metadata migration is needed:** saved query IDs,
root seals, seeds, attempts, source epochs, and scientific hashes remain intact.
The original `configs/eventwm_pipeline_release.json` is retained byte-for-byte.
Its historical candidate/benchmark annotations are supplemented by the native
release evidence in `docs/release_evidence`; they are not new performance claims.

The target is **300,000 total retained primary queries**, including 30,030
verified compatible legacy queries, by **12 October 2026, 09:20 Europe/London**.
It is not a TRAIN-only target. Only durably published completed roots with
immutable seals count. Proposals, failed roots, in-flight branches, native
scientific gates, and diagnostic panels do not count.

Execution remains COMBINED/COMBO_NO_RENDERER with fresh native physics and
copy-on-write prefix checkpoints, exact static root gating, legal arrived
histories, 8 movement paths × 4 fixed variants plus WAIT (33 primary queries),
3 native audit jobs, and the existing SR/collision/validity/censoring rules.
A completed safe query has SR=1; determined early native human collisions
remain SR=0. Unrelated incomplete non-collision outcomes are censored/invalid
and excluded from retained primary queries. Unexpected static collision and
scientific integrity failures remain collection-stopping conditions.

V4 changes the **TRAIN proposal law**: 87.5% static geometry-feasible proposals
and 12.5% original V3 support, with clearance threshold 0.38965485334396366 m.
The sampler uses frozen robot bounds and static NavMesh information; no future
human truth or SR label is used for proposal admission. Independent original
seeded V3 shadow coverage advances on original preparation success even when
the geometry proposal fails. Natural-reference and unseen-evaluation episodes
use original V3 shadow episodes, IDs and coverage. Actual TRAIN geometry
coverage is separate. Component probabilities and attempted/accepted strata
are recorded; importance weights have not been estimated. Do not interpret
TRAIN as a sample from the original V3 proposal distribution.

Scene split allocation is frozen: 12.5% held-out physical scenes per large
family; one eighth of eligible recipe opportunities is natural reference,
allocated before outcomes; all variants of a root share a split. Retained role
fractions can change with root acceptance. The combined index enforces unique
accepted root IDs, query IDs and seeds, and consistent physical scene roles.
Legacy source commit/master seed are `4b01a7a699462967172c9d61485dea4b40a41f78`
and 410000. New master seed is 731000000. New checkpoint
`production_source_epochs` preserves V3 pipeline commits `06e6afe…`, `44f0f95…`
and V4 `58b4b0a…`; each completed root retains its own source commit. The V4
switch followed 12,276 already published new V3 queries; those were not relabelled
or replayed.

## Measured performance and production health

All rates below count retained primary queries. Panels and live windows have
different workloads; the table is not a claim of a paired end-to-end speedup.

| Stage | Measurement | Result |
| --- | --- | --- |
| Original V3 exact-gate reference | 128-proposal panel, 165 retained, 280.114 s | 2,120.56 queries/hour |
| Original V3, larger paired native A/B | 128 proposals, 667 matched native jobs, 495 retained | Original 3,271.50/hour; first pipeline candidate 2,925.67/hour. Science matched; initial speed gate failed (0.894×). |
| V3 scheduler/storage release | Fixed C24 30-minute live window, 2,508 retained in 1,802.15 s | 5,010.01/hour; later long windows fell below deadline pace |
| V3 release epoch overall | 12,276 new retained, including startup and drains | 3,546.07/hour; this is not steady-state throughput |
| Direct canonical gzip | Same saved native records, validator/storage CPU | 112.28 → 108.24 CPU seconds, 1.037×; not collection speedup |
| V4 geometry diagnostic | Contiguous 128 TRAIN root proposals; native root-only | 9/128 → 20/128 accepted; 9.13 → 17.61 accepted roots/core-hour (1.929×). Proposal law changed. |
| V4 startup | 495 retained over first 635.05 s | 2,806.08/hour; startup-inclusive |
| V4 first 30 minutes | 2,904 retained in 1,806.56 s | 5,786.91/hour; startup-inclusive |
| V4 first 60 minutes | 5,940 retained in 3,608.79 s | 5,925.52/hour; startup-inclusive |
| V4 rolling 30 minutes at 15:27 London | 2,739 retained in 1,804.83 s | 5,463.34/hour |
| V4 rolling 60 minutes at 15:27 London | 6,270 retained in 3,607.83 s | 6,256.39/hour; preferred sustained estimate |

At **9 October 2026, 15:27:05 London**, production was running, PID 2444471:

- Total sealed retained queries: **49,071** (all deeply verified at this snapshot).
- Legacy: 30,030; new dataset: 19,041; V4 source epoch: 6,765.
  Of the V4 epoch, 6,072 queries came from TRAIN geometry-mixture roots;
  natural/unseen queries still use the V3 recipe.
- TRAIN 35,409; natural_reference 6,435; unseen_evaluation 7,227.
- SR positive 41,623; SR negative 7,448; duplicate accepted IDs: zero.
- Default effective native concurrency **C24**. Saved worker pool is 36 with
  12 idle slots, 4 preparers, 2 validators, prefetch 8, backlog 36. These saved
  settings must remain unchanged on resume. C24 is enforced through
  `native_limit.json`; it is not a change of the saved pool size.
- Native utilization is measured from queue busy-slot integrals; the snapshot
  contains the exact fraction and latest busy/limit values. Approximately 93%
  of C24 slots were occupied across the V4 epoch and the latest queue was 24/24.
- Free space: output filesystem 621.99 GB, archive filesystem 278.55 GB,
  aggregate **900.53 GB**. These are two partitions of the same NVMe device.
- Required pace: 3,808.77/hour. Rolling-hour projection: **11 October,
  07:33:33 London**. First-hour startup-inclusive projection was 11 October,
  09:44:52. These are projections, not completion receipts.

`status` recomputes rolling 15/30/60-minute windows from published counts,
restricted to the V4 epoch. It prefers 60 minutes, then 30; it reports shorter
or saved startup evidence explicitly if longer data are unavailable. It does
not wait for another measurement. `docs/release_evidence/production_snapshot.json`
is a dated snapshot; use `status` for live counts and projections.

Resolved bottlenecks were repeated scheduler scans of the full growing root
history, repeated decoding in unchanged native audits, duplicate canonical
serialization/compression, excess staging storage, and expensive invalid TRAIN
root proposals. The scheduler uses an incremental active index; validation
caches decoded native records while preserving audits; canonical gzip is written
once and verified; new rejected prefix evidence is losslessly packed; backlog 36
bounds staging. V4 improves TRAIN proposal feasibility while preserving native
truth and evaluation/reference sampling. Native physics and prefix/rollout work
remain the main costs; C36 did not improve the measured slot-probe rate over C24.
The root-only geometry diagnostic used a 900 s diagnostic limit; production
keeps its original 300 s timeout. That diagnostic is not a production benchmark.

## Dataset structure and integrity

Workstation locations are centralized in `configs/eventwm_300k_production.json`.

| Data | Location / format |
| --- | --- |
| New raw dataset | `/tmp/eventwm_v4_300k`; alias `datasets/raw/eventwm_v4_300k` |
| New root spill archive | `datasets/raw/eventwm_v4_300k_archive` → `/home/haodong/workspace/datasets/eventwm-falcon/raw/eventwm_v4_300k_archive` |
| Verified legacy raw data | `output/eventwm_v3_30k_c36_combined_exact_gate`; immutable, never copied or relabelled |
| Role-preserving combined index | `output/eventwm_300k_combined_index/verified.sqlite3` (`roots`, `examples`, `scenes`) |
| Continuous deep integrity report | `output/eventwm_300k_combined_index/integrity.json`; only sealed verified roots, may lag publication |
| Successor targets | `output/eventwm_300k_supervision/<root_id>.jsonl.gz` and `<root_id>.provenance.json` |
| Runtime checkpoint | New `pipeline.sqlite3` (WAL, synchronous FULL): `meta` and ordered `roots`; `manifest.json` is periodic export |
| Frozen replay inputs | New `recipes/<root_id>.json`, `catalogue/episodes/<root_id>.json`, `catalogue/cases/<root_id>.json`, `catalogue/scenes.json` |
| Accepted seals / query index | `root_seals/<root_id>.json`, `primary_indices/<root_id>.json` |
| Logs / telemetry | `output/eventwm_300k_logs/`; dataset `logs/<session>/scheduler_events.jsonl`, telemetry and worker logs |

For each accepted root, resolve its index row's `dataset` + `path`, usually
`roots/<root_id>/attempt_XX`. Some new root directories are symlinks into the
archive. Never break those links or move only one filesystem's portion.

| Inside the root attempt | Meaning |
| --- | --- |
| `legal_observations/history.json` | Legal arrived history, including structured tracked humans |
| `legal_observations/image_storage.json` | Explicit image-retention metadata; COMBO_NO_RENDERER model image input is None |
| `queries/queries.json` | Immutable executable query definitions / controls |
| `labels/labels.json` | SR, native human/static contact outcomes, social outcomes, event times, validity and censoring |
| `branch_truth/branches.jsonl.gz` | Full canonical native future traces and future arrived legal observations; supervision only |
| `roots/legal_root.json`, `roots/privileged_root.json` | Legal root interface and separate privileged native state |
| `roots/prefix_truth.json.gz`, `roots/replay.json` | Recorded prefix evidence and deterministic replay recipe |
| `audits/audits.json`, `audits/storage.json`, `audits/encoding.json` | Scientific audits, canonical/raw equality and encoding/hash receipts |
| `VALIDATED.json`, `COMPLETE.json` | Durable validation/publication receipts; completion also requires the external seal |

Training inputs are constructed by `collector.core.load_wm_input`: history,
query and image-retention metadata only. Future/privileged truth is restricted
to labels and supervision. The handoff audit read **49,071 actual inputs across
1,487 sealed roots** and passed this boundary. The native branch gate checked
132 retained queries/4 roots, plus audits, against the pinned scientific source;
all 13 required integrity gates passed. The separate `human_reactivity` diagnostic
reports FAIL because native humans are exogenous to robot motion; this is a
confirmed unchanged simulator limitation, not a failed retention gate.

Successor sidecars contain targets at 0.25/0.5/1/2/3 s, actual recorded poses/yaws,
finite-difference velocities from actual timestamps/positions, future arrived
observations, physical event indicators, and terminal state. Target times use the
nearest actually recorded sample and retain the actual timestamp. Censored
poses/velocities are absent with false masks; early collisions retain SR=0 with
observed physical-event masks. `sr_label` is full-query SR. Auxiliary `horizon_sr`
means collision-free observed horizon, not full-query SR. These target semantics
are unchanged in the release. Provenance sidecars now publish atomically; raw
truth and accepted primary IDs are untouched.

At the 15:27 snapshot legacy Successor backfill was running (24,453/30,030 queries).
The refreshed Successor audit passed 44,517 emitted queries across 1,349 roots.
Raw native future truth already exists for every retained query. Derived coverage
is tracked by `output/eventwm_72h_finalization_status.json` and may lag collection.
The finalizer requires the combined retention target, deep seals, complete sidecar
coverage, sidecar hashes, final input audit, and final Successor audit before
writing `output/eventwm_72h_MISSION_RESULT.json`. A target/count receipt alone is
not the completed supervised dataset.

## Production commands

Run as Haodong without root. The explicit Conda interpreter is selected by the
configuration; shell activation is optional. All normal actions use one launcher:

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-300k
./scripts/run_eventwm_300k.sh check
./scripts/run_eventwm_300k.sh status
./scripts/run_eventwm_300k.sh logs
./scripts/run_eventwm_300k.sh logs --follow
./scripts/run_eventwm_300k.sh attach
./scripts/run_eventwm_300k.sh verify
```

`check`, `status`, `logs`, and `verify` never restart production, write checkpoint
metadata, change concurrency, or signal processes. `verify` checks the existing
combined SQLite index and its deep-verification receipt; it does not reread the
entire raw dataset. Full seals continue to be checked by the integrity helper.
Detach from tmux with **Ctrl-b, then d**. Ctrl-c stops a log-follow command only;
do not press Ctrl-c in the collector's attached pane.

When the collector has stopped and the saved dataset is intact:

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-300k
./scripts/run_eventwm_300k.sh check
./scripts/run_eventwm_300k.sh resume
./scripts/run_eventwm_300k.sh status
```

For a genuinely fresh, empty configured output (not this existing dataset):

```bash
./scripts/run_eventwm_300k.sh start
```

Both `start` and `resume` **refuse while the existing collector is running**.
`start` rejects any existing checkpoint or unrecognized nonempty directory.
`resume` rejects missing or incompatible checkpoints instead of creating fresh
data. There is no reset command. Native dataset locks, a serialized launcher
lock, the tmux session, process identity checks, and frozen runner guards prevent
simultaneous collectors. Accepted query IDs are checked against legacy and new
completed data before recovery; the combined index adds UNIQUE constraints.

The configured total target is `retained_target`. For a different explicitly
planned mission use an explicit copy of the configuration and set
`EVENTWM_PRODUCTION_CONFIG=/absolute/path/to/config.json`; the launcher passes
that absolute configuration into durable tmux commands. Saved
`pipeline_settings`, including its target ceiling, must match on resume. Keep
the present target at 300,000 and its saved settings unchanged. Do not edit the
pinned scientific release or overwrite its output to change a target.

Graceful shutdown, only when intentionally requested by the operator:

```bash
./scripts/run_eventwm_300k.sh stop
./scripts/run_eventwm_300k.sh status
```

`stop` verifies host/PID/command/output/source identity and sends SIGTERM only to
the coordinator. Its frozen handler drains reserved preparation, native,
validation and publication work. It does not kill workers or delete checkpoints.
Poll until stopped; an in-flight root can require its normal timeout/retry and
publication time. Do not kill the tmux session to stop collection.

To restore missing support sessions after reboot or a helper crash:

```bash
./scripts/run_eventwm_300k.sh helpers
```

`resume` calls this automatically. Existing named helper sessions are left
running. Helpers include progress/deadline accounting, deep integrity indexing,
native metrics, sampler accounting, new and legacy Successor extraction, and
finalization. They run in tmux with persistent logs, low process/I/O priority,
per-helper locks, and retry receipts. They never automatically restart the
collector. No exploration or slot-response helper is launched.

Manual read-only raw/input and derived-target audits, if an integrity concern
warrants them (report files are additive):

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_production/audit_combined_inputs.py
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_production/audit_successor.py
```

## Recovery and confirmed limitations

- **SSH disconnection:** collector and support jobs remain in tmux. Reconnect,
  run `status`, and attach if needed. Do not run another collector.
- **Collector crash:** inspect `logs`, `status`, `verify`, and any
  `eventwm_72h_COLLECTOR_ALERT.json` or combined-index `FAILURE.json`. Run `check`.
  If scientific provenance and integrity pass, run `resume`. Completed sealed
  roots are preserved; incomplete work resumes from saved recipes/attempts under
  the existing maximum-two-attempt policy. The launcher does not delete a failure
  receipt or repair scientific data silently. Resolve an actual integrity failure
  before resuming.
- **Workstation reboot:** ext4 persists the files, but cv-w06's
  `/usr/lib/tmpfiles.d/tmp.conf` has `D /tmp ...` and boot setup runs tmpfiles
  with `--remove --boot`. An unprepared reboot can remove the dataset. Before
  a planned reboot, stop gracefully, wait for a stopped checkpoint, then run:

  ```bash
  ./scripts/run_eventwm_300k.sh preserve-for-reboot
  ```

  This refuses a live collector or existing backup, verifies checkpoint/IDs,
  stops only the named support sessions to avoid missing-output alarms,
  and atomically renames the complete dataset to
  `/var/tmp/eventwm_v4_300k_reboot_backup` on the same filesystem. No data are
  copied or overwritten. `/var/tmp` has no active cleanup rule for this path.
  After reboot:

  ```bash
  cd /home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-300k
  ./scripts/run_eventwm_300k.sh restore-after-reboot
  ./scripts/run_eventwm_300k.sh check
  ./scripts/run_eventwm_300k.sh verify
  ./scripts/run_eventwm_300k.sh resume
  ./scripts/run_eventwm_300k.sh status
  ```

  Restore refuses an existing output and verifies the saved source/settings,
  completed seals and IDs before an atomic rename to the original literal
  `/tmp` path. This preserves path-specific frozen storage behavior. Do not
  replace the native output with a permanent symlink to another directory.
  These commands were tested only in isolated temporary data during handoff;
  the live dataset was not moved. An unprepared reboot without a preserved
  backup can lose new raw data; the archive and combined index alone are not
  a complete backup. Missing output is a refusal, never fresh initialization.
  Native in-memory COW checkpoints disappear on reboot; unfinished roots
  replay saved episodes/prefixes under the existing recovery policy. An unrelated
  reused PID is never signalled. Inspect/close a stale dead collector tmux shell
  before resume; never kill a live collector session. There is no root service
  or automatic reboot launch.
- **Low disk space:** `check` projects remaining accepted data using the measured
  conservative pilot bytes/query plus 140 GB rejected-prefix reserve, 80 GB working
  reserve and 40 GB headroom; it also checks native filesystem and aggregate free
  floors. The frozen runtime resource guard drains when aggregate space reaches
  its capacity floor (128 GB with backlog 36/prefetch 8), or available RAM is below
  16 GiB. Stop gracefully if additional space must be provisioned. Do not delete
  sealed roots, checkpoint files, links or shared WM assets. Recheck capacity
  before resume. Two partitions share one NVMe; their free capacity is not I/O
  bandwidth from two drives.
- **Interrupted validation or packing:** run `check` and then `resume` after the
  collector is stopped. Existing recovery verifies COMPLETE/seals and validated
  receipts; raw/canonical equality and compressed/decoded SHA receipts protect
  repacking. Partial unsealed work is handled by the existing retry/recovery path.
  Do not manually rename temporary packing files to COMPLETE or remove receipts.
- **Interrupted derived-target extraction:** rerun `helpers`; extraction skips
  completed provenance markers and rebuilds unfinished additive sidecars. Legacy
  reuse checks the saved artifact hash. Finalization requires all covered hashes
  and audits, so partial extraction is not reported as complete.

Scientific limitations are exogenous native human motion and unestimated V4
TRAIN importance weights. Operational limitations are throughput variability,
finite shared-disk capacity, derived coverage lag, required pre-reboot preservation
of the /tmp dataset, and the explicit cv-w06 asset configuration. There is no unverified second workstation. No speculative
optimization, automatic collector restart, destructive reset, push or merge is
part of this handoff.

## Release validation and excluded work

Validation uses script/AST syntax checks, importable-module checks, existing
pipeline/combined/shadow unit tests, isolated launcher fault tests, source/hash
and configuration checks, read-only status/log/index checks, live duplicate
start/resume refusal, and checkpoint compatibility. Existing native science
and storage gates are reused with exact source/hash provenance; no second live
collector or new expensive native experiment was executed.

The first new status test exposed old progress records without timestamps; the
reader was fixed and the test passed. A direct import attempt of the standalone
worker and validator CLIs failed because they require `--collection-output` at entry; they are
syntax-checked as executables, while the six importable production modules
passed imports. There are no unresolved release-check failures. Details and
commands are in `docs/release_evidence/validation.json`.

Included: inherited validated V3/V4 pipeline and configurations; the consolidated
launcher; production monitoring/recovery/extraction/finalizer utilities; read-only
scientific audits; safety tests; handoff documentation and dated evidence.
Excluded from staging: main-worktree SR audit edits and reports, EventQueryWM and
WM training changes, unrelated archives, all other dirty worktrees, experimental
slot profilers/responders, geometry probes/launchers, old source-migration scripts,
new benchmark code, and obsolete alternative launchers in `output/`. These files
were preserved where they already existed. Historical experimental files already
in scientific commit 58b remain in the source tree but are not production entry
points. No unrelated experimental branch was merged.
