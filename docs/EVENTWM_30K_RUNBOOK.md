# Frozen EventWM V3 C36 collection

Run from `/home/haodong/workspace/projects/eventwm-falcon/output/root_admission_checkout`
on `research/eventwm-v3-root-admission-exploration`. The launcher verifies the
committed source/configuration hashes and the clean dd1400e scientific reference.
It refuses another collector, a changed manifest/commit, or an incompatible resume.
Preserved untracked scratch files are not imported or included in the release.

The configuration is fixed: COMBINED C36, one thread per worker, no visual
sensors/renderer, fresh root physics with the measured immutable asset cache,
isolated COW branch continuations, the exact state-dependent root clearance gate,
33 primary queries, both reverse audits and truncated-WAIT audit, unchanged V3
labels/static checker/legal history, and asynchronous unchanged validation.
The gate calls the canonical checker. Its diagnostic boundary is
`max(0, measured_footprint_radius - 0.25) + 0.03 + 0.01`, never fixed 0.38 m.

## Launch and inspect

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/output/root_admission_checkout
./scripts/run_eventwm_30k.sh check
./scripts/run_eventwm_30k.sh start
./scripts/run_eventwm_30k.sh status
./scripts/run_eventwm_30k.sh attach
```

Detach with Ctrl-b then d. Closing an SSH terminal does not stop tmux or the job.
The session is `eventwm_30k`. Logs are appended under `output/eventwm_30k_logs/`
in this worktree, with worker/validator/resource logs in the dataset's `logs/`.
Status reports retained count, cumulative active-collection q/h, failures and ETA.
It is a run average, not an instantaneous rate.

## Stop and resume

```bash
./scripts/run_eventwm_30k.sh stop
while tmux has-session -t '=eventwm_30k' 2>/dev/null; do sleep 5; done
./scripts/run_eventwm_30k.sh status
./scripts/run_eventwm_30k.sh resume
```

Stop sends SIGTERM to the verified coordinator only. It stops creating proposals
and drains prepared roots, all branches/audits and validation. Wait for
`stopped_by_user` and session exit before resuming. After reboot, `resume` uses
the same output, source commit, logical proposals, saved episodes and seeds.
Completed roots are deeply verified by the existing completion-marker/seal
mechanism and skipped; a completion published just before the manifest update
is recovered. A validated staged root is published only after its integrity
checks pass. All files are synced before the completion receipt.

Native COW checkpoints are in-memory, not simulator snapshots on disk. An
uncommitted interrupted root keeps its old attempt/evidence and is replayed from
its saved recipe/episode using the existing two-attempt retry policy. This can
repeat in-flight work; it never destroys completed progress or resamples that
logical proposal. A duplicate live worker makes resume fail rather than race it.
Unexpected science errors or corrupted seals stop the run visibly.

## Output and target

`/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_v3_30k_c36_combined_exact_gate`

This is a new dataset; Scale04 is not edited, resumed or deleted. The existing
952-scene family-interleaved sampler, roles, coverage accounting and master seed
are reused. Final data schema, primary indices and SR contract remain V3.
The target is at least 30,000 retained queries; complete root bundles are kept,
so the last root may take the count slightly above 30,000. No bundle is truncated.
The original 150 GB disk guard plus C36 reservations requires 222 GB available
at dispatch, and the 16 GiB available-RAM guard remains enabled.

## Frozen evidence

The paired 128-proposal C36 test retained 165 queries in 289.403 s without the
gate and 280.114 s with it (2,120.56 q/h). All 308 scientific job comparisons,
165 public WM inputs, 180 labels and 308 decoded episode records matched.
The source-preserving production integration test compared 76 jobs, 66 public
inputs and 72 primary/audit labels exactly, including safe/social/collision
outcomes and gate/legal failures. Completion-before-manifest recovery retained
all 66 queries with zero new Habitat jobs. Small-panel rates do not guarantee
sustained production speed. The 30k point estimate at 2,120.56 q/h is 14.15 h.

No mass collection or merge was performed while freezing this release.
