# Collector supervision, 11 October 2026

The collector stopped at 22:10:12 BST on 10 October. Its `STOP.json`,
`SUMMARY.json`, persistent `collector.log` and worker result agree: native branch
`b007`, query `q_4c518149379d`, root `v4g2_2cc61907376608bec99f`, scene
`hm3d_j6fHrce9pHR`, seed 731052107 reported a static feasibility violation at
3.960 seconds. The frozen coordinator raised at line 404 and returned exit code
2. The failed branch has no seal and was never retained. The saved checkpoint
was `interrupted_error`, with 127545 new plus 30030 legacy verified queries.

The preceding recovery repaired one scientific source epoch. It did not supervise
the collector: `_run` exited with the runner, tmux discarded its session, and
`run_helper` restarted only individual monitoring helpers. The deadline helper
wrote an alert but did not resume collection. Thus surviving monitors offered no
collector ownership or recovery guarantee. Kernel records in the failure window
showed no OOM, segfault or storage errors; repeated unrelated NVIDIA version
mismatch reports do not explain the explicit native integrity exception.

The guarded manual resume on 11 October used collector PID 3286596. Its frozen
verifier reread all completed root data before starting native work. It recovered
one previously validated root, raising retained publication to 157608 (127578
new plus 30030 legacy), then executed at C24 with approximately 75.6% host CPU.
At 18:54 BST the same query failed on attempt 02 with the same 3.960-second
static event. The static certificate had declared feasibility; execution recorded
a Spot link-19/stage contact with normal Y 0.799787104, below the unchanged 0.8
ground-support cutoff. This is a reproducible certification mismatch, requiring
a separate scientific repair release, not an operational restart loop.

The guard recorded collector death and blocked further automatic recovery. The
checkpoint remains valid and all 3865 pre-existing root seals and query IDs were
checked unchanged. Sustained production and the requested five-minute publication
verification were not achieved. Do not use repeated manual resumes to exhaust
the frozen interrupted-attempt policy and silently skip this known failure.
The separately authorized repair is now validated in science commit `eac1c27`.
See [the contact repair release](EVENTWM_STATIC_CONTACT_REPAIR.md) for the
confirmed manifold-refresh cause, bounded native gate and checkpoint transition.
Use the current pinned launcher after the explicit transition.

The operational guard runs independently in `eventwm_300k_recovery`. It checks
the actual runner PID, command, output and source commit every 15 seconds. It
waits while the collector tmux wrapper verifies sealed roots or cleans up workers.
After collector death, it records the collector's own STOP reason and permits at
most three resume attempts per hour, with the budget persisted across guard
process restarts. Every automatic resume uses the existing checkpoint preflight,
verified index and launcher lock, then the frozen runner's dataset lock. It
never edits root records, recipes, seeds, scientific code or sealed data.

Scientific/static/duplicate/seal/hash/source failures, failed preflight or index
integrity, missing checkpoint, intentional/resource stops and exhausted restart
budgets disable automatic recovery. `stop` disables recovery under the same
admission lock used for restart before signalling a graceful collector drain.
The guard remains idle after disabling so an explicit later resume can re-enable
it. Unexpected guard termination is visible as a missing tmux session and stale
heartbeat; this guard is not a system boot service.

From the operational recovery checkout:

```sh
scripts/run_eventwm_300k.sh status
scripts/run_eventwm_300k.sh recovery-status
scripts/run_eventwm_300k.sh verify
scripts/run_eventwm_300k.sh logs --follow
tail -F /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_logs/recovery.jsonl
scripts/run_eventwm_300k.sh stop
scripts/run_eventwm_300k.sh resume
```

`resume` re-enables the guard after a guarded manual launch. To adopt an already
running collector, use `scripts/run_eventwm_300k.sh recovery-start`. Neither
command resets the recent restart budget. Do not treat historical native metrics
or a monitor PID as proof of active execution; verify `active`, native workers,
and increasing retained publication counts.
