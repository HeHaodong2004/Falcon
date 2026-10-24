# Progress-first local executor

L4 follows the immutable selected query by default, attempts a small reactive
maneuver when useful, and returns to nominal execution when avoidance fails.
It never selects another goal, runs graph rerouting, emits ordinary BLOCKED,
uses STOP to wait, or ends a query through a local safety watchdog.

Read [EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md) for the current integration
boundary, [DIAGNOSIS.md](DIAGNOSIS.md) for measured historical failure mechanisms,
and [FINAL_REPORT.md](FINAL_REPORT.md) for paired results and readiness limits.
The previous release and every unsuccessful development run remain available.

The worktree is `/research-user/workspace/projects/eventwm-falcon-l4-social-executor-20261022`,
branch `research/l4-social-executor-20261022`, starting delivery `167a53a9`.
Use the existing environment without package upgrades or training:

```bash
source scripts/l4_env.sh
"$L4_PY" -m unittest discover -s tests/l4 -v

# Use new destinations; runners refuse existing directories.
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/social_executor_v1_20261022/navigation_public12.json \
  --variants B0,B1,B2 --output output/l4_public_replay
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/social_executor_v1_20261022/native_scenarios.json \
  --variants B0,B1,B2 --output output/l4_scenarios_replay
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/progress_first_20261023/short_queries.json \
  --variants B0,B1,B2 --output output/l4_short_replay
nice -n 19 ionice -c 3 "$L4_PY" scripts/l4_run_native.py \
  --protocol docs/l4/progress_first_20261023/additional8.json \
  --variants B0,B1,B2 --output output/l4_additional_replay

"$L4_PY" scripts/l4_progress_report.py \
  --protocol docs/l4/social_executor_v1_20261022/navigation_public12.json \
  --roots output/l4_public_replay \
  --historical docs/l4/social_executor_v1_20261022/artifacts/navigation_summary.json \
  --output output/l4_public_summary_replay.json
"$L4_PY" scripts/l4_audit_progress.py \
  --roots output/l4_public_replay output/l4_scenarios_replay \
          output/l4_short_replay output/l4_additional_replay \
  --output output/l4_progress_audit_replay.json
```

All runs use one numerical thread and low process/I/O priority. The 28-worker
collector must stay running. Protocols preserve native actions, metrics, human
controllers, geometry, seeds, episode membership and success thresholds. The
additional eight episodes were selected from public validation content in four
other certified scenes before candidate evaluation; no reserved confirmation
artifacts were opened. Short queries use fixed 3 m route prefixes with the
canonical Stage 1 adapter, not a simulated high-level policy. Their native
navigation metrics still refer to the original episode goal; query completion
is reported separately. Driver STOP after an isolated completed short-query
trial is outside the executor's feedback ledger.

Full native traces and development failures are retained under
`output/l4_progress_first_20261023/`. Each native arm stores the scenario, query,
actual observations/actions/poses/timestamps, native outcome labels and timings.
Source/protocol SHA256 receipts and summary evidence manifests bind those files.
Committed `artifacts/` contains the final summaries, per-case tables, audit,
profiling evidence and preservation receipt. The source receipts identify the
evaluated controller; documentation and reporting may be added afterward.

The frozen L2 checkpoint remains unchanged at the path documented in the prior
release (SHA256 `215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`).
No L2/L2.5/L3 implementation or collector configuration is changed.
