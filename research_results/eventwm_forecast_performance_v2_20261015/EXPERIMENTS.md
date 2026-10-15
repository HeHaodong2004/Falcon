# Experiment completion ledger

Final finite queue: 186 completed jobs, zero pending, zero blocked, no active research workers. The unique-output inventory also preserves superseded owners, failed attempts and unexecuted configurations. These are not 196 independent trained models.

| Inventory status | Output directories |
|---|---:|
| completed | 196 |
| failed | 4 |
| not_executed | 9 |
| paused_resource | 1 |
| cooperatively_stopped | 18 |
| completed_with_failed_children | 1 |

The full table is [EXPERIMENTS.csv](EXPERIMENTS.csv), including actual optimizer budgets and comparable development metrics where a run produced them. `experiment_ledger.json` and `attempt_compute_ledger.json` retain source receipt hashes and compute records.

Completed physical fitting totals 147,456 updates, 6,270,924 presentations and 30467.74 active seconds. Two GPUs may overlap; summed active seconds are not elapsed wall time. Calibration budgets and successful-invocation runtimes are separate in the table.

After lifecycle instrumentation, 160 non-owner attempts record 25050.82 summed wall seconds and 24693.54 CPU seconds. Largest recorded child RSS is 6.348 GiB. This partial ledger excludes earlier uninstrumented attempts and manual analyses and must not be presented as total campaign cost.

The eight new pilots, two intermediate runs and two full-data runs completed. Original v1 controls are reused immutable references and are not charged as newly trained v2 runs. The nonincremental stopped-law attempt remains paused/superseded with its evidence intact; its incremental replacement completed. The CV cache mismatch and old-population physics attempts failed before producing scores and have completed corrected replacements. The 500k configurations remain unexecuted because the gate is closed. All earlier research owners are inactive.

The bounded search is closed: all eight physical pilot slots were used, the promoted intermediate gain failed to persist at full scale, both full curves saturated within the planned schedule, and the final probability follow-up showed a calibration/ES tradeoff. Further fitting after confirmation would invalidate the untouched-bank claim. No such fitting occurred.
