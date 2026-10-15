# Physical training budgets and exact exposure

This table contains completed physical training only. Probability calibration,
occupancy fitting, software audits and unexecuted full-data configurations are
separate. Queries and windows are not independent human futures.

| Run | Fitting roots | Completed updates | Root presentations | Selected step | Active training seconds | Peak allocated GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| pilot_velocity | 1,024 | 8,192 | 262,144 | 4,608 | 1587.01 | 1.189 |
| pilot_window015 | 1,024 | 8,192 | 262,144 | 3,072 | 1596.43 | 1.189 |
| pilot_self | 1,024 | 8,192 | 262,144 | 7,168 | 1599.35 | 1.190 |
| pilot_occupancy2 | 1,024 | 8,192 | 262,144 | 4,608 | 1659.99 | 1.189 |
| pilot_arrival | 1,024 | 8,192 | 262,144 | 2,560 | 1662.52 | 0.490 |
| pilot_polar | 1,024 | 8,192 | 262,144 | 8,192 | 1682.94 | 1.142 |
| pilot_occupancy10 | 1,024 | 8,192 | 262,144 | 4,608 | 1660.40 | 1.189 |
| pilot_sampling | 1,024 | 8,192 | 262,144 | 3,072 | 1621.49 | 1.179 |
| intermediate_posterior | 4,096 | 16,384 | 524,288 | 6,656 | 3237.84 | 1.521 |
| intermediate_self | 4,096 | 16,384 | 524,288 | 9,728 | 3286.02 | 1.293 |
| full_self_s20261014 | 8,965 | 24,576 | 1,562,598 | 17,152 | 5500.33 | 3.222 |
| full_self_s20261015 | 8,965 | 24,576 | 1,562,598 | 15,104 | 5373.40 | 3.240 |

All eight new pilots use seed 20261014, batch 32 and 1,024 nested roots with
5,556 actual windows. The importance-corrected sampling pilot uses replicated
proposal slots, which are not additional windows. Both intermediate runs use
the same 4,096 nested roots, 22,175 actual windows, batch 32 and seed 20261014.
Full-data fitting retains 8,965 roots and 48,434 windows.

The selected step is chosen by raw three-second development ES within the
predeclared training budget. Post-training probability-law comparisons use
their own training-only calibration and fixed-budget development evaluation.
Active duration excludes time paused by the collector guard; it is not wall
time from initial launch to completion. Peak GPU memory is the allocated tensor
peak in the completing invocation, not a maximum across interrupted invocations
or total driver reservation. Resource probes and attempt ledgers supplement it.

Sampler audits replay shuffled root order, cursor, draws, RNG state and seen
roots against both last and selected optimizer checkpoints. The original v1
matched controls remain immutable; see ABLATIONS.md for their exact results.

New full-data physical runs were promoted in commit `0dcaa09` with seeds
20261014 and 20261015, 24,576 updates each and batch 64. Both schedules completed,
with all 8,965 fitting roots seen and 1,562,598 presentations per seed. Epoch-tail
batches explain why this is smaller than 24,576 times 64. The selected early
checkpoints remain distinct from the completed final optimizer checkpoints.
Exact sampler replay verifies every fitting window was presented in both
selected and final checkpoints. Seed 14 selected/final exposure is 1,090,589 /
1,562,598 presentations; seed 15 is 960,343 / 1,562,598. Every window appears
at least four times by either selected checkpoint, and at least 9 / 10 times
by the respective final checkpoints. Replay exactly matches cursor, root order,
sampler RNG, seen roots and presentation counters. Completed full-calibration
comparisons are reported separately from these raw selection curves.

The raw three-second selection curve made no new best over the final 7,424 and
9,472 updates respectively. There is no measured basis to extend this schedule.
These raw scores are checkpoint-selection diagnostics, not final model comparisons.
Across the eight pilots, two intermediate runs and two full runs, the campaign
has completed 147,456 physical optimizer updates and 6,270,924 root presentations.
Probability-fitting steps and the bounded resume software audit are separate.
