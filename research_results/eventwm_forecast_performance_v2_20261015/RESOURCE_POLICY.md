# Authorized training-priority resource policy

The user explicitly replaced the original throughput protection after repeated
pauses near 2,100 queries/hour. The old fixed reference was 3,615.965 q/h: three
rolling windows below 65% (~2,350 q/h) stopped research, then three above 85%
(~3,074 q/h), spanning 120 seconds, were required to recover. That hysteresis
kept healthy research idle even though collector production continued.

The initial authorized replacement was `training_priority_absolute_qph_v1_20261015`
in `4844a51`. After both physical schedules completed, the effective policy became
`training_priority_absolute_qph_v2_20261015`, with owner
`workflow_training_priority_v16`. Older owners remain stopped. The second revision
removes a later-stage starvation case: low throughput must not defer every
remaining CPU calibration/evaluation job. Thresholds and hard constraints are
unchanged; only physical-training priority can defer those auxiliary jobs.

| Condition | Action |
|---|---|
| At least 1,500 queries/hour | Target met |
| 1,000–1,500 queries/hour | Acceptable; keep training |
| Three valid windows below 1,000 | Use 90% research duty; keep bounded CPU progress eligible |
| Six / nine consecutive low windows | Bound duty at 80% / 60%; continue progress |
| Two valid windows above 1,500 | Clear legacy throughput pause and restore full duty |
| Collector PID/state/heartbeat failure | Stop research safely |
| Available RAM below 48 GiB or disk below 50 GiB | Stop research safely |
| New kernel OOM kill | Stop for inspection; do not silently clear the fault |
| Non-resource worker failure, including CUDA OOM | Preserve failure and checkpoint; inspect before retry |

Throughput alone does not create a new blanket suspension. Throttle sleeps are
bounded to five seconds and do not change optimizer, scheduler, RNG or sampling
semantics. GPU launches still require 8 GiB free; one trainer per GPU is retained.
The existing one-thread, nice-19, idle-IO research launch limits remain effective.
CPU load by itself is no longer treated as a hard failure.

Measurements use query-counter differences over collector heartbeat timestamps,
with windows of at least 600 and at most 900 seconds. Endpoints must belong to the
same healthy collector and have monotone counts. New decisions are spaced at
least 60 seconds; stale/repeated heartbeat observations do not count as new valid
windows. Invalid measurements do not count toward low/recovery streaks. Lifetime
`qph` remains diagnostic metadata and is not substituted for the rolling rate.
The original measurement history and old guard files are preserved for audit.

The scheduler prioritizes both `full_self_s20261014` and `full_self_s20261015`
until their planned physical schedules finish. It defers other jobs during that
period, including repeated snapshot/audit/preprocessing work. Subsequent bounded
calibration/evaluation dependencies remain queued. After physical training,
CPU work remains eligible even during low-throughput throttling. The v15 owner
allows at most two single-threaded CPU workers. Each guard applies the same
bounded work/rest duty as GPU jobs. Raw extraction
also requires healthy fresh evidence but does not treat slow throughput as a
blanket stop. The collector is never signalled or modified.

## Exact continuation and bounded verification

The previous owner and all its children were verified inactive before handoff.
The saved checkpoints, identities and progress files were copied to
`resource_policy_transition_20261015/` before resuming. The actual loader receipts
match those preserved bytes:

| Seed | Loaded update | Loaded presentations | Saved checkpoint SHA256 |
|---|---:|---:|---|
| 20261014 | 2,677 | 170,266 | `43d5527d29b6b7b26aca012ecd5762f44afd17d6a391bf317bf03517dec2b441` |
| 20261015 | 2,723 | 173,151 | `12aa59a800263e3bf17ebdafc64f68be0ce3f55c05b413aaf9bf70263e8f6b7e` |

All physical training-source fingerprints, configurations, dataset manifest and
fitting/development identities are unchanged. The original `guard.py` is included
in that fingerprint, so `policy_worker.py` installs the new guard explicitly at
runtime before importing the worker. Each attempt records the replacement source
hashes and the legacy guard hash in `runtime_policy.json` and an append-only
history. This records a scheduling change without rewriting or disguising the
saved optimizer identity. The unchanged trainer restores model weights, AdamW,
CPU/CUDA/sampler RNG, shuffled root order/cursor, counters and selected-best state.
Its learning rate remains the same deterministic function of saved update count.

Seven bounded guard/dispatch tests passed, covering the new recovery threshold,
the acceptable band, sustained-low-rate throttling, invalid/rapid measurements,
heartbeat-based rate arithmetic, and retained RAM/disk/health/OOM stops. The live
policy smoke check passed at 2,506.52 q/h with no hard failure. Both workers then
loaded their exact saved checkpoints and advanced to update 2,944.

The first post-resume resource receipt measured 1,400.20 q/h, with training still
running as intended. Over its three-second host sample, CPU busy was 76.97%,
I/O wait 0.17%, NVMe busy 6.03%, and writes 11.78 MiB/s. Each research worker used
approximately one CPU core. Available RAM was 184.28 GiB and disk 320.17 GiB.
Instantaneous GPU utilization was 16% / 49%; these are observations, not a steady
state average or an inference-latency claim. Later NVIDIA process accounting
showed roughly 13.3 / 19.4 GiB held by the two research processes; allocator-held
device memory is not interchangeable with PyTorch live-tensor peak allocation.

Receipts are `resource_policy_transition_20261015/live_policy_smoke.json`,
`after_resume_1792084385.json`, the two `resume_load_receipt.json` files, and
`owner_handoff_v12_v13.json`. Collector PID 3719917 remained healthy and unchanged.
No confirmation roots were fitted. The 495,000 validated-query scaling gate still
applies only to an explicitly expanded dataset; it does not gate these 350k runs.
Both physical schedules subsequently completed at 24,576 updates and 1,562,598
presentations per seed. Eleven bounded guard/dispatch/scaling-gate checks passed
for the second policy revision, including healthy 800 q/h CPU-extraction
eligibility and retained hard stops. Calibration, final evaluation and release
remain in progress.

A post-physical-training five-second probe at timestamp 1792090622 recorded
5,093.36 q/h over the guard's valid rolling window, 71.94% CPU busy, 0.20% IO
wait, no kernel OOM, 183.88 GiB available RAM and 311.65 GiB disk. GPU 0 was
2–8% utilized and GPU 1 0% during CPU evaluation; both physical schedules had
already completed. All four selected/final checkpoint hashes were reverified.
The queue had no blocked jobs and retained the same production PID. Receipt:
`resource_policy_transition_20261015/after_physical_completion_1792090622.json`.

The v15 and v16 owners separate CPU and GPU concurrency: at most two
single-threaded, nice-19/idle-IO CPU jobs, while retaining one job per GPU.
The measured 72% CPU utilization and 184 GiB available RAM support this bounded
post-training increase. The horizon-balanced probability control is scheduled
early because its result determines a possible final fitting-only follow-up.
All 125 child recipes and their optimizer/data identities are unchanged. Twelve
bounded guard/dispatch/scaling tests pass, including a check that two CPU slots
cannot admit a second worker on an occupied GPU. All adaptive throughput and
hard safety checks remain active. `owner_handoff_v14_v15_stop.json` verifies
the old owner and every child exited before launch; `owner_handoff_v14_v15.json`
verifies sole ownership. Collector throughput later measured 1,127.10 q/h,
inside the accepted band, with research continuing at full duty and no blocked
jobs. The horizon-balanced fit and subsequent independent CPU work resumed.
The v16 successor retains those same limits and runtime policy. Its verified
handoff receipt records 2,136.80 q/h, the unchanged collector PID, one CPU gate
fit and two independent GPU probability fits running, with no blocked jobs.
See `owner_handoff_v15_v16.json`; this is execution progress, not a completed
calibration or release claim.

## Completed schedules and observed adaptive behavior

Both physical seeds completed 24,576 updates and 1,562,598 presentations from
the exact paused states; all fitting roots and windows were used. The legacy
throughput threshold no longer blocks their work. A post-training calibration
probe at timestamp 1792093990 records 46.44% aggregate CPU use, 1.44% IO wait,
202.84 GiB available RAM, 306.70 GiB disk free, GPU utilization 2% / 20%, and
2,443.45 rolling collector queries/hour. No kernel OOM event was recorded.
These are interval/snapshot measurements, not campaign-wide utilization means.

Later, bursty completion counters produced three valid rolling windows below
1,000 q/h, reaching 191.34 q/h. The guard applied duty .9 and kept research
running. Two later valid windows above 1,500 cleared throttling; a subsequent
measurement was 2,890.94 q/h. The collector heartbeat stayed fresh, PID stayed
3719917, and completed roots increased. This exercises both sides of the new
hysteresis on real measurements without a blanket pause or production changes.
An isolated calibration CUDA fault was recovered after a bounded failing-chunk
reproduction and hash checks; its preserved evidence is separate from throughput
guarding. The new full physical runs were already complete and unaffected.

## Final finite queue and collector status

The v18 queue completed all 186 scheduled jobs with no pending, running or blocked jobs. Earlier owner and failed-attempt receipts remain preserved and are classified in EXPERIMENTS.md. The final queue rolling throughput was 1,600.94 q/h, above the new target, with no throttle or hard failure. A nearby two-second resource probe at timestamp 1792096605 recorded 86.11% CPU busy, .21% IO wait, 204.48 GiB available RAM, 303.75 GiB disk free, and GPU utilization 16% / 0%. The GPUs were no longer training because both planned full schedules had completed. NVMe writes were 34.30 MB/s with 9.98% device busy; no kernel OOM kills were recorded. These are snapshot/short-window observations, not campaign averages.

The final read-only scaling check found 457,314 validated queries and 13,858 roots, below 495,000, with no pending priority physical training. No 500k snapshot or fitting was attempted. The independent collector remains active; its worker allocation, output, configuration and target were not modified.
