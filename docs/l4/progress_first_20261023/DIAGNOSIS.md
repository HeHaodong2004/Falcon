# Diagnosis from retained actual traces

Source: delivery `167a53a92b93305306519f1e47c8393ffd0475ed`, executor
`e19431fb`, full traces in `output/l4_social_executor_v1_20261022/release_navigation`.
The [machine-readable replay](artifacts/previous_trace_diagnosis.json) binds
each trace by SHA256. Counts below measure executed native actions, not
simulated counterfactual successes.

| Mechanism | Measured evidence | Consequence and correction |
|---|---|---|
| Whole-plan rejection before execution | The old greedy proposal accepts only an entirely screened 32-action plan. Public B1 has 86 decisions overriding a nominal forward action. | A distant predicted problem can suppress an immediately useful step. Compute nominal first and only activate bounded avoidance for near-term risk. A nominal override alone does not prove counterfactual success. |
| Static rejection plus receding one-action search | `hm3d_3t8DB4Uzvkt_public_0`: 26,829 static candidate rejections, zero human rejections, 103 turns, zero forwards, 102 adjacent turn reversals. `hm3d_5cdEh9F2hJL_public_1`: 103 turns, zero forwards, 100 reversals, zero human rejections. | These are measured failures to execute under static screening, not insufficient human prediction. Failed cheap alternatives now preserve nominal execution. |
| Nonpersistent maneuver selection | Public B1 totals 1,159 adjacent turn reversals; `mp3d_2azQ1b91cZZ_public_2` alone has 250. The old preference is a soft cost; plans are regenerated each action. | A preference cannot enforce commitment. Commit a bounded side maneuver and require actual translation between attempts. |
| No-progress watchdog and empty-plan handoff | Six public episodes stop with `no_meaningful_progress`; two with `finite_search_no_admissible_action`. All eight use driver STOP after BLOCKED. | Early stopping reduced collision exposure while destroying completion. Remove both active-query handoffs and let native limits/outcomes govern. |
| Failed route rejoining | `hm3d_5cdEh9F2hJL_public_0`: 1.22 m final route deviation, one detour, zero rejoins. `mp3d_8194nk5LbLH_public_2`: 1.13 m deviation, one detour, zero rejoins. Old advancement required a sampled pose inside every waypoint disc. | Local detours can miss a waypoint and be directed backward toward it. Accept narrow ordered plane/adjacent-leg rejoin gates; keep endpoint and terminal-yaw tests unchanged. |
| Repeated full-body mesh/FK fallback | 1,101 of 1,752 public B1 control calls use the fallback. Public B1 P95 is 267.933 ms. Retained worst-state profiling spends 85.6% in triangle/link checks (469 candidate transitions per replay). | Fine geometry search is inappropriate for routine RL execution. Preserve geometry utilities and their regression witness, but remove the online mesh search and repeated calibration. |

These mechanisms interact. A nominally required turn can also be repeatedly
reversed by static search, so the 86 forward-override count understates the
loss of useful actions. The zero-forward traces directly establish the failure
without assuming how their unexecuted nominal trajectories would have ended.
Actual paired native evaluation is necessary to test recovery.

The old public B0/B1/B2 results (6/12, 1/12, 1/12 success) remain historical
reference. Their reduced B1/B2 contact counts are not accepted as improvement
when accompanied by eight early BLOCKED stops.
