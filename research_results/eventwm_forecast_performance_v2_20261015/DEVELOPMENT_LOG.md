# Development record

Fresh-run reproduction preparation now copies only explicit completed recipes,
remaps dependent artifacts into a new output directory, and retains scientific
settings plus immutable external data/control references. Three path tests and
a two-stage preparation smoke pass; no duplicate training was launched. The
selected-family recipe remains pending the completed development comparison.

The baseline is local committed source `c8e85e73949b77edba7421cdefaa196f4e1a589d`.
The original release, data snapshot, fitting/development memberships and collector
remain unchanged. Every experiment writes a unique directory; a checkpoint file
alone is not a completion receipt.

1. Reproduced selected baseline development metrics exactly, inspected source and
   reports, audited every fitting/development window, and preserved the original
   legal-input, recurrent and sampling tests. Observation age and two-second
   tracker retention describe different populations; no tracker change was made.
2. Closed five full-root fixed-mean probability controls. Age-aware covariance
   improves older-history density coverage and likelihood; conditional versus
   unconditional fitting, and origin versus all-window fitting, isolate effects.
3. Audited occupancy frames and support populations. The original total metric
   is valid; tracked-only diagnosis now restricts both prediction and target to
   the same supported humans. A geometry-only anonymous prior improves total
   occupancy; legal history bins do not justify their additional complexity.
4. Closed all eight 1k physical pilots at 8,192 updates each. The self-only,
   capacity-matched interaction control advances; velocity, arrival, polar,
   occupancy-loss and importance-corrected sampler controls are retained as
   measured negative or insufficient results. See ABLATIONS.md.
5. Fitted and implemented a persistent stopped-motion law. Full and half-weight
   controls quantify its likelihood/occupancy gains and ES costs. The full-root
   half-weight law reproduces its AP/NLL gains on the second released seed. It
   does not automatically satisfy the final ES tolerance. Recurrent correction,
   conditional coverage, physical tails and map-proxy limitations were audited.
6. Closed the matched 4k runs at 16,384 updates each. Commit `0dcaa09` freezes
   full-data promotion of the self-only family, with primary seed 20261014 and
   replicate 20261015. Origin and older-window evidence, not reserved scores,
   determines this choice. An actual CUDA interruption/resume test is bitwise
   equal for all optimizer, model, sampler and RNG state fields.
7. Both full-data workers have started their 24,576-update / batch-64 budgets.
   They initially paused and resumed under the original collector guard. The one/three-
   second calibration control is running through recoverable density chunks.
   Final training completion, development selection and release are pending.

Operational fixes preserve scientific identities. Verified density chunks avoid
rebuilding completed probability inputs after a guard pause. Attempt ledgers
record elapsed wall time, CPU cost, source/config hashes and exit state. Snapshot
payloads are sealed below the operational run directory so launcher logs remain
writable; these fixes did not reset guard history or thresholds. `CURRENT_OWNER` names the
sole active finite supervisor. Prior owners are stopped before handoff.

One wrong-CV-preparation attempt was rejected before scoring and preserved as a
failed attempt. The corrected CV diagnosis uses the declared .15-second cache.
Interrupted nonincremental stopped-law fitting is preserved and superseded by
the chunked run. Neither is silently counted as a completed scientific result.

The latest explicit validated-count receipt at this entry is 442,299 queries /
13,403 roots, below the 495,000-query scaling gate. This is not a 500k experiment.
The confirmation-only freeze will exclude every 350k root and physical identity;
new reserved target extraction requires the later frozen development selection.

8. Completed the confirmation-only incremental freeze: 444,774 validated queries
   and 13,478 roots, with the exact 350k parent preserved. No new forecasting
   targets or confirmation scores entered model selection. This is below the
   495,000-query expanded-training gate.
9. The user explicitly revised resource priorities to favor completing the
   physical models. Commit `4844a51` implements absolute throughput thresholds:
   target 1,500 q/h, acceptable 1,000–1,500, bounded throttling after three valid
   sub-1,000 windows, recovery after two above 1,500, and hard health/RAM/disk/OOM
   constraints. The old percentage policy is no longer effective. Both exact
   saved states were backed up and resumed under the sole v13 owner. Seven
   bounded tests and live integration passed. At the recorded verification,
   saved updates increased from 2,677/2,723 to 4,736/4,608 with unchanged training
   identities and finite weights. Auxiliary jobs are deferred during the two
   full-data physical schedules. See RESOURCE_POLICY.md and the machine receipt
   `resource_policy_transition_20261015/verification.json`.
10. Both full-data physical schedules completed at 24,576 updates, 8,965 roots
    seen and 1,562,598 root presentations per seed. Their best raw checkpoints
    are steps 17,152 and 15,104. Calibration/evaluation, not those preliminary
    raw values, will determine final selection. Completed best/last hashes and
    unchanged manifest/protocol identities passed the handoff verification.
11. After physical completion, the v13 owner and its sole calibration child
    stopped cooperatively before v14 launched. Commit `e781575` keeps one CPU
    stage eligible under soft low-throughput throttling, preventing starvation
    once all remaining work is CPU calibration/evaluation. Eleven bounded
    guard/dispatch/scaling-gate checks passed. No optimizer, data, collector or
    hard safety threshold changed. `owner_handoff_v13_v14.json` records the
    verified single owner, collector PID 3719917 and 2,766.70 q/h at handoff.
12. Prepared and tested the unchanged 495k continuation gate, without freezing
    more data. A read-only check found 448,107 verified queries. Standalone
    inference now quarantines unused legacy geometry weights while preserving
    trained map/occupancy paths; three bounded tests passed. Confirmation may
    reject the new family but cannot switch test seeds or promote a test-winning
    alternative; the only frozen fallback is the original primary/replicate.

- Both full physical exposure audits completed: all 48,434 windows were used,
  including by the selected early checkpoints. Full learning-curve artifacts
  are complete. Later calibration does not establish a full-physical-family
  improvement; raw best ES alone must not be used for promotion.
- Seed 15 horizon calibration encountered an isolated CUDA illegal-memory
  access before optimizer fitting. Kernel Xid 31 records the worker MMU fault.
  The first uncommitted 64-root chunk passed CPU parity and two synchronized
  CUDA executions; all 153 committed density parts passed SHA256 verification.
  Preserve the original failure and authorize one unchanged retry, which
  reuses those verified parts. No physical optimizer or collector was reset.
  Evidence: cuda_calibration_recovery/failed_chunk_probe.json and
  recovery_authorization.json. Root cause remains unproven.
- The development candidate under verification is the original fully trained
  physical family with all-window three-second age-aware conditional variance
  fitting and the fixed geometry anonymous prior, seeds 20261014/20261015.
  This is provisional, does not open confirmation, and explicitly retains
  central-coverage/short-horizon sharpness and occupancy-versus-CV limitations.

## Research closure and publication handoff

Both full physical seeds completed unchanged schedules, exposure replay and calibration. Development selection was committed before new confirmation extraction; the ten-model/two-bank panel completed. Paired confirmation shows probability/occupancy improvements but material natural ES2 and central-calibration costs. The hash-bound release review retains v1 primary/replicate and named alternatives without post-confirmation tuning. Protected physics diagnostics required a population-routing correction; original failures remain preserved, and 97 tests plus all four standalone checks passed again. The v18 queue completed all 186 jobs and exited. All research workers are inactive; production collection continues. The validated index remains below 495k. The next operation is a sealed archive-only research verification, followed by the separately authorized clean GitHub publication. No next-stage adapter implementation is included.
