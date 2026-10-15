# EventWM Forecasting v2: completed research findings

The preregistered primary remains the immutable v1 recurrent Learned Dynamics
Model (LDM), with its original independent replicate. The campaign found useful
probability and occupancy improvements, but no new model met the final
multiobjective promotion requirements. The age-aware family is retained as an
explicit alternative. This is a completed negative primary-promotion result,
not a claim that the incumbent dominates every probability metric.

Branch: `research/eventwm-forecast-performance-v2-20261015`.
Baseline: `c8e85e73949b77edba7421cdefaa196f4e1a589d`.
Artifacts: `/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_performance_v2_20261015`.
Release directory: `release_forecast_v2_20261015/` under that artifact root.
The sealed manifest records the exact source commit, checkpoint hashes, dataset
identities, tests and release decision. The separate release-validation receipt
records inference executed from the archived source after packaging.

## Main confirmation results

These are new, previously untouched 128-root natural and 128-root unseen banks,
opened only after the development family and checkpoint panel were committed.
Three-second trajectory support is 53/73 roots; occupancy AP support is 84/92.
They are not the old release's already-opened confirmation populations.

| Model, seed 14 | Natural ES3 | Unseen ES3 | Natural joint coverage3 | Natural AP3 | Unseen AP3 |
|---|---:|---:|---:|---:|---:|
| Retained v1 primary | 1.152000 | .783836 | .679245 | .075280 | .091217 |
| Age-aware probability alternative | 1.166005 | .791999 | .886792 | .090451 | .103533 |
| Original CV | 2.203736 | .963494 | .566038 | .078964 | .120514 |
| CV with same geometry prior | 2.203736 | .963494 | .566038 | .085315 | .126312 |

Nominal joint coverage is 58/65 = .892308. The alternative improves natural
coverage by .207547, with a paired scene-bootstrap 95% interval [.107096,.314815].
Its exact trajectory NLL improves by .313482 nats/coordinate on natural and
.047040 on unseen; both intervals exclude zero. Occupancy AP, Brier and log-score
improvements replicate across both seeds. However, natural ES2 worsens
9.72%/7.13% across the two seeds, with both intervals excluding zero. Natural
central coverage is excessive, and the unseen replicate's ES3 worsens 3.07%.
The original primary/replicate roles are retained under the declared 3% ES
tradeoff limit. Neither a favorable alternative nor a better test seed was
promoted after seeing confirmation. See [CONFIRMATION.md](CONFIRMATION.md) and
[the complete fixed panel](CONFIRMATION_TABLES.md).

## Completed work

Eight 1,024-root physical interventions, two matched 4,096-root runs and two
8,965-root full-data runs completed: 147,456 new physical optimizer updates and
6,270,924 root presentations. Each full run completed 24,576 updates and
1,562,598 presentations, using every one of 48,434 fitting windows. Exact resume
preserved the saved seed-14/15 states at 2,677/2,723 updates. No schedule,
architecture, data split or optimizer state was reset to escape the resource guard.
Full-data calibration, independent-seed comparisons, fixed-mean probability
controls, occupancy controls, recurrent audits and the frozen ten-model/two-bank
confirmation panel all completed. Physical training curves are included.

The revised guard targets 1,500 collector queries/hour, accepts 1,000–1,500,
throttles after three valid sub-1,000 windows, and restores full duty after two
valid above-1,500 windows. Healthy slow collection no longer stops all research.
Hard health, RAM, storage and OOM constraints remain. The collector was neither
restarted nor reconfigured. See [RESOURCE_POLICY.md](RESOURCE_POLICY.md).

## Research conclusions

1. **Strongest eligible primary:** the original width-128, six-mode recurrent
   LDM, with legal-history initial correction and persistent stochastic futures.
   `primary.pt` retains SHA256
   `d092b5d05d38be4619251e009d2aa2dafc691f3f57440473d72e87acf7887970`.
2. **Improvement over the release:** no primary forecasting improvement is
   claimed. The age-aware alternative has confirmed likelihood, occupancy and
   natural high-nominal coverage gains, with the ES/calibration costs above.
3. **Responsible interventions:** all-window root-balanced conditional variance
   fitting plus explicit age propagation improves old-history uncertainty;
   geometry-conditioned anonymous occupancy improves spatial probability quality.
   Controlled fitting separates these from changes to the physical mean.
4. **Negative results:** velocity revision, shorter motion window, polar dynamics,
   tracked-occupancy BCE weights 2/10, distinct-arrival encoding and older-window
   sampling did not qualify. Removing cross-agent messages helped at 4k roots
   but did not retain its advantage after two full-data calibrated runs.
5. **Probability reliability:** coherent finite joint densities and useful
   observation correction are verified. Neither family is uniformly calibrated:
   v1 under-covers old histories; the age alternative over-covers central natural
   sets and remains imperfect on unseen scenes. These are not epistemic ensembles.
6. **Occupancy versus CV:** the alternative exceeds original CV as a natural-bank
   AP point estimate, but not on unseen scenes. It is not a general CV-beating
   occupancy solution. Equal-prior CV is also reported.
7. **Recurrent update:** both candidate seeds and both fallback roles pass actual-
   arrival correction and direct/composed persistent-path audits. The delayed
   observation approximation is at most one .25s tick; no cross-ID matching exists.
8. **Remaining limits:** anonymous occupancy has no identity trajectories or joint
   temporal law; no complete scene posterior, wall-feasibility guarantee or
   independent scene-residual model is established. Five seconds extrapolates
   beyond the three-second physical objective. Sample geometry proxies can worsen.
9. **350k experiments:** all fitting and model selection use the frozen 348,777-query
   snapshot, with 8,965 independent fitting roots and 256 development roots.
   Repeated query branches are not independent human futures.
10. **500k experiments:** none. The 495,000 validated-query gate remains closed.
    The larger 444,774-query freeze was confirmation-only. The audited incremental
    continuation recipe is supplied and explicitly has not been trained.
11. **Readiness:** the retained LDM and alternatives are reproducible forecasting
    artifacts with a stable `encode_history / imagine / observe` interface. They
    are not the complete Layer-2 WM. A Query-Conditioned Belief Adapter, joint
    fine-tuning and downstream selector have not been implemented in this campaign.
12. **Most promising next LDM improvement:** separately learn state-estimation and
    future-process uncertainty across horizons. The measured old-history/central-
    coverage tradeoff motivates this hypothesis; it remains untested.

The next requested architectural stage is pretrained LDM → Query-Conditioned
Belief Adapter → joint fine-tuning → complete Layer-2 WM. This campaign does not
start that work. The adapter must preserve exogenous human/query independence.

## Evidence and reproduction

[PROTOCOL.md](PROTOCOL.md) and [SELECTION.md](SELECTION.md) preserve decisions made
before confirmation. [HYPOTHESES.md](HYPOTHESES.md), [AUDIT.md](AUDIT.md),
[ABLATIONS.md](ABLATIONS.md), [DEVELOPMENT_BENCHMARK.md](DEVELOPMENT_BENCHMARK.md)
and [TRAINING.md](TRAINING.md) record controls, negative results and actual budgets.
[MODEL.md](MODEL.md) contains the architecture diagram and interface semantics.
[DATA.md](DATA.md) documents support, identities and legally available inputs.
[REPRODUCE.md](REPRODUCE.md) gives exact local commands. [INFERENCE.md](INFERENCE.md),
[RECURRENT.md](RECURRENT.md), [PHYSICS.md](PHYSICS.md), [CALIBRATION.md](CALIBRATION.md)
and [OCCUPANCY.md](OCCUPANCY.md) describe verification and limitations.

Development and new-bank figures use baseline-only case selection, common axes,
common probability scales, coherent samples, and both ordinary and difficult
examples. They are illustrative, not substitutes for the complete tables.
