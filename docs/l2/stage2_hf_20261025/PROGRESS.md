# Stage 2 high-fidelity development progress

Source-only progress snapshot taken 2026-10-25T14:27:50.326112+00:00. This is unfinished research, not a trained release or a TARGET MET result.

Live development baseline: `3d2a0e4829f1071c53a322cfd8d864c732185313`. Publication is based on the existing public source-review commit `28b8512ee8ca95c94a45809fbd35e66d0f4562f7`, avoiding unpublished checkpoints and generated-data history. The live research checkout and its running jobs were not changed.

## Implemented

- Original-feature MLP objective/factorization pilots; compact two-head spatial predictor; frozen-prior spatial residual predictor; legal observation-recency extraction and point-feature predictors.
- Complete spatial/hybrid inference API, shared-map multi-trajectory execution, sparse hypothetical updates, immutable-input and deterministic future-graph checks.
- Full-domain pilot evaluation, exact original candidate generation, causal ancestor replay with sibling rollback, source/configuration hashes and bounded profiling.
- Recency uses starting-map observation times only. No future camera states, hidden geometry, Stage 1 forecasts or human states are prediction inputs.

## Completed pilot results

The common pilot evaluation contains 31 development Roots, 16 scenes and 666 full-domain windows. The released checkpoint at revelation threshold 0.20 scores **0.217413** here. These pilot scores are not full-development or protected-confirmation results.

| Trial | Best scored epoch | Decision | Label-aware IoU | Multiclass log loss |
|---|---|---|---:|---:|
| hybrid_bce | epoch04 | threshold_0.20 | 0.218188 | 0.022514 |
| hybrid_dice | epoch04 | threshold_0.30 | 0.213524 | 0.025850 |
| light_factor_focal | epoch04 | threshold_0.20 | 0.213859 | 0.023027 |
| light_factor_proper | epoch04 | threshold_0.20 | 0.215699 | 0.022910 |
| light_softmax_balanced | epoch02 | threshold_0.20 | 0.215967 | 0.022821 |
| spatial_bce | epoch08 | threshold_0.10 | 0.177642 | 0.025554 |
| spatial_dice | epoch08 | threshold_0.10 | 0.167822 | 0.027172 |
| time_revelation | epoch02 | threshold_0.20 | 0.218501 | 0.022683 |
| time_twohead | epoch04 | threshold_0.20 | 0.218899 | 0.022739 |

The best observed gain is small and does not establish a meaningful reconstruction improvement. The 0.35 milestone and 0.45 main target have not been reached. Negative trials remain preserved locally; compact results are included here.

## Still pending

- Final architecture/operating-point selection, selected full-data fit, and complete 47,128-window development evaluation.
- Final matched K=1/4/6 complete-inference latency, cold/warm measurements, integration gate and final checkpoint packaging.
- The recency point-model public inference adapter is not implemented yet; its feature extraction, training, evaluation and mathematical tests are included.
- New untouched confirmation, if valid data are available after freezing. The earlier opened 222 protected Roots must not be reused for training or selection.

## Running preparation at snapshot time

- `packed/full`: 78,016 / 105,838 Roots. Counts are a timestamped snapshot, not completion claims.
- `temporal/full`: 17,984 / 105,838 Roots. Counts are a timestamped snapshot, not completion claims.

All running training and preparation processes were left untouched. No simulator collection, Stage 1 modification or Layer 3 training was performed.

## Checks and external requirements

Basic syntax, whitespace, publication-file checks and two data-independent unit tests are recorded in `BASIC_CHECKS.json`. Earlier data-backed tests passed for candidate identity, CPU/CUDA covariates, actual future-graph equivalence, cache invalidation, map immutability and causal timestamp replay. These were not rerun as an extensive publication audit.

Datasets, checkpoints, caches, arrays, raw observations, predictions and large generated artifacts are intentionally excluded. Data-backed scripts retain the original laboratory path bindings and require the external immutable datasets and baseline weights. This publication is not a standalone trained-model package.

Example checks from the original environment:

```bash
python scripts/stage2_segment_run.py unittest \
  eventwm_graph.stage2_hf.test_time.CausalTimeFeatures.test_clock_shift_and_future_time_guard \
  eventwm_graph.stage2_hf.test_spatial.SpatialCorrectness.test_probabilities_and_shared_map
```

Training/evaluation entry points are `eventwm_graph.stage2_hf.light`, `spatial_train`, `hybrid`, `time_train`, and their corresponding evaluation modules. Exact completed-pilot configurations are in `configs/stage2_hf_20261025/`. Training configuration source hashes describe the code at execution time; this branch preserves the latest source snapshot.
