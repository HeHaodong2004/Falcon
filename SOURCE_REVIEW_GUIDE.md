# Source review guide

This is a clean, single-root-commit source snapshot, with no inherited research Git history. Primary source: `b0e662e37b5517cfff455ac7f6146d4df5c21f7a` (clean research checkout). The actual selected training and runtime source namespaces are also preserved under `research_reference/`. [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) records original and public SHA256 hashes for every copied file and the separate component commits.

## Implementation map

| Component | Actual implementation |
|---|---|
| Stage1 history-conditioned human world distribution | [conditioned.py](eventwm_graph/l2_two_stage/conditioned.py), [neural.py](eventwm_graph/l2_two_stage/neural.py), [population.py](eventwm_graph/l2_two_stage/population.py), [motion.py](eventwm_graph/l2_two_stage/motion.py), [context.py](eventwm_graph/l2_two_stage/context.py), [history_proposal.py](eventwm_graph/l2_two_stage/history_proposal.py) |
| Stage1 data/history and training | [data.py](eventwm_graph/l2_two_stage/data.py), [positive_history.py](eventwm_graph/l2_two_stage/positive_history.py), [fit_conditioned.py](eventwm_graph/l2_two_stage/fit_conditioned.py), [conditioned_training.py](eventwm_graph/l2_two_stage/conditioned_training.py) |
| Exact final Stage1 fit source | [archived fit_conditioned.py](research_reference/stage1_final_training/eventwm_graph/l2_two_stage/fit_conditioned.py), [source identity](research_reference/stage1_final_training/FINAL_SOURCE_IDENTITY.json), [orchestrator](scripts/final_stage1_research.py). This preserves the established full-data O recipe, including the documented sidecar-path execution fix; later pilot options in the live trainer are not evidence of final-model selection. |
| Stage1 evaluation | [evaluate.py](eventwm_graph/l2_two_stage/evaluate.py), [evaluate_current_population.py](eventwm_graph/l2_two_stage/evaluate_current_population.py), [final_stage1_evaluate.py](scripts/final_stage1_evaluate.py) |
| Stage2 data, features, network, training | [data.py](eventwm_graph/stage2/data.py), [features.py](eventwm_graph/stage2/features.py), [model.py](eventwm_graph/stage2/model.py), [train.py](eventwm_graph/stage2/train.py) |
| Stage2 API, graph update, evaluation, timing | [api.py](eventwm_graph/stage2/api.py), [map.py](eventwm_graph/partial/map.py), [evaluate.py](eventwm_graph/stage2/evaluate.py), [benchmark.py](eventwm_graph/stage2/benchmark.py), [tests](tests/stage2/test_graphdelta.py) |
| Graph observation, queries and execution | [partial/](eventwm_graph/partial/), especially `map.py`, `query.py`, `execution.py`; graph geometry also lives in [robotmap/](eventwm_graph/robotmap/) |
| Earlier L2 planning/sandbox contracts and adapters | [planning.py](eventwm_graph/l2_partial/planning.py), [queries.py](eventwm_graph/l2_partial/queries.py), [structural_adapter.py](eventwm_graph/l2_partial/structural_adapter.py), [l2_intermittent/](eventwm_graph/l2_intermittent/) |
| L2.5 Graph Compiler | [compiler.py](eventwm_graph/l25/compiler.py), [schema.py](eventwm_graph/l25/schema.py), [graph.py](eventwm_graph/l25/graph.py), [numeric.py](eventwm_graph/l25/numeric.py), [tests](tests/l25/test_compiler.py) |
| Existing execution/planner consumers | [executor.py](eventwm_graph/l4/executor.py), [search.py](eventwm_graph/l4/search.py), [forecast.py](eventwm_graph/l4/forecast.py), [falcon/](falcon/), [habitat-baselines/](habitat-baselines/) |

L2.5 is copied from clean component commit `e71cc0e564486d04cc5e48b31fbfab7b4b997f54`; L4 from `16901e05ce5cd9cc391355c6f4a0f17d920f83f4`. These already implemented components were on separate worktrees, not newly developed or integrated for this publication. Source provenance is per file. The existing Habitat source trees and their notices are included for tracing local simulator/policy dependencies, without external packages, meshes or visualization assets. The original root MIT license is retained; an older Habitat file refers to the root license as BSD-style in its existing header, which is preserved verbatim.

## Stage2 training examples and exact inputs/outputs

`data.branches` verifies checksum identities and reads `legal_root.json.gz`, `queries.json.gz`, `outcomes.json.gz`, and `future_observations.json.gz`. Input is the initial legal static PartialGraph plus executed robot trajectory rows `[absolute_time, x, y, z, yaw]`. Actual outcome DAG ancestry supplies time/duration/termination; planned trajectories do not substitute for executed records. Target is the final observed state/height at cells initially UNKNOWN. Occupied beats free within an observation, ego-base support occupation is ignored, and later free observations can clear occupation. Known-cell corrections are counted but outside this target scope. Protected confirmation payloads and privileged hidden geometry are not loaded.

Candidate generation uses initially UNKNOWN grid cells whose centers lie within 11 m of any executed pose. Existing known cells are not regenerated. Missing outside-domain cells remain UNKNOWN/no-update. The 47 columns, in exact zero-based order in `features.SceneFeatures.transform`, are:

| Columns | Computation |
|---|---|
| 0–3 | distances to four nearest executed poses / 10 |
| 4–7 | heading-relative facing cosine, clipped to [-1,1] |
| 8–11 | signed side/bearing component, clipped to [-1,1] |
| 12–15 | pose elapsed time / max(executed duration, 1e-4) |
| 16–24 | FREE indicators at the nine grid offsets in `OFFSETS` |
| 25–33 | BLOCKED indicators at the same offsets |
| 34–35 | nearest known FREE/BLOCKED distance-transform distances / 10 |
| 36–37 | cell-center x/z relative to first pose / 10 |
| 38–39 | cell-center x/z relative to final pose / 10 |
| 40 | nearest known-cell baseline height minus initial robot y |
| 41–46 | duration/5, x/z path length/5, resolution, final-minus-initial y, cosine and sine of heading change |

Ties, float32 rounding, padding when fewer than four poses exist, the exact grid-offset order, and no-known-cell fallback are defined by the source. There is no Stage1 prediction input.

The 4,004-parameter shared MLP is `47 → Linear(48) → SiLU → Linear(32) → SiLU → Linear(4)`. Outputs are UNKNOWN/FREE/BLOCKED logits plus one baseline-relative height residual. `data.sampling` takes up to 64 newly observed and 64 remaining-UNKNOWN candidates per Query, seeded by Root/Query identity. Exact inverse-inclusion weights and 1/Queries-per-Root weighting estimate full candidate-domain risk with equal Root weight. `train.run` uses six fixed epochs, seed 2410, batch 8192, AdamW lr .001, decay .0001, clip 5; loss is weighted cross entropy + .2 observed-cell smooth-L1 height loss. No threshold search or development selection was performed.

The source actually used for training is [research_reference/stage2_training/](research_reference/stage2_training/) (original identity `fc67e2ed3dfc88b1bc41dbfa41f01a77015b46a0193e3982ce6228a2fa64e36d`). The latest optimized runtime is [research_reference/stage2_runtime/](research_reference/stage2_runtime/) (original identity `7803aa3bec784a8dec3e55306bf9e284263577f383f7aec39c1f38ff42c4b90f`). Execution optimizations affect `api.py`, `features.py`, `benchmark.py`; they reuse feature calculations, stream inputs, tile independent network rows at 8192, and avoid redundant updater copies. They do not replace weights or the scientific scoring rules. Historical hashes refer to original unredacted source; the publication manifest explains location-only differences.

## Inference and deterministic graph construction

```python
from eventwm_graph.stage2.api import Stage2Predictor, apply_graph_delta
stage2 = Stage2Predictor(checkpoint_path, threads=1)  # weights intentionally not published
predicted_delta = stage2.predict(partial_graph=current_graph, trajectory=trajectory_segment)
predicted_future_graph = apply_graph_delta(current_graph, predicted_delta)
```

`predict_batch` accepts a list of the same input dictionaries and streams their independent cell rows. Each result includes candidate cells, full three-class probabilities, raw-argmax states, heights, UNKNOWN cells, sparse FREE/BLOCKED updates, timestamps/pose, base graph identity and checkpoint hash. A zero-duration segment emits no revelation. `apply_graph_delta` verifies identity, clock, unique initially UNKNOWN cells and finite heights; restores a private graph; calls the existing `PartialGraph.update`; and returns a hypothetical graph with prediction provenance. The input graph is unchanged. The full pre-existing map is never a network regeneration target.

## Evaluation and frozen results

[evaluate.py](eventwm_graph/stage2/evaluate.py) scores every Query from the original scene/Root-separated development split over the full candidate domain. It reports per-class precision/recall/F1, label-aware GraphDelta IoU and reveal-set IoU, multiclass Brier/log score, 15-bin calibration, observed-cell height MAE, substantial-information trajectories (at least 100 actual new cells), domain misses and deterministic update correctness. Scene-bootstrap confidence intervals use 2,000 draws, seed 2411; aggregate probability scores are Root-balanced. `benchmark.py` separates neural prediction API timing from prediction + deterministic update + sparse serialization, and records batching/cold startup/memory.

Frozen results, copied without rerunning training or scientific evaluation:

| Scene-balanced metric | FREE | BLOCKED |
|---|---:|---:|
| Precision | .4832518256 | .5592482419 |
| Recall | .1514334386 | .0557996750 |
| F1 | .2306040168 | .1014745983 |

Label-aware GraphDelta IoU: .0747000035; reveal-set IoU: .0965152963. Brier: .0158696009 versus persist-UNKNOWN .0201455593. Development support: 2,110 Roots, 108 scenes, 14,546 Queries. Complete prediction + update + sparse serialization P95: representative **70.347 ms**, dense **89.330 ms**; predict-only P95: **58.892 / 68.231 ms**. Dense complete-call P99 remains 216.196 ms. These are historical standalone measurements, not measured integrated L2 latency or results reproduced from this public snapshot.

See [frozen evaluation.json](docs/l2/final_stage1_stage2_20261024/release/stage2/evaluation.json), [performance.json](docs/l2/final_stage1_stage2_20261024/release/stage2/performance.json), [config.json](docs/l2/final_stage1_stage2_20261024/release/stage2/config.json), and the two final reports [A](docs/l2/final_stage1_stage2_20261024/A_FINAL_STAGE1_RESEARCH_RELEASE.md) / [B](docs/l2/final_stage1_stage2_20261024/B_STAGE2_RESEARCH.md). Stage1 is closed as a research checkpoint; population calibration did not pass the original gates. Collection was explicitly MANUALLY_STOPPED in the frozen final reports. No collection, training, tuning or evaluation rerun was performed for this publication.

## Implemented versus intended integration

Stage1 human-world prediction, Stage2 standalone learned GraphDelta prediction/update, existing earlier L2 planning contracts, the L2.5 compiler and L4 execution consumers are implemented as the source map indicates. Existing planner interfaces can be inspected in `l2_partial/planning.py` and L2.5 schema; no dedicated new Stage2-conditioned L3 policy or L3 RL training is claimed. The existing Habitat/Falcon policy sources are included, but they do not establish a newly trained Layer3 world-model policy.

The intended connection is: current legal graph + candidate executed motion → Stage2 GraphDelta → deterministic hypothetical graph → downstream query/graph consumers, alongside Stage1 human futures. This complete two-stage sandbox/consumer integration and its P95 ≤100 ms total latency have not been established by the standalone Stage2 results. The existing L2.5 compiler targets earlier belief/world-model interfaces; its presence does not prove it directly accepts the new Stage2 API. No architecture was invented to fill this gap.

## Reproduction limits, omissions and lightweight checks

No weights/checkpoints, collected datasets, protected confirmation data, raw future-observation DAGs, training caches, generated arrays, binary archives, original Git history, dependency environments, fonts, images, or third-party Gibson quality annotations are published. No required scientific Python/C++ implementation was omitted. Optional visualization/font consumers need their separately obtained assets. Detailed private/local review outputs are omitted; aggregate metrics are retained. Full training/evaluation reproduction and inference with the final trained weights require separately authorized data/checkpoints, and are not claimed here.

All copied text was examined by an automated whole-file artifact/credential/path/license scan and Python syntax parse; source fidelity is checked against original hashes, with only documented location redactions and Markdown publication annotations. Root MIT license and existing source notices are retained. Original research files and results are untouched. See `PUBLICATION_CHECKS.json` for exact validation outcomes and limitations. No scanner can establish an absolute absence of secrets; detected candidates were inspected before publication.

The isolated namespaces reproduce the existing lightweight import boundary without importing Habitat at package initialization. With Python 3.9, torch 1.12.1, numpy 1.23.5 and scipy 1.13.1 available:

```bash
SNAPSHOT=/path/to/clean-source-snapshot
cd /tmp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SNAPSHOT/research_reference/stage2_runtime" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s "$SNAPSHOT/tests/stage2" -v
```

The source-preserved offline entrypoints are `python -m eventwm_graph.stage2.data --split train`, `python -m eventwm_graph.stage2.train`, `python -m eventwm_graph.stage2.evaluate --checkpoint ...`, and `python -m eventwm_graph.stage2.benchmark --checkpoint ...`. They require absent artifacts and relocation of the original research paths in `common.py`/manifests. The local orchestration scripts also retain generic, redacted original installation paths; they are historical run records, not a promise of turnkey public execution. Do not execute collector orchestration to review Stage2.

## Exact review handoff

1. FREE recall (~15%): `eventwm_graph/stage2/data.py` (`branches`, `sampling`), `features.py` (`candidates`, `transform`), `train.py`, `api.py` (`predict_batch`).
2. BLOCKED recall (~5.6%): the same files, particularly occupied/free replay precedence, sampled strata, inverse-inclusion weights, height/class heads and final raw argmax.
3. Imbalance, target construction, representation, capacity, weighting versus decision rules: `data.py`, `features.py` (all 47 columns), `model.py`, `train.py`, `api.py`, plus frozen `release/stage2/config.json` and `evaluation.json` under the final-report directory.
4. Probabilities beyond argmax F1: `api.py` (returned probabilities), `evaluate.py` (`rates`, proper scores/calibration), and frozen `release/stage2/evaluation.json` (confusion, Brier, bins/support). UNKNOWN-heavy calibration is not proof of strong rare-class quality.
5. Query-conditioned World Model Sandbox feasibility: `stage2/api.py`, `partial/map.py`, `partial/query.py`, `l2_partial/planning.py`, `l25/compiler.py` and `l25/schema.py`. Examine the intended interface gap rather than assuming the connection is already validated.

These are review questions, not conclusions or fixes made during publication.
