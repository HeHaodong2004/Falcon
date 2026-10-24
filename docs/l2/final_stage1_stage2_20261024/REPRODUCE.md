> Source-publication note: this is a frozen local research report. Model weights, datasets and detailed generated evidence are deliberately excluded; reported results were not rerun for publication. Location references are redacted. See [SOURCE_REVIEW_GUIDE.md](../../../SOURCE_REVIEW_GUIDE.md).

# Reproduction and research use

Worktree: `/research-user/workspace/projects/eventwm-falcon-l2-two-stage-20261023`.
Python: `/research-user/miniforge3/envs/eventwm-falcon/bin/python`.
Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`. Exact executable sources are saved independently of Git and identified by SHA256 inventories.

## Frozen collection

`~/.local/bin/collectorctl status` reports the durable stopped state. Do not restart collection. Final fitting manifest:

`output/l2_two_stage_20261023/final_collection_20261024/manifest.json`

The immutable manifest references independent read-only Root payloads and retained parent/supplement copies. The complete source inventory, opaque-byte checksums, excluded/protected metadata, source journal and collector controls/logs are preserved under `output/research_transition_20261024/data_audit`. No protected payload needs to be decoded for reproduction.

The complete all-splits administrative snapshot is `output/research_transition_20261024/final_collection_all_splits/manifest.json`. It reuses independently verified fitting copies and adds independent read-only copies of every excluded/protected source Root. Its `training_manifest=False` flag is explicit; training continues to use only the fitting manifest above. Protected payloads are copied and hashed as opaque bytes.

The executed finalization entry point is `scripts/final_stage1_research.py`. It extends the audited parent preparation, creates legal positive-history/event sidecars and separate offline population targets, then runs the original fixed O trainer. Its saved training-source overlay changes only the hardcoded offline target-cache path to the supplied sidecar name. `FINAL_SOURCE_IDENTITY.json` preserves this exact difference from original source cfe4c5eb.

`scripts/extend_final_stage1_sidecars.py` reuses the unchanged 79,741-Root positive-history/event cache shards and runs the original transformation functions only for the 26,097 new Roots. Its merged manifests preserve parent hashes, source identities, exact Root/anchor accounting and new witness links. The orchestration process was suspended while its preparation child finished, then resumed automatically after these caches were sealed; no model fitting was interrupted or repeated. `reviews/review_final_stage1_sidecars.py` independently audits every reused/new payload and bitwise inherited joins.

## Stage 1

Final checkpoint: `output/l2_two_stage_20261023/experiments/final_stage1_research_20261024/last.pt`.
Exact training config, epoch-zero checkpoint, all eight epoch metrics, final optimizer/RNG state and completion certificate are in that directory. Load it directly for research; no further training is required.

Read `output/research_transition_20261024/inference_source_path.txt` for the immutable inference/evaluation source. Run Python from `/tmp`, set `EVENTWM_TWO_STAGE_BASE` to the worktree, and put that source and `output/l2_two_stage_20261023/deps` in `PYTHONPATH`. The original evaluation procedures are executed by `scripts/final_stage1_evaluate.py`; its command log is retained. Complete-call runtime, historical persistence, finite40 reliability, H+ and exact H0 drivers and independent audit live in `reviews/`. They preserve the original scientific thresholds and use new artifact destinations.

For an explicit future reproduction in a NEW experiment directory, use the saved training source and the following unchanged recipe. This command is documentation, not a request for another research training cycle:

```bash
CUDA_VISIBLE_DEVICES=1 python -m eventwm_graph.l2_two_stage.fit_conditioned REPRO_UNIQUE_NAME \
  --split train_final_collection_20261024 --sidecar train_final_collection_20261024 \
  --epochs 8 --full-fit --seed 413 --batch-size 48 --learning-rate .0002 \
  --history-fraction .5 --association-guidance \
  --initial /research-user/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/experiments/pilot_o_motion_guided_source_first/best.pt
```

The checkpoint is research-only. Consult the final Stage 1 report for actual failures; no scientific acceptance pass is implied.

Use the complete actual observation prefix, including valid empty frames. Pass **40 worlds explicitly**, since the historical predictor's default remains32. Initialization/certification is separate from initialized inference timing:

```python
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
from eventwm_graph.l2_two_stage.context import OnlineContext

stage1 = HistoryConditionedWorldModel(FINAL_STAGE1_CHECKPOINT)
context = OnlineContext.from_raw(actual_history, current_graph, executed_actions).with_association()
worlds = stage1.predict(context, actual_history[-1], horizon=3., worlds=40, seed=41)
# worlds.positions: float64 [40,64,N,2], world XZ metres.
# worlds.present masks padded/absent columns; worlds.times are relative seconds.
future_positions = worlds.positions_at([.5, 1., 2., 3.])
```

This produces coherent pedestrian worlds from actual observations and requires no new training. Stage2 is called independently with the legal current graph and robot trajectory.

## Stage 2

`python scripts/stage2_research.py` is the executed extraction/training/evaluation pipeline. It freezes code before processing, resumes checksum-bound completed preparation shards, trains once, and evaluates every original development Query. Preparation was resumed from four to twenty CPU workers after the collector stopped; the input/source identities and completed sample shards were unchanged. No training had started during this infrastructure adjustment.

Read `output/stage2_graphdelta_20261024/source_path.txt` for the immutable extraction/training/reference source. Read `runtime_source_path.txt` in the same directory for the optimized inference source. Set `EVENTWM_STAGE2_BASE` to the worktree and `PYTHONPATH` to the selected source; run from `/tmp`. Exact fixed protocol, sample weights, shard hashes, model/RNG/optimizer state, all epoch metrics and all per-Query evaluation rows remain under `output/stage2_graphdelta_20261024`.

`scripts/stage2_runtime_research.py` reuses the single trained checkpoint and evaluates execution-only optimizations. It preserves original full evaluation/benchmark artifacts, freezes the optimized source, reruns the same complete development cohort and timing workloads, and compares every scientific summary exactly. `optimized_runtime/` records the parent source, unchanged checkpoint SHA, comparison and completion proof. Candidate support, all47features, model parameters, class decisions and deterministic graph-update semantics are independently audited. No neural training is repeated.

Original cold measurements used repository namespaces. Optimized cold subprocesses run from `/tmp` with explicit frozen imports, and record their actual paths/hashes. Their cold predict/apply/wire time conservatively includes import-witness hashing overhead. Warm complete-call P99 tails remain in the ledgers. Batch sizes use different prefixes of the input pool; the reported throughput does not establish a matched batch speedup. Benchmark-process RSS includes retained workload examples and libraries as well as the model and temporary arrays.

```python
from eventwm_graph.stage2 import Stage2Predictor, apply_graph_delta

stage2 = Stage2Predictor("/absolute/path/to/output/stage2_graphdelta_20261024/training/final/last.pt")
# Each trajectory row is [absolute_time, x, y, z, yaw].
# The first row must start at partial_graph['timestamp'].
predicted_delta = stage2.predict(partial_graph=current_graph, trajectory=executed_segment)
predicted_future_graph = apply_graph_delta(current_graph, predicted_delta)
# Batch API: stage2.predict_batch([{'partial_graph': g, 'trajectory': path}, ...])
```

`candidate_cells`, `states`, `probabilities`, `heights` and `unknown_cells` are NumPy arrays. `static_updates` contains only predicted newly FREE/BLOCKED cells. UNKNOWN and cells outside the declared candidate domain produce no update. Hypothetical graphs and predicted cells carry explicit provenance and must not be substituted for actual sensor evidence. The input graph remains unchanged.

Evaluation and complete API benchmarking can be reproduced in fresh destinations:

```bash
python -m eventwm_graph.stage2.evaluate --checkpoint FINAL_CHECKPOINT --name REPRO_UNIQUE_EVAL
python -m eventwm_graph.stage2.benchmark --checkpoint FINAL_CHECKPOINT --name REPRO_UNIQUE_BENCH
```

The API uses no simulator, hidden map, pedestrian predictor, planner-in-the-loop training or reinforcement learning. End-to-end integrated Layer 2 P95 ≤100ms remains the overall target; component measurements alone do not establish integration performance.

After completion, `python scripts/verify_final_research.py` checks the checkpoint, configuration, source, metric and review artifact registry without training. Optional `--full-dataset` additionally rehashes the complete all-splits snapshot as opaque bytes; the retained independent full-byte audits already record that verification.
