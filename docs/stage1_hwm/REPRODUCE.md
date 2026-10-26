# Reproducing the Stage 1 human world model

All commands run from the worktree root with the research runner, which imports
`eventwm_graph` as a namespace (no Habitat import) and sets single-threaded BLAS.

```bash
PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python
RUN="$PY scripts/stage1_hwm_run.py"
```

Large data live outside `/home` (93% full): `/var/tmp/falcon-stage1-claude/`.

## 1. Legacy baseline (frozen source, bit-exact)

The legacy checkpoint is evaluated with its frozen inference source; the isolated base
`output/s1base` symlinks the legacy read-only inputs and owns a fresh `evaluation/` dir.

```bash
O=/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023
S=$O/frozen_sources/1ca38d61ddeab23a9022f045685ec5115f3a5a71463305855d10c6063bc188e0
LEG="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  EVENTWM_TWO_STAGE_BASE=$PWD/output/s1base NUMBA_CACHE_DIR=$PWD/output/s1base/output/l2_two_stage_20261023/numba_cache \
  PYTHONPATH=$S:$O/deps CUDA_VISIBLE_DEVICES="
# released legacy metrics (all 4,666 values reproduce exactly)
(cd /tmp && $LEG $PY -m eventwm_graph.l2_two_stage.evaluate repro_final_forecast --split dev_native_v2 \
  --fitting-split train_final_collection_20261024 --checkpoint $O/experiments/final_stage1_research_20261024/last.pt \
  --checkpoint $O/experiments/pilot_k_bounded_unique/best.pt --joint-set-score --worlds 40)
# legacy world banks at every native time (for the shared scoring harness)
(cd /tmp && $LEG $PY $PWD/eventwm_graph/stage1_hwm/bank_legacy.py $O/experiments/final_stage1_research_20261024/last.pt \
  /var/tmp/falcon-stage1-claude/banks/legacy_final)
# legacy complete-call latency with the shared protocol
(cd /tmp && $LEG $PY $PWD/eventwm_graph/stage1_hwm/benchmark_legacy.py $O/experiments/final_stage1_research_20261024/last.pt \
  $PWD/output/stage1_hwm/benchmark/legacy_NAME)
```

## 2. Data extraction (legal inputs + separate privileged labels)

```bash
$RUN eventwm_graph.stage1_hwm.extract dev --workers 12 --shard-roots 250          # 2,110 Roots, legacy anchors
$RUN eventwm_graph.stage1_hwm.extract train --workers 12 --stride 6 --shard-roots 1000   # 105,838 Roots
$RUN eventwm_graph.stage1_hwm.extract_queries                                       # sibling candidate robot paths
```

## 3. Training

```bash
$RUN eventwm_graph.stage1_hwm.train NAME --arch {full,fast,lite} --truncate --epochs E --batch 128 --device cuda:0 --workers 8
```

Selection: best scene-held-out training-split validation NLL (`output/stage1_hwm/runs/NAME/best.pt`).
Ablations: `--ablate map`, `--ablate vis`, `--hard-labels`, `--modes 1` (see `scripts/stage1_ablation_queue.sh`).

## 4. Development evaluation

```bash
$RUN eventwm_graph.stage1_hwm.bank_hwm output/stage1_hwm/runs/NAME/best.pt /var/tmp/falcon-stage1-claude/banks/NAME --device cuda:0
#   deployment CPU path instead:  --fast-cpu --device cpu
$RUN eventwm_graph.stage1_hwm.metrics /var/tmp/falcon-stage1-claude/banks/NAME NAME
$RUN eventwm_graph.stage1_hwm.compare legacy_final NAME          # side-by-side table
$RUN eventwm_graph.stage1_hwm.report NAME legacy_final           # paired scene-bootstrap CIs
```

## 5. Latency

```bash
CUDA_VISIBLE_DEVICES= $RUN eventwm_graph.stage1_hwm.benchmark output/stage1_hwm/runs/NAME/best.pt RUNNAME   # legacy protocol, 1 CPU thread
$RUN eventwm_graph.stage1_hwm.benchmark_batch output/stage1_hwm/runs/NAME/best.pt RUNNAME --device cuda:0    # B streams per call
```

## 6. Tests

```bash
$RUN tests/stage1_hwm/test_online_equivalence.py CHECKPOINT   # online API features == offline extraction
$RUN tests/stage1_hwm/test_batch_api.py CHECKPOINT            # batched == single features; bank validity, constraints
```
