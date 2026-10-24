# Layer 2.5 query outcomes v1

Runnable experimental integration of the frozen 500k L2 with a shared-context,
per-query typed graph compiler. **Research decision: REVISE for unrestricted
deployment; usable for restricted L3 interface experiments.** Nominal enclosing-
body risk is supported as an approximation. Native mesh contact, official
geodesic PSC, L4 completion and navigation success remain unsupported.

- [Baseline gate](BASELINE_VERIFICATION.md): real checkpoint/API/native calls,
  399 unchanged regression tests before implementation.
- [Architecture](ARCHITECTURE.md): graph, joint-world aggregation, sparse branches,
  masks, ownership and bounded caches.
- [Interface contract](INTERFACE_CONTRACT.md): exact L3/L4 responsibilities and
  unresolved timing/compiler integration.
- [Scientific validation](SCIENTIFIC_VALIDATION.md): matching nominal targets,
  calibration/ranking, censoring, uncertainty and inherited defects.
- [Performance](PERFORMANCE.md): workloads, all latency percentiles, failures,
  cache rates, sensitivity and fallback.
- [Final report](FINAL_REPORT.md): independent acceptance decisions.
- [Review record](REVIEW.md): inspection findings, fixes and evidence boundaries.

Implementation: [compiler.py](../../../eventwm_graph/l25/compiler.py),
[schema.py](../../../eventwm_graph/l25/schema.py),
[graph.py](../../../eventwm_graph/l25/graph.py),
[numeric.py](../../../eventwm_graph/l25/numeric.py),
[tests](../../../tests/l25/test_compiler.py).

## Reproduce on this machine

```bash
cd /research-user/workspace/projects/eventwm-falcon-l25-query-outcomes-20261022
PY=/research-user/miniforge3/envs/eventwm-falcon/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/output/l25_query_outcomes_v1_20261022/cache/numba"
export MPLCONFIGDIR="$PWD/output/l25_query_outcomes_v1_20261022/cache/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
nice -n 19 ionice -c 3 "$PY" -m unittest discover -s tests/l25 -v
nice -n 19 ionice -c 3 "$PY" scripts/l25_verify_baseline.py \
  --output output/l25_query_outcomes_v1_20261022/baseline_reproduce.json
nice -n 19 ionice -c 3 "$PY" scripts/l25_run.py demo \
  --output output/l25_query_outcomes_v1_20261022/demo_reproduce
```

The demo is a real native development root: arrived legal history → actual L2
belief → base graph → existing native compiled candidates → shared joint bank
→ per-query outcomes → typed graph → one consumer logit per candidate. It saves
JSON, scores and two standalone plots. No training, simulator execution or RL
is invoked. Supplied query IDs select references held by the upstream compiler.

Bounded performance/scientific reproduction (about minutes, CPU only):

```bash
nice -n 19 ionice -c 3 "$PY" scripts/l25_run.py benchmark --roots 60 \
  --warm-repeats 5 --cold-repeats 2 \
  --output output/l25_query_outcomes_v1_20261022/benchmark_reproduce
nice -n 19 ionice -c 3 "$PY" scripts/l25_run.py science --roots 60 \
  --output output/l25_query_outcomes_v1_20261022/science_reproduce
nice -n 19 ionice -c 3 "$PY" scripts/l25_ablate.py \
  --output output/l25_query_outcomes_v1_20261022/ablation_reproduce
```

All artifact-producing commands require a fresh output file/directory; change
the suffix to rerun. Benchmark/ablation use non-blocking advisory lock
`/tmp/eventwm-research-inference-benchmark.lock`. No GPU is used. Other research
sessions can use this same lock for expensive inference work. No collector lock,
configuration or process is modified.

The default checkpoint and frozen development snapshot are absolute read-only
paths declared in `scripts/l25_verify_baseline.py`. On another machine, supply
`--checkpoint` and relocate the frozen development `OLD` path in the reproduction
driver; preserve input hashes. The online package itself has no machine-specific
paths, dataset reader or checkpoint selection. Construct it with a loaded model.
Existing L2 packaged dependencies under `docs/l2/` are required by that frozen
model and remain unchanged.

For all 399 inherited tests and numerical oracles, use a fresh tag:

```bash
nice -n 19 ionice -c 3 "$PY" - <<'PYCODE'
from pathlib import Path
import eventwm_graph.l2_intermittent.validate as v
v.OUT = Path('output/l25_query_outcomes_v1_20261022').resolve()
v.run('inherited_reproduce')
PYCODE
```

Full raw runs live in
`/research-user/workspace/projects/eventwm-falcon-l25-query-outcomes-20261022/output/l25_query_outcomes_v1_20261022/`.
Small frozen results, raw timing/label ledgers, input hashes, graph example and
plots are committed under [artifacts](artifacts/). Reproduction uses development
roots only; protected confirmation data are neither opened nor unsealed.

Check the committed artifact hashes and packet invariants:

```bash
"$PY" scripts/l25_package.py --check
```

The default demo's three short queries have zero sampled risk. A second,
explicitly qualitative contrast example is provided as
[illustrative_decision_packet.json](artifacts/illustrative_decision_packet.json),
[graph](artifacts/illustrative_context_graph.png) and
[outcomes](artifacts/illustrative_query_outcomes.png). Its root was selected by
predicted candidate-risk contrast, without truth labels; it is not a cohort
performance claim. Recreate it from the preserved scientific predictions:

```bash
nice -n 19 ionice -c 3 "$PY" scripts/l25_visualize.py \
  --results output/l25_query_outcomes_v1_20261022 \
  --output output/l25_query_outcomes_v1_20261022/illustration_reproduce
```
