"""Read-only frozen L2 gate. No fitting or confirmation data access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
CHECKPOINT = Path('/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt')
OLD = Path('/research-user/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021')


def helpers():
    path = BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py'
    spec = importlib.util.spec_from_file_location('frozen_l2_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    args = p.parse_args()
    import numpy as np
    import torch
    from eventwm_graph.l2_intermittent.api_example import example
    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
    from eventwm_graph.l2_intermittent.observations import predict_observations
    b = helpers()
    torch.set_num_threads(1)
    tick = time.perf_counter()
    model = IntermittentWorldModel.load(args.checkpoint)
    load_ms = 1000*(time.perf_counter()-tick)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert digest == '215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223'
    ck = torch.load(args.checkpoint, map_location='cpu')
    smoke = example(checkpoint=args.checkpoint)
    rows = np.load(OLD/'dev.npy', allow_pickle=True).tolist()
    # Selection uses only development-root metadata, before legal payload reads.
    selection = b.select_cohort(rows, 8)
    dense = sorted(rows, key=lambda r: (-max((h.n for h in r['filtered_hypotheses']), default=0), r['root_id']))[:3]
    ids = {r['root_id'] for r in selection}
    for r in dense:
        if r['root_id'] not in ids:
            selection.append(dict(root_id=r['root_id'], scene_id=r['scene_id'], tags=['dense'], history_length=r['history_length']))
    calls, checks = [], []
    for meta in selection:
        folder = OLD/'snapshot/roots'/meta['root_id']
        legal = b.read(folder/'legal_root.json.gz')
        queries = b.read(folder/'queries.json.gz')
        prior = model.initialize(legal['history'][:-1], b.prefix_snapshot(legal))
        updated = model.observe(prior, legal['history'][-1])
        future = model.predict_shared(updated, 3., 16, 731)
        events = [predict_observations(model, updated, q, future) for q in queries[:2]]
        cells = updated.graph.materialize()
        assert len(queries) >= 2 and queries[0]['query_id'] != queries[1]['query_id']
        assert abs(future.times[0]-updated.timestamp) < 1e-9
        assert np.isclose(future.weights.sum(), 1) and np.isfinite(future.times).all()
        checks.append(dict(root_id=meta['root_id'], graph_namespace=updated.graph.namespace,
                           graph_cells=len(cells), graph_version=updated.graph.version,
                           timestamp=updated.timestamp, future_shape=list(future.positions.shape),
                           hidden_shape=list(future.hidden_positions.shape), query_ids=[q['query_id'] for q in queries[:2]],
                           statuses=[e['status'] for e in events], shared_future_identity=future.identity()))
        query = max(queries, key=lambda q: (len(q['compiled']['actions']), q['query_id']))
        b.one_call(model, prior, legal['history'][-1], query, meta, 'unprimed', 16, 'prime', 0)
        for cache in ('warm', 'dynamic_cold'):
            for rep in range(3):
                calls.append(b.one_call(model, prior, legal['history'][-1], query, meta, cache, 16, 'primary', rep))
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
                  configuration=ck['configuration'], parameter_count=sum(p.numel() for p in model.learned.network.parameters()),
                  model_version=model.model_version, model_load_ms=load_ms,
                  synthetic_smoke=smoke, native_checks=checks, calls=calls,
                  quantiles=b.quantiles(calls), checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest,
                  scope='8 first development roots + 3 densest, 3 warm/3 dynamic cold repeats; not the original 83-root benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), quantiles=result['quantiles'], native_roots=len(checks)), indent=2))


if __name__ == '__main__':
    main()
