"""Reproducible native demo, complete-call benchmark and offline risk audit.

Use one CPU thread and idle priority. All input datasets/checkpoints are read
only. Inference uses legal prefixes only; privileged DEVELOPMENT future labels
are opened only after forecasts are serialized by the scientific audit mode.
"""
import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from l25_verify_baseline import CHECKPOINT, OLD, helpers


def dump(path, value):
    from eventwm_graph.l25.schema import jsonable
    with Path(path).open('x') as f:
        json.dump(jsonable(value), f, indent=2, allow_nan=False)


def quantiles(calls):
    import numpy as np
    result = {}
    for cohort in ('representative', 'dense', 'expanded'):
        for cache in ('warm', 'dynamic_cold'):
            part = [c for c in calls if c['cache'] == cache and (cohort == 'expanded' or (cohort == 'dense' and 'dense12' in c['tags']) or (cohort == 'representative' and 'first60' in c['tags']))]
            if not part:
                continue
            metrics = {}
            for name in ('observe_ms', 'context_ms', 'forecast_ms', 'incremental_ms', 'full_ms', 'candidate_throughput_per_s'):
                values = [c[name] for c in part if c.get('complete')]
                metrics[name] = dict(zip(('p50', 'p95', 'p99'), map(float, np.quantile(values, [.5, .95, .99])))) if values else None
            result[cohort+':'+cache] = dict(attempts=len(part), complete=sum(c.get('complete', False) for c in part),
                                           roots=len({c['root_id'] for c in part}), metrics=metrics,
                                           candidate_counts=dict(Counter(c.get('candidates') for c in part)),
                                           branch_counts=dict(Counter(c.get('branches') for c in part)),
                                           unsupported_candidates=sum(c.get('unsupported', 0) for c in part),
                                           censored_candidates=sum(c.get('censored', 0) for c in part),
                                           budget_truncated_queries=sum(c.get('budget_truncated', 0) for c in part))
    return result


def prefix(model, b, meta):
    folder = OLD/'snapshot/roots'/meta['root_id']
    legal = b.read(folder/'legal_root.json.gz')
    queries = b.read(folder/'queries.json.gz')
    # Same prefix conversion/seed as the historical complete-call benchmark.
    from eventwm_graph.l2_120k.model import load_model
    from eventwm_graph.l2_intermittent.common import seed_for
    baseline = BASE/'docs/l2/graph_wm_120k_20261021/artifacts'
    legacy_model = load_model(population=model.population, observation=str(baseline/'checkpoints/frozen_observation_fit.json'),
                              motion=json.loads((baseline/'motion.json').read_text()))
    legacy = legacy_model._initialize_history(legal['history'][:-1], b.prefix_snapshot(legal))
    prior = model.from_legacy(legacy, seed=seed_for(meta['root_id'], 913))
    return prior.fork(receipt=model._receipt(legal['history'][-2])), legal, queries


def public_goal(legal):
    # The native root stores only relative pointgoal in some releases. Read its
    # documented XYZ goal if available; otherwise use the arrived robot position
    # as an explicitly declared interface-only public goal, never a hidden goal.
    import numpy as np
    for key in ('goal_position', 'goal'):
        value = legal.get(key)
        if isinstance(value, list) and len(value) == 3:
            return value, key
    pose = legal['history'][-1]['robot_pose']
    pointgoal = legal.get('pointgoal', legal.get('pointgoal_with_gps_compass'))
    # partial.collect.root_input records episode.goals[0].position as XYZ.
    if isinstance(pointgoal, list) and len(pointgoal) == 3:
        return pointgoal, 'legal_root.pointgoal XYZ (partial.collect root_input)'
    if isinstance(pointgoal, dict) and 'goal_position' in pointgoal:
        return pointgoal['goal_position'], 'pointgoal.goal_position'
    return list(np.asarray(pose)[:3]), 'interface fixture: arrived ego XYZ as public goal; task-progress claims excluded'


def choose(queries, count):
    return sorted(queries, key=lambda q: (-len(q['compiled']['actions']), q['query_id']))[:count]


def draw(packet, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    s = packet.context.spatial
    fig, ax = plt.subplots(figsize=(9, 7))
    for a, b in s.edge_index.T:
        ax.plot(s.positions[[a, b], 0], s.positions[[a, b], 1], color='#d4dce4', linewidth=.6)
    ax.scatter(s.positions[:-1, 0], s.positions[:-1, 1], c=s.base_features[:-1, 5], cmap='Blues', s=12)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(packet.outcomes))))
    for o, color in zip(packet.outcomes, colors):
        nodes = [r['node'] for r in o['route_occurrences']]
        path = s.positions[nodes]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, label=o['query_id'][:16])
    ax.scatter([packet.context.ego[0]], [packet.context.ego[2]], marker='*', color='black', s=100, label='arrived ego')
    ax.set(xlabel='world X (m)', ylabel='world Z (m)', title='Arrived graph with ordered nominal query regions')
    ax.axis('equal'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'context_graph.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [o['query_id'][:12] for o in packet.outcomes]
    for ax, key, title in zip(axes, ('p_nominal_collision', 'p_social_euclidean_proxy', 'nominal_duration'),
                              ('Nominal body-envelope collision', 'Horizontal social-distance proxy', 'Nominal duration (s)')):
        values = [o[key] if o[key] is not None else 0 for o in packet.outcomes]
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Finite L2 joint-bank estimates; L4 completion and episode success unsupported', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'query_outcomes.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('demo', 'benchmark', 'science'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--roots', type=int, default=60)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--warm-repeats', type=int, default=5)
    parser.add_argument('--cold-repeats', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.roots <= 60 or not 1 <= args.candidates <= 24:
        raise ValueError('bounded development workload required')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(BASE/'output/l25_query_outcomes_v1_20261022'):
        raise ValueError('outputs must remain inside isolated L2.5 result directory')
    args.output.mkdir(parents=True, exist_ok=False)
    # Shared, non-destructive advisory lock for research inference benchmarks.
    # CPU only: no GPU work or device allocation is performed.
    lock = open('/tmp/eventwm-research-inference-benchmark.lock', 'a+')
    if args.mode == 'benchmark':
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
    from eventwm_graph.l2_intermittent.common import seed_for, truth_at
    from eventwm_graph.l25 import QueryOutcomeCompiler, CompiledReference, ComputeBudget
    from eventwm_graph.l25.graph import message_passing_logits
    from eventwm_graph.l25.numeric import joint_events
    from eventwm_graph.l25.validation import risk_metrics
    torch.set_num_threads(1)
    b = helpers()
    start = time.perf_counter()
    model = IntermittentWorldModel.load(args.checkpoint)
    load_ms = (time.perf_counter()-start)*1000
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != '215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223':
        raise ValueError('unexpected checkpoint')
    sources = list((BASE/'eventwm_graph/l25').glob('*.py'))+[Path(__file__)]
    hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows = np.load(OLD/'dev.npy', allow_pickle=True).tolist()
    selection = b.select_cohort(rows, args.roots)
    # Reject any roots outside the established train-source development split.
    manifest = json.loads((OLD/'snapshot/manifest.json').read_text())
    allowed = {r['root_id'] for r in manifest['rows'] if r['split'] == 'dev' and r['source_split'] == 'train'}
    assert {r['root_id'] for r in selection} <= allowed
    if args.mode == 'demo':
        selection = selection[:1]
    dump(args.output/'cohort.json', selection)
    calls, errors, predictions, labels, censored = [], [], [], [], []
    startup = []
    cache_probe = []
    saved = False
    with (args.output/'calls.jsonl').open('x') as call_file:
        for index, meta in enumerate(selection):
            try:
                ti = time.perf_counter()
                prior, legal, queries = prefix(model, b, meta)
                prefix_ms = (time.perf_counter()-ti)*1000
                query_inputs = choose(queries, args.candidates)
                arrived = legal['history'][-1]
                refs = [CompiledReference.from_native(q, arrived['timestamp']) for q in query_inputs]
                goal, goal_source = public_goal(legal)
                compiler = QueryOutcomeCompiler(model)
                budget = ComputeBudget(worlds=args.worlds, seed=seed_for(meta['root_id'], 927), soft_budget_ms=1e6)

                def invoke(cache, repeat):
                    if cache == 'dynamic_cold':
                        compiler.clear_dynamic_caches()
                        b.clear_dynamic_caches(model)
                    began = time.perf_counter()
                    updated = model.observe(prior, arrived)
                    t1 = time.perf_counter()
                    ctx = compiler.build_context(updated, updated.graph, updated.robot_pose, goal)
                    t2 = time.perf_counter()
                    packet = compiler.evaluate_candidates(ctx, refs, replace(budget, seed=seed_for(meta['root_id'], 927+repeat)))
                    end = time.perf_counter()
                    tm = packet.metadata['timings_ms']
                    record = dict(root_id=meta['root_id'], scene_id=meta['scene_id'], tags=meta['tags'], cache=cache, repeat=repeat,
                                  complete=True, candidates=len(refs), worlds=args.worlds, branches=len(packet.branches),
                                  static_cells=ctx.metadata['static_cells'], spatial_nodes=len(ctx.spatial.ids),
                                  graph_nodes=len(packet.graph['node_features']), edges=packet.graph['edge_index'].shape[1],
                                  tracked_max=max(c.n for c in updated.components), stored_timestamps=len(packet.metadata['future_times']),
                                  observe_ms=(t1-began)*1000, context_ms=(t2-t1)*1000, forecast_ms=tm['forecast_ms'],
                                  full_ms=(end-began)*1000,
                                  incremental_ms=(end-t1)*1000-tm['forecast_ms'],
                                  candidate_throughput_per_s=len(refs)/(end-began),
                                  unsupported=sum(not o['candidate_legal'] for o in packet.outcomes),
                                  censored=sum(o['censored_running'] for o in packet.outcomes),
                                  budget_truncated=packet.metadata['budget_truncated_queries'], cache_statistics=packet.metadata['cache'],
                                  stage_timings=tm, goal_source=goal_source, rss_bytes=b.rss())
                    return record, packet

                first, packet = invoke('unprimed', 0)
                startup.append(dict(root_id=meta['root_id'], prefix_reconstruction_ms=prefix_ms, first_call=first))
                if not saved:
                    packet.save(args.output/'decision_packet.json')
                    logits, mask = message_passing_logits(packet.graph)
                    dump(args.output/'consumer.json', dict(query_ids=packet.graph['query_ids'], logits=logits, candidate_mask=mask))
                    if args.mode == 'demo':
                        draw(packet, args.output)
                    saved = True
                if args.mode == 'benchmark':
                    for cache, repeats in [('warm', args.warm_repeats), ('dynamic_cold', args.cold_repeats)]:
                        for rep in range(repeats):
                            record, packet = invoke(cache, rep)
                            calls.append(record)
                            call_file.write(json.dumps(record, allow_nan=False)+'\n'); call_file.flush()
                    # Fixed-belief cache probe: explicit repeated output request,
                    # separate from live complete calls with changing tape seed.
                    ctx = packet.context
                    probe_budget = replace(budget, seed=17)
                    compiler.evaluate_candidates(ctx, refs, probe_budget)
                    t = time.perf_counter(); cached = compiler.evaluate_candidates(ctx, refs, probe_budget)
                    cache_probe.append(dict(root_id=meta['root_id'], elapsed_ms=(time.perf_counter()-t)*1000,
                                            cache=cached.metadata['cache'], candidates=len(refs)))
                elif args.mode == 'science':
                    updated = packet.context.belief
                    predictions_for_root = []
                    for horizon in (.5, 1., 2., 3.):
                        evaluated = compiler.evaluate_candidates(packet.context, refs, replace(budget, horizon=horizon, branch_queries=0))
                        for ref, outcome in zip(refs, evaluated.outcomes):
                            rec = dict(root_id=meta['root_id'], scene_id=meta['scene_id'], query_id=ref.query_id, horizon=horizon,
                                       timestamp=updated.timestamp, covered_duration=outcome['covered_duration'],
                                       nominal_collision=outcome['p_nominal_collision'], social_proxy=outcome['p_social_euclidean_proxy'],
                                       source_belief=updated.fingerprint(), future_identity=evaluated.metadata['future_identity'])
                            predictions_for_root.append(rec)
                    # Seal forecasts before opening offline labels; truth never
                    # enters context, proposals, weights, caches or branches.
                    dump(args.output/('predictions_'+meta['root_id']+'.json'), predictions_for_root)
                    predictions.extend(predictions_for_root)
                    with np.load(OLD/'snapshot/roots'/meta['root_id']/'privileged_future.npz') as f:
                        truth_times = f['timestamps'].copy()
                        truth = f['positions'][..., [0, 2]].copy()
                    for record in predictions_for_root:
                        ref = next(q for q in refs if q.query_id == record['query_id'])
                        n = int(round(record['covered_duration']/.048))
                        target_times = ref.times[1:n+1]
                        indices = np.searchsorted(truth_times, target_times-1e-7)
                        if record['nominal_collision'] is None or target_times[-1] > truth_times[-1]+1e-7:
                            censored.append({**record, 'status': 'UNSUPPORTED_OR_TRUTH_HORIZON_CENSORED'})
                            continue
                        exact = not np.any(indices >= len(truth_times)) and np.max(abs(truth_times[indices]-target_times)) <= 1e-5
                        if not exact:
                            censored.append({**record, 'status': 'EXACT_NATIVE_CLOCK_UNSUPPORTED',
                                             'note': 'actual native tape can include 56ms renewal intervals; nominal clock is 48ms'})
                        # Auxiliary time-aligned path labels use the SAME existing
                        # L2 offline linear truth interpolation, never extrapolate.
                        # They are not native contact or exact execution labels.
                        aligned = np.stack([truth_at(truth_times, truth, t) for t in target_times])
                        label_sets = [('interpolated', aligned)]
                        if exact:
                            label_sets.append(('exact', truth[indices]))
                        for alignment, positions in label_sets:
                            hit, separation = joint_events(positions[None], ref.poses[None, 1:n+1])
                            for kind, prediction, label in [('nominal_envelope', record['nominal_collision'], bool(hit.any())),
                                                            ('social_horizontal_proxy', record['social_proxy'], bool((separation < 1.).any()))]:
                                labels.append(dict(root_id=meta['root_id'], scene_id=meta['scene_id'], query_id=ref.query_id,
                                                   horizon=record['horizon'], kind=kind+':'+alignment, prediction=prediction, label=label,
                                                   covered_duration=record['covered_duration']))
            except Exception as exc:
                import traceback
                errors.append(dict(root_id=meta['root_id'], error=repr(exc), traceback=traceback.format_exc()))
            if (index+1) % 10 == 0 or index+1 == len(selection):
                print(json.dumps(dict(roots=index+1, selected=len(selection), errors=len(errors), calls=len(calls))), flush=True)
    result = dict(mode=args.mode, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash, model_version=model.model_version,
                  source_sha256=hashes, sources_unchanged=all(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h for p, h in hashes.items()),
                  checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==checkpoint_hash,
                  environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                                   torch_threads=torch.get_num_threads(), nice=os.nice(0), gpu_used=False),
                  import_and_process_startup='external /usr/bin/time measurement; model load below excludes imports',
                  model_load_ms=load_ms, startup=startup, errors=errors, quantiles=quantiles(calls), cache_probes=cache_probe,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  wall_seconds=time.perf_counter()-start,
                  workload='actual arrived observation + arrived context + ONE shared future bank + all candidate risk/social/time/progress/optical features + one coarse event/two successors + graph tensor export; GNN outside timed call',
                  budget_semantics='branches soft budget 1e6 ms during benchmark; no outputs disabled for speed; native prefix cap 120 endpoints explicit',
                  latency_gates=dict(incremental_p95_ms=10., full_call_p95_ms=100.),
                  confirmation_payloads_opened=False)
    if args.mode == 'science':
        dump(args.output/'labels.json', labels)
        dump(args.output/'censored.json', censored)
        result['science'] = {kind+':'+alignment+':'+str(h): risk_metrics([r for r in labels if r['kind']==kind+':'+alignment and r['horizon']==h])
                             for kind in ('nominal_envelope', 'social_horizontal_proxy') for alignment in ('exact', 'interpolated') for h in (.5, 1., 2., 3.)}
        result['scientific_scope'] = 'observed DEVELOPMENT exogenous human futures on matching nominal robot references; envelope and horizontal proxies, not mesh contacts/official PSC/L4 execution/navigation success; no tuning'
    dump(args.output/'run.json', result)
    print(json.dumps(dict(output=str(args.output), errors=errors, quantiles=result['quantiles']), indent=2), flush=True)
    if errors:
        raise RuntimeError('retained root errors; see run.json')


if __name__ == '__main__':
    main()
