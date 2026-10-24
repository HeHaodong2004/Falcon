"""Bounded CPU query/world/branch sensitivity; no training or private labels."""
import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l25_run import helpers, prefix, choose, public_goal, dump
from l25_verify_baseline import OLD, CHECKPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = open('/tmp/eventwm-research-inference-benchmark.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
    from eventwm_graph.l25 import QueryOutcomeCompiler, CompiledReference, ComputeBudget
    torch.set_num_threads(1)
    model = IntermittentWorldModel.load(CHECKPOINT)
    b = helpers()
    rows = np.load(OLD/'dev.npy', allow_pickle=True).tolist()
    cohort = b.select_cohort(rows, 60)
    selected = [cohort[5], next(r for r in cohort if 'dense12' in r['tags'])]
    calls = []
    for meta in selected:
        prior, legal, queries = prefix(model, b, meta)
        actual = model.observe(prior, legal['history'][-1])
        compiler = QueryOutcomeCompiler(model)
        context = compiler.build_context(actual, actual.graph, actual.robot_pose, public_goal(legal)[0])
        refs = [CompiledReference.from_native(q, actual.timestamp) for q in choose(queries, 8)]
        for count in (1, 4, 8):
            candidates = refs[:count]
            for worlds in (8, 16, 32, 64):
                for branch_queries in (0, 1):
                    for rep in range(4):
                        compiler.caches['future'].clear()
                        compiler.caches['numeric'].clear()
                        compiler.caches['branches'].clear()
                        budget = ComputeBudget(worlds=worlds, seed=927+rep, branch_queries=branch_queries, soft_budget_ms=1e6)
                        began = time.perf_counter()
                        packet = compiler.evaluate_candidates(context, candidates, budget)
                        elapsed = (time.perf_counter()-began)*1000
                        calls.append(dict(root_id=meta['root_id'], dense='dense12' in meta['tags'], candidates=len(candidates),
                                          requested_candidates=count, worlds=worlds, branch_queries=branch_queries,
                                          branch_count=len(packet.branches), repeat=rep, evaluation_ms=elapsed,
                                          incremental_ms=packet.metadata['timings_ms']['incremental_ms'],
                                          forecast_ms=packet.metadata['timings_ms']['forecast_ms'],
                                          nominal_collision=[o['p_nominal_collision'] for o in packet.outcomes]))
        # Correct soft-budget fallback: retain every core outcome, put all
        # supported unexpanded event mass in the explicit residual, zero branches.
        full = compiler.evaluate_candidates(context, refs, ComputeBudget(soft_budget_ms=1e6))
        fallback = compiler.evaluate_candidates(context, refs, ComputeBudget(soft_budget_ms=0.))
        assert [o['p_nominal_collision'] for o in full.outcomes] == [o['p_nominal_collision'] for o in fallback.outcomes]
        informative = any(o['candidate_legal'] and o['finite_event_mi_nats'] > 1e-8 for o in full.outcomes)
        assert not fallback.branches and fallback.metadata['budget_truncated_queries'] == int(informative)
    summary = []
    keys = sorted({(c['dense'], c['requested_candidates'], c['worlds'], c['branch_queries']) for c in calls})
    for dense, count, worlds, branch_queries in keys:
        part = [c for c in calls if (c['dense'], c['requested_candidates'], c['worlds'], c['branch_queries']) == (dense, count, worlds, branch_queries)]
        summary.append(dict(dense=dense, requested_candidates=count, actual_candidates=part[0]['candidates'],
                            worlds=worlds, branch_queries=branch_queries,
                            p50_evaluation_ms=float(np.quantile([c['evaluation_ms'] for c in part], .5)),
                            p95_evaluation_ms=float(np.quantile([c['evaluation_ms'] for c in part], .95)),
                            p95_incremental_ms=float(np.quantile([c['incremental_ms'] for c in part], .95))))
    dump(args.output/'ablation.json', dict(scope='two fixed legal development roots; four repeats; candidate evaluation including fresh forecast, EXCLUDES actual observation and context build; exploratory sensitivity, not full-call gate',
                                          selected=selected, calls=calls, summary=summary,
                                          fallback_core_equality=True, fallback_hard_deadline_supported=False))
    print(json.dumps(dict(output=str(args.output), calls=len(calls), fallback_core_equality=True)))


if __name__ == '__main__':
    main()
