"""One qualitative contrast example selected from predictions, never labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l25_run import helpers, prefix, choose, public_goal, draw, dump
from l25_verify_baseline import CHECKPOINT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import json
    import numpy as np
    import torch
    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
    from eventwm_graph.l2_intermittent.common import seed_for
    from eventwm_graph.l25 import QueryOutcomeCompiler, CompiledReference, ComputeBudget
    from eventwm_graph.l25.graph import message_passing_logits
    torch.set_num_threads(1)
    cohort = json.loads((args.results/'science_release/cohort.json').read_text())
    contrasts = []
    for path in sorted((args.results/'science_release').glob('predictions_*.json')):
        rows = [r for r in json.loads(path.read_text()) if r['horizon'] == 3. and r['nominal_collision'] is not None]
        if rows:
            contrasts.append((max(r['nominal_collision'] for r in rows)-min(r['nominal_collision'] for r in rows), rows[0]['root_id']))
    contrast, root_id = max(contrasts)
    meta = next(r for r in cohort if r['root_id'] == root_id)
    model = IntermittentWorldModel.load(CHECKPOINT)
    prior, legal, queries = prefix(model, helpers(), meta)
    actual = model.observe(prior, legal['history'][-1])
    compiler = QueryOutcomeCompiler(model)
    context = compiler.build_context(actual, actual.graph, actual.robot_pose, public_goal(legal)[0])
    refs = [CompiledReference.from_native(q, actual.timestamp) for q in choose(queries, 8)]
    packet = compiler.evaluate_candidates(context, refs, ComputeBudget(seed=seed_for(root_id, 927), soft_budget_ms=1e6))
    packet.save(args.output/'decision_packet.json')
    logits, mask = message_passing_logits(packet.graph)
    dump(args.output/'consumer.json', dict(query_ids=packet.graph['query_ids'], logits=logits, mask=mask))
    dump(args.output/'selection.json', dict(root=meta, rule='maximum predicted nominal collision contrast across legal candidate queries, deterministic root-ID tie break',
                                           contrast=contrast, scope='qualitative illustration, not representative performance or independent validation',
                                           truth_labels_used_for_selection=False, source_belief=actual.fingerprint(),
                                           graph_feature_shape=list(packet.graph['node_features'].shape)))
    draw(packet, args.output)
    print(json.dumps(dict(root_id=root_id, contrast=contrast, candidates=len(refs), output=str(args.output))))


if __name__ == '__main__':
    main()
