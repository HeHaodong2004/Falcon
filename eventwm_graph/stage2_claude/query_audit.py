"""Step 1 of the query-ranking benchmark: dataset suitability audit (development split, no predictions).

Every development Root holds Queries and their recorded Outcomes. An Outcome is a path of executed nodes in the
Root's prefix DAG (Queries that share an action prefix share nodes); every Outcome starts from the Root's
decision state (same partial graph, legal history and sensor frames). The evaluation windows of the original
benchmark are deduplicated (start node, end node) segments: per Query the full path plus 0.5 / 1 / 2 s
sub-windows at random start steps, i.e. subdivisions of the same Outcome, not independent queries.

Comparison sets used by the benchmark (only saved nodes, never interpolated):
  full  : each Query's complete executed path from the Root (durations differ between Queries);
  H s   : each Query truncated at its saved node closest to t0 + H, kept only if that node lies within
          `tol` seconds of t0 + H (Queries that ended earlier have no outcome at H and are excluded).
Two Queries whose (truncated) paths end at the same node are the SAME outcome and form no pair.
"""
import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import get_context

import numpy as np

from eventwm_graph.stage2_repair.diagnose import OUT, source_rows
from eventwm_graph.stage2_segment.data import load

HORIZONS = (.5, 1., 2.)
TOL = .05          # s; one discrete action is ~0.048 s


def outcomes_at(legal, queries, outcomes, nodes, horizon, tol=TOL):
    """[(query_id, end_node, end_step, duration, termination)] of the comparable outcomes at a horizon ('full' or s)."""
    t0 = legal['partial_graph']['timestamp']
    out = []
    for q, o in zip(queries, outcomes):
        ids = o['node_ids']
        if not ids:
            continue
        times = np.array([nodes[i]['observation']['timestamp'] for i in ids])
        if horizon == 'full':
            b = len(ids)
        else:
            j = int(np.argmin(abs(times-(t0+horizon))))
            if abs(times[j]-(t0+horizon)) > tol:
                continue
            b = j+1
        out.append((q['query_id'], ids[b-1], b, float(times[b-1]-t0), o['summary']['termination']))
    return out


def audit_root(row):
    legal, queries, outcomes, nodes, children = load(row)
    t0 = legal['partial_graph']['timestamp']
    info = dict(root_id=row['root_id'], scene_id=row['scene_id'], queries=len(queries), nodes=len(nodes),
                history_frames=len(legal['history']), history_seconds=float(t0-legal['history'][0]['timestamp']),
                terminations=Counter(o['summary']['termination'] for o in outcomes),
                durations=[float(nodes[o['node_ids'][-1]]['observation']['timestamp']-t0) if o['node_ids'] else 0. for o in outcomes],
                empty_paths=sum(not o['node_ids'] for o in outcomes),
                same_decision_time=all(q['decision_timestamp'] == t0 for q in queries))
    for h in ('full',)+HORIZONS:
        sel = outcomes_at(legal, queries, outcomes, nodes, h)
        distinct = {}
        for qid, end, b, dur, term in sel:
            distinct.setdefault(end, []).append(qid)
        n = len(distinct)
        info['h_%s' % h] = dict(queries_with_outcome=len(sel), distinct_outcomes=n, pairs=n*(n-1)//2,
                                shared_outcome_queries=len(sel)-n)
    return info


def main(workers=6, out=None):
    rows, _ = source_rows('dev')
    with get_context('spawn').Pool(workers) as pool:
        infos = pool.map(audit_root, rows, chunksize=8)
    summary = dict(dev_roots=len(infos), scenes=len({i['scene_id'] for i in infos}),
                   queries_total=sum(i['queries'] for i in infos),
                   queries_per_root=dict(sorted(Counter(i['queries'] for i in infos).items())),
                   terminations=dict(sum((i['terminations'] for i in infos), Counter())),
                   empty_paths=sum(i['empty_paths'] for i in infos),
                   all_queries_share_decision_time=all(i['same_decision_time'] for i in infos),
                   history_seconds=dict(zip(('p10', 'p50', 'p90'), np.percentile([i['history_seconds'] for i in infos], [10, 50, 90]).round(3).tolist())),
                   duration_seconds=dict(zip(('p10', 'p50', 'p90', 'max'), [*np.percentile(sum((i['durations'] for i in infos), []), [10, 50, 90]).round(3).tolist(),
                                                                              round(max(max(i['durations']) for i in infos), 3)])))
    for h in ('full',)+HORIZONS:
        k = 'h_%s' % h
        elig = [i for i in infos if i[k]['distinct_outcomes'] >= 2]
        summary[k] = dict(eligible_roots=len(elig), eligible_scenes=len({i['scene_id'] for i in elig}),
                          outcomes_in_eligible_roots=sum(i[k]['distinct_outcomes'] for i in elig),
                          pairs=sum(i[k]['pairs'] for i in elig),
                          queries_with_outcome=sum(i[k]['queries_with_outcome'] for i in infos),
                          queries_merged_as_shared_outcomes=sum(i[k]['shared_outcome_queries'] for i in infos),
                          roots_excluded_fewer_than_two_outcomes=len(infos)-len(elig))
    print(json.dumps(summary, indent=1))
    if out:
        (OUT.parent/'claude_stage2'/'query_bench').mkdir(parents=True, exist_ok=True)
        path = OUT.parent/'claude_stage2'/'query_bench'/out
        path.write_text(json.dumps(dict(summary=summary, roots=[dict(i, terminations=dict(i['terminations'])) for i in infos]), indent=1))
        print('wrote', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--out', default='audit.json')
    main(**vars(p.parse_args()))
