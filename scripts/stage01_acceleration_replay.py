#!/usr/bin/env python
"""Original independent fresh-Env/observer replay with extended source binding.

The replay, legal observer, graph/query reconstruction, metrics and tolerances
are unchanged. Only source_versions knows about the additional producer files.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eventwm_graph.config import file_digest, write_json
from eventwm_graph.partial import replay_review
from eventwm_graph.partial.acceleration import bind, source_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+'); p.add_argument('--output', required=True)
    p.add_argument('--max-queries', type=int, default=4); p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    def sources():
        return {**replay_review.source_versions(), **source_identity(),
                'acceleration/replay_adapter': file_digest(__file__)}
    review = bind(replay_review.review_root, source_versions=sources)
    began = time.perf_counter(); reports = []
    for folder in args.roots:
        try:
            reports.append(review(folder, args.max_queries, args.repeats))
        except Exception as error:
            reports.append(dict(folder=str(Path(folder).resolve()), status='FAIL', exception=repr(error)))
    result = dict(schema='stage01_acceleration_native_replay_1',
                  status='PASS' if reports and all(r['status'] == 'PASS' for r in reports) else 'FAIL',
                  implementation=source_identity(), seconds=time.perf_counter()-began, roots=reports,
                  branches=sum(len(r.get('branches', [])) for r in reports),
                  action_endpoints=sum(b.get('executed_steps', 0) for r in reports for b in r.get('branches', [])),
                  actual_reference='Original fresh NativeEpisode/Env.step and original GeometryObserver; original full validator and replay tolerances',
                  training_started=False, production_started=False)
    write_json(args.output, result)
    print(result['status'], result['branches'], result['action_endpoints'], args.output)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
