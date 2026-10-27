"""Sibling candidate robot paths of each development Root (downstream ranking probe).

Every Root stores several Query branches executed from the same legal state
against the same exogenous humans. Their executed robot poses form a set of
candidate trajectories that one shared Stage 1 bank must score jointly.
Only robot poses are read here; humans come from the privileged truth.
"""
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from .extract import DEV_MANIFEST, DEV_ROOTS, read


def root_queries(root_id):
    folder = DEV_ROOTS / root_id
    queries = read(folder / 'queries.json.gz');outcomes = read(folder / 'outcomes.json.gz')
    nodes = read(folder / 'future_observations.json.gz')
    legal = read(folder / 'legal_root.json.gz');start = legal['history'][-1]
    out = []
    for q, o in zip(queries, outcomes):
        assert q['query_id'] == o['query_id']
        rows = [[start['timestamp'], *start['robot_pose']]] + [[nodes[i]['observation']['timestamp'], *nodes[i]['observation']['robot_pose']] for i in o['node_ids']]
        out.append(dict(query_id=q['query_id'], poses=np.array(rows, np.float64), actions=len(o['node_ids'])))
    return root_id, out


def main():
    m = json.loads(DEV_MANIFEST.read_text())
    roots = [r['root_id'] for r in m['rows'] if r['split'] == 'dev']
    with ProcessPoolExecutor(12) as pool:
        result = dict(pool.map(root_queries, roots, chunksize=16))
    dest = Path('/var/tmp/falcon-stage1-claude/hwm_data/dev_queries.pkl')
    pickle.dump(result, open(dest, 'wb'))
    n = [len(v) for v in result.values()]
    print(json.dumps(dict(roots=len(result), queries=int(sum(n)), per_root=[int(min(n)), float(np.mean(n)), int(max(n))])))


if __name__ == '__main__':
    main()
