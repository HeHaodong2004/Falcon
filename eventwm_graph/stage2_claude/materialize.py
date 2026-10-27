"""Complete hypothetical future graphs from decoded candidate rows (linear merge).

Same result as `stage2_repair.materialize.future_from_columns` (same checks,
errors, headers and table), but when the base table is in canonical key order
(its known-cell codes strictly increase) the predicted rows, which are ordered
and disjoint from the known cells, are merged in linear time instead of
re-sorting the whole table. Otherwise the original function is used.
"""
import numpy as np
from types import MappingProxyType

from eventwm_graph.stage2_repair.materialize import future_from_columns
from eventwm_graph.stage2_segment.native import Table, NativePartialGraph

SOURCE = 'predicted_stage2_research'


def future_from_columns_merged(base, keys, states, heights, now, pose, checkpoint):
    if base is None or not base.valid_topology:
        return None
    known = base.known_codes
    if not len(keys) or (len(known) > 1 and not np.all(known[1:] > known[:-1])):
        return future_from_columns(base, keys, states, heights, now, pose, checkpoint)
    old = float(base.headers['timestamp']); now = float(now)
    if not np.isfinite(now) or now < old:
        raise ValueError('invalid hypothetical clock')
    if now == old:
        raise ValueError('starting-only path cannot reveal cells')
    if not np.all((states == 1) | (states == 2)) or not np.isfinite(heights).all():
        raise ValueError('invalid increment')
    biased = (keys.astype(np.int64)+np.int64(2147483648)).astype(np.uint64)
    codes = (biased[:, 0] << np.uint64(32)) | biased[:, 1]
    if np.any(codes[1:] <= codes[:-1]):
        raise ValueError('increment keys must be unique and ordered')
    pos = np.searchsorted(known, codes); valid = pos < len(known)
    if np.any(known[pos[valid]] == codes[valid]):
        raise ValueError('known-cell update forbidden')
    t = base.table; names = list(t.sources)
    if SOURCE not in names:
        names.append(SOURCE)
    n, m = len(known), len(keys)
    new_at = pos+np.arange(m)
    fresh = np.zeros(n+m, bool); fresh[new_at] = True
    old_at = np.flatnonzero(~fresh)

    def merged(a, b, dtype):
        out = np.empty((n+m,)+np.shape(a)[1:], dtype)
        out[old_at] = a; out[new_at] = b
        return out

    table = Table(merged(t.keys, keys.astype(np.int64), np.int64),
                  merged(t.states, states, np.result_type(t.states, states)),
                  merged(t.heights, heights, np.result_type(t.heights, heights)),
                  merged(t.first, np.full(m, now), np.result_type(t.first, np.float64)),
                  merged(t.last, np.full(m, now), np.result_type(t.last, np.float64)),
                  merged(t.source_ids, np.full(m, names.index(SOURCE), np.int32), np.result_type(t.source_ids, np.int32)), names)
    headers = dict(base.headers)
    headers.update(timestamp=now, ego_support=tuple(pose[:3]))
    visited = headers['visited']
    if not visited or np.linalg.norm((np.asarray(pose[:3])-np.asarray(visited[-1]['position']))[[0, 2]]) >= .55:
        headers['visited'] = (*visited, MappingProxyType(dict(position=tuple(pose[:3]), timestamp=now)))
    headers['revision'] += 1
    headers.update(hypothetical=True, prediction_checkpoint_sha256=checkpoint)
    return NativePartialGraph(headers, table)
