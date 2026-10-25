"""Owned future-map construction with one cell copy and unchanged semantics.

The inherited updater replaces rows; predicted updates must address initially
UNKNOWN cells. Copy scalar cell records using the C dict copier, retain their
owned coordinate lists, and serialize them once. All topology pruning still
uses the exact existing compiled supercover implementation.
"""
import copy
from collections.abc import Mapping
import numpy as np
from eventwm_graph.stage2.common import SCHEMA
from eventwm_graph.stage2_quality.geometry import graph_identity
from eventwm_graph.stage2_quality.update import FastPartialGraph


FIELDS = frozenset(('cell','state','height','first_seen','last_seen','source'))

def own(value, thaw=False):
    if isinstance(value, Mapping):
        return {k: own(v, thaw) for k, v in value.items()}
    if isinstance(value, list):
        return [own(v, thaw) for v in value]
    if isinstance(value, tuple):
        items = (own(v, thaw) for v in value)
        return list(items) if thaw else tuple(items)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return copy.deepcopy(value)


def apply_graph_delta(current, delta, *, verified_identity=None):
    """verified_identity is private synchronous-call reuse, never a cache key.

    Public standalone callers omit it. SegmentPredictor supplies the identity
    just computed from this same caller-owned graph inside the synchronous API.
    """
    if delta.get('schema') != SCHEMA or delta.get('predicted') is not True:
        raise ValueError('invalid predicted GraphDelta')
    identity = graph_identity(current) if verified_identity is None else verified_identity
    if delta['base_graph_identity'] != identity:
        raise ValueError('GraphDelta base graph mismatch')
    thaw = not isinstance(current, dict)
    graph = FastPartialGraph(current['resolution'], current['height'], current['clearance'])
    graph.timestamp = current['timestamp']; graph.revision = current['revision']
    graph.ego_support = np.asarray(current['ego_support']).copy() if current.get('ego_support') is not None else None
    graph.visited = own(current['visited'], thaw)
    graph.global_nodes = own(current.get('global_nodes', {}), thaw)
    graph.global_edges = own(current.get('global_edges', []), thaw)
    graph.cells = {}
    for cell in current['cells']:
        # A legal record's fixed fields are scalar. Extensions retain the
        # reference deep-copy behavior, including frozen Sandbox mappings.
        if len(cell) == 6 and FIELDS == cell.keys() and isinstance(cell['source'], str):
            row = cell.copy() if isinstance(cell, dict) else dict(cell)
            row['cell'] = list(cell['cell'])
        else:
            row = own(cell, thaw)
            row['cell'] = list(cell['cell'])
        graph.cells[tuple(row['cell'])] = row
    now = float(delta['timestamp'])
    if now < graph.timestamp:
        raise ValueError('GraphDelta clock regressed')
    updates = delta['static_updates']; seen = set()
    for row in updates:
        key = tuple(row['cell'])
        if key in seen or graph.state(key) != 0 or row['state'] not in (1, 2) or row['timestamp'] != now or not np.isfinite(row['height']):
            raise ValueError('delta must update unique initially UNKNOWN cells only')
        seen.add(key)
    if now == graph.timestamp:
        if updates:
            raise ValueError('zero-duration trajectory cannot reveal cells')
    else:
        graph.update(dict(timestamp=now, robot_pose=delta['robot_pose'], static_updates=updates, tracks=[]))
    cells = []
    for key, row in sorted(graph.cells.items()):
        if 'cell' not in row:
            row['cell'] = list(key)
        cells.append(row)
    return dict(schema='causal_partial_cspace_graph_1', resolution=graph.resolution, height=graph.height,
        clearance=graph.clearance, timestamp=graph.timestamp, revision=graph.revision,
        ego_support=graph.ego_support.tolist() if graph.ego_support is not None else None,
        cells=cells, visited=graph.visited, global_nodes=graph.global_nodes, global_edges=graph.global_edges,
        hypothetical=True, prediction_checkpoint_sha256=delta['checkpoint_sha256'])
