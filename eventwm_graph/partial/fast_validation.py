"""Full accepted root checks with one graph reconstruction per unique DAG node.

The original validator performs every integrity, legality, action, time, query,
label, summary, split and root-graph check. Graph deltas and successor revisions
are additionally verified here for every node/outcome, reusing shared prefixes.
Nothing is counted as validated until this entire function succeeds.
"""
from pathlib import Path

from . import storage
from .fast_graph import IncrementalPartialGraph


def _bind(function, **values):
    from types import FunctionType
    return FunctionType(function.__code__, {**function.__globals__, **values}, function.__name__,
                        function.__defaults__, function.__closure__)


def validate_root(folder, reconstruct=True):
    parsed = {}
    def read(path):
        path = Path(path)
        if path not in parsed:
            parsed[path] = storage.read_gzip(path)
        return parsed[path]
    online = _bind(storage.online_input, read_gzip=read)
    reference = _bind(storage.validate_root, read_gzip=read, online_input=online)
    # This flag only delegates the two graph-reconstruction checks to the DAG
    # traversal below; all remaining accepted checks still execute unchanged.
    result = reference(folder, reconstruct=False)
    if not reconstruct:
        return result
    folder = Path(folder)
    root = read(folder/'legal_root.json.gz')
    nodes = read(folder/'future_observations.json.gz')
    outcomes = read(folder/'outcomes.json.gz')
    children = {}
    for node in nodes:
        children.setdefault(node['parent'], []).append(node['node_id'])
    revisions = {None: root['partial_graph']['revision']}
    checked = set()

    def visit(parent, graph):
        successors = children.get(parent, [])
        if not successors:
            return
        snapshot = graph.snapshot() if len(successors) > 1 else None
        for branch, nid in enumerate(successors):
            if nid in checked:
                raise ValueError('invalid_frame_DAG_cycle')
            if branch:
                graph = IncrementalPartialGraph.restore(snapshot)
            node = nodes[nid]
            if graph.update(node['observation']) != node['graph_delta']:
                raise ValueError('graph_delta_not_reconstructable')
            revisions[nid] = graph.revision
            checked.add(nid)
            visit(nid, graph)
    visit(None, IncrementalPartialGraph.restore(root['partial_graph']))
    if checked != set(range(len(nodes))):
        raise ValueError('orphan_future_observation')
    for outcome in outcomes:
        last = outcome['node_ids'][-1] if outcome['node_ids'] else None
        if outcome['successor']['graph_revision'] != revisions[last]:
            raise ValueError('successor_not_reconstructable')
    return result
