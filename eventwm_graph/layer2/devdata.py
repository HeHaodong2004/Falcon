"""Frozen development data for integrated Layer 2 tests and benchmarks (read-only).

Decision states are rebuilt from the frozen development snapshot (2,110 Roots, 108 scenes,
the split of both stages' development results) and the frozen 256 px legal range packets:

* the Root decision: ``legal_root.history`` frames, the Root's legal partial graph and the
  range frames of those observations;
* a later decision on a recorded branch: history + the arrived branch observations, the
  legal graph replayed from their certificates (every recorded ``graph_delta`` verified) and
  the range frames of exactly those observations.

Candidates are the recorded executions of the alternative Queries issued at the same Root
(robot poses of the recorded branch nodes, native 0.048 s clock). Future observations are
read only to build candidates and evaluation labels, never as model inputs.
"""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover - plain json is slower but identical
    _loads = json.loads

SNAPSHOT = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot')
STAGE2_OUT = Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-claude-handoff-20261026/output')
SENSOR = STAGE2_OUT/'stage2_repair_20261025/full_range_branches_256'
REFERENCE = STAGE2_OUT/'claude_stage2/query_bench/predictions/v17a.json.gz'
RANGE_PROTOCOL = 'ideal_current_jaw_range_dynamic_occlusion_1'


def dev_rows():
    m = json.loads((SNAPSHOT/'manifest.json').read_text())
    return [dict(r, folder=str(SNAPSHOT/'roots'/r['root_id'])) for r in m['rows'] if r['split'] == 'dev']


def read_member(row, name):
    raw = (Path(row['folder'])/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['files'][name]:
        raise ValueError('frozen source checksum mismatch: %s/%s' % (row['root_id'], name))
    return _loads(gzip.decompress(raw))


def reference_records():
    """Stage 2 v17_a per-outcome records of the standalone query benchmark (keyed by Root, end node)."""
    data = json.loads(gzip.decompress(REFERENCE.read_bytes()))
    return {(r['root_id'], r['end_node']): r for r in data['records']}


class DevRoot:
    """One frozen development Root: legal history, recorded Query branches and range frames."""

    def __init__(self, row):
        self.row = row
        self.root_id = row['root_id']
        self.scene_id = row['scene_id']
        self.legal = read_member(row, 'legal_root.json.gz')
        self.queries = read_member(row, 'queries.json.gz')
        self.outcomes = read_member(row, 'outcomes.json.gz')
        self.nodes = read_member(row, 'future_observations.json.gz')
        sensor = SENSOR/(self.root_id+'.npz')
        receipt = json.loads(sensor.with_suffix('.json').read_text())
        if receipt['identity']['files'] != row['files']:
            raise ValueError('range packet was produced from a different Root')
        with np.load(sensor) as z:
            self.sensor = {k: z[k] for k in z.files}
        self.history_count = int(self.sensor['history_count'])
        if self.history_count != len(self.legal['history']):
            raise ValueError('range history length differs from the legal history')
        g = self.legal['partial_graph']
        if g['timestamp'] != self.legal['history'][-1]['timestamp']:
            raise ValueError('Root clock mismatch')
        for i, n in enumerate(self.nodes):
            if n['node_id'] != i or (n['parent'] is not None and not 0 <= n['parent'] < i):
                raise ValueError('invalid branch DAG')
            if int(self.sensor['parents'][i]) != (-1 if n['parent'] is None else n['parent']):
                raise ValueError('range branch DAG differs from the observation DAG')

    # ------------------------------------------------------------------ legal decision states
    def ancestors(self, node):
        chain = []
        while node is not None:
            chain.append(node)
            node = self.nodes[node]['parent']
        return chain[::-1]

    def frames(self, node=None):
        return list(self.legal['history'])+[self.nodes[i]['observation'] for i in self.ancestors(node)]

    def packet(self, node=None):
        take = np.r_[np.arange(self.history_count), np.asarray(self.ancestors(node), np.int64)+self.history_count].astype(np.int64)
        s = self.sensor
        return dict(ranges=s['depth'][take], origins=s['origins'][take], rotations=s['rotations'][take],
                    timestamps=s['timestamps'][take], sensor_protocol=RANGE_PROTOCOL)

    def graph(self, node=None):
        """Legal graph after the arrived observations up to ``node`` (Root graph for None)."""
        if node is None:
            return self.legal['partial_graph']
        from eventwm_graph.stage2_quality.update import FastPartialGraph
        g = FastPartialGraph.restore(self.legal['partial_graph'])
        for i in self.ancestors(node):
            if g.update(self.nodes[i]['observation']) != self.nodes[i]['graph_delta']:
                raise ValueError('recorded graph_delta replay mismatch at node %d' % i)
        return g.snapshot()

    def decision_state(self, node=None, stream=None):
        from .state import DecisionState
        return DecisionState(self.frames(node), self.graph(node), self.packet(node), stream=stream or self.root_id,
                             metadata=dict(root_id=self.root_id, scene_id=self.scene_id, node=node))

    # ------------------------------------------------------------------ recorded candidates
    def trajectory(self, node_ids, start=None):
        """N x 5 [t, x, y, z, yaw] from the decision node (Root for None) along recorded nodes."""
        first = self.legal['history'][-1] if start is None else self.nodes[start]['observation']
        rows = [[float(first['timestamp']), *first['robot_pose']]]
        rows += [[float(self.nodes[i]['observation']['timestamp']), *self.nodes[i]['observation']['robot_pose']] for i in node_ids]
        return np.asarray(rows, np.float64)

    def root_candidates(self):
        """Distinct recorded Query executions from the Root: list of (end_node, query_id, trajectory)."""
        out = {}
        for q, o in zip(self.queries, self.outcomes):
            ids = o['node_ids']
            if ids and ids[-1] not in out:
                out[ids[-1]] = (ids[-1], q['query_id'], self.trajectory(ids))
        return [out[k] for k in sorted(out)]

    def continuation_candidates(self, node):
        """Distinct recorded continuations from a branch node (rest of the Queries through it)."""
        out = {}
        for q, o in zip(self.queries, self.outcomes):
            ids = o['node_ids']
            if node in ids and ids[-1] != node and ids[-1] not in out:
                rest = ids[ids.index(node)+1:]
                out[ids[-1]] = (ids[-1], q['query_id'], self.trajectory(rest, start=node))
        return [out[k] for k in sorted(out)]

    def target(self, end_node, start=None):
        """Evaluation labels only: cells first known at ``end_node`` that were unknown at ``start``."""
        initial = self.graph(start)
        known = {tuple(c['cell']) for c in initial['cells']}
        end = self.graph(end_node)
        return {tuple(c['cell']): (int(c['state']), float(c['height'])) for c in end['cells'] if tuple(c['cell']) not in known}

    def children(self):
        ch = defaultdict(list)
        for n in self.nodes:
            ch[n['parent']].append(n['node_id'])
        return ch


def confusion(delta, target):
    """3x3 label-aware confusion (rows truth UNKNOWN/FREE/BLOCKED, cols prediction) on the candidate domain."""
    from eventwm_graph.stage2_repair.full_evaluate import labels
    y, _, missing = labels(delta['candidate_cells'], target)
    cm = np.bincount(3*y.astype(np.int64)+delta['states'], minlength=9).reshape(3, 3)
    return cm, missing


def label_aware_iou(cm):
    """Mean of FREE and BLOCKED IoU (the Stage 2 label-aware increment IoU for one window)."""
    out = []
    for c in (1, 2):
        tp = cm[c, c]; fp = cm[:, c].sum()-tp; fn = cm[c, :].sum()-tp
        out.append(tp/(tp+fp+fn) if tp+fp+fn else np.nan)
    return float(np.nanmean(out)) if not np.all(np.isnan(out)) else np.nan
