"""Versioned in-process contract and JSON-safe, finite tensor export."""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.partial.query import NOMINAL_STEP_SECONDS

CONTRACT = 'eventwm.l25l4.contract.v1'
VERSION = 'l25-query-outcomes-1'


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def readonly(value, dtype=None):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def immutable(value):
    if isinstance(value, np.ndarray):
        return readonly(value)
    if isinstance(value, Mapping):
        return MappingProxyType({k: immutable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(v) for v in value)
    return value


@dataclass(frozen=True)
class ComputeBudget:
    horizon: float = 3.
    worlds: int = 16
    seed: int = 731
    max_candidates: int = 24
    branch_queries: int = 1
    max_branches: int = 2
    min_branch_probability: float = .001
    soft_budget_ms: float = 100.

    def __post_init__(self):
        if not np.isfinite(self.horizon) or not .048 <= self.horizon <= 30:
            raise ValueError('forecast horizon must be .048..30 seconds')
        for k, lo, hi in [('worlds', 1, 256), ('max_candidates', 1, 24), ('branch_queries', 0, 4), ('max_branches', 0, 8)]:
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError('invalid '+k)
        if not np.isfinite(self.min_branch_probability) or not 0 <= self.min_branch_probability < 1:
            raise ValueError('invalid pruning threshold')
        if not np.isfinite(self.soft_budget_ms) or self.soft_budget_ms < 0:
            raise ValueError('invalid soft budget')


@dataclass(frozen=True)
class CompiledReference:
    """Compatibility view of Stage 0/1's existing compiler, never a compiler.

    The caller owns kinematic validity and reference generation. Content digest
    prevents cached results surviving a same-ID route/timing mutation.
    """
    query_id: str
    reference_hash: str
    content_hash: str
    start_timestamp: float
    poses: np.ndarray
    actions: tuple
    target: Any
    terminal_orientation: Any
    reference_route: Any
    completion_rule: Any
    execution_limit: Any
    compiler_version: str
    graph_revision: int
    _native_json: str = field(repr=False)

    @classmethod
    def from_native(cls, query, start_timestamp):
        if query.get('executor_version') != EXECUTOR_VERSION:
            raise ValueError('adapter requires the existing native compiler version')
        qid = query['query_id']
        if not isinstance(qid, str) or not qid:
            raise ValueError('nonempty stable string query_id required')
        c = query['compiled']
        if c.get('ok') is False:
            raise ValueError('native compiler did not certify this reference')
        p = np.asarray(c['planned_poses'], dtype=float)
        actions = tuple(c['actions'])
        if p.shape != (len(actions)+1, 4) or not 1 <= len(actions) <= 180 or not np.isfinite(p).all():
            raise ValueError('invalid native reference shape/length')
        if any(a not in ('forward', 'left', 'right') for a in actions):
            raise ValueError('unsupported native action')
        if not np.isfinite(start_timestamp):
            raise ValueError('invalid reference start time')
        if 'nominal_seconds' in c and not np.isclose(c['nominal_seconds'], len(actions)*NOMINAL_STEP_SECONDS, atol=1e-8, rtol=0):
            raise ValueError('foreign nominal clock; timestamp adapter required')
        points = query.get('waypoints')
        target = query.get('target', points[-1] if points else None)
        if target is not None and (np.asarray(target).shape != (3,) or not np.isfinite(np.asarray(target, float)).all()):
            raise ValueError('target must be supplied finite XYZ, or absent')
        orientation = query.get('terminal_orientation', query.get('final_yaw'))
        route = query.get('reference_route', c.get('route'))
        rule = query.get('completion_rule')
        if rule is None and 'position_tolerance' in c and 'yaw_tolerance' in c:
            rule = dict(source='native compiler tolerances; not an L4 completion model',
                        position_tolerance=c['position_tolerance'], yaw_tolerance=c['yaw_tolerance'])
        content = dict(poses=p, actions=actions, target=target, terminal_orientation=orientation,
                       reference_route=route, completion_rule=rule,
                       execution_limit=query.get('execution_limit'), executor=EXECUTOR_VERSION,
                       graph_revision=c['graph_revision'])
        h = digest(content)
        # Store input by value so the adapter cannot hold a mutable foreign dict.
        raw = json.dumps(jsonable(query), sort_keys=True, allow_nan=False)
        return cls(qid, str(query.get('reference_hash', h)), h, float(start_timestamp), readonly(p), actions,
                   immutable(target), orientation, immutable(route),
                   immutable(rule), immutable(query.get('execution_limit')),
                   str(query.get('compiler_version', EXECUTOR_VERSION)), int(c['graph_revision']), raw)

    @property
    def duration(self):
        return len(self.actions)*NOMINAL_STEP_SECONDS

    @property
    def times(self):
        return self.start_timestamp+np.arange(len(self.poses))*NOMINAL_STEP_SECONDS

    def native(self):
        return json.loads(self._native_json)

    def verify(self):
        content = dict(poses=self.poses, actions=self.actions, target=self.target,
                       terminal_orientation=self.terminal_orientation, reference_route=self.reference_route,
                       completion_rule=self.completion_rule, execution_limit=self.execution_limit,
                       executor=EXECUTOR_VERSION, graph_revision=self.graph_revision)
        if digest(content) != self.content_hash:
            raise ValueError('compiled reference content changed after adaptation')


@dataclass(frozen=True)
class ContextGraph:
    belief: Any = field(repr=False)
    belief_version: str
    graph_version: str
    model_version: str
    timestamp: float
    ego: np.ndarray
    goal: np.ndarray
    spatial: Any
    features: np.ndarray
    feature_names: tuple
    tracks: tuple
    metadata: dict


@dataclass(frozen=True)
class SparseConditionalBranch:
    query_id: str
    event: str
    probability: float
    timestamp: float
    ego: np.ndarray
    successor: Any = field(repr=False)
    summary: dict
    root_executable: bool = False


@dataclass(frozen=True)
class DecisionPacket:
    context: ContextGraph
    outcomes: tuple
    branches: tuple
    graph: dict
    metadata: dict

    def to_dict(self):
        """Portable summaries; opaque finite carriers remain in process only."""
        return jsonable(dict(schema=VERSION, contract=CONTRACT,
            context=dict(belief_version=self.context.belief_version, graph_version=self.context.graph_version,
                         model_version=self.context.model_version, timestamp=self.context.timestamp,
                         ego=self.context.ego, goal=self.context.goal, metadata=self.context.metadata),
            outcomes=self.outcomes,
            branches=[dict(query_id=b.query_id, event=b.event, probability=b.probability, timestamp=b.timestamp,
                           ego=b.ego, summary=b.summary, root_executable=False) for b in self.branches],
            graph=self.graph, metadata=self.metadata))

    def save(self, path):
        with open(path, 'x') as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False)
