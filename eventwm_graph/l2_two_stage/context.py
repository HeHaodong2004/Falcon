"""Immutable, incrementally certified online sensor history and partial graph.

Only arrived observations update this context.  Grid certificates describe robot
configuration space, not human material walls.  Missing grid cells remain
UNKNOWN.  No simulator, query branch, label reader or complete-map API is used.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
import math
from types import MappingProxyType
import time

import numpy as np

from eventwm_graph.partial.map import PartialGraph
from .types import (LegalContext, LegalHistory, SealedObservation,
                    StaticCertificateTable, freeze)
from .motion import MotionSummary
from .view import ViewSummary
from .graphfacts import OrderedCellFacts

_VERSION = 'causal_online_context_two_stage_1'
_CERTIFICATE = object()
_ACTIONS = frozenset(('forward', 'left', 'right', 'wait', 'stop'))
_NODE_KEYS = frozenset(('id', 'cell', 'position', 'kind', 'created_at'))
_NODE_KINDS = frozenset(('junction', 'corridor_turn', 'corridor_transition',
                         'visibility_support', 'historical_decision',
                         'frontier_viewpoint'))
_EDGE_KEYS = frozenset(('source', 'target', 'path', 'length', 'revision',
                         'certificate'))
_EDGE_CERTIFICATES = frozenset(('known_free_cell_supercover',
                               'known_free_Astar_component_backbone'))


def _xyz(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('invalid ' + name)
    return result


def _geometry(graph, cells):
    """The existing legal geometry implementation; cells are observed facts."""
    grid = PartialGraph(graph['resolution'], graph['height'], graph['clearance'])
    grid.cells = cells
    grid.timestamp = graph['timestamp']
    grid.revision = graph['revision']
    grid.ego_support = np.asarray(graph['ego_support'], dtype=float)
    return grid


def _segment_support(grid, a, b):
    """Certify the path height over every intersected observed grid square.

    The shared grid helper checks endpoint height and crossed cell states.
    A crossed FREE certificate at another height does not support this path.
    Slab intersections retain sloped paths and corner-touch supercover cells.
    """
    if not grid.segment_free(a, b):
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = b - a
    covered = grid.segment_cells(a, b)
    for key in covered:
        low, high = 0., 1.
        for axis, coordinate in ((0, key[0]), (2, key[1])):
            left, right = coordinate * grid.resolution, (coordinate + 1) * grid.resolution
            if abs(delta[axis]) < 1e-14:
                if not left - 1e-10 <= a[axis] <= right + 1e-10:
                    return None
            else:
                enter, leave = sorted(((left-a[axis])/delta[axis],
                                      (right-a[axis])/delta[axis]))
                low, high = max(low, enter), min(high, leave)
        if low > high + 1e-10:
            return None
        # Closed-cell contact is deliberate: a corner touching an unsupported
        # height must not be treated as certified configuration space.
        low, high = np.clip((low, high), 0., 1.)
        height = grid.cells[key]['height']
        if max(abs(a[1]+delta[1]*low-height),
               abs(a[1]+delta[1]*high-height)) > .22 + 1e-10:
            return None
    return frozenset(covered)


def _guidance_support(grid, a, b):
    """Certify a source2.5D route guide, including a terrain-following witness.

    ``PartialGraph.route`` string-pulls XZ guidance between measured-height
    waypoints. The installed ``compile_query`` then sets every forward step's
    Y to the destination cell's measured height; it does not execute the long
    endpoint-to-endpoint3D chord. A terrain hump can therefore invalidate that
    chord while the arrived configuration-space connection remains supported.

    The fallback uses only the guide's exact FREE-cell supercover. Every one
    of those cells must belong to one connected component under the existing
    ``PartialGraph.adjacency`` height/corner policy, with each local3D link
    independently checked by ``_segment_support``. No outside detour, UNKNOWN
    bridge, unsupported waypoint height or isolated wrong-height cell is used.
    This is an abstract connectivity certificate, not an executed path length.
    """
    direct = _segment_support(grid, a, b)
    if direct is not None:
        return direct
    if not grid.segment_free(a, b):
        return None
    corridor = frozenset(grid.segment_cells(a, b))
    source, target = grid.key(a), grid.key(b)
    if source not in corridor or target not in corridor:
        return None
    attachments = (_segment_support(grid, a, grid.center(source)),
                   _segment_support(grid, grid.center(target), b))
    if any(link is None or not link <= corridor for link in attachments):
        return None
    # Reuse the installed graph policy on this bounded corridor, rather than
    # rebuilding the whole scene adjacency or inventing a new height tolerance.
    local = PartialGraph(grid.resolution, grid.height, grid.clearance)
    local.cells = {key: grid.cells[key] for key in corridor}
    local.timestamp, local.revision = grid.timestamp, grid.revision
    adjacency = local.adjacency
    reached, pending = {source}, [source]
    while pending:
        key = pending.pop()
        for neighbor, _ in adjacency.get(key, ()):
            if neighbor in reached:
                continue
            support = _segment_support(local, local.center(key),
                                       local.center(neighbor))
            if support is None or not support <= corridor:
                continue
            reached.add(neighbor)
            pending.append(neighbor)
    if target not in reached or reached != corridor:
        return None
    return corridor


def _certify_graph_metadata(graph, history):
    """Check abstraction paths against arrived known support, never an oracle."""
    for key in ('resolution', 'height', 'clearance', 'timestamp'):
        if not math.isfinite(float(graph[key])):
            raise ValueError('nonfinite graph ' + key)
    if graph['resolution'] <= 0 or graph['clearance'] <= 0:
        raise ValueError('nonpositive graph scale')
    if (isinstance(graph['revision'], bool) or
            not isinstance(graph['revision'], int) or graph['revision'] < 0):
        raise ValueError('invalid graph revision')
    now = history.frames[-1]['timestamp']
    if not np.allclose(_xyz(graph['ego_support'], 'ego support'),
                       history.frames[-1]['robot_pose'][:3], rtol=0, atol=1e-7):
        raise ValueError('ego support differs from actual robot pose')
    frame_by_time = {ob['timestamp']: ob for ob in history.frames}
    for visited in graph['visited']:
        if set(visited) != {'position', 'timestamp'}:
            raise ValueError('unknown visited-pose field')
        t = float(visited['timestamp'])
        if not math.isfinite(t) or t not in frame_by_time:
            raise ValueError('visited pose has no arrived timestamp')
        if not np.allclose(_xyz(visited['position'], 'visited pose'),
                           frame_by_time[t]['robot_pose'][:3], rtol=0, atol=1e-7):
            raise ValueError('visited pose has no arrived pose provenance')
    nodes = graph['global_nodes']
    for key, node in nodes.items():
        if set(node) != _NODE_KEYS or node['id'] != key:
            raise ValueError('unknown or inconsistent topology node schema')
        if node['kind'] not in _NODE_KINDS:
            raise ValueError('unknown topology node kind')
        t = float(node['created_at'])
        if not math.isfinite(t) or t > now + 1e-7 or t not in frame_by_time:
            raise ValueError('topology node has no arrived creation timestamp')
        _xyz(node['position'], 'topology node position')
    grid = _geometry(graph, history.cells)
    seen = set()
    supports = []
    for edge in graph['global_edges']:
        if set(edge) != _EDGE_KEYS:
            raise ValueError('unknown topology edge schema')
        source, target = edge['source'], edge['target']
        if source not in nodes or target not in nodes or source == target:
            raise ValueError('invalid topology edge endpoints')
        pair = tuple(sorted((source, target)))
        if pair in seen:
            raise ValueError('duplicate topology edge')
        seen.add(pair)
        revision = edge['revision']
        if (isinstance(revision, bool) or not isinstance(revision, int) or
                not 0 <= revision <= history.revision):
            raise ValueError('noncausal topology edge revision')
        if edge['certificate'] not in _EDGE_CERTIFICATES:
            raise ValueError('unknown topology edge certificate')
        path = np.asarray(edge['path'], dtype=float)
        if path.ndim != 2 or path.shape[1:] != (3,) or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('invalid topology edge path')
        if (not np.allclose(path[0], nodes[source]['position'], rtol=0, atol=1e-7) or
                not np.allclose(path[-1], nodes[target]['position'], rtol=0, atol=1e-7)):
            raise ValueError('edge path does not join its declared nodes')
        length = float(edge['length'])
        measured = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())
        if not math.isfinite(length) or length < 0 or abs(length - measured) > 1e-7 * max(1., measured):
            raise ValueError('incorrect topology path length')
        covered = set()
        for a, b in zip(path, path[1:]):
            segment = _guidance_support(grid, a, b)
            if segment is None:
                raise ValueError('topology path crosses UNKNOWN or obstructed support')
            covered.update(segment)
        supports.append(frozenset(covered))
    return tuple(supports)


@dataclass(frozen=True, init=False)
class OnlineContext:
    """Validated immutable actual facts; no predicted graph updates are admitted.

    Construct with ``from_inputs``. ``ingest`` validates and seals the new sensor
    frame, appends its executed action, and derives graph cells/revision in the
    timed online operation. Initial full-prefix certification is a separate cost.
    Dynamic model cache resets do not erase these legally accumulated facts.
    """
    _legal_history: LegalHistory
    actions: tuple
    partial_graph: Mapping
    _cell_order: tuple
    _edge_supports: tuple
    _motion_summary: MotionSummary
    _view_summary: ViewSummary
    _cell_facts: OrderedCellFacts
    processing_ms: Mapping
    _positive_history: object = None
    _association_graph: object = None
    version: str = _VERSION

    @property
    def history(self):
        self.validate_certificate()
        return self._legal_history.frames

    @property
    def legal_history(self):
        self.validate_certificate()
        return self._legal_history

    @property
    def certified_cells(self):
        self.validate_certificate()
        return self._legal_history.cells

    @property
    def motion_summary(self):
        self.validate_certificate()
        self._motion_summary.validate_certificate()
        return self._motion_summary

    @property
    def view_summary(self):
        self.validate_certificate()
        self._view_summary.validate_certificate()
        return self._view_summary

    @property
    def cell_facts(self):
        self.validate_certificate()
        self._cell_facts.validate_certificate()
        return self._cell_facts

    def validate_certificate(self):
        if getattr(self, '_certificate', None) is not _CERTIFICATE:
            raise ValueError('uncertified online context')
        return self

    @classmethod
    def _verified(cls, history, actions, graph, order, supports, profile, motion,
                  positive_history=None, association_graph=None,view_summary=None,cell_facts=None):
        result = object.__new__(cls)
        if view_summary is None:view_summary=ViewSummary.from_frames(history.frames)
        if cell_facts is None:cell_facts=OrderedCellFacts.from_cells(order,history.cells)
        for key, value in dict(_legal_history=history, actions=tuple(actions),
                partial_graph=graph, _cell_order=order, _edge_supports=supports,
                _motion_summary=motion,
                _view_summary=view_summary,
                _cell_facts=cell_facts,
                _positive_history=positive_history, _association_graph=association_graph,
                processing_ms=MappingProxyType(dict(profile)), version=_VERSION,
                _certificate=_CERTIFICATE).items():
            object.__setattr__(result, key, value)
        return result

    def with_association(self):
        """Certify the full actual positive history once, during initialization.

        Subsequent ingest calls update this immutable state from the caller's
        PRE-arrival context. A model's mutable cache cannot hide online work.
        """
        self.validate_certificate()
        if self._association_graph is not None:
            return self
        from .conditioned import history_from_context
        from .association import AssociationGraph
        begin=time.perf_counter();positive=history_from_context(self)
        graph=AssociationGraph.from_history(positive)
        profile=dict(self.processing_ms,association_initialization_ms=(time.perf_counter()-begin)*1000)
        return type(self)._verified(self.legal_history,self.actions,self.partial_graph,
            self._cell_order,self._edge_supports,profile,self.motion_summary,positive,graph,self._view_summary,self._cell_facts)

    def _append_association(self,result):
        if self._association_graph is None:
            return result
        from .positive_history import PositiveSetHistory,positive_xyz
        import numpy as np
        begin=time.perf_counter();old=self._positive_history;frame=result.history[-1]
        xyz=positive_xyz(frame)
        positive=PositiveSetHistory(old.root_id,old.scene_id,old.canonical_query_id,len(result.history)-1,
            np.r_[old.timestamps,float(frame['timestamp'])],
            np.r_[old.positive_offsets,old.positive_offsets[-1]+len(xyz)],
            np.concatenate((old.positive_xyz,xyz)),
            np.r_[old.observation_valid,frame.get('observation_valid',True)],
            np.r_[old.human_sensor_available,frame.get('human_sensor_available',True)])
        graph=self._association_graph.append(positive)
        elapsed=(time.perf_counter()-begin)*1000
        profile=dict(result.processing_ms,association_update_ms=elapsed)
        profile['complete_update_ms']=profile.get('complete_update_ms',0.)+elapsed
        return type(self)._verified(result.legal_history,result.actions,result.partial_graph,
            result._cell_order,result._edge_supports,profile,result.motion_summary,positive,graph,result._view_summary,result._cell_facts)

    @classmethod
    def from_inputs(cls, history, obs, partial_graph, actions=()):
        """Fully certify an initial prefix or an explicitly replaced raw graph."""
        begin = time.perf_counter()
        if isinstance(history, cls):
            history.validate_certificate()
            history = history.legal_history
        certified = history if isinstance(history, LegalHistory) else LegalHistory.from_frames(history)
        # LegalContext checks the LegalHistory token, all provenance and clocks.
        checked = LegalContext.from_inputs(certified, obs, partial_graph, actions)
        if not certified.frames or checked.history[-1]['timestamp'] > certified.frames[-1]['timestamp']:
            certified = certified.append(obs)
        if len(actions) != len(checked.history) - 1:
            raise ValueError('every history transition needs an executed action')
        supports = _certify_graph_metadata(partial_graph, certified)
        order = tuple(tuple(cell['cell']) for cell in partial_graph['cells'])
        graph = {key: freeze(value) for key, value in partial_graph.items() if key != 'cells'}
        graph['cells'] = tuple(certified.cells[key] for key in order)
        motion = MotionSummary.from_frames(certified.frames)
        views = ViewSummary.from_frames(certified.frames)
        facts = OrderedCellFacts.from_cells(order,certified.cells)
        elapsed = (time.perf_counter() - begin) * 1000
        return cls._verified(certified, actions, MappingProxyType(graph), order, supports,
                             dict(initialization_ms=elapsed), motion,view_summary=views,cell_facts=facts)

    @classmethod
    def from_raw(cls, history, partial_graph, actions=()):
        """Convenience factory when the final actual frame is already included."""
        frames = history.frames if isinstance(history, LegalHistory) else tuple(history)
        if not frames:
            raise ValueError('initial context requires an actual observation')
        return cls.from_inputs(history if isinstance(history, LegalHistory) else frames,
                               frames[-1], partial_graph, actions)

    def ingest(self, obs, executed_action, partial_graph=None):
        """Return a new context from exactly one newly arrived actual frame.

        No graph delta or future graph is accepted. Old global topology is
        preserved while certified; newly obstructed old free support clears
        all links, matching conservative offline-anchor reconstruction. A raw
        replacement graph, if supplied, undergoes full certification.
        """
        self.validate_certificate()
        begin = time.perf_counter()
        if executed_action not in _ACTIONS:
            raise ValueError('unknown executed action')
        sealed = SealedObservation.from_raw(obs)
        sealed_done = time.perf_counter()
        history = self._legal_history.append(sealed)
        appended = time.perf_counter()
        actions = self.actions + (executed_action,)
        if partial_graph is not None:
            for key in ('resolution', 'height', 'clearance'):
                if partial_graph[key] != self.partial_graph[key]:
                    raise ValueError('replacement changes the episode grid contract')
            result = type(self).from_inputs(history, sealed, partial_graph, actions)
            profile = dict(result.processing_ms, seal_ms=(sealed_done-begin)*1000,
                           append_ms=(appended-sealed_done)*1000,
                           complete_update_ms=(time.perf_counter()-begin)*1000)
            return self._append_association(type(self)._verified(history, actions, result.partial_graph,
                                       result._cell_order, result._edge_supports, profile,
                                       result.motion_summary,view_summary=result._view_summary,cell_facts=result._cell_facts))
        motion = self.motion_summary.append(sealed)
        motion_done=time.perf_counter()
        views=self.view_summary.append(sealed)
        summarized = time.perf_counter()
        old_cells, new_cells = self._legal_history.cells, history.cells
        updates = sealed['static_updates']
        if isinstance(updates, StaticCertificateTable):
            # Avoid constructing another ephemeral mapping for every row. Ego
            # support-only keys may be included: append leaves those cells
            # unchanged, so they cannot invalidate a node/link or its height.
            touched = {tuple(key) for key in updates.cells.tolist()}
        else:
            touched = {tuple(update['cell']) for update in updates
                       if update['source'] != 'occupied_ego_base_support'}
        invalidated, changed_heights = set(), set()
        for key in touched:
            old, new = old_cells.get(key), new_cells.get(key)
            if old is not None and old['state'] == 1 and new is not None:
                if new['state'] != 1:
                    invalidated.add(key)
                elif abs(old['height'] - new['height']) > 1e-8:
                    changed_heights.add(key)
        # append copies an insertion-ordered certified dictionary and never
        # removes keys. Its final added entries are exactly the old full scan's
        # new-key subsequence, even when raw FREE/OCCUPIED rows collide.
        added = len(new_cells) - len(old_cells)
        order = self._cell_order + tuple(islice(reversed(new_cells), added))[::-1]
        graph = dict(self.partial_graph)
        graph.update(timestamp=sealed['timestamp'], revision=history.revision,
                     ego_support=sealed['robot_pose'][:3])
        nodes = self.partial_graph['global_nodes']
        retained = {key: node for key, node in nodes.items()
                    if new_cells[tuple(node['cell'])]['state'] == 1 and
                    abs(node['position'][1] - new_cells[tuple(node['cell'])]['height']) <= 1e-8}
        graph['global_nodes'] = MappingProxyType(retained)
        edges, supports = [], []
        if not invalidated:
            grid = _geometry(graph, new_cells) if changed_heights else None
            for edge, support in zip(self.partial_graph['global_edges'], self._edge_supports):
                if edge['source'] not in retained or edge['target'] not in retained:
                    continue
                if changed_heights.intersection(support):
                    if not all(_guidance_support(grid, a, b) is not None
                               for a, b in zip(edge['path'], edge['path'][1:])):
                        continue
                edges.append(edge)
                supports.append(support)
        graph['global_edges'] = tuple(edges)
        facts_begin=time.perf_counter()
        facts, graph['cells'] = self.cell_facts.update_with_rows(
            order, new_cells, touched, self.partial_graph['cells'])
        end = time.perf_counter()
        return self._append_association(type(self)._verified(history, actions, MappingProxyType(graph), order,
                tuple(supports), dict(seal_ms=(sealed_done-begin)*1000,
                    append_ms=(appended-sealed_done)*1000,
                    motion_ms=(motion_done-appended)*1000,view_update_ms=(summarized-motion_done)*1000,
                    cell_facts_update_ms=(end-facts_begin)*1000,
                    graph_ms=(end-summarized)*1000, complete_update_ms=(end-begin)*1000),
                motion,view_summary=views,cell_facts=facts))
