"""Incremental recertification of already published observed graph paths.

Every cell update and delta uses the accepted PartialGraph.update bytecode.
An existing certified path only needs recertification if its exact grid support
or former singleton ego support changed. No private geometry is available here.
"""
import numpy as np

from .map import PartialGraph


class IncrementalPartialGraph(PartialGraph):
    @classmethod
    def restore(cls, snapshot):
        graph = super().restore(snapshot)
        # A legacy snapshot contains no deferred path-check debt. Tiny height
        # changes can leave paths temporarily unchecked until the next revision.
        graph._paths_unchecked = True
        return graph

    def update(self, observation):
        edges = self.global_edges
        if not edges:
            self._deferred_geometry = set()
            return super().update(observation)
        keys = {tuple(u['cell']) for u in observation['static_updates']
                if u['source'] != 'occupied_ego_base_support'}
        previous = {k: (self.cells[k]['state'], self.cells[k]['height']) if k in self.cells else None for k in keys}
        old_ego = self.ego_support
        # Let the reference apply all observations, clocks, precedence, topology
        # revision and node removal. Replace only its exhaustive path scan.
        self.global_edges = []
        try:
            delta = super().update(observation)
        except BaseException:
            self.global_edges = edges
            raise
        affected = {k for k, old in previous.items()
                    if old != ((self.cells[k]['state'], self.cells[k]['height']) if k in self.cells else None)}
        # Even sub-1e-8 height updates must enter affected when a different cell
        # triggers the reference's path scan. Do not key this by revision alone.
        if old_ego is not None and not np.array_equal(old_ego, self.ego_support):
            affected.add(self.key(old_ego))
        affected.update(getattr(self, '_deferred_geometry', set()))
        if not delta['changed_cells']:
            self._deferred_geometry = affected
            self.global_edges = edges
            return delta
        self._deferred_geometry = set()
        unchecked = getattr(self, '_paths_unchecked', False)
        self._paths_unchecked = False
        if not hasattr(self, '_edge_support'):
            self._edge_support = {}
        retained = []
        for edge in edges:
            if edge['source'] not in self.global_nodes or edge['target'] not in self.global_nodes:
                continue
            path = edge['path']; key = tuple(tuple(p) for p in path)
            cover = self._edge_support.get(key)
            if cover is None:
                cells = {self.key(p) for p in path}
                for a, b in zip(path, path[1:]):
                    cells.update(self.segment_cells(a, b))
                cover = self._edge_support[key] = frozenset(cells)
            if (not unchecked and not affected.intersection(cover)) or all(self.segment_free(a, b) for a, b in zip(path, path[1:])):
                retained.append(edge)
        self.global_edges = retained
        return delta
