"""Legal decision context: the arrived observation stream of ONE episode in the Layer 2 formats.

``LegalStream`` accumulates, step by step, exactly what the robot has legally observed:

* the legal observation rows of the querygen ``LegalSensor`` (the project's ideal geometry sensor; identical rows
  with the exact fast sensor), converted to the dataset frame format Layer 2 was validated on
  (``static_updates`` materialised from ``fastsense.UpdateBlock`` arrays to the reference dicts);
* the causal partial graph ``causal_partial_cspace_graph_1``, updated with ``PartialGraph.update`` from those rows'
  certificates (the same function that built every Stage 1 / Stage 2 development graph);
* one measured range frame per row (``ideal_current_jaw_range_dynamic_occlusion_1``: the environment-side sensor
  extension Stage 2 consumes), checked against the row's clock and camera pose.

``decision_state()`` packages them as a ``layer2.state.DecisionState``.  Nothing here reads simulator state, the
navmesh, scene geometry, identities or future observations; predicted (hypothetical) content is never accepted
(``DecisionState`` rejects it).  Frames and range frames are append-only: arrays handed to an earlier decision state
are never written again.
"""
import math

import numpy as np

from eventwm_graph.layer2.state import DecisionState, RANGE_PROTOCOL, CLOCK_TOLERANCE, POSE_TOLERANCE

RESOLUTION = 0.1016884229464652      # dataset static-certificate grid (robot collision reference / 6)
RANGE_SIZE = 256
FRAME_KEYS = ('timestamp', 'robot_pose', 'camera_origin', 'camera_rotation', 'tracks', 'static_updates',
              'observation_valid', 'sensor_protocol')


class ContextError(ValueError):
    """The observation stream violates the legal-context contract (order, alignment, format)."""


class CompactUpdates:
    """One frame's static updates kept as the sensor's arrays (``fastsense.UpdateBlock`` content).

    A read-only sequence of the reference update dicts (dataset format: cell, state, height, timestamp, source;
    identical content and order): iteration, ``len``, indexing and ``==`` with a list behave as the list does, but the
    dicts are built afresh on each access and never cached.  A long history then holds four arrays per frame instead of
    ~1,600 Python objects that every cyclic garbage collection would traverse (150-215 ms pauses at 345k-754k objects,
    measured).  ``tolist()`` gives the plain list.
    """
    __slots__ = ('cells', 'states', 'heights', 'sources', 'timestamp')

    def __init__(self, block):
        self.cells, self.states, self.heights = block.cells, block.states, block.heights
        self.sources, self.timestamp = block.sources, float(block.timestamp)

    def __iter__(self):
        from eventwm_graph.querygen.fastsense import SOURCES
        t = self.timestamp
        for c, st, h, k in zip(self.cells.tolist(), self.states.tolist(), self.heights.tolist(), self.sources.tolist()):
            yield dict(cell=[int(c[0]), int(c[1])], state=int(st), height=float(h), timestamp=t, source=SOURCES[int(k)])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        from eventwm_graph.querygen.fastsense import SOURCES
        if isinstance(i, slice):
            return self.tolist()[i]
        c = self.cells[i]
        return dict(cell=[int(c[0]), int(c[1])], state=int(self.states[i]), height=float(self.heights[i]),
                    timestamp=self.timestamp, source=SOURCES[int(self.sources[i])])

    def tolist(self):
        return list(iter(self))

    def __eq__(self, other):
        return self.tolist() == (other.tolist() if isinstance(other, CompactUpdates) else list(other))

    def __repr__(self):
        return 'CompactUpdates(%d updates at t=%.3f)' % (len(self), self.timestamp)


def layer2_frame(row, compact=True):
    """One legal observation row -> Layer 2 frame (dataset format).

    The row is not mutated.  ``fastsense.UpdateBlock`` static updates become a ``CompactUpdates`` sequence of their
    reference dicts (``compact``) or the plain list; poses become float lists as in the frozen development frames.
    """
    missing = [k for k in FRAME_KEYS if k not in row]
    if missing:
        raise ContextError('observation row lacks legal fields %s' % missing)
    if row.get('predicted') or row.get('hypothetical'):
        raise ContextError('predicted rows are not observations')
    su = row['static_updates']
    if hasattr(su, 'sources'):
        updates = CompactUpdates(su) if compact else list(su.dicts())
    else:
        updates = list(su)
    return dict(timestamp=float(row['timestamp']), robot_pose=[float(v) for v in row['robot_pose']],
                camera_origin=[float(v) for v in row['camera_origin']],
                camera_rotation=[[float(v) for v in r] for r in row['camera_rotation']],
                tracks=row['tracks'], static_updates=updates, observation_valid=bool(row['observation_valid']),
                sensor_protocol=row['sensor_protocol'])


class LegalStream:
    """Append-only legal history of one robot/episode (Layer 2 view).

    ``observe(row, range_frame)`` after every environment step (and once at the reset);
    ``decision_state()`` at a decision.  ``range_frame`` is a mapping with ``range_image`` (or ``ranges``)
    [S, S] float32 Euclidean metres (NaN = no return within 10 m), ``timestamp``, ``camera_origin``,
    ``camera_rotation`` and ``sensor_protocol`` -- e.g. ``CurrentRangeSensor.measure()``.
    """

    def __init__(self, stream='default', resolution=RESOLUTION, range_size=RANGE_SIZE, capacity=256,
                 global_graph=False, compact=True):
        from eventwm_graph.stage2_quality.update import FastPartialGraph
        self._graph_cls = FastPartialGraph
        self.stream = str(stream)
        self.resolution = float(resolution)
        self.size = int(range_size)
        self.global_graph = bool(global_graph)
        self.compact = bool(compact)
        self.frames = []
        self.graph = None
        self._n = 0
        self._alloc(int(capacity))
        self._snapshot = None           # (timestamp, revision, n) -> cached legal graph mapping
        self.updates = 0

    # ------------------------------------------------------------------ storage
    def _alloc(self, capacity):
        old = getattr(self, '_ranges', None)
        ranges = np.empty((capacity, self.size, self.size), np.float32)
        origins = np.empty((capacity, 3), np.float64)
        rotations = np.empty((capacity, 3, 3), np.float64)
        times = np.empty(capacity, np.float64)
        if old is not None and self._n:
            n = self._n
            ranges[:n] = self._ranges[:n]; origins[:n] = self._origins[:n]
            rotations[:n] = self._rotations[:n]; times[:n] = self._times[:n]
        # earlier decision states keep views of the previous buffers: those are never written again
        self._ranges, self._origins, self._rotations, self._times = ranges, origins, rotations, times

    def __len__(self):
        return self._n

    @property
    def timestamp(self):
        return float(self.frames[-1]['timestamp']) if self.frames else None

    # ------------------------------------------------------------------ ingest
    def observe(self, row, range_frame):
        frame = layer2_frame(row, self.compact)
        t = frame['timestamp']
        if not math.isfinite(t) or (self.frames and t <= self.frames[-1]['timestamp']):
            raise ContextError('observations must arrive in strictly increasing time')
        if range_frame is None:
            raise ContextError('every observation needs its measured range frame (Layer 2 range packet contract)')
        if range_frame.get('sensor_protocol') != RANGE_PROTOCOL:
            raise ContextError('range frame protocol must be %r' % RANGE_PROTOCOL)
        img = range_frame['range_image'] if 'range_image' in range_frame else range_frame['ranges']
        img = np.asarray(img, np.float32)
        if img.shape != (self.size, self.size):
            raise ContextError('range frame must be %d x %d' % (self.size, self.size))
        if abs(float(range_frame['timestamp'])-t) > CLOCK_TOLERANCE:
            raise ContextError('range frame clock %.9f differs from the observation %.9f' % (float(range_frame['timestamp']), t))
        o = np.asarray(range_frame['camera_origin'], np.float64)
        r = np.asarray(range_frame['camera_rotation'], np.float64)
        if np.max(np.abs(o-np.asarray(frame['camera_origin']))) > POSE_TOLERANCE or \
                np.max(np.abs(r-np.asarray(frame['camera_rotation']))) > POSE_TOLERANCE:
            raise ContextError('range frame was not taken from the observation camera pose')
        finite = img[np.isfinite(img)]
        if len(finite) and (finite.min() < 0. or finite.max() > 10.00001):
            raise ContextError('range values must be Euclidean metres within 10 m (NaN = no return)')
        if self.graph is None:
            self.graph = self._graph_cls(self.resolution, float(frame['robot_pose'][1]))
        self.graph.update(frame)
        self.updates += 1
        if self._n == len(self._times):
            self._alloc(2*len(self._times))
        n = self._n
        self._ranges[n] = img; self._origins[n] = o; self._rotations[n] = r; self._times[n] = t
        self._n = n+1
        self.frames.append(frame)
        return frame

    # ------------------------------------------------------------------ views
    def range_packet(self):
        n = self._n
        return dict(ranges=self._ranges[:n], origins=self._origins[:n], rotations=self._rotations[:n],
                    timestamps=self._times[:n], sensor_protocol=RANGE_PROTOCOL)

    def graph_snapshot(self):
        """The legal ``causal_partial_cspace_graph_1`` at the latest observation (cached until the next one).

        Equivalent to ``PartialGraph.snapshot()`` (cell rows hold only scalars and the source string, so a shallow
        row copy is a deep copy); with ``global_graph`` the frontier skeleton is rebuilt at the current pose first
        (``PartialGraph.rebuild_global``, as the development collector did at its Roots).
        """
        if self.graph is None:
            raise ContextError('no observation yet')
        g = self.graph
        key = (g.timestamp, g.revision, self._n)
        if self._snapshot is not None and self._snapshot[0] == key:
            return self._snapshot[1]
        if self.global_graph:
            g.rebuild_global(np.asarray(self.frames[-1]['robot_pose'][:3], np.float64))
        import copy
        snap = {'schema': 'causal_partial_cspace_graph_1', 'resolution': g.resolution, 'height': g.height,
                'clearance': g.clearance, 'timestamp': g.timestamp, 'revision': g.revision,
                'ego_support': g.ego_support.tolist() if g.ego_support is not None else None,
                'cells': [{'cell': list(k), **v} for k, v in sorted(g.cells.items())],
                'visited': copy.deepcopy(g.visited),
                'global_nodes': copy.deepcopy(g.global_nodes), 'global_edges': copy.deepcopy(g.global_edges)}
        self._snapshot = (key, snap)
        return snap

    def decision_state(self, metadata=None):
        """``DecisionState`` at the latest observation (validated by Layer 2)."""
        if not self.frames:
            raise ContextError('no observation yet')
        return DecisionState(tuple(self.frames), self.graph_snapshot(), self.range_packet(), stream=self.stream,
                             metadata=dict(metadata or {}))
