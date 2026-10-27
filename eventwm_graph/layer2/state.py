"""Shared legal decision state and candidate trajectories for the unified Layer 2.

A ``DecisionState`` is everything Layer 2 is allowed to see at one decision time t:

* ``frames``: the arrived legal observation dicts (oldest first, ``frames[-1]`` is the
  current frame at t). Stage 1 consumes their robot/camera poses, tracker positives and
  static certificates; Stage 2 consumes their ``tracks`` as the aligned legal track history.
* ``partial_graph``: the legal ``causal_partial_cspace_graph_1`` at t (built only from those
  frames' certificates). Hypothetical (predicted) graphs are rejected.
* ``range_packet``: the measured past/current range frames
  (``ideal_current_jaw_range_dynamic_occlusion_1``), one per arrived frame, same clock and
  same camera poses (checked).

One state is shared by every candidate of a decision: the human bank is predicted once and
every candidate's map increment is conditioned on the same graph, ranges and tracks.

Candidate trajectories use one canonical format, ``N x 5 [t, x, y, z, yaw]`` with absolute
times (same clock as the frames), world metres (y up; the floor plane is x/z, which is also the
grid of the partial graph), the robot-pose yaw convention of the frames, and ``row 0`` equal to
the current robot pose at t.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
from typing import Any, Optional, Sequence, Tuple

import numpy as np

RANGE_PROTOCOL = 'ideal_current_jaw_range_dynamic_occlusion_1'
GRAPH_SCHEMA = 'causal_partial_cspace_graph_1'
PREDICTED_SOURCE_PREFIX = 'predicted'
CLOCK_TOLERANCE = 1e-7       # s, same tolerance as both stages
POSE_TOLERANCE = 1e-5        # m / rad; packet poses are float64 copies of the frame poses


class LegalityError(ValueError):
    """An input would give Layer 2 privileged, future or hypothetical information."""


def is_hypothetical_graph(graph):
    """True for any graph carrying predicted content (Stage 2 future graphs or predicted cells)."""
    if not isinstance(graph, Mapping):
        return False
    if graph.get('hypothetical') or 'prediction_checkpoint_sha256' in graph:
        return True
    columns = getattr(graph, 'cell_columns', None)
    if columns is not None:
        names = columns['source_names']
        used = set(np.unique(np.asarray(columns['source_id'])).tolist()) if len(columns['source_id']) else set()
        return any(str(names[i]).startswith(PREDICTED_SOURCE_PREFIX) for i in used)
    return any(str(c.get('source', '')).startswith(PREDICTED_SOURCE_PREFIX) for c in graph.get('cells', ()))


def _packet_arrays(packet):
    if not isinstance(packet, Mapping):
        raise TypeError('range_packet must be a mapping')
    if packet.get('sensor_protocol') != RANGE_PROTOCOL:
        raise LegalityError('range packet protocol must be %r' % RANGE_PROTOCOL)
    ranges = np.asarray(packet['ranges'], np.float32)
    origins = np.asarray(packet['origins'], np.float64)
    rotations = np.asarray(packet['rotations'], np.float64)
    timestamps = np.asarray(packet['timestamps'], np.float64)
    return dict(ranges=ranges, origins=origins, rotations=rotations, timestamps=timestamps, sensor_protocol=RANGE_PROTOCOL)


@dataclass(frozen=True)
class DecisionState:
    frames: Tuple[Mapping, ...]
    partial_graph: Mapping
    range_packet: Mapping
    stream: str = 'default'
    metadata: Mapping = field(default_factory=dict)   # opaque caller tags (episode, scene, root id); never read by the models

    def __post_init__(self):
        object.__setattr__(self, 'frames', tuple(self.frames))
        object.__setattr__(self, 'range_packet', _packet_arrays(self.range_packet))
        self.validate()

    # ------------------------------------------------------------------ derived views
    @property
    def timestamp(self) -> float:
        return float(self.frames[-1]['timestamp'])

    @property
    def robot_pose(self) -> np.ndarray:
        """Current robot pose [x, y, z, yaw] (world metres, frame yaw convention)."""
        return np.asarray(self.frames[-1]['robot_pose'], np.float64)

    @property
    def start_pose(self) -> np.ndarray:
        """Row 0 every candidate trajectory must start with: [t, x, y, z, yaw]."""
        return np.r_[self.timestamp, self.robot_pose]

    @property
    def track_history(self):
        """Legal track rows aligned with the range frames (Stage 2 input)."""
        return [f['tracks'] for f in self.frames]

    @property
    def resolution(self) -> float:
        return float(self.partial_graph['resolution'])

    def identity(self) -> str:
        """Cheap content fingerprint of the clock/pose/packet geometry (diagnostic, not a cache key)."""
        h = hashlib.sha256(b'layer2-decision-state-1')
        h.update(np.float64([self.timestamp, *self.robot_pose, len(self.frames), len(self.partial_graph['cells'])]).tobytes())
        for k in ('origins', 'rotations', 'timestamps'):
            h.update(np.ascontiguousarray(self.range_packet[k]).tobytes())
        h.update(str(self.stream).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ validation
    def validate(self):
        if not self.frames:
            raise ValueError('a decision state needs at least the current frame')
        g = self.partial_graph
        if not isinstance(g, Mapping) or g.get('schema') != GRAPH_SCHEMA:
            raise ValueError('partial_graph must be a %s mapping' % GRAPH_SCHEMA)
        if is_hypothetical_graph(g):
            raise LegalityError('predicted (hypothetical) graphs are never legal Layer 2 inputs; '
                                'use the legal graph observed at the decision time')
        times = np.asarray([float(f['timestamp']) for f in self.frames], np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('frames must have finite, strictly increasing timestamps')
        for f in self.frames:
            if f.get('predicted') or f.get('hypothetical'):
                raise LegalityError('predicted frames are not observations')
            for key in ('robot_pose', 'camera_origin', 'camera_rotation', 'tracks', 'static_updates'):
                if key not in f:
                    raise ValueError('frame lacks legal field %r' % key)
        now = times[-1]
        if abs(float(g['timestamp'])-now) > CLOCK_TOLERANCE:
            raise ValueError('graph clock %.9f differs from the current frame %.9f' % (float(g['timestamp']), now))
        p = self.range_packet
        n = len(self.frames)
        if p['ranges'].ndim != 3 or len(p['ranges']) != n or p['origins'].shape != (n, 3) or p['rotations'].shape != (n, 3, 3) \
                or p['timestamps'].shape != (n,):
            raise ValueError('range packet must hold exactly one frame per arrived observation (%d)' % n)
        if np.max(np.abs(p['timestamps']-times)) > CLOCK_TOLERANCE:
            raise ValueError('range frames and observation frames are on different clocks')
        if np.max(np.abs(p['origins']-np.asarray([f['camera_origin'] for f in self.frames], np.float64))) > POSE_TOLERANCE or \
                np.max(np.abs(p['rotations']-np.asarray([f['camera_rotation'] for f in self.frames], np.float64))) > POSE_TOLERANCE:
            raise ValueError('range frames were not taken from the observation camera poses')
        if not np.isfinite(self.robot_pose).all() or self.robot_pose.shape != (4,):
            raise ValueError('current robot pose must be finite [x, y, z, yaw]')
        return self


def as_trajectory(candidate, state: DecisionState) -> np.ndarray:
    """Validate one candidate against the decision state; returns a float64 N x 5 copy."""
    poses = candidate.poses if isinstance(candidate, CandidateTrajectory) else candidate
    p = np.array(poses, np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 5 or not len(p) or not np.isfinite(p).all():
        raise ValueError('candidate trajectories are N x 5 [t, x, y, z, yaw] arrays')
    if len(p) > 1 and np.any(np.diff(p[:, 0]) <= 0):
        raise ValueError('candidate times must increase')
    if abs(p[0, 0]-state.timestamp) > CLOCK_TOLERANCE:
        raise ValueError('candidate must start at the decision time (absolute clock)')
    if not np.allclose(p[0, 1:], state.robot_pose, rtol=0, atol=1e-7):
        raise ValueError('candidate must start at the current robot pose')
    return p


@dataclass(frozen=True)
class CandidateTrajectory:
    """One hypothetical robot execution (e.g. a Query rolled out by an L2.5 planner + dynamics model)."""
    poses: Any                       # N x 5 [t, x, y, z, yaw], absolute time, row 0 = current pose
    query_id: Optional[str] = None   # opaque caller id, echoed in the output
    metadata: Mapping = field(default_factory=dict)


def trajectory_from_poses(poses, start_time, step=0.048):
    """Planner/dynamics rollout poses [T+1, 4] (x, y, z, yaw; row 0 = current pose, one row per
    ``step`` seconds) -> canonical N x 5 candidate with absolute times ``start_time + i * step``."""
    p = np.asarray(poses, np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not len(p):
        raise ValueError('rollout poses are [T+1, 4] (x, y, z, yaw)')
    return np.c_[float(start_time)+np.arange(len(p))*float(step), p]


def normalise_candidates(candidates: Sequence, state: DecisionState):
    out = []
    for i, c in enumerate(candidates):
        if isinstance(c, CandidateTrajectory):
            out.append((as_trajectory(c, state), c.query_id, dict(c.metadata)))
        else:
            out.append((as_trajectory(c, state), None, {}))
    return out
