"""Explicit numerical allowlist. Index metadata never enters tensorize or the model."""
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import numpy as np
import torch

from collector.core import load_wm_input

INPUT_VERSION = 'eventquerywm-legal-tensors-v1'
HUMAN_FEATURES = ('relative_x', 'relative_z', 'velocity_x', 'velocity_z',
                  'visible_now', 'velocity_valid', 'valid', 'observation_age',
                  'relative_time', 'dt', 'present', 'track_age', 'log_observation_count')
ROBOT_FEATURES = ('world_x', 'world_y', 'world_z', 'sin_yaw', 'cos_yaw',
                  'relative_x', 'relative_z', 'relative_time', 'dt',
                  'previous_linear_command', 'previous_angular_command', 'tracking_available')
QUERY_FEATURES = ('relative_x', 'relative_z', 'time', 'velocity_x', 'velocity_z',
                  'sin_relative_yaw', 'cos_relative_yaw', 'distance_travelled',
                  'angular_velocity', 'time_fraction')


@dataclass
class ModelInput:
    humans: torch.Tensor       # [N,T,13] or [B,N,T,13]
    human_padding: torch.Tensor  # [N] or [B,N], True = absent
    robot: torch.Tensor        # [T,12] or [B,T,12]
    history_padding: torch.Tensor  # [T] or [B,T], True = absent
    query: torch.Tensor        # [16,10] or [B,16,10]

    def to(self, device):
        return ModelInput(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def select(self, indices):
        return ModelInput(**{f.name: getattr(self, f.name)[indices] for f in fields(self)})


def tensorize(legal):
    """Only accepts the canonical loader envelope; nested extras are never read.

    Coordinates are planar in the final robot's frame, except the explicitly legal
    world robot pose. No imagery, identifiers, label, variant name, or index values
    become numerical features. Keep the full arrived history (no resampling).
    """
    if set(legal) != {'history', 'history_images', 'query'}:
        raise ValueError('Expected only the canonical load_wm_input envelope')
    history, query = legal['history'], legal['query']
    if not history:
        raise ValueError('Empty arrived history')
    times = np.asarray([h['timestamp'] for h in history], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('History clock must be finite and strictly increasing')
    origin = np.asarray(history[-1]['robot_position'], dtype=np.float64)
    yaw = float(history[-1]['robot_yaw'])
    # Habitat forward = (cos(yaw), -sin(yaw)) in world x/z.
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def local(value, velocity=False):
        value = np.asarray(value, dtype=np.float64)
        return rotation @ (value if velocity else value-origin)[[0, 2]]

    ids = sorted({t['track_id'] for h in history for t in h['human_tracks']})
    humans = np.zeros((len(ids), len(history), len(HUMAN_FEATURES)), dtype=np.float32)
    robot = []
    for j, h in enumerate(history):
        dt = 0. if j == 0 else times[j]-times[j-1]
        rel_time = times[j]-times[-1]
        robot.append([*h['robot_position'], np.sin(h['robot_yaw']), np.cos(h['robot_yaw']),
                      *local(h['robot_position']), rel_time, dt, *h['previous_command'],
                      float(h['tracking_available'])])
        tracks = {t['track_id']: t for t in h['human_tracks']}
        if len(tracks) != len(h['human_tracks']):
            raise ValueError('Duplicate track identity in a history frame')
        for i, identity in enumerate(ids):
            humans[i, j, 8:10] = [rel_time, dt]
            if identity not in tracks:
                continue
            t = tracks[identity]
            valid = bool(t['valid'])
            velocity_valid = valid and bool(t['velocity_valid'])
            position = local(t['position']) if valid else [0., 0.]
            velocity = local(t['estimated_velocity'], True) if velocity_valid else [0., 0.]
            humans[i, j] = [*position, *velocity, float(t['visible_now']),
                            float(velocity_valid), float(valid), t['observation_age'],
                            rel_time, dt, 1., times[j]-t['first_seen_time'],
                            np.log1p(t['num_observations'])]

    # This is the precommitted nominal query, never a branch's executed future.
    trajectory = np.asarray(query['nominal_timed_trajectory'], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 5 or len(trajectory) < 2:
        raise ValueError('Expected nominal [time,x,y,z,yaw] query trajectory')
    qt = trajectory[:, 0]
    if qt[0] != 0 or np.any(np.diff(qt) <= 0) or not np.isfinite(trajectory).all():
        raise ValueError('Invalid nominal query clock/coordinates')
    sample_times = np.linspace(0., qt[-1], 16)
    xy = np.asarray([local(p) for p in trajectory[:, 1:4]])
    heading = np.unwrap(trajectory[:, 4])-yaw
    segment_velocity = np.diff(xy, axis=0)/np.diff(qt)[:, None]
    angular_velocity = np.diff(heading)/np.diff(qt)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    # At a knot use the next interval; final token uses the final interval.
    segment = np.clip(np.searchsorted(qt, sample_times, side='right')-1, 0, len(qt)-2)
    heading_samples = np.interp(sample_times, qt, heading)
    query_tensor = np.column_stack([
        np.interp(sample_times, qt, xy[:, 0]), np.interp(sample_times, qt, xy[:, 1]),
        sample_times, segment_velocity[segment], np.sin(heading_samples),
        np.cos(heading_samples), np.interp(sample_times, qt, distance),
        angular_velocity[segment], sample_times/qt[-1]])
    arrays = [humans, np.asarray(robot, dtype=np.float32), query_tensor.astype(np.float32)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite legal model input')
    return ModelInput(torch.from_numpy(arrays[0]), torch.zeros(len(ids), dtype=torch.bool),
                      torch.from_numpy(arrays[1]), torch.zeros(len(history), dtype=torch.bool),
                      torch.from_numpy(arrays[2]))


def load_sample(row):
    """Index selects a path/query and label, but is never passed to tensorize."""
    if not row['sr_valid'] or row['sr_label'] not in (0, 1):
        raise ValueError('Expected a valid authoritative binary SR label')
    if not row['branch_id'].startswith('b'):
        raise ValueError('Expected a primary sample')
    target = int(row['sr_label'])
    return tensorize(load_wm_input(Path(row['dataset'])/row['root_path'], row['query_id'])), target


def collate(inputs):
    b = len(inputs)
    n = max(1, max(x.humans.shape[0] for x in inputs))
    t = max(x.robot.shape[0] for x in inputs)
    result = ModelInput(torch.zeros(b, n, t, len(HUMAN_FEATURES)), torch.ones(b, n, dtype=torch.bool),
                        torch.zeros(b, t, len(ROBOT_FEATURES)), torch.ones(b, t, dtype=torch.bool),
                        torch.stack([x.query for x in inputs]))
    for i, x in enumerate(inputs):
        ni, ti = x.humans.shape[:2]
        result.humans[i, :ni, :ti] = x.humans
        result.human_padding[i, :ni] = x.human_padding
        result.robot[i, :ti] = x.robot
        result.history_padding[i, :ti] = x.history_padding
    return result


def input_hash(value):
    """Hash actual FP32 features and masks before batch-dependent padding."""
    digest = hashlib.sha256(INPUT_VERSION.encode())
    for f in fields(value):
        array = getattr(value, f.name).detach().cpu().contiguous().numpy()
        digest.update(f.name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
