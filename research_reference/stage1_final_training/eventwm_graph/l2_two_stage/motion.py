"""Exact deterministic summaries of arrived human and robot observations.

These are parameter-independent online input facts, not a forecast cache. Every
actual frame is retained separately by OnlineContext. No candidate query or
future trajectory/observation enters this state.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .types import SealedObservation, immutable_array

_CERTIFICATE = object()
_VERSION = 'arrived_motion_summary_two_stage_1'


def _positives(ob):
    if not ob.get('observation_valid', True) or not ob.get('human_sensor_available', True):
        return []
    return [tr for tr in ob['tracks'] if tr.get('visible_now') and tr.get('valid', True)
            and abs(float(tr['last_seen_time'])-float(ob['timestamp'])) < 1e-7]


@dataclass(frozen=True, init=False)
class MotionSummary:
    memory: Mapping
    valid: int
    empty: int
    invalid: int
    distance: float
    yaw_motion: float
    minimum_count: int
    frame_count: int
    timestamp: float
    robot_pose: tuple
    version: str = _VERSION

    def validate_certificate(self):
        if getattr(self, '_certificate', None) is not _CERTIFICATE:
            raise ValueError('uncertified motion summary')
        return self

    @classmethod
    def _verified(cls, memory, valid, empty, invalid, distance, yaw_motion,
                  minimum_count, frame_count, timestamp, robot_pose):
        result = object.__new__(cls)
        values = dict(memory=MappingProxyType(memory), valid=valid, empty=empty,
                      invalid=invalid, distance=distance, yaw_motion=yaw_motion,
                      minimum_count=minimum_count, frame_count=frame_count,
                      timestamp=timestamp, robot_pose=tuple(robot_pose),
                      version=_VERSION, _certificate=_CERTIFICATE)
        for key, value in values.items():
            object.__setattr__(result, key, value)
        return result

    @classmethod
    def from_frames(cls, frames):
        result = cls._verified({}, 0, 0, 0, 0., 0., 0, 0, float('-inf'), (0., 0., 0., 0.))
        for ob in frames:
            result = result.append(ob)
        return result

    def append(self, obs):
        """Use exactly the arithmetic/order of the full-history replay fallback."""
        self.validate_certificate()
        ob = SealedObservation.from_raw(obs)
        timestamp = float(ob['timestamp'])
        if timestamp <= self.timestamp:
            raise ValueError('noncausal motion-summary append')
        distance, yaw_motion = self.distance, self.yaw_motion
        if self.frame_count:
            previous = self.robot_pose
            distance += np.linalg.norm(np.asarray(ob['robot_pose'])[[0, 2]] - np.asarray(previous)[[0, 2]])
            yaw_motion += abs(np.angle(np.exp(1j*(ob['robot_pose'][3]-previous[3]))))
        tracks = _positives(ob)
        valid, empty, invalid = self.valid, self.empty, self.invalid
        if ob.get('observation_valid', True) and ob.get('human_sensor_available', True):
            valid += 1
            empty += not bool(tracks)
        else:
            invalid += 1
        memory = dict(self.memory)
        for track in tracks:
            key = str(track['track_id'])
            t = float(ob['timestamp'])
            position = np.asarray(track['position'])[[0, 2]]
            velocity = np.zeros(2)
            acceleration = np.zeros(2)
            if key in memory:
                old = memory[key]
                dt = t-old['last']
                velocity = (position-old['pos'])/dt
                acceleration = (velocity-old['velocity'])/max(dt, .048)
                count = old['count']+1
                mean = old['mean']+(position-old['mean'])/count
                first = old['first']
                observed_path = old['path']+np.linalg.norm(position-old['pos'])
            else:
                count = 1
                mean = position.copy()
                first = t
                observed_path = 0.
                if track.get('velocity_valid') and track.get('estimated_velocity') is not None:
                    velocity = np.asarray(track['estimated_velocity'])[[0, 2]]
            memory[key] = MappingProxyType(dict(pos=immutable_array(position), last=t,
                velocity=immutable_array(velocity), acceleration=immutable_array(acceleration),
                count=count, mean=immutable_array(mean), first=first, path=observed_path))
        return type(self)._verified(memory, valid, empty, invalid, distance, yaw_motion,
            max(self.minimum_count, len(tracks)), self.frame_count+1, timestamp,
            ob['robot_pose'])
