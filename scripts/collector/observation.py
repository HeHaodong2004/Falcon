"""Ideal legal perception. Only the Habitat adapter may inspect sensor-world GT.

Detections contain no simulator identity. Tracking consumes only these detections.
All world coordinates use Habitat x/y/z (y up); no GT velocities are read.
"""
import copy
import math
from dataclasses import asdict, dataclass
import numpy as np

TRACK_FIELDS = ('track_id', 'position', 'estimated_velocity', 'velocity_valid',
                'visible_now', 'valid', 'first_seen_time', 'last_seen_time',
                'observation_age', 'num_observations')


def legal_tracks(tracks):
    """Deep, nested whitelist; track IDs are bookkeeping, not model features."""
    return [{k: copy.deepcopy(t[k]) for k in TRACK_FIELDS} for t in tracks]


@dataclass(frozen=True)
class ObservationConfig:
    horizontal_fov_deg: float
    vertical_fov_deg: float
    max_range_m: float
    torso_height_m: float = 1.0
    los_tolerance_m: float = .02
    track_max_age_s: float = 2.0
    association_gate_m: float = .8
    prediction_horizon_s: float = .5
    velocity_max_gap_s: float = .5
    min_velocity_dt_s: float = 1e-6
    sensor_uuid: str = 'agent_0_articulated_agent_jaw_depth'

    def __post_init__(self):
        if not (0 < self.horizontal_fov_deg < 180 and 0 < self.vertical_fov_deg < 180):
            raise ValueError('FOV must be in (0,180) degrees')
        for key, value in asdict(self).items():
            if key != 'sensor_uuid' and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'Invalid observation config {key}')

    @classmethod
    def from_depth_sensor(cls, sensor):
        hfov = float(sensor.hfov)
        vfov = math.degrees(2*math.atan(math.tan(math.radians(hfov/2))*sensor.height/sensor.width))
        return cls(hfov, vfov, float(sensor.max_depth))


@dataclass(frozen=True)
class HumanDetection:
    timestamp: float
    position: tuple
    visible: bool = True
    measurement_valid: bool = True


class VisibilityModel:
    def __init__(self, config):
        self.config = config

    def evaluate(self, origin, rotation, base_position, active, line_of_sight):
        """rotation is camera-to-world; camera looks down local -Z.

        line_of_sight(origin, torso) returns (clear, privileged ray witness).
        No GT metadata other than active status / current position is accepted.
        """
        if not active:
            return False, {'reason': 'inactive'}
        p = np.asarray(base_position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            return False, {'reason': 'invalid_position'}
        target = p + [0., self.config.torso_height_m, 0.]
        delta = target-np.asarray(origin)
        distance = float(np.linalg.norm(delta))
        witness = dict(torso=target.tolist(), range_m=distance)
        if distance > self.config.max_range_m or distance < 1e-9:
            return False, dict(witness, reason='out_of_range')
        local = np.asarray(rotation).T @ delta
        horizontal = math.degrees(math.atan2(local[0], -local[2]))
        vertical = math.degrees(math.atan2(local[1], -local[2]))
        witness.update(horizontal_angle_deg=horizontal, vertical_angle_deg=vertical)
        if (local[2] >= 0 or abs(horizontal) > self.config.horizontal_fov_deg/2 or
                abs(vertical) > self.config.vertical_fov_deg/2):
            return False, dict(witness, reason='outside_fov')
        clear, ray = line_of_sight(np.asarray(origin), target)
        return bool(clear), dict(witness, reason='visible' if clear else 'static_occlusion', ray=ray)


class IdealHumanDetector:
    def __init__(self, config):
        self.visibility = VisibilityModel(config)

    def detect(self, timestamp, origin, rotation, sensor_world, line_of_sight):
        detections, diagnostics = [], []
        for human in sensor_world:
            visible, witness = self.visibility.evaluate(origin, rotation, human['position'],
                                                       human['active'], line_of_sight)
            # GT identity exists only in a separate privileged diagnostic record.
            diagnostics.append(dict(witness, diagnostic_id=human.get('diagnostic_id'),
                                    gt_position=list(human['position']), detected=visible))
            if visible:
                detections.append(HumanDetection(float(timestamp), tuple(map(float,human['position']))))
        # Simulator iteration order/hidden slots cannot determine legal track IDs.
        detections.sort(key=lambda d: d.position)
        return detections, diagnostics


class LegalHumanTracker:
    def __init__(self, config):
        self.config, self.tracks, self.next_id = config, {}, 0
        self.timestamp = None

    def update(self, timestamp, detections):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise ValueError('Tracker timestamps must be finite and nondecreasing')
        if timestamp == self.timestamp:
            return legal_tracks(self.tracks.values())  # root observed twice; never double count
        self.timestamp = timestamp
        self.tracks = {k: t for k,t in self.tracks.items()
                       if timestamp-t['last_seen_time'] <= self.config.track_max_age_s}
        ds = []
        for d in detections:
            p = np.asarray(d.position, dtype=float)
            if (d.measurement_valid and d.visible and d.timestamp == timestamp and
                    p.shape == (3,) and np.isfinite(p).all()):
                ds.append(d)
        ds.sort(key=lambda d:d.position)
        # Deterministic greedy minimum-distance matching, gated in world XYZ.
        # Prediction is ONLY for association; output position stays last observed.
        pairs = []
        for track_id,t in self.tracks.items():
            t['visible_now'] = False
            t['observation_age'] = timestamp-t['last_seen_time']
            pred = np.asarray(t['position'], dtype=float)
            if t['velocity_valid']:
                pred = pred+np.asarray(t['estimated_velocity'])*min(t['observation_age'],self.config.prediction_horizon_s)
            for i,d in enumerate(ds):
                distance = float(np.linalg.norm(pred-d.position))
                if distance <= self.config.association_gate_m:
                    pairs.append((distance, track_id, i))
        used_tracks, used_detections = set(), set()
        for _,tid,i in sorted(pairs):
            if tid in used_tracks or i in used_detections:
                continue
            used_tracks.add(tid); used_detections.add(i)
            t, d = self.tracks[tid], ds[i]
            dt = timestamp-t['last_seen_time']
            velocity = None
            if self.config.min_velocity_dt_s < dt <= self.config.velocity_max_gap_s:
                velocity = ((np.asarray(d.position)-t['position'])/dt).tolist()
            t.update(position=list(d.position), estimated_velocity=velocity, velocity_valid=velocity is not None,
                     visible_now=True, last_seen_time=timestamp, observation_age=0.,
                     num_observations=t['num_observations']+1)
        for i,d in enumerate(ds):
            if i in used_detections:
                continue
            tid = f'track_{self.next_id:06d}'; self.next_id += 1
            self.tracks[tid] = dict(track_id=tid, position=list(d.position), estimated_velocity=None,
                velocity_valid=False, visible_now=True, valid=True, first_seen_time=timestamp,
                last_seen_time=timestamp, observation_age=0., num_observations=1)
        return legal_tracks(self.tracks.values())


class HabitatHumanObserver:
    """Privileged sensor adapter; raycasts are read-only, no RNG or stepping.

    Stage + STATIC rigid objects occlude. Articulated agents and non-static rigid
    objects are excluded: this protocol explicitly models static occlusion only.
    """
    def __init__(self, sim, config):
        self.sim, self.config = sim, config
        self.detector, self.tracker = IdealHumanDetector(config), LegalHumanTracker(config)
        self.diagnostics = []
        self.last_timestamp = None
        self.last_tracks = []

    def camera_pose(self):
        node = self.sim._sensors[self.config.sensor_uuid]._sensor_object.node
        transform = node.absolute_transformation()
        return np.asarray(transform.translation, dtype=float), np.asarray(transform.rotation(), dtype=float)

    def ignored_object_ids(self):
        import habitat_sim
        ids = set()
        for agent in self.sim.agents_mgr.articulated_agents_iter:
            ids.add(agent.sim_obj.object_id)
            ids.update(agent.sim_obj.link_object_ids.keys())
        for obj in self.sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
            if obj.motion_type != habitat_sim.physics.MotionType.STATIC:
                ids.add(obj.object_id)
        return ids

    def line_of_sight(self, origin, target):
        import habitat_sim
        delta = target-origin; length = float(np.linalg.norm(delta))
        ray = habitat_sim.geo.Ray(origin, delta/length)
        ignored = self.ignored_object_ids()
        hits = self.sim.cast_ray(ray, max_distance=length).hits
        for hit in sorted(hits, key=lambda h:h.ray_distance):
            if hit.object_id not in ignored and hit.ray_distance < length-self.config.los_tolerance_m:
                return False, dict(object_id=hit.object_id, distance_m=float(hit.ray_distance),
                                   point=np.asarray(hit.point).tolist())
        return True, None

    def observe(self, timestamp, active_count):
        if timestamp == self.last_timestamp:
            return legal_tracks(self.last_tracks)
        origin, rotation = self.camera_pose()
        world = [dict(active=True, position=np.asarray(self.sim.get_agent_data(i).articulated_agent.base_pos).tolist(),
                      diagnostic_id=i) for i in range(1,active_count+1)]
        ds, diagnostics = self.detector.detect(timestamp,origin,rotation,world,self.line_of_sight)
        tracks = self.tracker.update(timestamp,ds)
        self.diagnostics.append(dict(timestamp=float(timestamp), privileged_overlay=True,
            camera_origin=origin.tolist(), camera_rotation=rotation.tolist(), humans=diagnostics,
            legal_detection_count=len(ds), legal_tracks=copy.deepcopy(tracks)))
        self.last_timestamp, self.last_tracks = timestamp, tracks
        return legal_tracks(tracks)
