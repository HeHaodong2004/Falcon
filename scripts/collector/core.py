"""Simulator-free query, legal-input, storage and offline-label contracts."""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class LegalRoot:
    root_id: str
    position: tuple
    yaw: float
    control_dt: float
    integration_dt: float

    @classmethod
    def load(cls, value):
        # Deliberately enumerate fields; never forward arbitrary simulator dicts.
        return cls(value['root_id'], tuple(value['robot_position']),
                   float(value['robot_yaw']), float(value['control_dt']),
                   float(value['integration_dt']))


@dataclass(frozen=True)
class Query:
    """Canonical bytes are the immutable executable, including all controls."""
    serialized: str

    @classmethod
    def make(cls, value):
        return cls(canonical(value))

    @property
    def data(self):
        return json.loads(self.serialized)

    @property
    def sha256(self):
        return hashlib.sha256(self.serialized.encode()).hexdigest()


def legal_observation(observations, position, yaw, timestamp, previous_command, human_tracks=None):
    # Tracks must come from the legal observer, never arbitrary task sensors.
    from collector.observation import legal_tracks
    allowed = ('agent_0_articulated_agent_jaw_rgb', 'agent_0_articulated_agent_jaw_depth')
    images = {k: np.asarray(observations[k]).copy() for k in allowed}
    meta = dict(timestamp=float(timestamp), robot_position=list(map(float, position)),
                robot_yaw=float(yaw), previous_command=list(previous_command),
                human_tracks=legal_tracks(human_tracks) if human_tracks is not None else [],
                tracking_available=human_tracks is not None,
                sensor_keys=list(allowed))
    return meta, images


def compile_motion(root, path, target, direction, wait, scale, config):
    dt = root.control_dt
    commands, trajectory = [], [[0., *root.position, root.yaw]]
    pos, yaw = np.array(root.position, dtype=float), root.yaw

    def append(v, w, steps):
        nonlocal pos, yaw
        for _ in range(steps):
            commands.append([v * dt / root.integration_dt, w * dt / root.integration_dt])
            yaw += w * dt
            pos += np.array([math.cos(yaw), 0., -math.sin(yaw)]) * v * dt
            trajectory.append([len(commands)*dt, *pos.tolist(), yaw])

    append(0., 0., int(math.ceil(wait/dt - 1e-9)))
    for end in path[1:]:
        delta = np.asarray(end) - pos
        distance = float(np.linalg.norm(delta[[0, 2]]))
        if distance < 1e-5:
            continue
        heading = math.atan2(-delta[2], delta[0])
        angle = (heading-yaw+math.pi) % (2*math.pi)-math.pi
        n = max(1, math.ceil(abs(angle)/(config['yaw_speed']*dt)))
        append(0., angle/(n*dt), n)
        n = max(1, math.ceil(distance/(config['speed']*scale*dt)))
        append(distance/(n*dt), 0., n)
    if not commands:
        append(0., 0., 1)
    body = dict(schema_version=1, root_id=root.root_id, direction=direction,
                local_target=target, spatial_path=path, nominal_timed_trajectory=trajectory,
                executable_controls=commands, control_units=['native_m_per_s', 'native_rad_per_s'],
                control_dt=dt, integration_dt=root.integration_dt, wait_duration=wait,
                timing_policy='one_fixed_control_per_environment_step; native_clock_authoritative',
                clock_quantization_tolerance=.008001,
                effective_wait_duration=math.ceil(wait/dt-1e-9)*dt, speed_scale=scale,
                planned_duration=len(commands)*dt,
                static_path_length=sum(float(np.linalg.norm((np.asarray(b)-a)[[0,2]])) for a,b in zip(path,path[1:])),
                robot_spec={'research_radius': .25, 'navmesh_radius': .25, 'height': 1.41},
                generation_config=config)
    body['query_id'] = 'q_' + digest(body)[:12]
    return Query.make(body)


def generate_queries(root, planner):
    """Capability boundary: only a LegalRoot and static pathfinder adapter."""
    cfg = dict(directions=8, target_radius=1.5, speed=.7, yaw_speed=1.5,
               max_projection=1.5, max_path_length=4., human_blind=True)
    queries, rejected, seen = [], [], set()
    for i in range(8):
        angle = root.yaw + 2*math.pi*i/8
        local = [cfg['target_radius']*math.cos(2*math.pi*i/8),
                 cfg['target_radius']*math.sin(2*math.pi*i/8)]
        target = np.asarray(root.position) + [cfg['target_radius']*math.cos(angle),0,-cfg['target_radius']*math.sin(angle)]
        path = planner.path(root.position, target, cfg)
        if path is None:
            rejected.append(dict(direction=i, reason='static_invalid'))
            continue
        key = tuple(np.round(path[-1], 3))
        if key in seen:
            rejected.append(dict(direction=i, reason='duplicate_projected_target'))
            continue
        seen.add(key)
        for wait, scale in ((0.,1.),(.5,1.),(1.,1.),(0.,.5)):
            queries.append(compile_motion(root,path,local,i,wait,scale,cfg))
    horizon = max(q.data['planned_duration'] for q in queries) if queries else 4.
    queries.append(compile_motion(root,[list(root.position)], [0.,0.], 'WAIT', horizon, 0., cfg))
    return queries, rejected


def build_labels(branch, near_threshold=.5):
    rows = branch['truth']
    times = np.array([r['timestamp'] for r in rows])
    clearance = []
    for row in rows:
        r = np.asarray(row['robot']['position'])[[0,2]]
        ds = [float(np.linalg.norm(r-np.asarray(h['position'])[[0,2]]))-.25-.30 for h in row['humans']]
        clearance.append(min(ds) if ds else float('inf'))
    disc = np.asarray(clearance) <= 0
    native = np.array([r['native_human_contact'] for r in rows], dtype=bool)
    static = np.array([r['native_static_contact'] for r in rows], dtype=bool)
    event = disc | native
    completed = bool(branch['completed'])
    def first(mask):
        return float(times[np.flatnonzero(mask)[0]]-times[0]) if mask.any() else None
    return dict(branch_id=branch['branch_id'], query_id=branch['query_id'],
                native_robot_human_contact=bool(native.any()), research_disc_contact=bool(disc.any()),
                human_collision_event=bool(event.any()), first_event_time=first(event),
                first_native_event_time=first(native), first_disc_event_time=first(disc),
                static_contact=bool(static.any()), first_static_event_time=first(static),
                minimum_human_clearance=min(clearance) if np.isfinite(min(clearance)) else None,
                near_threshold=near_threshold,
                near_exposure_seconds=float(np.sum(np.diff(times)*(np.asarray(clearance)[1:] < near_threshold))),
                exposure_convention='right_endpoint_piecewise_constant_native_samples',
                completed=completed, censored=not completed,
                full_query_safe=(not event.any() and not static.any()) if completed else None,
                event_observed=bool(event.any()), planned_duration=branch['planned_duration'],
                observed_duration=float(times[-1]-times[0]), termination_reason=branch['termination_reason'])


def load_wm_input(root_dir, query_id):
    """Explicit public loader never opens roots/privileged or branch_truth."""
    root_dir = Path(root_dir)
    history = read(root_dir/'legal_observations/history.json')
    fields = ('timestamp','robot_position','robot_yaw','previous_command','human_tracks',
              'tracking_available','sensor_keys')
    history = [{k: row[k] for k in fields} for row in history]
    from collector.observation import legal_tracks
    for row in history:
        row['human_tracks'] = legal_tracks(row['human_tracks'])
    query = next(q['query'] for q in read(root_dir/'queries/queries.json') if q['query']['query_id']==query_id)
    query_fields = ('schema_version','root_id','direction','local_target','spatial_path',
                    'nominal_timed_trajectory','executable_controls','control_units','control_dt',
                    'integration_dt','wait_duration','timing_policy','clock_quantization_tolerance',
                    'effective_wait_duration','speed_scale','planned_duration','static_path_length',
                    'robot_spec','generation_config','query_id')
    query = {k: query[k] for k in query_fields}
    query['robot_spec'] = {k: query['robot_spec'][k] for k in ('research_radius','navmesh_radius','height')}
    query['generation_config'] = {k: query['generation_config'][k] for k in
        ('directions','target_radius','speed','yaw_speed','max_projection','max_path_length','human_blind')}
    return dict(history=history, history_images=str(root_dir/'legal_observations/history_images.npz'), query=query)
