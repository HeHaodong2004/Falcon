"""Legal online state: array partial map, human tracks, proprioceptive history.

Everything here is built only from legal observation rows produced by the
project's ideal geometry sensor (``partial.fast_observer``): FREE/BLOCKED
configuration-space certificates, exact-torso detections in the 90 deg jaw
frustum with static+human occlusion, the robot's own pose/actions, and the
public PointGoal.  UNKNOWN stays UNKNOWN.  ``GridMap.apply`` reproduces
``partial.map.PartialGraph.update`` exactly (verified by tests), so a
``PartialGraph`` for Stage 2 can always be exported.
"""
import copy
import math
import numpy as np

UNKNOWN, FREE, BLOCKED = 0, 1, 2
RESOLUTION = 0.1016884229464652  # robot collision_reference/6, as in all Stage 1/2 data
EGO_SOURCE = 'occupied_ego_base_support'


class GridMap:
    """Growable dense certificate grid with PartialGraph update semantics."""

    def __init__(self, resolution=RESOLUTION, center=(0., 0.), half_cells=160):
        self.res = float(resolution)
        cx, cz = math.floor(center[0]/self.res), math.floor(center[1]/self.res)
        self.ox, self.oz = cx-half_cells, cz-half_cells     # key of index (0,0)
        n = 2*half_cells+1
        self.state = np.zeros((n, n), dtype=np.uint8)
        self.height = np.zeros((n, n), dtype=np.float64)
        self.first_seen = np.full((n, n), np.nan, dtype=np.float32)
        self.last_seen = np.full((n, n), np.nan, dtype=np.float32)
        self.revision = 0
        self.timestamp = None
        self.base_height = None

    # ---------------------------------------------------------- geometry
    @property
    def shape(self):
        return self.state.shape

    def key(self, x, z):
        return math.floor(x/self.res), math.floor(z/self.res)

    def index(self, x, z):
        return math.floor(x/self.res)-self.ox, math.floor(z/self.res)-self.oz

    def center_xz(self, i, j):
        return (i+self.ox+.5)*self.res, (j+self.oz+.5)*self.res

    def inside(self, i, j):
        return 0 <= i < self.state.shape[0] and 0 <= j < self.state.shape[1]

    def state_at(self, x, z):
        i, j = self.index(x, z)
        return int(self.state[i, j]) if self.inside(i, j) else UNKNOWN

    def _ensure(self, keys):
        if not len(keys):
            return
        kmin = keys.min(axis=0); kmax = keys.max(axis=0)
        lo_x, lo_z = kmin[0]-self.ox, kmin[1]-self.oz
        hi_x, hi_z = kmax[0]-self.ox, kmax[1]-self.oz
        n0, n1 = self.state.shape
        if lo_x >= 0 and lo_z >= 0 and hi_x < n0 and hi_z < n1:
            return
        pad = 64
        add_lo_x = max(0, -lo_x)+(pad if lo_x < 0 else 0)
        add_lo_z = max(0, -lo_z)+(pad if lo_z < 0 else 0)
        add_hi_x = max(0, hi_x-n0+1)+(pad if hi_x >= n0 else 0)
        add_hi_z = max(0, hi_z-n1+1)+(pad if hi_z >= n1 else 0)
        width = ((add_lo_x, add_hi_x), (add_lo_z, add_hi_z))
        self.state = np.pad(self.state, width)
        self.height = np.pad(self.height, width)
        self.first_seen = np.pad(self.first_seen, width, constant_values=np.nan)
        self.last_seen = np.pad(self.last_seen, width, constant_values=np.nan)
        self.ox -= add_lo_x; self.oz -= add_lo_z

    # ------------------------------------------------------------ update
    def apply(self, row):
        """Exactly ``PartialGraph.update`` for one observation row (vectorised).

        Within a frame BLOCKED wins over FREE for the same key; a later FREE
        certificate may clear a BLOCKED cell; the proprioceptive ego cell is
        ignored.  Revision increments iff any cell state or height changes.
        """
        t = float(row['timestamp'])
        if not math.isfinite(t) or (self.timestamp is not None and t <= self.timestamp):
            raise ValueError('Map observations must arrive in strict timestamp order')
        allu = row['static_updates']
        if any(float(u['timestamp']) != t for u in allu):
            raise ValueError('Future/stale static evidence is forbidden')
        ups = [u for u in allu if u['source'] != EGO_SOURCE]
        changed = 0
        if ups:
            keys = np.asarray([u['cell'] for u in ups], dtype=np.int64)
            states = np.asarray([int(u['state']) for u in ups], dtype=np.uint8)
            if not np.isin(states, (FREE, BLOCKED)).all():
                raise ValueError('Only measured free/occupied cells may be published')
            heights = np.asarray([float(u['height']) for u in ups], dtype=np.float64)
            self._ensure(keys)
            ii = keys[:, 0]-self.ox; jj = keys[:, 1]-self.oz
            flat = ii*self.state.shape[1]+jj
            blk = states == BLOCKED
            # last occurrence wins per key (as sequential dict assignment)
            order = np.arange(len(flat))
            bflat, bidx = np.unique(flat[blk][::-1], return_index=True)
            bsel = order[blk][::-1][bidx]
            fmask = (~blk) & ~np.isin(flat, bflat)
            fflat, fidx = np.unique(flat[fmask][::-1], return_index=True)
            fsel = order[fmask][::-1][fidx]
            sel = np.concatenate([bsel, fsel])
            fl = flat[sel]
            a, b = fl//self.state.shape[1], fl % self.state.shape[1]
            prev = self.state[a, b]
            old_h = self.height[a, b]
            new_s = states[sel]; new_h = heights[sel]
            changed = int(((prev != new_s) | ((prev != UNKNOWN) & (np.abs(old_h-new_h) > 1e-8))).sum())
            self.state[a, b] = new_s
            self.height[a, b] = new_h
            fs = self.first_seen[a, b]
            self.first_seen[a, b] = np.where(np.isnan(fs), t, fs)
            self.last_seen[a, b] = t
        if changed:
            self.revision += 1
        self.timestamp = t
        self.base_height = float(row['robot_pose'][1])
        return changed

    def copy(self):
        g = GridMap.__new__(GridMap)
        g.res, g.ox, g.oz = self.res, self.ox, self.oz
        g.state, g.height = self.state.copy(), self.height.copy()
        g.first_seen, g.last_seen = self.first_seen.copy(), self.last_seen.copy()
        g.revision, g.timestamp, g.base_height = self.revision, self.timestamp, self.base_height
        return g

    def known_counts(self):
        return int((self.state == FREE).sum()), int((self.state == BLOCKED).sum())

    def to_partial_graph(self, ego_pose=None):
        """Export the same certificates as a ``partial.map.PartialGraph``."""
        from ..partial.map import PartialGraph
        g = PartialGraph(self.res, self.base_height if self.base_height is not None else 0.)
        for i, j in np.argwhere(self.state != UNKNOWN):
            g.cells[(int(i+self.ox), int(j+self.oz))] = {
                'state': int(self.state[i, j]), 'height': float(self.height[i, j]),
                'first_seen': float(self.first_seen[i, j]), 'last_seen': float(self.last_seen[i, j]),
                'source': 'visible_native_cspace_square_and_clearance_halo' if self.state[i, j] == FREE
                          else 'observed_static_wall_inflated_025m'}
        g.timestamp, g.revision = self.timestamp, self.revision
        if ego_pose is not None:
            g.ego_support = np.asarray(ego_pose[:3], dtype=float).copy()
        return g


class LegalState:
    """The robot's complete legal knowledge at one decision/control instant."""

    def __init__(self, goal, start_pose, start_time, resolution=RESOLUTION):
        self.goal = np.asarray(goal, dtype=float)          # public PointGoal (world frame)
        self.map = GridMap(resolution, center=(start_pose[0], start_pose[2]))
        self.tracks = []
        self.time = float(start_time)
        self.start_time = float(start_time)
        self.pose = np.asarray(start_pose, dtype=float).copy()
        self.poses = [self.pose.copy()]
        self.times = [self.time]
        self.actions = []
        self.steps = 0
        self.track_log = []          # (time, tracks) legal human history for Stage 1

    def observe(self, row, action=None):
        if action is not None:
            self.actions.append(action)
            self.steps += 1
        self.map.apply(row)
        self.tracks = row['tracks']
        self.time = float(row['timestamp'])
        self.pose = np.asarray(row['robot_pose'], dtype=float)
        self.poses.append(self.pose.copy())
        self.times.append(self.time)
        self.track_log.append((self.time, row['tracks']))

    def velocity(self, window=1):
        """Measured planar velocity (m/s) and yaw rate over the last steps."""
        if len(self.poses) < 2:
            return np.zeros(2), 0.
        k = min(window, len(self.poses)-1)
        a, b = self.poses[-1-k], self.poses[-1]
        dt = max(self.times[-1]-self.times[-1-k], 1e-6)
        dyaw = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
        return np.array([(b[0]-a[0])/dt, (b[2]-a[2])/dt]), dyaw/dt

    def goal_vector(self):
        d = self.goal[[0, 2]]-self.pose[[0, 2]]
        return float(np.hypot(*d)), math.atan2(-d[1], d[0])

    def copy(self):
        s = LegalState.__new__(LegalState)
        s.goal = self.goal
        s.map = self.map.copy()
        s.tracks = self.tracks
        s.time, s.start_time, s.pose = self.time, self.start_time, self.pose.copy()
        s.poses, s.times, s.actions = list(self.poses), list(self.times), list(self.actions)
        s.steps = self.steps
        s.track_log = list(self.track_log)
        return s


class _ShallowCopy:
    """Module shim: ``deepcopy`` identity for read-only observation rows."""
    @staticmethod
    def deepcopy(x):
        return x


HEIGHT_QUANTUM = 0.01   # m; branch-pure private caches (see LegalSensor.pure_caches)


def _pure_observer_class():
    """FastGeometryObserver whose private support / probe certificates are pure
    functions of (cell, quantised base height).

    The accepted observer caches static support certificates per 0.1 m height
    *layer* but computes them at the exact base height of the first request, so
    on sloped floors the cached value depends on which pose asked first.  With
    many Oracle branches sharing one cache this made a branch's legal map depend
    on the search history (found by a HOLDOUT replay mismatch).  Here the height
    is quantised to 1 cm before computing and the cache namespace carries the
    quantised height, so no value is reused across heights.  On flat floors the
    certificates are identical to the accepted observer's.
    """
    from ..partial.fast_observer import FastGeometryObserver

    class PureFastGeometryObserver(FastGeometryObserver):
        def _support(self, key, base_height):
            return super()._support(key, round(float(base_height)/HEIGHT_QUANTUM)*HEIGHT_QUANTUM)

        def _targets(self, key, height):
            return super()._targets(key, round(float(height)/HEIGHT_QUANTUM)*HEIGHT_QUANTUM)
    return PureFastGeometryObserver


class LegalSensor:
    """Wrapper around the project's ideal geometry sensor with copyable state.

    The private static-support/probe caches are sensor-side simulation caches
    (static geometry evaluated once per scene); they never enter the legal
    state.  The per-branch sensor state (tracker, certified-cell set) does.
    ``pure_caches`` (default False = the accepted observer, as frozen in
    querygen-freeze-v1) makes those caches branch-history independent.
    """

    def __init__(self, task, resolution=RESOLUTION, private_cache=None, pure_caches=False):
        from types import SimpleNamespace
        from habitat_baselines.config.default import get_config
        from scripts.collector.observation import ObservationConfig
        from ..partial.fast_observer import FastGeometryObserver
        from .native import CONFIGS
        jaw = get_config(CONFIGS[task.spec.dataset]).habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor
        self.config = ObservationConfig.from_depth_sensor(jaw)
        rt = SimpleNamespace(sim=task.sim, observation_config=self.config, count=task.human_count,
                             episode=SimpleNamespace(scene_id=task.episode.scene_id,
                                                     info={'graph_navmesh': 'native_benchmark_navmesh'}))
        cls = _pure_observer_class() if pure_caches else FastGeometryObserver
        self.observer = cls(rt, resolution, private_cache=private_cache if private_cache is not None else {})
        self.task = task
        self.pure_caches = pure_caches
        self._namespace0 = self.observer.cache_namespace
        # The observer deep-copies each full row into ``last_row`` only to
        # serve repeated timestamps; rows are never mutated and timestamps
        # never repeat here, so that copy (~30% of sensing time) is skipped.
        from ..partial import fast_observer as _fo
        if not isinstance(_fo.copy, _ShallowCopy):
            _fo.copy = _ShallowCopy()

    def observe(self, timestamp):
        if self.pure_caches:
            h = round(float(self.task.robot.base_pos[1])/HEIGHT_QUANTUM)
            self.observer.cache_namespace = self._namespace0+(('qheight', h),)
        return self.observer.observe(timestamp)

    def state(self):
        o = self.observer
        return (copy.deepcopy(o.tracker), set(o.seen), o.last_timestamp, o.last_row)

    def restore(self, st):
        o = self.observer
        o.tracker = copy.deepcopy(st[0]); o.seen = set(st[1])
        o.last_timestamp, o.last_row = st[2], st[3]
