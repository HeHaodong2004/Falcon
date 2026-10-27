"""Incremental legal observation state shared by offline extraction and online inference.

Consumes only arrived frames: robot/camera pose, tracker positives and static
C-space certificates. Produces a robot-centred world-aligned map raster, a
visibility-age raster (approximate 2-D line of sight over certified BLOCKED
cells) and per-tracker-key observation histories. Tracker keys are bookkeeping
labels, not physical identities. Nothing here reads simulator state.
"""
import math
import numpy as np

COARSE = 0.2          # map raster cell size (m)
CROP = 128            # map raster side (cells) -> +-12.8 m
VIS_POOL = 2          # visibility raster is CROP/VIS_POOL at 0.4 m
GRID = 400            # rolling global grid side (cells) -> +-40 m
MARGIN = 90           # re-centre when the robot is within MARGIN cells of an edge
HISTORY = 16          # retained positives per key
FOV = math.radians(45.)
RANGE = 10.
RAYS = np.linspace(-FOV, FOV, 91)
STEPS = np.arange(0.15, RANGE + 1e-9, 0.15)
AGE_STEP = 0.05       # visibility age quantum (s); 254 cap, 255 never seen
ROBOT_LAGS = np.arange(9) * .25
FREE, BLOCKED = 1, 2
USE_KERNELS = False   # numba raycast (identical result); enabled by the online API


def positives(frame):
    """Exactly the legacy positive rule (features.positives)."""
    if not frame.get('observation_valid', True) or not frame.get('human_sensor_available', True):
        return []
    now = float(frame['timestamp'])
    return [tr for tr in frame['tracks'] if tr.get('visible_now') and tr.get('valid', True)
            and abs(float(tr['last_seen_time']) - now) < 1e-7]


def camera_yaw(rotation):
    forward = -np.asarray(rotation, np.float64)[:, 2]
    return math.atan2(forward[2], forward[0])


class LegalState:
    """Causal state after each arrived frame; ``ingest`` must see frames in order."""

    def __init__(self, resolution):
        self.resolution = float(resolution)
        self.fine = {}
        self.origin = None          # integer coarse index of grid corner
        self.blocked = None
        self.free = None
        self.last_visible = None
        self.keys = {}              # key -> dict(rows=[(t,x,z)], velocity, valid_velocity, first)
        self.robot = []             # (t, x, z, yaw, y)
        self.timestamp = None
        self.frames = 0
        self.valid_frames = 0
        self.empty_frames = 0
        self.max_simultaneous = 0

    # ----- rolling coarse grid -------------------------------------------------
    def _coarse_index(self, x, z):
        return math.floor(x / COARSE), math.floor(z / COARSE)

    def _allocate(self, cx, cz):
        self.origin = (cx - GRID // 2, cz - GRID // 2)
        self.blocked = np.zeros((GRID, GRID), np.int16)
        self.free = np.zeros((GRID, GRID), np.int16)
        old = self.last_visible
        self.last_visible = np.full((GRID, GRID), -np.inf, np.float32)
        return old

    def _recentre(self, x, z):
        cx, cz = self._coarse_index(x, z)
        if self.origin is None:
            self._allocate(cx, cz)
            return
        ix, iz = cx - self.origin[0], cz - self.origin[1]
        if MARGIN <= ix < GRID - MARGIN and MARGIN <= iz < GRID - MARGIN:
            return
        old_origin, old_visible = self.origin, self._allocate(cx, cz)
        dx, dz = old_origin[0] - self.origin[0], old_origin[1] - self.origin[1]
        src_x = slice(max(0, -dx), min(GRID, GRID - dx)); dst_x = slice(max(0, dx), min(GRID, GRID + dx))
        src_z = slice(max(0, -dz), min(GRID, GRID - dz)); dst_z = slice(max(0, dz), min(GRID, GRID + dz))
        self.last_visible[dst_x, dst_z] = old_visible[src_x, src_z]
        if self.fine:   # vectorised recount (same per-cell arithmetic as _count)
            keys = np.array(list(self.fine.keys()), np.float64);states = np.fromiter(self.fine.values(), np.int64, len(self.fine))
            r = self.resolution
            i = np.floor((keys[:, 0] + .5) * r / COARSE).astype(np.int64) - self.origin[0]
            j = np.floor((keys[:, 1] + .5) * r / COARSE).astype(np.int64) - self.origin[1]
            ok = (i >= 0) & (i < GRID) & (j >= 0) & (j < GRID)
            for state, target in ((BLOCKED, self.blocked), (FREE, self.free)):
                m = ok & (states == state)
                np.add.at(target, (i[m], j[m]), 1)

    def _fine_coarse(self, key):
        r = self.resolution
        return (math.floor((key[0] + .5) * r / COARSE) - self.origin[0],
                math.floor((key[1] + .5) * r / COARSE) - self.origin[1])

    def _count(self, key, state, sign):
        i, j = self._fine_coarse(key)
        if 0 <= i < GRID and 0 <= j < GRID:
            if state == BLOCKED:self.blocked[i, j] += sign
            elif state == FREE:self.free[i, j] += sign

    # ----- frame ingestion -----------------------------------------------------
    def ingest(self, frame):
        t = float(frame['timestamp'])
        if self.timestamp is not None and t <= self.timestamp:
            raise ValueError('frames must arrive in strict time order')
        pose = frame['robot_pose']
        self._recentre(float(pose[0]), float(pose[2]))
        updates = frame['static_updates']
        occupied = {tuple(u['cell']) for u in updates if u['state'] == BLOCKED}
        fine = self.fine
        for u in updates:
            if u['source'] == 'occupied_ego_base_support':continue
            key = tuple(u['cell']);state = int(u['state'])
            if state == FREE and key in occupied:continue
            old = fine.get(key)
            if old == state:continue
            if old is not None:self._count(key, old, -1)
            fine[key] = state;self._count(key, state, 1)
        valid = frame.get('observation_valid', True) and frame.get('human_sensor_available', True)
        if frame.get('observation_valid', True):
            self._visibility(frame['camera_origin'], frame['camera_rotation'], t)
        found = positives(frame)
        self.frames += 1
        if valid:
            self.valid_frames += 1;self.empty_frames += not found
        self.max_simultaneous = max(self.max_simultaneous, len(found))
        for tr in found:
            key = str(tr['track_id']);p = tr['position']
            row = (t, float(p[0]), float(p[2]))
            memory = self.keys.get(key)
            if memory is None:
                velocity = None
                if tr.get('velocity_valid') and tr.get('estimated_velocity') is not None:
                    v = tr['estimated_velocity'];velocity = (float(v[0]), float(v[2]))
                self.keys[key] = dict(rows=[row], tracker_velocity=velocity, first=t, n=1)
            else:
                memory['rows'].append(row);memory['n'] += 1
                if len(memory['rows']) > 64:del memory['rows'][:-64]
                v = tr.get('estimated_velocity')
                memory['tracker_velocity'] = (float(v[0]), float(v[2])) if tr.get('velocity_valid') and v is not None else None
        self.robot.append((t, float(pose[0]), float(pose[2]), float(pose[3]), float(pose[1])))
        self.timestamp = t

    def _visibility(self, origin, rotation, t):
        if USE_KERNELS:
            from .kernels import raycast
            yaw = camera_yaw(rotation);angles = yaw + RAYS
            raycast(self.blocked, self.last_visible, float(origin[0]), float(origin[2]), np.cos(angles), np.sin(angles),
                    STEPS, COARSE, int(self.origin[0]), int(self.origin[1]), np.float32(t))
            return
        ox, oz = float(origin[0]), float(origin[2])
        yaw = camera_yaw(rotation)
        angles = yaw + RAYS
        xs = ox + np.cos(angles)[:, None] * STEPS[None]
        zs = oz + np.sin(angles)[:, None] * STEPS[None]
        ix = np.floor(xs / COARSE).astype(np.int64) - self.origin[0]
        iz = np.floor(zs / COARSE).astype(np.int64) - self.origin[1]
        inside = (ix >= 0) & (ix < GRID) & (iz >= 0) & (iz < GRID)
        ixc = np.clip(ix, 0, GRID - 1);izc = np.clip(iz, 0, GRID - 1)
        hit = (self.blocked[ixc, izc] > 0) & inside
        first = np.where(hit.any(1), hit.argmax(1), hit.shape[1])
        visible = (np.arange(hit.shape[1])[None] < first[:, None]) & inside
        self.last_visible[ixc[visible], izc[visible]] = t

    # ----- anchor features -----------------------------------------------------
    def crops(self):
        """Map codes [128,128] uint8, visibility age codes [64,64] uint8, world corner."""
        t, x, z = self.robot[-1][:3]
        cx, cz = self._coarse_index(x, z)
        i0, j0 = cx - CROP // 2 - self.origin[0], cz - CROP // 2 - self.origin[1]
        corner = np.array([(cx - CROP // 2) * COARSE, (cz - CROP // 2) * COARSE], np.float64)
        if 0 <= i0 and i0 + CROP <= GRID and 0 <= j0 and j0 + CROP <= GRID:   # usual case: no padding
            blocked = self.blocked[i0:i0 + CROP, j0:j0 + CROP];free = self.free[i0:i0 + CROP, j0:j0 + CROP]
            state = (free > 0).astype(np.uint8);state[blocked > 0] = BLOCKED
            seen = self.last_visible[i0:i0 + CROP, j0:j0 + CROP]
        else:
            state = np.zeros((CROP, CROP), np.uint8);seen = np.full((CROP, CROP), -np.inf, np.float32)
            a0, b0 = max(0, i0), max(0, j0);a1, b1 = min(GRID, i0 + CROP), min(GRID, j0 + CROP)
            if a1 > a0 and b1 > b0:
                blocked = self.blocked[a0:a1, b0:b1] > 0;free = self.free[a0:a1, b0:b1] > 0
                state[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = np.where(blocked, BLOCKED, np.where(free, FREE, 0))
                seen[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = self.last_visible[a0:a1, b0:b1]
        assert VIS_POOL == 2   # 2x2 max pool via strided slices (exact, avoids a slow strided reduce)
        pooled = np.maximum(np.maximum(seen[0::2, 0::2], seen[1::2, 0::2]), np.maximum(seen[0::2, 1::2], seen[1::2, 1::2]))
        age = self.timestamp - pooled
        with np.errstate(invalid='ignore'):
            code = np.minimum(254, np.round(age / AGE_STEP))
        code[~np.isfinite(age)] = 255
        return state, code.astype(np.uint8), corner

    def robot_history(self):
        times = np.array([r[0] for r in self.robot]);rows = np.array([r[1:4] for r in self.robot])
        wanted = self.timestamp - ROBOT_LAGS
        index = np.clip(np.searchsorted(times, wanted + 1e-9) - 1, 0, len(times) - 1)
        return rows[index].astype(np.float32)

    def key_features(self):
        """Per-key arrays in first-seen order (legacy ``features.extract`` key order)."""
        keys = tuple(self.keys)
        n = len(keys);now = self.timestamp
        hist = np.zeros((n, HISTORY, 3), np.float32);mask = np.zeros((n, HISTORY), bool)
        last = np.zeros((n, 2), np.float64);last_t = np.zeros(n, np.float64)
        velocity = np.zeros((n, 2), np.float32);velocity_valid = np.zeros(n, bool)
        count = np.zeros(n, np.int32);first = np.zeros(n, np.float64)
        for i, key in enumerate(keys):
            m = self.keys[key];rows = m['rows']
            tail = rows[-HISTORY:];k = len(tail)
            hist[i, HISTORY - k:] = [(r[0] - now, r[1], r[2]) for r in tail];mask[i, HISTORY - k:] = True
            last[i] = rows[-1][1:];last_t[i] = rows[-1][0]
            if len(rows) >= 2 and 1e-6 < rows[-1][0] - rows[-2][0] <= .5:
                dt = rows[-1][0] - rows[-2][0]
                velocity[i] = ((rows[-1][1] - rows[-2][1]) / dt, (rows[-1][2] - rows[-2][2]) / dt);velocity_valid[i] = True
            elif m['tracker_velocity'] is not None:
                velocity[i] = m['tracker_velocity'];velocity_valid[i] = True
            count[i] = m['n'];first[i] = m['first']
        return dict(keys=keys, hist=hist, hist_mask=mask, last=last, last_t=last_t, velocity=velocity,
                    velocity_valid=velocity_valid, count=count, first=first)

    def summary(self):
        return np.array([self.frames, self.valid_frames, self.empty_frames, self.max_simultaneous], np.float32)
