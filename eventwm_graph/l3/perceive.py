"""Perception-faithful nominal people for imagination (Layer 3 option).

The Sandbox's imagined planner normally perceives every tracked person of the nominal world at every imagined step.
The real planner perceives people only through the 90 deg jaw camera: a person is detected when inside the frustum
(10 m) with a clear line of sight, and a lost track is extrapolated at constant velocity for at most 2 s
(``control.human_estimates``).  ``VisibleWorld`` reproduces that inside the imagination: it holds one world's
people (tracked AND hidden slots of a Stage 1 world, or the analysis-only true future), and at every imagined step
returns the legal-format tracks the robot would have from its IMAGINED pose (frustum, legal-map occlusion, last-seen
memory).  State resets when a rollout starts (k == 0) and carries through the default continuation.

Layer 3 option ``L3_CAM_TRACKER=1`` (default off; per instance ``VisibleWorld(..., tracker=True)``): the imagined
camera + tracker EMULATE the robot's real perception pipeline (``scripts/collector/observation.py``, fast path
``querygen/fastsense.py``) instead of the simplified memory above -- the robot's own software and sensor spec, so
legal:

* camera: the jaw camera's pose relative to the base, read from the legal observation row at t (``camera_origin`` /
  ``camera_rotation`` vs ``robot_pose``; ``CameraRig``) and carried rigidly with the imagined base.  The arm, and with
  it the camera, is fixed relative to the base within an episode but differs between episodes (train state bank,
  53 episodes: 0.61-0.86 m above the base pose, yaw offset -15..+17 deg, pitch -12..+32 deg), so no fixed mount fits;
* detection: the person's torso point (base + ``torso_height_m`` = 1 m; the base on the robot's floor -- tracked
  people at their legal track height, everyone else at the robot's base height, as real rows show: person base y -
  robot base y median 0.00 m, p95 0.19 m) inside the 3-D 90 x 90 deg / 10 m frustum (the real ``in_frustum``), and a
  clear line of sight on the legal map (the real 3-D ray test needs scene geometry, which is not legal);
* tracking: the REAL ``LegalHumanTracker`` (greedy gated association, position = last detection, velocity only from
  detections <= 0.5 s apart, new tracks without velocity, sequential ids, 2 s age limit) seeded at k = 0 with the
  robot's legal tracks at t (ids, positions, velocities, last-seen times; the next id beyond every id in the legal
  track log -- exactly the real tracker's state at t).
"""
import copy
import math
import os
import re

import numpy as np

from eventwm_graph.querygen.dynamics import STEP_SECONDS

from .infofeat import _blocked_lookup, _los

HFOV = math.radians(90.)
RANGE = 10.
# camera mount (Layer 3 option L3_CAM_MOUNT=1, default off: the camera at the base centre, looking along the base
# heading).  The real jaw depth camera (robot spec, measured from the sensor rows' camera_origin / camera_rotation):
# 0.52 m ahead of the base centre, 0.02 m to its right, yawed ~4.5 deg to the right -- people beside the robot's front
# that a centre camera would see are outside the real frustum.
CAM_MOUNT = __import__('os').environ.get('L3_CAM_MOUNT') == '1'
CAM_FWD, CAM_RIGHT, CAM_YAW = 0.52, 0.02, math.radians(-4.5)

# tracker emulation (L3_CAM_TRACKER=1, default off): see the module docstring
CAM_TRACKER = os.environ.get('L3_CAM_TRACKER') == '1'
# the jaw depth sensor (falcon_{hm3d,mp3d}_task.yaml: 256 x 256 px, hfov 90 deg -> vfov 90 deg, max depth 10 m), as
# ``ObservationConfig.from_depth_sensor`` builds the real sensor's config; tracker parameters = the dataclass defaults
SENSOR_FOV_DEG = (90., 90.)
SENSOR_RANGE_M = 10.
# fallback camera rig when no legal observation row is given: per-episode medians over the TRAIN state bank
# (output/statebank/orca_train, 53 episodes; forward / right / up offsets from the base pose in m, yaw offset (left +)
# and pitch (up +) in deg).  Callers that hold the decision's legal row (DecisionState / runtime state) pass it.
RIG_FALLBACK = dict(fwd=0.517, right=0.005, up=0.731, yaw=-0.05, pitch=9.18)
_TRACK_ID = re.compile(r'^track_(\d+)$')


def _rot_y(a):
    """Rotation about world +y by ``a`` (the base frame: local +x = forward (cos yaw, -sin yaw), +z = right)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _legal_row(source):
    """The latest legal observation row (dict with robot_pose / camera_origin / camera_rotation) held by ``source``:
    a row or Layer 2 frame, a ``DecisionState`` (its last frame) or a querygen runtime state (its sensor's last row)."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source if 'camera_origin' in source else None
    frames = getattr(source, 'frames', None)
    if frames:
        return _legal_row(frames[-1])
    sensor = getattr(source, 'sensor', None)
    if isinstance(sensor, (tuple, list)) and len(sensor) >= 4:
        return _legal_row(sensor[3])
    return None


class CameraRig:
    """The jaw camera's pose in the base frame (``offset`` [3], ``rotation`` [3, 3] camera-to-base; the camera looks
    down its local -Z, as ``scripts.collector.observation``): ``pose(base)`` -> (origin, camera-to-world rotation)
    for an imagined base pose [x, y, z, yaw]."""

    def __init__(self, offset, rotation, source='row'):
        self.offset = np.asarray(offset, np.float64).reshape(3)
        self.rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        self.source = source

    @classmethod
    def from_row(cls, row):
        p = np.asarray(row['robot_pose'], np.float64)
        Rb = _rot_y(float(p[3]))
        o = np.asarray(row['camera_origin'], np.float64)
        R = np.asarray(row['camera_rotation'], np.float64)
        return cls(Rb.T @ (o-p[:3]), Rb.T @ R, 'row')

    @classmethod
    def fallback(cls):
        f = RIG_FALLBACK
        yaw, pitch = math.radians(f['yaw']), math.radians(f['pitch'])
        c, s = math.cos(pitch), math.sin(pitch)
        Rx = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        return cls([f['fwd'], f['up'], f['right']], _rot_y(yaw-math.pi/2) @ Rx, 'fallback')

    @classmethod
    def of(cls, source):
        row = _legal_row(source)
        return cls.from_row(row) if row is not None else cls.fallback()

    def pose(self, base):
        Rb = _rot_y(float(base[3]))
        return np.asarray(base[:3], np.float64)+Rb @ self.offset, Rb @ self.rotation


def next_track_id(legal):
    """The real tracker's next sequential id at t: one beyond every id it has ever published (legal track log)."""
    best = -1
    for t in getattr(legal, 'tracks', ()) or ():
        m = _TRACK_ID.match(str(t.get('track_id')))
        if m:
            best = max(best, int(m.group(1)))
    for _, trs in getattr(legal, 'track_log', ()) or ():
        for t in trs:
            m = _TRACK_ID.match(str(t.get('track_id')))
            if m:
                best = max(best, int(m.group(1)))
    return best+1


def sensor_config():
    from scripts.collector.observation import ObservationConfig
    return ObservationConfig(SENSOR_FOV_DEG[0], SENSOR_FOV_DEG[1], SENSOR_RANGE_M)


def seeded_tracker(legal, config=None, next_id=None):
    """The REAL ``LegalHumanTracker`` in the state the robot's tracker had at t: its published tracks (``legal.tracks``
    -- every field the tracker keeps), clock t and next id."""
    from scripts.collector.observation import LegalHumanTracker
    tr = LegalHumanTracker(config or sensor_config())
    for t in legal.tracks:
        last = float(t['last_seen_time'])
        d = dict(track_id=str(t['track_id']), position=[float(v) for v in t['position']],
                 estimated_velocity=copy.deepcopy(t.get('estimated_velocity')),
                 velocity_valid=bool(t.get('velocity_valid', False)), visible_now=bool(t.get('visible_now', False)),
                 valid=bool(t.get('valid', True)), first_seen_time=float(t.get('first_seen_time', last)),
                 last_seen_time=last, observation_age=float(t.get('observation_age', 0.)),
                 num_observations=int(t.get('num_observations', 1)))
        tr.tracks[d['track_id']] = d
    tr.timestamp = float(legal.time)
    tr.next_id = next_track_id(legal) if next_id is None else int(next_id)
    return tr


def camera_pose(pose):
    """(x, z, yaw) of the imagined camera for a base pose [x, y, z, yaw]."""
    if not CAM_MOUNT:
        return float(pose[0]), float(pose[2]), float(pose[3])
    c, s = math.cos(float(pose[3])), math.sin(float(pose[3]))
    return (float(pose[0])+CAM_FWD*c+CAM_RIGHT*s, float(pose[2])-CAM_FWD*s+CAM_RIGHT*c, float(pose[3])+CAM_YAW)


class Discovery:
    """Progressive map discovery inside imagination: at every imagined step the cells of a ``truth`` map inside the
    imagined camera frustum (``rng`` m, line of sight through the truth map's BLOCKED cells) are written into ``work``
    -- the imagined legal map the executor plans on -- as the real sensor certifies cells while the robot moves.
    ANALYSIS: truth = a privileged map (the real robot's map at the end of the window); a legal variant would use a
    predicted map (Layer 2).  ``work`` is modified in place (the imagined legal state holds it); ``reset`` restores
    the map at t."""

    def __init__(self, work, truth, rng=5.0):
        self.work, self.truth, self.rng = work, truth, float(rng)
        self.base = work.copy()
        self.blocked = _blocked_lookup(truth)
        ii, jj = np.nonzero(truth.state != 0)
        self.ki, self.kj = ii+truth.ox, jj+truth.oz
        self.cx, self.cz = (self.ki+.5)*truth.res, (self.kj+.5)*truth.res
        self.cs, self.ch = truth.state[ii, jj], truth.height[ii, jj]
        self.revealed = 0

    def reset(self):
        b = self.base
        w = self.work
        w.ox, w.oz = b.ox, b.oz
        w.state, w.height = b.state.copy(), b.height.copy()
        w.first_seen, w.last_seen = b.first_seen.copy(), b.last_seen.copy()
        w.revision = b.revision
        self.revealed = 0

    def reveal(self, pose, t):
        if not len(self.cx):
            return
        px, pz, pyaw = camera_pose(pose)
        dx, dz = self.cx-px, self.cz-pz
        near = np.flatnonzero((np.abs(dx) <= self.rng) & (np.abs(dz) <= self.rng))
        if not len(near):
            return
        dx, dz = dx[near], dz[near]
        fw = dx*math.cos(pyaw)-dz*math.sin(pyaw)
        lf = -dx*math.sin(pyaw)-dz*math.cos(pyaw)
        keep = (np.hypot(dx, dz) <= self.rng) & (np.abs(np.arctan2(lf, fw)) <= HFOV/2)
        idx = near[keep]
        if not len(idx):
            return
        w = self.work
        keys = np.stack([self.ki[idx], self.kj[idx]], 1)
        w._ensure(keys)
        wi, wj = keys[:, 0]-w.ox, keys[:, 1]-w.oz
        diff = w.state[wi, wj] != self.cs[idx]
        if not diff.any():
            return
        idx, wi, wj = idx[diff], wi[diff], wj[diff]
        clear = _los(self.blocked, np.full(len(idx), px), np.full(len(idx), pz), self.cx[idx], self.cz[idx])
        if not clear.any():
            return
        idx, wi, wj = idx[clear], wi[clear], wj[clear]
        w.state[wi, wj] = self.cs[idx]
        w.height[wi, wj] = self.ch[idx]
        fs = w.first_seen[wi, wj]
        w.first_seen[wi, wj] = np.where(np.isnan(fs), t, fs)
        w.last_seen[wi, wj] = t
        w.revision += 1
        self.revealed += len(idx)


class VisibleWorld:
    def __init__(self, legal, bank, world, steps, step_s=STEP_SECONDS, camera=None, tracker=None):
        """``camera``: the decision's legal observation row (or a DecisionState / runtime state holding it) for the
        tracker emulation's camera rig; ``tracker``: emulate the real camera + tracker (default ``L3_CAM_TRACKER``).
        Both are ignored by the default camera model."""
        self.legal = legal
        self.now = float(legal.time)
        self.steps = int(steps)
        self.step = float(step_s)
        keys = [str(k) for k in getattr(bank, 'track_keys', ())]
        rel = np.arange(0, self.steps+1)*self.step
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*self.step)
        inside = rel[(rel > 0) & (rel <= horizon+1e-9)]
        if len(keys):
            P = np.asarray(bank.sandbox_positions(inside), np.float64)[int(world)]        # [Ti, N, 2]
            cur = np.asarray(bank.sandbox_positions(np.asarray([1e-3])), np.float64)[int(world), 0]
            P = np.concatenate([cur[None], P], 0)
            if len(P) < len(rel):
                vel = (P[-1]-P[-2])/self.step if len(P) >= 2 else np.zeros_like(P[-1])
                extra = np.arange(1, len(rel)-len(P)+1)*self.step
                P = np.concatenate([P, P[-1][None]+extra[:, None, None]*vel[None]], 0)
        else:
            P = np.zeros((len(rel), 0, 2))
        self.P = P                                                                          # [steps+1, N, 2]
        self.ids = [k[len('track:'):] if k.startswith('track:') else 'h%d' % n for n, k in enumerate(keys)]
        y = float(legal.pose[1])+0.9
        self.y = y
        # what the nominal-humans interface exposes for clearance statistics
        N = self.P.shape[1]
        self.positions = np.zeros((self.steps+1, N, 3))
        self.positions[:, :, 0] = np.where(np.isfinite(self.P[..., 0]), self.P[..., 0], 1e6)
        self.positions[:, :, 1] = y
        self.positions[:, :, 2] = np.where(np.isfinite(self.P[..., 1]), self.P[..., 1], 1e6)
        self.blocked = _blocked_lookup(legal.map)
        self._init = {str(t['track_id']): t for t in legal.tracks if t.get('valid', True)}
        self.mode = 'visible'
        self.fallback = 0
        self.discovery = None            # optional Discovery: progressive map certification from the imagined pose
        self.rec = None                  # ANALYSIS (shortprobe): a list receiving (k, tracks) of every tracks_at call
        self.emulate = CAM_TRACKER if tracker is None else bool(tracker)
        if self.emulate:
            # L3_CAM_TRACKER: the real camera rig + detection rules + LegalHumanTracker (module docstring)
            self.config = sensor_config()
            self.rig = CameraRig.of(camera)
            self.next_id = next_track_id(legal)
            th = {str(t['track_id']): float(t['position'][1]) for t in legal.tracks}
            self.hy = np.asarray([th.get(i, float(legal.pose[1])) for i in self.ids], np.float64)
        self.reset()

    def fork(self, discovery=None, blocked=None):
        """A copy sharing the people's positions, with its own tracker memory (and optional map discovery / occlusion
        map) -- one per candidate when every candidate discovers its own map."""
        import copy
        w = copy.copy(self)
        w.discovery = discovery
        w.rec = None
        if blocked is not None:
            w.blocked = blocked
        w.reset()
        return w

    def reset(self):
        if self.emulate:
            # the real tracker as it was at t (its own memory per rollout)
            self.tr = seeded_tracker(self.legal, self.config, self.next_id)
            from scripts.collector.observation import legal_tracks
            self.out = self._mark(legal_tracks(self.tr.tracks.values()))
            return
        # last-seen memory: id -> (time, position xz, velocity xz); start from the robot's real tracks
        self.mem = {}
        for tid, t in self._init.items():
            p = np.asarray(t['position'], float)
            v = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), float)
            self.mem[tid] = (float(t['last_seen_time']), np.array([p[0], p[2]]), np.array([v[0], v[2]]))

    def tracks(self, k):
        """Fallback (no pose): the remembered tracks at step k (k == 0 starts a new rollout)."""
        if int(k) == 0:
            self.reset()
        if self.emulate:
            return self.out
        return self._format(int(k), set())

    @staticmethod
    def _mark(tracks):
        for t in tracks:
            t['hypothetical'] = True
        return tracks

    def _emulate(self, k, pose, t_now):
        """L3_CAM_TRACKER: the real detection rules from the imagined camera, then one real tracker update."""
        if k == 0 or t_now <= self.tr.timestamp:
            return self.out                       # the tracker already holds time t (the legal tracks)
        from eventwm_graph.partial.observer import in_frustum
        from scripts.collector.observation import HumanDetection
        dets = []
        if self.P.shape[1]:
            pts = self.P[k]
            idx = np.flatnonzero(np.isfinite(pts).all(1))
            if len(idx):
                origin, R = self.rig.pose(pose)
                base = np.stack([pts[idx, 0], self.hy[idx], pts[idx, 1]], 1)
                torso = base.copy()
                torso[:, 1] += self.config.torso_height_m
                inside = in_frustum(torso, origin, R, self.config)
                idx, base = idx[inside], base[inside]
                if len(idx):
                    clear = _los(self.blocked, np.full(len(idx), origin[0]), np.full(len(idx), origin[2]),
                                 pts[idx, 0], pts[idx, 1])
                    dets = [HumanDetection(t_now, tuple(float(v) for v in b)) for b in base[clear]]
        self.out = self._mark(self.tr.update(t_now, sorted(dets, key=lambda d: d.position)))
        return self.out

    def tracks_at(self, k, pose):
        k = int(min(k, self.steps))
        if k == 0:
            self.reset()
            if self.discovery is not None:
                self.discovery.reset()
        t_now = self.now+k*self.step
        if self.discovery is not None and k > 0:
            self.discovery.reveal(pose, t_now)
        if self.emulate:
            out = self._emulate(k, pose, t_now)
            if self.rec is not None:
                self.rec.append((k, out))
            return out
        visible = set()
        if self.P.shape[1]:
            pts = self.P[k]
            ok = np.isfinite(pts).all(1)
            cx, cz, cyaw = camera_pose(pose)
            dx, dz = pts[:, 0]-cx, pts[:, 1]-cz
            fw = dx*math.cos(cyaw)-dz*math.sin(cyaw)
            lf = -dx*math.sin(cyaw)-dz*math.cos(cyaw)
            inside = ok & (np.abs(np.arctan2(lf, fw)) <= HFOV/2) & (np.hypot(dx, dz) <= RANGE)
            idx = np.flatnonzero(inside)
            if len(idx):
                clear = _los(self.blocked, np.full(len(idx), cx), np.full(len(idx), cz), pts[idx, 0], pts[idx, 1])
                for n in idx[clear]:
                    tid = self.ids[n]
                    prev = self.mem.get(tid)
                    v = (pts[n]-prev[1])/max(t_now-prev[0], self.step) if prev is not None else np.zeros(2)
                    self.mem[tid] = (t_now, pts[n].copy(), v)
                    visible.add(tid)
        out = self._format(k, visible)
        if self.rec is not None:
            self.rec.append((k, out))
        return out

    def _format(self, k, visible):
        t_now = self.now+k*self.step
        out = []
        for tid, (ts, p, v) in self.mem.items():
            if t_now-ts > 2.0:
                continue
            out.append(dict(track_id=tid, position=[float(p[0]), self.y, float(p[1])],
                            estimated_velocity=[float(v[0]), 0., float(v[1])], velocity_valid=True,
                            visible_now=tid in visible, valid=True, first_seen_time=self.now, last_seen_time=ts,
                            observation_age=t_now-ts, num_observations=5, hypothetical=True))
        return out


class TimedReveal:
    """ANALYSIS (privileged): the real robot's map stream replayed by time -- at imagined time t, every cell the real
    branch's legal map ``final`` had certified by t (``first_seen <= t``) is written into ``work`` with its final state,
    whatever the imagined pose (fidelity ladder: map knowledge as the real executor had it)."""

    def __init__(self, work, final, only=None):
        self.work, self.final = work, final
        self.base = work.copy()
        keep = (final.state != 0) if only is None else (final.state == int(only))     # only: 1 FREE / 2 BLOCKED cells
        ii, jj = np.nonzero(keep & np.isfinite(final.first_seen))
        order = np.argsort(final.first_seen[ii, jj], kind='stable')
        ii, jj = ii[order], jj[order]
        self.t = final.first_seen[ii, jj].astype(np.float64)
        self.ki, self.kj = ii+final.ox, jj+final.oz
        self.cs, self.ch = final.state[ii, jj], final.height[ii, jj]
        self.n = 0
        self.revealed = 0

    def reset(self):
        b, w = self.base, self.work
        w.ox, w.oz = b.ox, b.oz
        w.state, w.height = b.state.copy(), b.height.copy()
        w.first_seen, w.last_seen = b.first_seen.copy(), b.last_seen.copy()
        w.revision = b.revision
        self.n = 0
        self.revealed = 0

    def reveal(self, pose, t):
        m = int(np.searchsorted(self.t, float(t)+1e-6, side='right'))
        if m <= self.n:
            return
        sel = slice(self.n, m)
        self.n = m
        w = self.work
        keys = np.stack([self.ki[sel], self.kj[sel]], 1)
        w._ensure(keys)
        wi, wj = keys[:, 0]-w.ox, keys[:, 1]-w.oz
        diff = w.state[wi, wj] != self.cs[sel]
        if not diff.any():
            return
        wi, wj = wi[diff], wj[diff]
        w.state[wi, wj] = self.cs[sel][diff]
        w.height[wi, wj] = self.ch[sel][diff]
        fs = w.first_seen[wi, wj]
        w.first_seen[wi, wj] = np.where(np.isnan(fs), t, fs)
        w.last_seen[wi, wj] = t
        w.revision += 1
        self.revealed += int(diff.sum())


class ReplayTracks:
    """ANALYSIS (privileged): the REAL tracker's output replayed into imagination -- at imagined step k the executor
    receives the tracks the real robot held k steps after the decision (the real branch's ``track_log``), whatever
    the imagined pose (fidelity ladder: perception as the real executor had it).  Optional ``discovery``
    (``TimedReveal`` / ``Discovery``) is advanced every step as in ``VisibleWorld``."""

    def __init__(self, track_log, start_index, steps, now, step_s=STEP_SECONDS):
        self.log = [tr for _, tr in track_log[start_index:]] or [[]]
        self.steps = int(steps)
        self.now = float(now)
        self.step = float(step_s)
        self.positions = np.zeros((self.steps+1, 0, 3))
        self.discovery = None
        self.rec = None                  # ANALYSIS (shortprobe): a list receiving (k, tracks) of every tracks_at call

    def fork(self, discovery=None, blocked=None):
        import copy
        w = copy.copy(self)
        w.discovery = discovery
        w.rec = None
        return w

    def tracks(self, k):
        return list(self.log[min(int(k), len(self.log)-1)])

    def tracks_at(self, k, pose):
        k = int(min(k, self.steps))
        if k == 0 and self.discovery is not None:
            self.discovery.reset()
        if self.discovery is not None and k > 0:
            self.discovery.reveal(pose, self.now+k*self.step)
        out = self.tracks(k)
        if self.rec is not None:
            self.rec.append((k, out))
        return out


def predicted_truth(legal_map, delta):
    """LEGAL truth map for ``Discovery``: the legal map at t with Layer 2's (Stage 2) predicted states written into its
    UNKNOWN cells (``delta``: a Stage 2 increment -- candidate_cells keys in the legal map's key space, states 1 FREE /
    2 BLOCKED).  Hypothetical; never merged into the real legal state."""
    truth = legal_map.copy()
    if delta is None:
        return truth
    st = np.asarray(delta['states'])
    m = st > 0
    if not m.any():
        return truth
    keys = np.asarray(delta['candidate_cells'], np.int64)[m]
    truth._ensure(keys)
    i, j = keys[:, 0]-truth.ox, keys[:, 1]-truth.oz
    unknown = truth.state[i, j] == 0
    truth.state[i[unknown], j[unknown]] = st[m][unknown].astype(np.uint8)
    return truth


def predicted_truth_union(legal_map, deltas):
    """LEGAL truth map from SEVERAL Stage 2 increments (every candidate's predicted cells, one predicted local map):
    predicted states are written into the legal map's UNKNOWN cells; where predictions disagree BLOCKED wins
    (conservative for planning and contact).  Hypothetical; never merged into the real legal state."""
    truth = legal_map.copy()
    for delta in deltas:
        if delta is None:
            continue
        st = np.asarray(delta['states'])
        m = st > 0
        if not m.any():
            continue
        keys = np.asarray(delta['candidate_cells'], np.int64)[m]
        truth._ensure(keys)
        i, j = keys[:, 0]-truth.ox, keys[:, 1]-truth.oz
        unknown = legal_map.state_at_keys(keys) == 0 if hasattr(legal_map, 'state_at_keys') else None
        if unknown is None:
            unknown = np.zeros(len(keys), bool)
            li, lj = keys[:, 0]-legal_map.ox, keys[:, 1]-legal_map.oz
            ok = (li >= 0) & (li < legal_map.state.shape[0]) & (lj >= 0) & (lj < legal_map.state.shape[1])
            unknown[~ok] = True
            unknown[ok] = legal_map.state[li[ok], lj[ok]] == 0
        cur = truth.state[i, j]
        new = st[m].astype(np.uint8)
        write = unknown & ((cur == 0) | (new == 2))
        truth.state[i[write], j[write]] = new[write]
    return truth
