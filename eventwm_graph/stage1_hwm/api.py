"""Online Stage 1 human world model with the legacy ``predict`` signature.

``HumanWorldModel.predict(history, obs, partial_graph=None, horizon=3.,
worlds=40, seed=None, actions=(), executed_action=None)`` accepts the raw
arrived frames (list of observation dicts) or a legacy ``OnlineContext`` and
returns ``HumanWorlds``: a ``JointHumanWorlds``-compatible IID joint bank
(positions [W,T,N,2] world XZ metres, present [W,N], relative times, uniform
weights) plus the exact count law and per-key existence probabilities.

Legal inputs only: frames are replayed through ``LegalState`` (incrementally
when a call extends the previously seen history). ``partial_graph`` and
``actions`` are accepted for interface compatibility; static evidence is read
from the frames' own certificates, which define the same graph.
"""
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Tuple
import numpy as np
import torch

from .legal_state import LegalState, camera_yaw, HISTORY
from .data import collate, to_torch
from .model import HWM, STEP, SEGMENTS
from .sampler import sample

VERSION = 'stage1_hwm_joint_human_worlds_1'
DEFAULT_RESOLUTION = .1016884229464652   # dataset static-certificate grid; overridden by partial_graph


class _LiveStore:
    """One-anchor view with the ``Store`` fields ``collate`` reads (no labels)."""

    def __init__(self, state, episode_start):
        r = state.robot[-1]
        frame = state.last_frame
        mp, vis, corner = state.crops()
        self.a = dict(robot=np.array([[r[1], r[2], r[3], r[4]]], np.float32), robot_hist=state.robot_history()[None],
                      cam=np.array([[frame['camera_origin'][0], frame['camera_origin'][2], camera_yaw(frame['camera_rotation'])]], np.float32),
                      corner=corner[None], map=mp[None], vis=vis[None], summary=state.summary()[None],
                      t=np.array([state.timestamp]), anchor_root=np.zeros(1, int), tau_mask=np.zeros((1, 13), bool))
        kf = state.key_features();n = len(kf['keys'])
        self.keys = kf['keys']
        self.k = {'k_' + name: kf[name] for name in ('hist', 'hist_mask', 'last', 'last_t', 'velocity', 'velocity_valid', 'count', 'first')}
        self.k['k_legacy_presence'] = np.full(n, np.nan, np.float32);self.k['k_phys'] = np.full(n, -1, np.int16)
        self.h = dict(h_hum=np.zeros((0, 13, 2), np.float32))
        self.key_off = np.array([0, n]);self.hum_off = np.array([0, 0])
        self.episode_start = np.array([episode_start])


@dataclass(frozen=True)
class HumanWorlds:
    timestamp: float
    times: np.ndarray
    weights: np.ndarray
    positions: np.ndarray          # [W,T,N,2] world XZ
    present: np.ndarray            # [W,N]
    track_keys: Tuple[str, ...]
    unseen: np.ndarray             # [N] True for residual (never-keyed) slots
    poisson_rate: float            # expected residual count (legacy field name)
    support: Tuple[str, ...]
    minimum_count: int
    count_log_probability: np.ndarray   # exact log P(total present = c)
    existence_probability: np.ndarray   # per track key
    knots: np.ndarray = field(repr=False, default=None)   # [W,N,13,2] world, 0.25 s
    law: object = None
    version: str = VERSION

    def positions_at(self, relative_times):
        """Exact piecewise-linear evaluation at any times in [0, 3.028] s."""
        t = np.asarray(relative_times, np.float64)
        if t.ndim != 1 or np.any(t < 0) or np.any(t > 3.028001):raise ValueError('times outside [0, 3.028] s')
        s = np.clip(np.floor(t / STEP).astype(int), 0, SEGMENTS - 2);f = (t - s * STEP) / STEP
        a = self.knots[:, :, s];b = self.knots[:, :, s + 1]
        return (a + f[None, None, :, None] * (b - a)).transpose(0, 2, 1, 3)

    def validate(self):
        k, t, n, d = self.positions.shape
        assert d == 2 and self.present.shape == (k, n) and len(self.times) == t and len(self.track_keys) == n
        assert np.all(np.diff(self.times) > 0) and abs(self.times[0]) < 1e-8 and np.isfinite(self.positions).all()
        assert abs(self.weights.sum() - 1) < 1e-8
        return self


class HumanWorldModel:
    def __init__(self, checkpoint, device='cpu'):
        state = torch.load(checkpoint, map_location='cpu')
        self.network = HWM(**state['config'].get('model', {}))
        self.network.load_state_dict(state['model']);self.network.eval().requires_grad_(False)
        self.device = torch.device(device);self.network.to(self.device)
        self.checkpoint_sha256 = hashlib.sha256(open(checkpoint, 'rb').read()).hexdigest()
        self._state = None;self._first = None;self._episode_start = None
        self._resolution = DEFAULT_RESOLUTION;self._resolution_used = None;self._last = None
        self.last_profile = {}

    # ----- legal state maintenance ---------------------------------------------
    @staticmethod
    def _fingerprint(frame):
        return (float(frame['timestamp']), tuple(float(v) for v in frame['robot_pose']), tuple(float(v) for v in frame['camera_origin']))

    def _sync(self, frames, resolution):
        """Reuse the incremental state only when this history extends the one already ingested."""
        reuse = False
        if self._state is not None and self._resolution_used == resolution and self._fingerprint(frames[0]) == self._first:
            last = self._state.timestamp
            at = [i for i, f in enumerate(frames) if abs(float(f['timestamp']) - last) < 1e-9]
            reuse = len(at) == 1 and self._fingerprint(frames[at[0]]) == self._last
        if reuse:
            new = frames[at[0] + 1:]
        else:
            self._state = LegalState(resolution);self._first = self._fingerprint(frames[0])
            self._episode_start = float(frames[0]['timestamp']);self._resolution_used = resolution
            new = list(frames)
        for f in new:
            self._state.ingest(f);self._state.last_frame = f
        self._last = self._fingerprint(frames[-1])

    def predict(self, history, obs, partial_graph=None, horizon=3., worlds=40, seed=None, actions=(), executed_action=None, times=None):
        if not 0 < horizon <= 3.028001 or worlds < 2:raise ValueError('supported horizon (0,3] and >=2 joint worlds required')
        start = time.perf_counter()
        frames = list(history.history) if hasattr(history, 'history') and not isinstance(history, (list, tuple)) else list(history)
        if not frames or float(obs['timestamp']) > float(frames[-1]['timestamp']) + 1e-9:frames.append(obs)
        elif abs(float(obs['timestamp']) - float(frames[-1]['timestamp'])) > 1e-9:raise ValueError('observation older than history')
        if partial_graph is not None and abs(float(partial_graph['timestamp']) - float(frames[-1]['timestamp'])) > 1e-7:
            raise ValueError('graph clock differs from arrived observation')
        if partial_graph is not None:self._resolution = float(partial_graph['resolution'])
        self._sync(frames, self._resolution)
        synced = time.perf_counter()
        live = _LiveStore(self._state, self._episode_start)
        batch = to_torch(collate(live, [0]), self.device)
        if seed is None:
            h = hashlib.sha256(b'stage1-hwm-1');h.update(np.float64(self._state.timestamp).tobytes())
            for k in ('last', 'last_t'):h.update(np.ascontiguousarray(live.k['k_' + k]).tobytes())
            h.update(live.a['robot'].tobytes());seed = int.from_bytes(h.digest()[:8], 'little')
        g = torch.Generator(device=self.device);g.manual_seed(int(seed) % (2 ** 63))
        out = sample(self.network, batch, worlds, g)
        sampled = time.perf_counter()
        knots = out['knots'][0].cpu().numpy().astype(np.float64);present = out['present'][0].cpu().numpy()
        rot = batch['rot'][0].cpu().numpy().astype(np.float64);origin = batch['origin'][0].cpu().numpy().astype(np.float64)
        knots = np.einsum('ji,...j->...i', rot, knots) + origin                 # world [W,N,13,2]
        n_keys = len(live.keys);K = out['n_keys']
        col = batch['key_column'][0].cpu().numpy()
        order = [int(np.flatnonzero(col == i)[0]) for i in range(n_keys) if np.any(col == i)]
        hidden = list(range(K, knots.shape[1]))
        keep = order + hidden
        knots = knots[:, keep];present = present[:, keep]
        if times is None:times = np.linspace(0., float(horizon), 64)
        times = np.asarray(times, np.float64)
        s = np.clip(np.floor(times / STEP).astype(int), 0, SEGMENTS - 2);f = (times - s * STEP) / STEP
        positions = (knots[:, :, s] + f[None, None, :, None] * (knots[:, :, s + 1] - knots[:, :, s])).transpose(0, 2, 1, 3)
        keys = tuple('track:' + live.keys[c] for c in [int(col[i]) for i in order]) + tuple('residual:%d' % i for i in range(len(hidden)))
        p_exist = out['p_exist'][0].cpu().numpy()[order]
        count_lp = out['count_logp'][0].cpu().numpy()
        result = HumanWorlds(float(self._state.timestamp), times, np.full(worlds, 1. / worlds), positions, present, keys,
                             np.r_[np.zeros(len(order), bool), np.ones(len(hidden), bool)],
                             float(present[:, len(order):].sum(1).mean()) if hidden else 0.,
                             ('IID_JOINT_WORLDS', 'TRACK_KEYS_ARE_NOT_IDENTITIES', 'UNKNOWN_GEOMETRY_LEARNED_PRIOR',
                              'MONTE_CARLO_BANK_NOT_SAFETY_CERTIFICATE', 'HUMANS_EXOGENOUS_TO_ROBOT'),
                             int(self._state.max_simultaneous), count_lp, p_exist, knots=knots).validate()
        end = time.perf_counter()
        self.last_profile = dict(state_ms=(synced - start) * 1000, sample_ms=(sampled - synced) * 1000,
                                 output_ms=(end - sampled) * 1000, complete_ms=(end - start) * 1000)
        return result
