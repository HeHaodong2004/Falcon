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
import json
from pathlib import Path
import math
import time
from typing import Tuple
import numpy as np
import torch

from .legal_state import LegalState, camera_yaw, HISTORY
from .data import collate, collate_one, to_torch
from .model import HWM, STEP, SEGMENTS, fold_batchnorm
from .sampler import sample
from eventwm_graph.l2_two_stage.types import JointHumanWorlds

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
class HumanWorlds(JointHumanWorlds):
    """``JointHumanWorlds``-compatible IID joint bank with the exact count law.

    Inherited fields: timestamp, times [T] (relative s), weights [W], positions [W,T,N,2]
    world XZ, present [W,N], track_keys, unseen [N] (residual slots), poisson_rate (here:
    the model's expected residual count before the legal truncation), support, minimum_count (legal lower bound
    enforced exactly), law (None: the law is the network + this sampler), version.
    """
    count_log_probability: np.ndarray = None   # exact log P(total present = c), c = 0..23
    existence_probability: np.ndarray = None   # per track key
    knots: np.ndarray = field(repr=False, default=None)   # [W,N,13,2] world, 0.25 s knots

    def sandbox_positions(self, relative_times):
        """[W,T,N,2] world XZ at exact relative times with absent people set to +inf (L2.5 convention)."""
        p = np.array(self.positions_at(relative_times), np.float64)
        p[~np.broadcast_to(self.present[:, None, :], p.shape[:3])] = np.inf
        return p

    def evaluate_candidates(self, robot_paths, lengths=None):
        """Score K candidate robot paths with this ONE shared bank (no re-prediction).

        robot_paths [K,T,4] = (x, y, z, yaw) at relative times ``(1..T) * 0.048 s`` (the L2.5
        nominal action clock); ``lengths`` [K] valid steps. Returns the L2.5 joint-world
        reduction (collision envelope / 1 m social events, first-event masses, minimum
        distances), computed per world so human-human and time correlations are kept.
        """
        from eventwm_graph.l25.numeric import aggregate_joint
        from eventwm_graph.partial.query import NOMINAL_STEP_SECONDS
        robots = np.asarray(robot_paths, np.float64)
        K, T = robots.shape[:2]
        lengths = np.full(K, T) if lengths is None else np.asarray(lengths, int)
        rel = np.arange(1, T + 1) * NOMINAL_STEP_SECONDS
        if rel[-1] > float(self.times[-1]) + 1e-9:raise ValueError('candidates exceed the bank horizon')
        return aggregate_joint(self.sandbox_positions(rel), robots, lengths, np.asarray(self.weights))

    def positions_at(self, relative_times):
        """Exact piecewise-linear evaluation at any times in [0, 3.028] s."""
        t = np.asarray(relative_times, np.float64)
        if t.ndim != 1 or np.any(t < 0) or np.any(t > 3.028001):raise ValueError('times outside [0, 3.028] s')
        s = np.clip(np.floor(t / STEP).astype(int), 0, SEGMENTS - 2);f = (t - s * STEP) / STEP
        a = self.knots[:, :, s];b = self.knots[:, :, s + 1]
        return (a + f[None, None, :, None] * (b - a)).transpose(0, 2, 1, 3)


def _fingerprint(frame):
    return (float(frame['timestamp']), tuple(float(v) for v in frame['robot_pose']), tuple(float(v) for v in frame['camera_origin']))


class _Stream:
    """Incremental legal state of one robot/episode stream."""

    def __init__(self):
        self.state = None;self.first = None;self.last = None;self.episode_start = None;self.resolution = None

    def sync(self, frames, resolution):
        """Reuse the incremental state only when this history extends the one already ingested."""
        reuse = False
        if self.state is not None and self.resolution == resolution and _fingerprint(frames[0]) == self.first:
            at = [i for i, f in enumerate(frames) if abs(float(f['timestamp']) - self.state.timestamp) < 1e-9]
            reuse = len(at) == 1 and _fingerprint(frames[at[0]]) == self.last
        if reuse:
            new = frames[at[0] + 1:]
        else:
            self.state = LegalState(resolution);self.first = _fingerprint(frames[0])
            self.episode_start = float(frames[0]['timestamp']);self.resolution = resolution
            new = list(frames)
        for f in new:
            self.state.ingest(f);self.state.last_frame = f
        self.last = _fingerprint(frames[-1])


class _MultiStore:
    """Concatenate one-anchor live stores into a batch view for ``collate``."""

    def __init__(self, lives):
        self.a = {k: np.concatenate([l.a[k] for l in lives]) for k in lives[0].a}
        self.a['anchor_root'] = np.arange(len(lives))
        self.k = {k: np.concatenate([l.k[k] for l in lives]) for k in lives[0].k}
        self.h = lives[0].h
        self.key_off = np.r_[0, np.cumsum([l.key_off[1] for l in lives])]
        self.hum_off = np.zeros(len(lives) + 1, int)
        self.episode_start = np.concatenate([l.episode_start for l in lives])


class HumanWorldModel:
    """Legacy-signature Stage 1 API. ``stream`` separates concurrent robots/episodes."""

    def __init__(self, checkpoint, device='cpu'):
        state = torch.load(checkpoint, map_location='cpu')
        self.network = HWM(**state['config'].get('model', {}))
        self.network.load_state_dict(state['model']);self.network.eval().requires_grad_(False)
        if self.network.arch in ('fast', 'lite', 'lite2'):fold_batchnorm(self.network)
        self.device = torch.device(device);self.network.to(self.device)
        # CPU deployment path: one TorchScript engine + NumPy sampling (same law, fewer dispatches).
        from . import legal_state
        legal_state.USE_KERNELS = True   # numba raycast: identical visibility raster (tested), fewer dispatches
        self.fast = None
        if self.device.type == 'cpu' and self.network.arch in ('fast', 'lite', 'lite2'):
            from .infer import FastSampler
            self.fast = FastSampler(self.network)
        self.checkpoint_sha256 = hashlib.sha256(open(checkpoint, 'rb').read()).hexdigest()
        self.streams = {}
        self.last_profile = {}

    @staticmethod
    def _frames(history, obs, partial_graph):
        frames = list(history.history) if hasattr(history, 'history') and not isinstance(history, (list, tuple)) else list(history)
        if not frames or float(obs['timestamp']) > float(frames[-1]['timestamp']) + 1e-9:frames.append(obs)
        elif abs(float(obs['timestamp']) - float(frames[-1]['timestamp'])) > 1e-9:raise ValueError('observation older than history')
        if partial_graph is not None and abs(float(partial_graph['timestamp']) - float(frames[-1]['timestamp'])) > 1e-7:
            raise ValueError('graph clock differs from arrived observation')
        return frames

    def predict(self, history, obs, partial_graph=None, horizon=3., worlds=40, seed=None, actions=(), executed_action=None,
                times=None, stream='default'):
        return self.predict_batch([dict(history=history, obs=obs, partial_graph=partial_graph, stream=stream)],
                                  horizon=horizon, worlds=worlds, seed=seed, times=times)[0]

    def predict_batch(self, requests, horizon=3., worlds=40, seed=None, times=None):
        """One network call for several streams. Each request: history, obs, partial_graph, stream.

        Worlds of different requests are independent; with a shared generator the draws of
        one request depend on the batch composition (seed derived from all requests).
        """
        if not 0 < horizon <= 3.028001 or worlds < 2:raise ValueError('supported horizon (0,3] and >=2 joint worlds required')
        start = time.perf_counter();lives = [];streams = []
        if len({r.get('stream', 'default') for r in requests}) != len(requests):raise ValueError('one request per stream per call')
        for r in requests:
            frames = self._frames(r['history'], r['obs'], r.get('partial_graph'))
            st = self.streams.setdefault(r.get('stream', 'default'), _Stream())
            graph = r.get('partial_graph')
            resolution = float(graph['resolution']) if graph is not None else (st.resolution or DEFAULT_RESOLUTION)
            st.sync(frames, resolution);streams.append(st)
            lives.append(_LiveStore(st.state, st.episode_start))
        synced = time.perf_counter()
        if self.fast is not None and len(lives) == 1:
            batch = collate_one(lives[0], 0)            # bitwise-identical single-anchor fast path
        else:
            batch = to_torch(collate(_MultiStore(lives) if len(lives) > 1 else lives[0], np.arange(len(lives)), trim=True), self.device)
        if seed is None:
            h = hashlib.sha256(b'stage1-hwm-1')
            for st, live in zip(streams, lives):
                h.update(np.float64(st.state.timestamp).tobytes());h.update(live.a['robot'].tobytes())
                for k in ('last', 'last_t'):h.update(np.ascontiguousarray(live.k['k_' + k]).tobytes())
            seed = int.from_bytes(h.digest()[:8], 'little')
        if self.fast is not None and len(requests) == 1:
            out = self.fast(batch, worlds, np.random.default_rng(int(seed) % (2 ** 63)))
            out = dict(out, knots=torch.from_numpy(out['knots'][None]), present=torch.from_numpy(out['present'][None]),
                       p_exist=torch.from_numpy(np.asarray(out['p_exist'])[None]), count_logp=out['count_logp'][None])
        else:
            g = torch.Generator(device=self.device);g.manual_seed(int(seed) % (2 ** 63))
            out = sample(self.network, batch, worlds, g)
        sampled = time.perf_counter()
        if times is None:times = np.linspace(0., float(horizon), 64)
        times = np.asarray(times, np.float64)
        si = np.clip(np.floor(times / STEP).astype(int), 0, SEGMENTS - 2);fr = (times - si * STEP) / STEP
        all_knots = out['knots'].cpu().numpy().astype(np.float64);all_present = out['present'].cpu().numpy()
        rots = batch['rot'].cpu().numpy().astype(np.float64);origins = batch['origin'].cpu().numpy().astype(np.float64)
        cols = batch['key_column'].cpu().numpy();p_all = out['p_exist'].cpu().numpy();K = out['n_keys']
        results = []
        for j, (st, live) in enumerate(zip(streams, lives)):
            col = cols[j];n_keys = len(live.keys)
            slots = np.flatnonzero(col >= 0);order = slots[np.argsort(col[slots], kind='stable')]
            hidden = K + np.flatnonzero(all_present[j, :, K:].any(0))
            keep = np.r_[order, hidden].astype(int)
            knots = np.einsum('ji,...j->...i', rots[j], all_knots[j][:, keep]) + origins[j]
            present = all_present[j][:, keep]
            positions = (knots[:, :, si] + fr[None, None, :, None] * (knots[:, :, si + 1] - knots[:, :, si])).transpose(0, 2, 1, 3)
            keys = tuple('track:' + live.keys[int(col[i])] for i in order) + tuple('residual:%d' % i for i in range(len(hidden)))
            result = HumanWorlds(float(st.state.timestamp), times, np.full(worlds, 1. / worlds), positions, present, keys,
                                 np.r_[np.zeros(len(order), bool), np.ones(len(hidden), bool)],
                                 float(out['expected_residual'][j]),
                                 ('IID_JOINT_WORLDS', 'TRACK_KEYS_ARE_NOT_IDENTITIES', 'UNKNOWN_GEOMETRY_LEARNED_PRIOR',
                                  'MONTE_CARLO_BANK_NOT_SAFETY_CERTIFICATE', 'HUMANS_EXOGENOUS_TO_ROBOT'),
                                 int(st.state.max_simultaneous), None, VERSION, out['count_logp'][j], p_all[j][order], knots).validate()
            assert (present.sum(1) >= st.state.max_simultaneous).all()
            results.append(result)
        end = time.perf_counter()
        self.last_profile = dict(state_ms=(synced - start) * 1000, sample_ms=(sampled - synced) * 1000,
                                 output_ms=(end - sampled) * 1000, complete_ms=(end - start) * 1000, batch=len(requests))
        return results


SELECTED = Path(__file__).resolve().parents[2] / 'docs/stage1_hwm/SELECTED.json'


def load_selected(config=SELECTED, device='cpu'):
    """Construct the selected Stage 1 model from its configuration (checkpoint SHA-256 verified).

    The configuration fixes worlds/times/horizon used for every reported result; callers pass
    them to ``predict`` (defaults already equal 40 worlds, 64 times, 3 s).
    """
    cfg = json.loads(Path(config).read_text())
    ck = Path(cfg['checkpoint'])
    if hashlib.sha256(ck.read_bytes()).hexdigest() != cfg['checkpoint_sha256']:
        raise ValueError('checkpoint differs from the selected configuration')
    model = HumanWorldModel(ck, device)
    model.configuration = cfg
    return model
