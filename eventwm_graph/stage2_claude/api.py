"""Deployable legal Stage 2 predictor: voxel-simulation features + BEV U-Net.

Interface (unchanged from the range-backed Stage 2 API)::

    predictor.predict_segments(partial_graph=G_t, trajectories=[tau_1..tau_K],
                               range_observations=packet, track_history=tracks,
                               materialize=True) -> [delta_1..delta_K]

Inputs are exactly the current legal PartialGraph, the supplied Nx5
[time,x,y,z,yaw] base trajectories (sharing the current starting pose), the
measured past/current range packet (ideal_current_jaw_range_dynamic_occlusion_1)
and the aligned legal track history. No scene mesh, navmesh, simulator,
future camera or future observation is accepted.

Each delta carries the ORIGINAL candidate universe (initially UNKNOWN cells
within 11 m of the supplied positions, original order), three-state
probabilities, hard states, heights, a concrete sparse increment and (by
default) the complete hypothetical future graph. Known cells are retained
exactly; predictions are marked hypothetical and never certify the real graph.
"""
from collections import OrderedDict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
import torch.nn.functional as F

from eventwm_graph.stage2.common import SCHEMA
from eventwm_graph.stage2_quality.features import candidate_mask
from eventwm_graph.stage2_segment.prepared import CompiledGraph, content_key
from eventwm_graph.stage2_segment.native import Base
from eventwm_graph.stage2_repair.range_history import validate
from eventwm_graph.stage2_repair.range_identity import identity as range_identity
from eventwm_graph.stage2_repair.increment import SparseIncrement
from .materialize import future_from_columns_merged as future_from_columns
from eventwm_graph.stage2_repair.diagnose import sha
from .features import GroupContext, crop_box, build_inputs, human_arrays, current_humans, channel_index, MAP_CHANNELS, QUERY_CHANNELS, RADIUS
from .model import UNet, GraphRunner, build_model
from .geometry import upload

PROTOCOL = 'ideal_current_jaw_range_dynamic_occlusion_1'


def fast_content_key(graph):
    """Full-content cache key: complete pickle encoding, XXH3-128 digest.

    Same coverage as stage2_segment.prepared.content_key (every field, dtype
    and container identity); only the digest function differs. Non-dict
    mappings use the original typed-pickle reference path.
    """
    import pickle
    import ctypes as ct
    from eventwm_graph.stage2_repair.range_identity import LIB
    if not isinstance(graph, dict) or LIB is None:
        return content_key(graph)
    try:
        raw = pickle.dumps(graph, protocol=5)
    except TypeError:
        return content_key(graph)
    state = LIB.XXH3_createState()
    try:
        LIB.XXH3_128bits_reset(state)
        LIB.XXH3_128bits_update(state, ct.c_char_p(raw), len(raw))
        d = LIB.XXH3_128bits_digest(state)
        return 'full-pickle-xxh3:%016x%016x' % (d.high64, d.low64)
    finally:
        LIB.XXH3_freeState(state)
FRAME_MARGIN = 19.   # m around the starting pose kept in the reusable legal map


try:
    from eventwm_graph.stage2_repair.fast_native_base import FastColumnsMixin, FastNativeBase

    class FastCompiledGraph(FastColumnsMixin, CompiledGraph):
        pass
except Exception:  # pragma: no cover - portable fallback
    FastCompiledGraph, FastNativeBase = CompiledGraph, Base


def _rows(keys, mask):
    """keys[mask] for contiguous (n,2) int32 keys via one 64-bit row view."""
    return keys.view(np.int64).reshape(-1)[mask].view(np.int32).reshape(-1, 2)


class LegalVoxelPredictor:
    predictor_schema = 'claude_legal_voxel_unet_1'

    def __init__(self, checkpoint, *, device='cuda:0', threshold=None, cache_size=4, cuda_graphs=True, lanes=None):
        saved = torch.load(checkpoint, map_location='cpu')
        config = saved['config']
        self.channels = channel_index(config)
        cin = len(config['map_channels'])+len(config['query_channels'])
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.model = build_model(cin, config).to(self.device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        # Inference: F16 weights/activations, channels-last layout, CUDA graphs.
        # lanes > 1: the per-query graphs of one call overlap on separate streams
        # (identical kernels per query, so results do not depend on the lane).
        lanes = int(saved.get('inference', {}).get('lanes', 1) if lanes is None else lanes)
        self.runner = GraphRunner(self.model, self.device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384),
                                  capture=cuda_graphs, channels=cin, lanes=lanes)
        self.model = self.runner.model     # the inference copy (BatchNorm folded, F16)
        self.lanes = lanes
        self.config = config
        decision = dict(saved.get('decision', {}))
        if threshold is not None:
            decision = dict(rule='reveal', threshold=float(threshold), blocked_threshold=.5)
        self.rule = decision.get('rule', 'reveal')
        if self.rule == 'reveal':
            self.threshold = float(decision.get('threshold', .5))
            self.blocked_threshold = float(decision.get('blocked_threshold', .5))
            self.decision_rule = 'revelation_threshold_then_conditional_blocked'
        elif self.rule == 'class':
            self.threshold = float(decision['free_threshold'])
            self.blocked_threshold = float(decision['blocked_class_threshold'])
            self.decision_rule = 'class_probability_thresholds_free_blocked'
        else:
            raise ValueError('unsupported decision rule')
        self.checkpoint_sha256 = sha(checkpoint)
        self.cache_size = cache_size
        self.graph_cache = OrderedDict()
        self.map_cache = OrderedDict()
        self.map_quick = {}
        self._hasher = ThreadPoolExecutor(1)
        self.last_batch_profile = {}
        self._pinned = None

    def clear_cache(self):
        self.graph_cache.clear()
        self.map_cache.clear()
        self.map_quick.clear()

    # ------------------------------------------------------------------ graph
    def prepare_graph(self, graph):
        key = fast_content_key(graph)
        if key in self.graph_cache:
            self.graph_cache.move_to_end(key)
            return self.graph_cache[key]
        p = FastCompiledGraph(graph)
        p.future_base = None
        p.future_base_ready = False
        p.unknown_grids = OrderedDict()
        self.graph_cache[key] = p
        while len(self.graph_cache) > self.cache_size:
            self.graph_cache.popitem(last=False)
        return p

    @staticmethod
    def _ensure_future_base(prepared, graph):
        """Native hypothetical-graph base (built lazily so it overlaps GPU work)."""
        if not prepared.future_base_ready:
            if prepared.native_eligible:
                try:
                    prepared.future_base = FastNativeBase(graph, prepared)
                except (TypeError, ValueError, OverflowError):
                    prepared.future_base = None
            prepared.future_base_ready = True
        return prepared.future_base

    def _candidates(self, prepared, path):
        """Exact original candidate universe and order for one trajectory."""
        res = prepared.resolution
        lo, hi = crop_box(path, res)
        gridkey = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
        if gridkey not in prepared.unknown_grids:
            shape = hi-lo
            if int(np.prod(shape)) > 4000000:
                raise ValueError('trajectory domain exceeds four million cells')
            states = np.zeros(tuple(shape), np.uint8)
            ij = prepared.keys.astype(np.int64)-lo
            ok = np.all((ij >= 0) & (ij < shape), axis=1)
            states[ij[ok, 0], ij[ok, 1]] = 1
            prepared.unknown_grids[gridkey] = np.argwhere(states == 0).astype(np.int32)+lo.astype(np.int32)
            while len(prepared.unknown_grids) > 16:
                prepared.unknown_grids.popitem(last=False)
        unknown = prepared.unknown_grids[gridkey]
        xy = np.unique(np.asarray(path)[:, [1, 3]].astype(np.float32), axis=0)
        return unknown[candidate_mask(unknown, xy, res)], lo, hi

    # -------------------------------------------------------------- legal map
    def _context(self, prepared, graph, packet, track_history, paths):
        ranges = np.asarray(packet['ranges'], np.float32)
        origins = np.asarray(packet['origins'], np.float64)
        rotations = np.asarray(packet['rotations'], np.float64)
        timestamps = np.asarray(packet['timestamps'], np.float64)
        args = (ranges, origins, rotations, timestamps, track_history, prepared.resolution, 1)
        # The full-content identity decides every cache hit. A cheap necessary condition (graph identity,
        # packet shape, camera poses and times) tells whether any cached map can match at all; if none
        # can, the full identity is hashed on a helper thread while the map is built.
        quick = (prepared.identity, ranges.shape, hashlib.sha1(np.ascontiguousarray(origins).tobytes()+
                 np.ascontiguousarray(rotations).tobytes()+np.ascontiguousarray(timestamps).tobytes()).hexdigest())
        pending = None
        if quick in self.map_quick.values():
            ident = range_identity(*args)
            key = (ident, prepared.identity)
            ctx = self.map_cache.get(key)
        else:
            pending = self._hasher.submit(range_identity, *args)
            ctx = None
        res = prepared.resolution
        start = paths[0][0]
        need_lo = np.min([crop_box(p, res)[0] for p in paths], 0)
        need_hi = np.max([crop_box(p, res)[1] for p in paths], 0)
        if ctx is None or np.any(need_lo < ctx.lo) or np.any(need_hi > ctx.lo+np.asarray(ctx.shape)):
            center = np.floor(np.asarray([start[1], start[3]], np.float64)/res).astype(np.int64)
            margin = int(np.ceil(FRAME_MARGIN/res))
            lo = np.minimum(center-margin, need_lo)
            hi = np.maximum(center+margin, need_hi)
            humans = human_arrays(track_history) if self.config.get('human_filter') else None
            ctx = GroupContext(prepared.keys, prepared.values, ranges, origins, rotations, start, res, self.device, lo, hi, humans=humans,
                               splat=bool(self.config.get('splat', True)), sub=int(self.config.get('sub', 1)),
                               render=int(self.config.get('render', 96)),
                               current_humans=current_humans(track_history) if self.config.get('human_occluders') else None,
                               near=float(self.config.get('near', 0.)),
                               human_motion=current_humans(track_history, motion=True)
                               if 'human_future' in self.config.get('query_channels', ()) else None)
            if pending is not None:
                ident = pending.result()
                key = (ident, prepared.identity)
            self.map_cache[key] = ctx
            self.map_quick[key] = quick
            while len(self.map_cache) > 2:
                old, _ = self.map_cache.popitem(last=False)
                self.map_quick.pop(old, None)
        self.map_cache.move_to_end(key)
        return ctx, ident

    # ---------------------------------------------------------------- predict
    def predict(self, *, partial_graph, trajectory, range_observations, track_history, materialize=True):
        return self.predict_segments(partial_graph=partial_graph, trajectories=[trajectory], range_observations=range_observations,
                                     track_history=track_history, materialize=materialize)[0]

    @torch.no_grad()
    def predict_segments(self, *, partial_graph, trajectories, range_observations, track_history, materialize=True):
        tick = time.perf_counter()
        packet = range_observations
        if packet.get('sensor_protocol') != PROTOCOL:
            raise ValueError('only measured Euclidean 90-degree range protocol is validated')
        now0 = float(partial_graph['timestamp'])
        validate(packet['ranges'], packet['origins'], packet['rotations'], packet['timestamps'], now0)
        if abs(float(np.asarray(packet['timestamps'])[-1])-now0) > 1e-7:
            raise ValueError('current sensor packet must reach graph start time')
        if len(track_history) != len(packet['ranges']):
            raise ValueError('legal track/range history alignment')
        paths = [np.asarray(p, np.float64) for p in trajectories]
        if not paths:
            return []
        for p in paths:
            if p.ndim != 2 or p.shape[1] != 5 or not len(p) or not np.isfinite(p).all():
                raise ValueError('supplied Nx5 [time,x,y,z,yaw] trajectories required')
            if len(p) > 1 and np.any(np.diff(p[:, 0]) <= 0):
                raise ValueError('trajectory timestamps must increase')
            if abs(p[0, 0]-now0) > 1e-7:
                raise ValueError('trajectory must begin at current graph time')
        if any(not np.allclose(p[0], paths[0][0], rtol=0, atol=1e-7) for p in paths):
            raise ValueError('queries must share current starting pose')
        prepared = self.prepare_graph(partial_graph)
        prepared_at = time.perf_counter()
        res = prepared.resolution
        boxes = [crop_box(p, res) for p in paths]
        if any(int(np.prod(hi-lo)) > 4000000 for lo, hi in boxes):
            raise ValueError('trajectory domain exceeds four million cells')
        ctx, ident = self._context(prepared, partial_graph, packet, track_history, paths)
        context_at = time.perf_counter()
        x, mask, fr = build_inputs(ctx, paths, [(lo, hi-lo) for lo, hi in boxes])
        if self.channels is not None:
            x = x[:, self.channels]
        features_at = time.perf_counter()
        logits = self.runner(x, frames=fr[:, 2:])
        # CPU work that does not depend on predictions overlaps queued GPU work.
        if materialize:
            self._ensure_future_base(prepared, partial_graph)
        # Exact candidate universe (row-major == original argwhere order),
        # decoded on device; one pinned device->host transfer.
        q = len(paths)
        idx = torch.nonzero(mask)
        n = int(idx.shape[0])
        dev = self.device
        meta = upload(np.concatenate([np.asarray([b[0] for b in boxes], np.int64).reshape(-1),
                                      np.asarray([len(p) > 1 for p in paths], np.int64)]), dev)
        lo_t = meta[:2*q].view(q, 2)
        moving = meta[2*q:].bool()
        qi_t = idx[:, 0]
        keys_t = (idx[:, 1:]+lo_t[qi_t]).to(torch.int32).contiguous()
        rb = torch.sigmoid(logits[qi_t, :, idx[:, 1], idx[:, 2]].float())
        r = torch.where(moving[qi_t], rb[:, 0], torch.zeros_like(rb[:, 0]))
        b = rb[:, 1]
        prob_t = torch.stack([1-r, r*(1-b), r*b], 1).contiguous()
        if self.rule == 'reveal':
            states_t = torch.where(r > self.threshold, torch.where(b > self.blocked_threshold, 2, 1), 0).to(torch.uint8)
        else:
            pf, pb = prob_t[:, 1], prob_t[:, 2]
            states_t = torch.where((pf > self.threshold) | (pb > self.blocked_threshold),
                                   torch.where(pb-self.blocked_threshold > pf-self.threshold, 2, 1), 0).to(torch.uint8)
        counts_t = mask.view(q, -1).sum(1, dtype=torch.int64)
        nbytes = n*8+n*12+n+q*8
        if self._pinned is None or self._pinned.numel() < nbytes:
            self._pinned = torch.empty(max(nbytes, 1 << 22), dtype=torch.uint8, pin_memory=True)
        host = self._pinned
        dev_buf = torch.cat([keys_t.view(torch.uint8).reshape(-1), prob_t.view(torch.uint8).reshape(-1), states_t.reshape(-1),
                             counts_t.view(torch.uint8).reshape(-1)])
        host[:nbytes].copy_(dev_buf, non_blocking=True)
        torch.cuda.current_stream(dev).synchronize()
        raw = host[:nbytes].numpy()
        keys_all = raw[:n*8].view(np.int32).reshape(n, 2).copy()
        prob_all = raw[n*8:n*20].view(np.float32).reshape(n, 3).copy()
        states_all = raw[n*20:n*21].copy()
        counts = raw[n*21:n*21+q*8].view(np.int64)
        network_at = time.perf_counter()
        domains = []
        at = 0
        for qi in range(q):
            c = int(counts[qi])
            domains.append((keys_all[at:at+c], prob_all[at:at+c], states_all[at:at+c]))
            at += c
        out = []
        for qi, (path, (keys, prob, states)) in enumerate(zip(paths, domains)):
            heights = np.full(len(keys), np.float32(path[0, 2]), np.float32)
            chosen = np.flatnonzero(states)
            nk, ns, nh = _rows(keys, chosen), states[chosen], heights[chosen]
            now = float(path[-1, 0])
            updates = SparseIncrement(nk, ns, nh, now)
            delta = dict(schema=SCHEMA, predictor_schema=self.predictor_schema, predicted=True, base_graph_identity=prepared.identity,
                         base_timestamp=now0, timestamp=now, robot_pose=path[-1, 1:].tolist(), static_updates=updates,
                         candidate_cells=keys, probabilities=prob, states=states, heights=heights, unknown_cells=_rows(keys, states == 0),
                         outside_domain='UNKNOWN_NO_UPDATE', checkpoint_sha256=self.checkpoint_sha256, increment_format='columns',
                         observation_scope='initially_UNKNOWN_static_cells_only', probability_columns=['UNKNOWN', 'FREE', 'BLOCKED'],
                         prediction_task='segment_conditioned_legal_observation_1',
                         provenance=dict(hypothetical=True, privileged_runtime_inputs=False, sensor_extension=True,
                                         input='current_legal_graph_supplied_base_path_and_measured_past_current_range_history',
                                         range_history_identity=ident, checkpoint_sha256=self.checkpoint_sha256,
                                         trajectory_sha256=hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(),
                                         decision=self.decision_rule, revelation_threshold=self.threshold,
                                         blocked_threshold=self.blocked_threshold, implementation=type(self).__name__))
            if materialize:
                future = future_from_columns(prepared.future_base, nk, ns, nh, now, path[-1, 1:].tolist(), self.checkpoint_sha256)
                if future is None:
                    from eventwm_graph.stage2_segment.update import apply_graph_delta
                    delta['future_partial_graph'] = apply_graph_delta(partial_graph, delta)
                    delta['future_format'] = 'snapshot'
                else:
                    delta['future_partial_graph'] = future
                    delta['future_format'] = 'native_columns'
            out.append(delta)
        done = time.perf_counter()
        self.last_batch_profile = dict(queries=len(paths), prepare_seconds=prepared_at-tick,
                                       legal_map_seconds=context_at-prepared_at, query_features_seconds=features_at-context_at,
                                       network_seconds=network_at-features_at, decode_materialize_seconds=done-network_at,
                                       total_seconds=done-tick, serialization_included=False, sensor_acquisition_included=False)
        return out
