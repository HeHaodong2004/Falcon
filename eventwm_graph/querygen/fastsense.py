"""Exact fast legal geometry sensor (v13 throughput).

``fast_observer_class(base)`` derives an observer from the accepted one (``legal._pure_observer_class()`` or
``partial.fast_observer.FastGeometryObserver``) that publishes the SAME observation rows -- the same detections and
tracks, the same certified FREE cells in the same order, the same inflated wall cells in the same order, the same
ego cell -- and leaves the tracker and the certified-cell set in the same state.  Checked observation by observation
against the reference (``tests/querygen/test_fastsense.py``).  Savings:

1. Static-occlusion memo.  A support probe ray is blocked iff some non-ignored hit lies before its end; actors can
   only add hits, so a probe blocked by the stage (or a STATIC rigid object) stays blocked from the same exact camera
   origin whatever moves -- the cell cannot be certified from there.  Support rays are cast through the same native
   castRay (all hits) and classified; such cells are skipped at later observations from that origin (a waiting
   robot, branches through the same pose).  The memo is static geometry only (no actor, no time) and private to the
   sensor, like the support-certificate caches.
2. Inflated wall cells in one vectorised pass (the reference loops over ~100 wall hits x 49 cells in Python with the
   same float64 arithmetic; the scalar-norm boundary rule is kept), published as an ``UpdateBlock`` (arrays; the
   reference dict list is materialised only if someone iterates it).  ``GridMap.apply`` and the prefix cache read
   the arrays directly.
3. The probe-index height list is kept as an array (the reference converts it at every observation).

Numerical note: every array operation here was checked to be row-independent in this NumPy/OpenBLAS build (matmul
(N,3)@(3,3), axis norms, batched ray lengths), so evaluating a subset gives the reference's values for that subset.
"""
import copy
import functools
import math
import time
from collections import OrderedDict

import numpy as np

from scripts.collector.observation import HumanDetection
from ..partial.fast_observer import StaticProbeIndex
from ..partial.map import KNOWN_FREE, KNOWN_OCCUPIED
from ..partial.observer import SENSOR_VERSION, in_frustum

FREE_SOURCE = 'visible_native_cspace_square_and_clearance_halo'
WALL_SOURCE = 'observed_static_wall_inflated_025m'
EGO_SOURCE = 'occupied_ego_base_support'
SOURCES = (FREE_SOURCE, WALL_SOURCE, EGO_SOURCE)
SRC_FREE, SRC_WALL, SRC_EGO = 0, 1, 2

MEMO_ORIGINS = 4096          # LRU bound on memoised camera origins per observer
IGNORED, STATIC, OTHER = 0, 1, 2


class UpdateBlock:
    """One observation's static updates as arrays, in publication order (immutable).

    ``cells`` int64 [N,2], ``states`` uint8 [N], ``heights`` float64 [N], ``sources`` uint8 [N] (index into
    SOURCES), ``timestamp``.  Iteration / indexing / len give the reference dicts."""
    __slots__ = ('cells', 'states', 'heights', 'sources', 'timestamp', '_dicts')

    def __init__(self, cells, states, heights, sources, timestamp):
        self.cells, self.states, self.heights, self.sources = cells, states, heights, sources
        self.timestamp = float(timestamp)
        self._dicts = None
        for a in (cells, states, heights, sources):
            a.setflags(write=False)

    def dicts(self):
        if self._dicts is None:
            t = self.timestamp
            self._dicts = [dict(cell=[int(c[0]), int(c[1])], state=int(s), height=float(h), timestamp=t,
                                source=SOURCES[int(k)])
                           for c, s, h, k in zip(self.cells.tolist(), self.states.tolist(), self.heights.tolist(),
                                                 self.sources.tolist())]
        return self._dicts

    def __iter__(self):
        return iter(self.dicts())

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.dicts()[i]

    def __eq__(self, other):
        return list(self) == list(other)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def __reduce__(self):
        return (UpdateBlock, (np.array(self.cells), np.array(self.states), np.array(self.heights),
                              np.array(self.sources), self.timestamp))


def _lookup(index, keys, observer, height):
    """``StaticProbeIndex.lookup`` (identical evaluation order and caching), also returning the probe-row ids."""
    indices = np.full(len(keys), -1, dtype=np.int32)
    groups = []
    tile_keys = keys // 64
    for x in range(int(tile_keys[:, 0].min()), int(tile_keys[:, 0].max())+1):
        for z in range(int(tile_keys[:, 1].min()), int(tile_keys[:, 1].max())+1):
            selected = np.flatnonzero((tile_keys[:, 0] == x) & (tile_keys[:, 1] == z))
            if not len(selected):
                continue
            tile = index.tiles.setdefault((x, z), np.full((64, 64), -1, dtype=np.int32))
            local = keys[selected] % 64
            indices[selected] = tile[local[:, 0], local[:, 1]]
            groups.append((tile, selected, local))
    missing = np.flatnonzero(indices == -1)
    observer.performance['indexed_support_reuses'] += len(keys)-len(missing)
    for i in missing:
        key = (int(keys[i, 0]), int(keys[i, 1]))
        value = observer._targets(key, height)
        if value is None:
            indices[i] = -2
        else:
            center, probes = value
            indices[i] = len(index.targets)
            index.targets.append(probes); index.heights.append(float(center[1]))
            index.array = None
    for tile, selected, local in groups:
        tile[local[:, 0], local[:, 1]] = indices[selected]
    selected = np.flatnonzero(indices >= 0)
    if not len(selected):
        return selected, np.empty((0, 63, 3)), np.empty(0), selected
    if index.array is None:
        index.array = np.asarray(index.targets)
        index.array.setflags(write=False)
    h = getattr(index, '_qg_heights', None)
    if h is None or len(h) != len(index.heights):
        h = index._qg_heights = np.asarray(index.heights)
    rows = indices[selected]
    return selected, index.array[rows], h[rows], rows


@functools.lru_cache(maxsize=None)
def fast_observer_class(base):
    """Observer class with the exact fast paths on top of ``base`` (a FastGeometryObserver subclass)."""

    class FastSenseObserver(base):
        def __init__(self, runtime, resolution, private_cache=None, native_batch=True):
            super().__init__(runtime, resolution, private_cache, native_batch)
            import habitat_sim
            self._memo = OrderedDict()
            self.performance.update(memo_skipped=0, memo_entries=0, classified_rays=0)
            ignored = set(self.robot_ids) | set(self.inactive_ids)
            static = {int(habitat_sim.stage_id)}
            try:
                for obj in self.sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
                    if obj.motion_type == habitat_sim.physics.MotionType.STATIC:
                        static.add(int(obj.object_id))
            except Exception:
                pass
            static -= ignored | set(self.all_actor_ids)
            ids = sorted(ignored | static | set(self.all_actor_ids))
            self._id_lo = min(ids+[-1])
            table = np.full(max(ids+[0])-self._id_lo+1, OTHER, dtype=np.int8)
            for i in static:
                table[i-self._id_lo] = STATIC
            for i in ignored:
                table[i-self._id_lo] = IGNORED
            self._id_table = table
            self._ignored_sorted = sorted(ignored)

        # ---------------------------------------------------------------- rays
        def _classify(self, ids):
            j = ids.astype(np.int64)-self._id_lo
            out = np.full(len(ids), OTHER, dtype=np.int8)
            ok = (j >= 0) & (j < len(self._id_table))
            out[ok] = self._id_table[j[ok]]
            return out

        def _cast_support(self, origin, targets):
            """(clear[N], static_blocked[N]) for support probes: exactly ``_clear_many(origin, targets)``'s result
            (strict 0.015 m endpoint rule, robot and inactive agents ignored) plus whether a STATIC hit blocks."""
            vectors = np.asarray(targets)-origin
            lengths = np.sqrt((vectors[:, None, :] @ vectors[:, :, None])[:, 0, 0])
            valid = lengths >= 1e-7
            clear = np.ones(len(targets), dtype=bool)
            sblock = np.zeros(len(targets), dtype=bool)
            if valid.any():
                L = lengths[valid]
                out = self.batch.cast_rays(origin, vectors[valid]/L[:, None], L)
                counts = np.diff(out['offsets'])
                if counts.sum():
                    ray = np.repeat(np.arange(len(L)), counts)
                    before = out['distances'] < np.repeat(L-.015, counts)
                    cls = self._classify(out['object_ids'])
                    blocking = before & (cls != IGNORED)
                    b = np.zeros(len(L), dtype=bool); b[ray[blocking]] = True
                    s = np.zeros(len(L), dtype=bool); s[ray[blocking & (cls == STATIC)]] = True
                    clear[valid] = ~b
                    sblock[valid] = s
                self.performance['rays'] += int(valid.sum())
                self.performance['classified_rays'] += int(valid.sum())
                self.performance['batch_calls'] += 1
            return clear, sblock

        def _memo_get(self, key):
            v = self._memo.get(key)
            if v is not None:
                self._memo.move_to_end(key)
            return v

        def _memo_add(self, key, rows):
            old = self._memo.get(key)
            new = np.unique(rows) if old is None else np.union1d(old, rows)
            self._memo[key] = new
            self._memo.move_to_end(key)
            if len(self._memo) > MEMO_ORIGINS:
                self._memo.popitem(last=False)

        # ------------------------------------------------------------ observe
        def observe(self, timestamp=None):
            from ..robotmap.execution import signed_yaw
            t = float(self.sim.get_world_time() if timestamp is None else timestamp)
            if t == self.last_timestamp:
                return copy.deepcopy(self.last_row)
            if self.last_timestamp is not None and t < self.last_timestamp:
                raise ValueError('Future/rewound observer state')
            began = time.perf_counter()
            robot = self.sim.get_agent_data(0).articulated_agent
            pose = [*np.asarray(robot.base_pos, dtype=float).tolist(), signed_yaw(robot)]
            origin, rotation = self.camera.camera_pose()
            detections = []
            for i, ids in enumerate(self.human_ids, 1):
                p = np.asarray(self.sim.get_agent_data(i).articulated_agent.base_pos, dtype=float)
                target = p+[0, self.config.torso_height_m, 0]
                if in_frustum(target[None], origin, rotation, self.config)[0]:
                    self.performance['human_rays'] += 1
                    if self._clear_many(origin, target[None], ids)[0]:
                        detections.append(HumanDetection(t, tuple(p.tolist())))
            tracks = self.tracker.update(t, sorted(detections, key=lambda d: d.position))
            r, radius = self.resolution, self.config.max_range_m
            ix0, ix1 = math.floor((origin[0]-radius)/r), math.floor((origin[0]+radius)/r)
            iz0, iz1 = math.floor((origin[2]-radius)/r), math.floor((origin[2]+radius)/r)
            ix, iz = np.meshgrid(np.arange(ix0, ix1+1), np.arange(iz0, iz1+1), indexing='ij')
            keys = np.stack([ix.ravel(), iz.ravel()], axis=1)
            centers = np.stack([(keys[:, 0]+.5)*r, np.full(len(keys), pose[1]+.55), (keys[:, 1]+.5)*r], axis=1)
            keys = keys[in_frustum(centers, origin, rotation, self.config)]
            self.performance['frustum_candidates'] += len(keys)
            if self.seen:
                seen = np.asarray(list(self.seen), dtype=np.int64)
                inside = ((seen[:, 0] >= ix0) & (seen[:, 0] <= ix1) &
                          (seen[:, 1] >= iz0) & (seen[:, 1] <= iz1))
                seen = seen[inside]
                seen_mask = np.zeros((ix1-ix0+1, iz1-iz0+1), dtype=bool)
                seen_mask[seen[:, 0]-ix0, seen[:, 1]-iz0] = True
                keys = keys[~seen_mask[keys[:, 0]-ix0, keys[:, 1]-iz0]]
            free_cells = np.empty((0, 2), dtype=np.int64)
            free_heights = np.empty(0, dtype=np.float64)
            if len(keys):
                ikey = ('static_probe_index_v1', self.cache_namespace, round(pose[1]/.1))
                index = self.cache.setdefault(ikey, StaticProbeIndex())
                selected, targets, heights, rows = _lookup(index, keys, self, pose[1])
                candidates = keys[selected]
                if len(candidates):
                    mkey = (ikey, np.asarray(origin, dtype=np.float64).tobytes())
                    known = self._memo_get(mkey)
                    live = np.arange(len(candidates))
                    if known is not None:
                        live = live[~np.isin(rows, known)]
                        self.performance['memo_skipped'] += len(candidates)-len(live)
                    if len(live):
                        mask = in_frustum(targets[live].reshape(-1, 3), origin, rotation,
                                          self.config).reshape(len(live), -1).all(axis=1)
                        remaining = live[mask]
                    else:
                        remaining = live
                    blocked = []
                    for probe in range(targets.shape[1]):
                        if not len(remaining):
                            break
                        self.performance['support_rays'] += len(remaining)
                        clear, sblock = self._cast_support(origin, targets[remaining, probe])
                        if sblock.any():
                            blocked.append(rows[remaining[sblock]])
                        remaining = remaining[clear]
                    if blocked:
                        self._memo_add(mkey, np.concatenate(blocked))
                        self.performance['memo_entries'] = len(self._memo)
                    free_cells = candidates[remaining]
                    free_heights = heights[remaining]
                    self.seen.update(map(tuple, free_cells.tolist()))
            directions = []
            for x in np.linspace(-.95, .95, 35):
                for y in (-.5, 0., .5):
                    d = rotation @ np.array([x, y, -1.]); d /= np.linalg.norm(d)
                    directions.append(d)
            self.performance['wall_rays'] += len(directions)
            wall_hits = self.batch.nearest_hits(origin, np.asarray(directions), np.full(len(directions), radius),
                                                self._ignored_sorted)
            self.performance['rays'] += len(directions)
            self.performance['batch_calls'] += 1
            wall_cells = self._wall_cells(wall_hits, pose, r)
            if len(wall_cells):
                self.seen.difference_update(map(tuple, wall_cells.tolist()))
            ego = np.asarray([[math.floor(pose[0]/r), math.floor(pose[2]/r)]], dtype=np.int64)
            nf, nw = len(free_cells), len(wall_cells)
            cells = np.concatenate([free_cells.reshape(-1, 2), wall_cells.reshape(-1, 2), ego]).astype(np.int64)
            states = np.concatenate([np.full(nf, KNOWN_FREE, dtype=np.uint8), np.full(nw, KNOWN_OCCUPIED, dtype=np.uint8),
                                     np.asarray([KNOWN_FREE], dtype=np.uint8)])
            hts = np.concatenate([np.asarray(free_heights, dtype=np.float64), np.full(nw+1, pose[1], dtype=np.float64)])
            srcs = np.concatenate([np.full(nf, SRC_FREE, dtype=np.uint8), np.full(nw, SRC_WALL, dtype=np.uint8),
                                   np.asarray([SRC_EGO], dtype=np.uint8)])
            updates = UpdateBlock(cells, states, hts, srcs, t)
            row = dict(timestamp=t, robot_pose=pose, tracks=tracks, camera_origin=origin.tolist(),
                       camera_rotation=rotation.tolist(), static_updates=updates,
                       sensor_protocol=SENSOR_VERSION, observation_valid=True)
            self.last_timestamp, self.last_row = t, row
            self.performance['visibility_seconds'] += time.perf_counter()-began
            self.performance['observations'] += 1
            return row

        def _wall_cells(self, wall_hits, pose, r):
            """Inflated wall cells of the reference loop (wall-ray order, then offset order), vectorised."""
            valid = np.asarray(wall_hits['valid'], dtype=bool)
            ids = np.asarray(wall_hits['object_ids'])
            pts = np.asarray(wall_hits['points'])
            keep = valid.copy()
            if self.all_actor_ids:
                keep &= ~np.isin(ids, np.fromiter(self.all_actor_ids, dtype=np.int64))
            p64 = pts.astype(np.float64)
            keep &= (pose[1]+.2 < p64[:, 1]) & (p64[:, 1] < pose[1]+1.4)
            sel = np.flatnonzero(keep)
            if not len(sel):
                return np.empty((0, 2), dtype=np.int64)
            P = p64[sel]
            key = np.stack([np.floor(P[:, 0]/r), np.floor(P[:, 2]/r)], axis=1).astype(np.int64)
            nearby = key[:, None, :]+self._wall_offsets[None, :, :]                       # [W, K, 2]
            disp = np.stack([(nearby[..., 0]+.5)*r-P[:, None, 0], (nearby[..., 1]+.5)*r-P[:, None, 2]], axis=-1)
            dist = np.linalg.norm(disp, axis=-1)                                          # [W, K]
            near = np.argwhere(np.abs(dist-.25) <= 1e-14)
            for w, j in near:
                p = pts[sel[w]]
                c = np.asarray([(nearby[w, j, 0]+.5)*r, pose[1], (nearby[w, j, 1]+.5)*r])
                dist[w, j] = np.linalg.norm((c-p)[[0, 2]])
            return nearby[dist <= .25]

    FastSenseObserver.__name__ = 'FastSense' + base.__name__
    return FastSenseObserver
