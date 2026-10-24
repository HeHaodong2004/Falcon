"""Exact batched implementation of the accepted ideal geometry observer.

All caches here are private simulator data. They contain static certificates
and their fixed probe coordinates, never an online graph or dynamic mask.
Candidate order, probe order, native intersection rules and published evidence
remain those of :mod:`observer`. Dynamic rays use the current collision world.
"""
import copy
import math
import time

import numpy as np

from scripts.collector.observation import HumanDetection
from .map import KNOWN_FREE, KNOWN_OCCUPIED
from .observer import GeometryObserver, SENSOR_VERSION, in_frustum


class StaticProbeIndex:
    """Lazy private spatial index; -1 untested, -2 unsupported, >=0 certificate.

    The first request still evaluates the accepted support rule at the actual
    base height, in exactly the original cell order. This preserves its existing
    layer-cache evaluation law without repeatedly dispatching known failures.
    """
    def __init__(self):
        self.tiles = {}
        self.targets = []
        self.heights = []
        self.array = None

    def lookup(self, keys, observer, height):
        indices = np.full(len(keys), -1, dtype=np.int32)
        groups = []
        tile_keys = keys // 64
        for x in range(int(tile_keys[:, 0].min()), int(tile_keys[:, 0].max())+1):
            for z in range(int(tile_keys[:, 1].min()), int(tile_keys[:, 1].max())+1):
                selected = np.flatnonzero((tile_keys[:, 0] == x) & (tile_keys[:, 1] == z))
                if not len(selected):
                    continue
                tile = self.tiles.setdefault((x, z), np.full((64, 64), -1, dtype=np.int32))
                local = keys[selected] % 64
                indices[selected] = tile[local[:, 0], local[:, 1]]
                groups.append((tile, selected, local))
        missing = np.flatnonzero(indices == -1)
        observer.performance['indexed_support_reuses'] += len(keys)-len(missing)
        # Do not group first evaluations by tile: global reference key order is
        # significant to the existing lazy static certificate cache.
        for i in missing:
            key = (int(keys[i, 0]), int(keys[i, 1]))
            value = observer._targets(key, height)
            if value is None:
                indices[i] = -2
            else:
                center, probes = value
                indices[i] = len(self.targets)
                self.targets.append(probes); self.heights.append(float(center[1]))
                self.array = None
        for tile, selected, local in groups:
            tile[local[:, 0], local[:, 1]] = indices[selected]
        selected = np.flatnonzero(indices >= 0)
        if not len(selected):
            return selected, np.empty((0, 63, 3)), np.empty(0)
        if self.array is None:
            self.array = np.asarray(self.targets)
            self.array.setflags(write=False)
        return selected, self.array[indices[selected]], np.asarray(self.heights)[indices[selected]]


def ray_lengths(vectors):
    # Each 1x3 @ 3x1 is the scalar dot used by np.linalg.norm(vector), not the
    # differently rounded elementwise reduction used by norm(..., axis=1).
    return np.sqrt((vectors[:, None, :] @ vectors[:, :, None])[:, 0, 0])


class FastGeometryObserver(GeometryObserver):
    def __init__(self, runtime, resolution, private_cache=None, native_batch=True):
        super().__init__(runtime, resolution, private_cache)
        # Extra dependencies prevent sharing certificates between differently
        # configured scenes or observation contracts. The worker's map identity
        # already binds the geometry, navmesh, native contract and robot assets.
        identity = getattr(runtime, 'private_geometry_identity', None)
        self.cache_namespace = (*self.cache_namespace, SENSOR_VERSION, identity)
        self.probe_cache = self.cache.setdefault(('static_probe_coordinates_v1', self.cache_namespace), {})
        self.batch = None
        if native_batch:
            from .native_batch import for_simulator
            self.batch = for_simulator(self.sim)
        self.performance.update(support_hits=0, support_misses=0, probe_hits=0,
                                probe_misses=0, frustum_candidates=0, batch_calls=0,
                                support_rays=0, human_rays=0, wall_rays=0, indexed_support_reuses=0)
        r = self.resolution
        halo_radius = .25 + r/math.sqrt(2)
        self._halo = np.asarray([[math.cos(a)*halo_radius, 0., math.sin(a)*halo_radius]
                                 for a in np.linspace(0., 2*math.pi, 16, endpoint=False)])
        span = int(math.ceil(.25/r))+1
        self._wall_offsets = np.asarray([(dx, dz) for dx in range(-span, span+1)
                                          for dz in range(-span, span+1)], dtype=np.int64)

    def _targets(self, key, height):
        ck = (self.cache_namespace, key[0], key[1], round(height/.1))
        self.performance['support_hits' if ck in self.cache else 'support_misses'] += 1
        support = self._support(key, height)
        if support is None:
            return None
        if ck in self.probe_cache:
            self.performance['probe_hits'] += 1
            return support[0], self.probe_cache[ck]
        center, points = support
        # Addition and concatenation order deliberately match the scalar code.
        halo = center + self._halo
        probes = np.concatenate([points[[0, 1, 2, 3, 4]], halo])
        targets = np.concatenate([probes+[0, .025, 0], probes+[0, .7, 0], probes+[0, 1.4, 0]])
        targets.setflags(write=False)
        self.probe_cache[ck] = targets
        self.performance['probe_misses'] += 1
        return center, targets

    def _clear_many(self, origin, targets, ignore=()):
        if self.batch is None:
            return np.asarray([self.clear_ray(origin, p, ignore) for p in targets], dtype=bool)
        vectors = np.asarray(targets)-origin
        lengths = ray_lengths(vectors)
        valid = lengths >= 1e-7
        result = np.ones(len(targets), dtype=bool)
        if valid.any():
            ignored = self.robot_ids | self.inactive_ids | set(ignore)
            result[valid] = self.batch.clear_rays(origin, vectors[valid]/lengths[valid, None],
                                                  lengths[valid], sorted(ignored))
            self.performance['rays'] += int(valid.sum())
            self.performance['batch_calls'] += 1
        return result

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
        candidates, targets, heights = np.empty((0, 2), dtype=np.int64), [], []
        if len(keys):
            index = self.cache.setdefault(('static_probe_index_v1', self.cache_namespace, round(pose[1]/.1)), StaticProbeIndex())
            selected, targets, heights = index.lookup(keys, self, pose[1])
            candidates = keys[selected]
        updates = []
        if len(candidates):
            # Same 2-D point matrix operations as the reference, in one call.
            mask = in_frustum(targets.reshape(-1, 3), origin, rotation, self.config).reshape(len(candidates), -1).all(axis=1)
            remaining = np.flatnonzero(mask)
            # Wavefront short circuit: blocked cells stop at the same probe as
            # the scalar observer. Casting all 63 probes for blocked cells is
            # substantially slower on cluttered scenes.
            for probe in range(targets.shape[1]):
                if not len(remaining):
                    break
                self.performance['support_rays'] += len(remaining)
                clear = self._clear_many(origin, targets[remaining, probe])
                remaining = remaining[clear]
            for index in remaining:
                key = tuple(map(int, candidates[index]))
                updates.append(dict(cell=list(key), state=KNOWN_FREE, height=heights[index], timestamp=t,
                                    source='visible_native_cspace_square_and_clearance_halo'))
                self.seen.add(key)
        directions = []
        for x in np.linspace(-.95, .95, 35):
            for y in (-.5, 0., .5):
                d = rotation @ np.array([x, y, -1.]); d /= np.linalg.norm(d)
                directions.append(d)
        self.performance['wall_rays'] += len(directions)
        if self.batch is not None:
            wall_hits = self.batch.nearest_hits(origin, np.asarray(directions),
                                                np.full(len(directions), radius),
                                                sorted(self.robot_ids | self.inactive_ids))
            self.performance['rays'] += len(directions)
            self.performance['batch_calls'] += 1
            wall_points = wall_hits['points']
            wall_ids = [int(i) if v else None for i, v in zip(wall_hits['object_ids'], wall_hits['valid'])]
        else:
            import habitat_sim
            wall_points, wall_ids = [], []
            for d in directions:
                hits = self.sim.cast_ray(habitat_sim.geo.Ray(origin, d), max_distance=radius).hits
                self.performance['rays'] += 1
                hits = sorted((h for h in hits if h.object_id not in self.robot_ids and
                               h.object_id not in self.inactive_ids), key=lambda h: h.ray_distance)
                wall_ids.append(hits[0].object_id if hits else None)
                wall_points.append(np.asarray(hits[0].point) if hits else np.zeros(3))
        for p, object_id in zip(wall_points, wall_ids):
            if object_id is None or object_id in self.all_actor_ids:
                continue
            p = np.asarray(p)
            if not pose[1]+.2 < p[1] < pose[1]+1.4:
                continue
            key = np.asarray([math.floor(p[0]/r), math.floor(p[2]/r)])
            nearby = key + self._wall_offsets
            # Native point components are retained; no ray-hit interpolation.
            displacement = np.stack([(nearby[:, 0]+.5)*r-p[0], (nearby[:, 1]+.5)*r-p[2]], axis=1)
            distances = np.linalg.norm(displacement, axis=1)
            # Scalar reference at floating-point boundary, retaining <= .25.
            near = np.flatnonzero(np.abs(distances-.25) <= 1e-14)
            for j in near:
                c = np.asarray([(nearby[j, 0]+.5)*r, pose[1], (nearby[j, 1]+.5)*r])
                distances[j] = np.linalg.norm((c-p)[[0, 2]])
            for raw in nearby[distances <= .25]:
                k = (int(raw[0]), int(raw[1]))
                updates.append(dict(cell=list(k), state=KNOWN_OCCUPIED, height=pose[1], timestamp=t,
                                    source='observed_static_wall_inflated_025m'))
                self.seen.discard(k)
        ego_key = (math.floor(pose[0]/r), math.floor(pose[2]/r))
        updates.append(dict(cell=list(ego_key), state=KNOWN_FREE, height=pose[1], timestamp=t,
                            source='occupied_ego_base_support'))
        row = dict(timestamp=t, robot_pose=pose, tracks=tracks, camera_origin=origin.tolist(),
                   camera_rotation=rotation.tolist(), static_updates=updates,
                   sensor_protocol=SENSOR_VERSION, observation_valid=True)
        self.last_timestamp, self.last_row = t, copy.deepcopy(row)
        self.performance['visibility_seconds'] += time.perf_counter()-began
        self.performance['observations'] += 1
        return row
