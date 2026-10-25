"""OFFLINE ORACLE ONLY: privileged geometry/camera diagnostic.

The real WM cannot access this engine's hidden scene/navmesh inputs. Its
accuracy and latency cannot satisfy legal-input Stage 2 acceptance. Use it
only to diagnose observation semantics and geometric upper bounds.
"""
from collections import OrderedDict
import hashlib
import time
import numpy as np
import torch
from .geometry_scene import SceneAsset
from .materialize import future_from_columns
from .increment import SparseIncrement
from eventwm_graph.stage2_segment.prepared import CompiledGraph, content_key
from eventwm_graph.stage2_segment.native import Base
from eventwm_graph.stage2_quality.features import candidate_mask
from eventwm_graph.stage2_segment.update import apply_graph_delta
from eventwm_graph.stage2.common import SCHEMA


class GeometryPredictor:
    def __init__(self, scene, navmesh, resolution, sensor, *, device='cuda:0', cache_size=4):
        self.asset = SceneAsset(scene, navmesh, resolution, sensor, device)
        self.kernel = self.asset.upload()
        self.device = torch.device(device)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.identity = self.asset.identity
        self.startup_seconds = self.asset.prepare_seconds
        self.last_batch_profile = {}

    def close(self):
        self.asset.close()

    def clear_cache(self):
        self.cache.clear()

    def prepare_graph(self, graph):
        key = content_key(graph)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        p = CompiledGraph(graph)
        if p.resolution != self.asset.resolution:
            raise ValueError('scene certificate resolution differs from graph')
        if abs(float(graph['clearance'])-.25)>1.e-9:
            raise ValueError('scene certificates require the declared 0.25m clearance')
        p.future_base=None
        if p.native_eligible:
            try:p.future_base=Base(graph,p)
            except(TypeError,ValueError,OverflowError):pass
        if self.cache_size:
            self.cache[key] = p
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return p

    def predict(self, *, partial_graph, trajectory, camera_origins, camera_rotations, materialize=True):
        return self.predict_segments(partial_graph=partial_graph, trajectories=[trajectory], camera_trajectories=[dict(origins=camera_origins, rotations=camera_rotations)], materialize=materialize)[0]

    def predict_segments(self, *, partial_graph, trajectories, camera_trajectories, materialize=True):
        tick = time.perf_counter()
        prepared = self.prepare_graph(partial_graph)
        paths = [np.asarray(p, np.float64) for p in trajectories]
        if not paths:
            return []
        if len(paths) != len(camera_trajectories):
            raise ValueError('one camera trajectory required per base trajectory')
        views, steps, dedup = [], [], {}
        for path, camera in zip(paths, camera_trajectories):
            if path.ndim != 2 or path.shape[1] != 5 or not len(path) or not np.isfinite(path).all():
                raise ValueError('finite [time,x,y,z,yaw] trajectory required')
            if np.any(np.diff(path[:, 0]) <= 0) or abs(path[0, 0]-float(partial_graph['timestamp'])) > 1.e-7:
                raise ValueError('invalid trajectory clock')
            origin, rotation = np.asarray(camera['origins'], np.float64), np.asarray(camera['rotations'], np.float64)
            if origin.shape != (len(path), 3) or rotation.shape != (len(path), 3, 3) or not np.isfinite(origin).all() or not np.isfinite(rotation).all():
                raise ValueError('finite camera origin/rotation at every supplied pose required')
            if not np.allclose(np.swapaxes(rotation, 1, 2)@rotation, np.eye(3), atol=2.e-5, rtol=0):
                raise ValueError('camera rotations must be orthonormal')
            seq = []
            packed=np.concatenate([origin,rotation.reshape(-1,9),path[:,2:3]],axis=1)
            for c in packed[1:]:
                key = c.tobytes()
                if key not in dedup:
                    dedup[key] = len(views)
                    views.append(c)
                seq.append(dedup[key])
            steps.append(np.asarray(seq, np.int32))
        xy = np.concatenate([p[:, [1, 3]].astype(np.float32) for p in paths])
        r = prepared.resolution
        lo = np.floor((xy.min(0)-11)/r).astype(np.int32)-4
        hi = np.floor((xy.max(0)+11)/r).astype(np.int32)+5
        shape = hi.astype(np.int64)-lo
        if int(np.prod(shape)) > 4000000 and len(paths)>1:
            return [self.predict(partial_graph=partial_graph, trajectory=p, camera_origins=c['origins'], camera_rotations=c['rotations'], materialize=materialize) for p,c in zip(paths,camera_trajectories)]
        if int(np.prod(shape)) > 4000000:
            raise ValueError('trajectory domain exceeds original four-million-cell limit')
        unknown = np.ones(tuple(shape), np.uint8)
        ij = prepared.keys-lo
        inside = np.all((ij>=0)&(ij<shape), axis=1)
        at = ij[inside]
        unknown[at[:, 0], at[:, 1]] = 0
        gridkeys = np.argwhere(unknown).astype(np.int32)+lo
        candidates = []
        masks = {}
        for path in paths:
            points = np.unique(path[:, [1, 3]].astype(np.float32), axis=0)
            key = points.tobytes()
            if key not in masks:
                masks[key] = gridkeys[candidate_mask(gridkeys, points, r)]
            candidates.append(masks[key])
        prepared_at = time.perf_counter()
        cameras = np.array(views, np.float64).reshape(-1, 13)
        codes = self.kernel.compute(unknown, lo, cameras, candidates, steps)
        geometry_at = time.perf_counter()
        out = []
        for path, keys, code in zip(paths, candidates, codes):
            states = np.where(code>0, 1, np.where(code<0, 2, 0)).astype(np.uint8)
            # SparseIncrement's graph interface stores F32 heights. Decode
            # once to that precision so the full graph and sparse update are
            # numerically identical, including roundoff in probe_y - .025.
            height = np.full(len(keys), prepared.height, np.float32)
            free, blocked = code>0, code<0
            height[free] = self.asset.arrays['targets'][code[free]-1, 0, 1]-.025
            height[blocked] = cameras[-code[blocked]-1, 12]
            p = np.zeros((len(keys), 3), np.float32)
            p[np.arange(len(keys)), states] = 1.
            selected = np.flatnonzero(states)
            nk, ns, nh = keys[selected], states[selected], height[selected]
            now = float(path[-1, 0])
            updates = SparseIncrement(nk, ns, nh, now)
            result = dict(schema=SCHEMA, predictor_schema='static_scene_trial_camera_increment_1', predicted=True, checkpoint_sha256=self.identity,
                base_graph_identity=prepared.identity, base_timestamp=float(partial_graph['timestamp']), timestamp=now, robot_pose=path[-1,1:].tolist(), static_updates=updates,
                candidate_cells=keys, probabilities=p, states=states, heights=height, unknown_cells=keys[states==0], outside_domain='UNKNOWN_NO_UPDATE', probability_columns=['UNKNOWN','FREE','BLOCKED'],
                observation_scope='initially_UNKNOWN_static_cells_only', increment_format='columns', prediction_task='segment_conditioned_legal_observation_1',
                provenance=dict(hypothetical=True, oracle_only=True, deployable=False, input_contract='ORACLE_PRIVILEGED_static_scene_navmesh_and_trial_camera_poses_1', scene_asset_sha256=self.identity, dynamic_occlusion='not_modeled', trajectory_sha256=hashlib.sha256(path.astype('<f8').tobytes()).hexdigest()))
            if materialize:
                future = future_from_columns(prepared.future_base, nk, ns, nh, now, path[-1,1:].tolist(), self.identity)
                if future is None:
                    result['future_partial_graph'] = apply_graph_delta(partial_graph, result)
                    result['future_format'] = 'snapshot'
                else:
                    result['future_partial_graph'] = future
                    result['future_format'] = 'native_columns'
            out.append(result)
        self.last_batch_profile = dict(queries=len(paths), distinct_camera_views=len(views), graph_candidates_and_input_seconds=prepared_at-tick,
            geometry_transfer_and_reduce_seconds=geometry_at-prepared_at, decode_and_materialize_seconds=time.perf_counter()-geometry_at, total_seconds=time.perf_counter()-tick,
            materialized_future_graphs=len(paths) if materialize else 0, serialization_included=False)
        return out
