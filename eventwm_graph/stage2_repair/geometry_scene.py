"""Query-independent scene assets for the revised Stage 2 contract.

Every supported navigation cell on every projected floor is prepared, without
reading graphs, trial paths, observations or target labels. Scene preparation
is startup work, analogous to loading model weights, and is timed separately.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import time
import numpy as np
from numba import njit
from .diagnose import OUT, sha, save, read
from .geometry_reference import NativeStaticPredictor
from eventwm_graph.l4.mesh_visibility import StaticVisibility


@njit(cache=True)
def projected_cells(triangles, resolution):
    xx, zz, yy = [], [], []
    for t in triangles:
        ax, az = t[0, 0], t[0, 2]
        bx, bz = t[1, 0]-ax, t[1, 2]-az
        cx, cz = t[2, 0]-ax, t[2, 2]-az
        det = bx*cz-bz*cx
        if abs(det) < 1.e-12:
            continue
        lo = int(np.ceil(np.min(t[:, 0])/resolution-.5))
        hi = int(np.floor(np.max(t[:, 0])/resolution-.5))
        lowz = int(np.ceil(np.min(t[:, 2])/resolution-.5))
        highz = int(np.floor(np.max(t[:, 2])/resolution-.5))
        if (hi-lo+1)*(highz-lowz+1) > 10000000:
            raise ValueError('unsupported navmesh raster extent')
        for x in range(lo, hi+1):
            for z in range(lowz, highz+1):
                dx, dz = (x+.5)*resolution-ax, (z+.5)*resolution-az
                u, v = (dx*cz-dz*cx)/det, (bx*dz-bz*dx)/det
                if u >= -1.e-8 and v >= -1.e-8 and u+v <= 1.+1.e-8:
                    xx.append(x)
                    zz.append(z)
                    yy.append(t[0, 1]+u*(t[1, 1]-t[0, 1])+v*(t[2, 1]-t[0, 1]))
    return np.asarray(xx), np.asarray(zz), np.asarray(yy)


class SceneAsset:
    def __init__(self, scene, navmesh, resolution, sensor, device='cuda:0'):
        tick = time.perf_counter()
        for key,value in [('max_range_m',10.),('horizontal_fov_deg',90.),('vertical_fov_deg',90.)]:
            if abs(float(sensor[key])-value)>1.e-7:raise ValueError('unsupported geometry sensor '+key)
        self.reference = NativeStaticPredictor(scene, navmesh, resolution, sensor)
        self.resolution = float(resolution)
        identity = dict(schema='static_certificate_scene_1', scene_sha256=sha(self.reference.scene), navmesh_sha256=sha(self.reference.navmesh), resolution=resolution, source_sha256=sha(__file__), support_source_sha256=sha(Path(__file__).parents[1]/'partial/observer.py'), probe_source_sha256=sha(Path(__file__).parents[1]/'partial/fast_observer.py'), bank_source_sha256=sha(Path(__file__).with_name('geometry_reference.py')), mesh_source_sha256=sha(Path(__file__).parents[1]/'l4/mesh_visibility.py'))
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = OUT/'geometry/assets'/self.identity
        directory.mkdir(parents=True, exist_ok=True)
        receipt, arrays = directory/'identity.json', directory/'arrays.npz'
        with (directory/'build.lock').open('a+b')as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            self._read_or_build(receipt,arrays,identity,tick)
            self.metadata=read(receipt)
        self.prepare_seconds = time.perf_counter()-tick
        self.device = device
        self.gpu = None

    def _read_or_build(self,receipt,arrays,identity,tick):
        if receipt.exists():
            metadata = read(receipt)
            if metadata['identity'] != identity or sha(arrays) != metadata['arrays_sha256']:
                raise ValueError('stale scene asset')
            with np.load(arrays, allow_pickle=False) as z:
                self.arrays = {k: z[k] for k in z.files}
        else:
            sim = self.reference.geometry.sim
            vertices = np.asarray(sim.pathfinder.build_navmesh_vertices(), np.float64)
            indices = np.asarray(sim.pathfinder.build_navmesh_vertex_indices(), np.int64)
            navtri = vertices[indices.reshape(-1, 3)]
            x, z, h = projected_cells(navtri, self.resolution)
            candidates = np.column_stack([x, z, np.rint(h/.1).astype(np.int64)])
            _, take = np.unique(candidates, axis=0, return_index=True)
            keys, targets = [], []
            for i in take:
                key = (int(x[i]), int(z[i]))
                value = self.reference.bank._targets(key, float(h[i]))
                if value is not None:
                    keys.append(key)
                    targets.append(value[1])
            mesh = StaticVisibility.from_native(self.reference.geometry)
            triangles = mesh.triangles.astype(np.float32)
            e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
            packed = np.concatenate([triangles[:, 0], e1, e2], axis=1)
            self.arrays = dict(keys=np.asarray(keys, np.int32).reshape(-1, 2), targets=np.asarray(targets, np.float64).reshape(-1, 63, 3), triangles=packed)
            for name, data in zip(['bounds', 'left', 'right', 'first', 'count', 'indices'], mesh.ray_bvh):
                self.arrays[name] = np.asarray(data, np.float32 if name == 'bounds' else np.int32)
            np.savez(arrays, **self.arrays)
            save(receipt, dict(identity=identity, arrays_sha256=sha(arrays), support_candidates=len(take), certificates=len(keys), triangles=len(packed), prepare_seconds=time.perf_counter()-tick, mesh_metadata=dict(mesh.metadata)))

    def upload(self):
        if self.gpu is None:
            from .geometry_cuda import GeometryKernel
            self.gpu = GeometryKernel(self.arrays, self.resolution, self.device)
        return self.gpu

    def close(self):
        self.reference.close()
