"""Static stage-mesh BVH, native sensor parity, and conservative box bounds.

This loader deliberately supports an explicit subset of native stage assets:
uncompressed GLB2 indexed triangle geometry, standard affine node transforms,
and a stage with no additional rigid/articulated collision objects. Unsupported
compression, sparse/quantized positions, skins, morph targets and instancing
extensions fail explicitly. Native initialization attributes supply the actual
collision asset and frame. The index is immutable and contains no human state.

Point visibility implements the existing torso/frustum/range/LOS contract; it
does not introduce sensor noise. A box certificate uses affine halfspace extrema
and full triangle/convex-hull intersection, never only corner visibility rays.
Unknown boxes remain [0,1]. Uniform-box subdivision retains unresolved volume
in its upper/lower probability bounds, without renormalizing it away.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import itertools
import json
import math
import struct
from types import MappingProxyType

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


VERSION = 'native-stage-glb-bvh-visibility-bounds-2-native-edge-recovery'


class UnsupportedGeometry(ValueError):
    pass


def _readonly(value, dtype=None):
    result = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def _node_transform(node):
    if any(key in node for key in ('skin', 'weights', 'extensions')):
        raise UnsupportedGeometry('Skin, morph, or extended node geometry is unsupported')
    if 'matrix' in node:
        if any(k in node for k in ('translation', 'rotation', 'scale')):
            raise UnsupportedGeometry('Mixed matrix and TRS node transform')
        matrix = np.asarray(node['matrix'], float).reshape(4, 4, order='F')
    else:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(node.get('rotation', (0, 0, 0, 1))).as_matrix()
        matrix[:3, :3] *= np.asarray(node.get('scale', (1, 1, 1)))[None, :]
        matrix[:3, 3] = node.get('translation', (0, 0, 0))
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-12):
        raise UnsupportedGeometry('Non-affine or invalid node transform')
    return matrix


def load_glb_triangles(path):
    """Return local asset triangles plus reproducible format metadata."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    if len(raw) < 20 or struct.unpack_from('<III', raw) != (0x46546c67, 2, len(raw)):
        raise UnsupportedGeometry('Only complete GLB version2 assets are supported')
    chunks = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from('<II', raw, offset)
        offset += 8
        if offset+length > len(raw) or kind in chunks:
            raise UnsupportedGeometry('Malformed or repeated GLB chunk')
        chunks[kind] = raw[offset:offset+length]
        offset += length
    if 0x4e4f534a not in chunks or 0x004e4942 not in chunks:
        raise UnsupportedGeometry('GLB JSON and BIN chunks are required')
    doc = json.loads(chunks[0x4e4f534a])
    data = chunks[0x004e4942]
    if len(doc.get('buffers', [])) != 1 or 'uri' in doc['buffers'][0]:
        raise UnsupportedGeometry('External or multiple geometry buffers are unsupported')
    if doc.get('skins') or doc.get('animations'):
        raise UnsupportedGeometry('Animated collision stage is unsupported')
    compressed = {'KHR_draco_mesh_compression', 'EXT_meshopt_compression',
                  'EXT_mesh_gpu_instancing', 'KHR_mesh_quantization'}
    if compressed.intersection(doc.get('extensionsUsed', [])):
        raise UnsupportedGeometry('Compressed, quantized or instanced geometry is unsupported')
    for view in doc.get('bufferViews', []):
        if view.get('extensions'):
            raise UnsupportedGeometry('Extended geometry buffer views are unsupported')
    formats = {5121: np.dtype('u1'), 5123: np.dtype('<u2'),
               5125: np.dtype('<u4'), 5126: np.dtype('<f4')}

    def accessor(index, *, position=False):
        a = doc['accessors'][index]
        if a.get('sparse') or a.get('extensions') or a.get('normalized'):
            raise UnsupportedGeometry('Sparse, normalized or extended accessors are unsupported')
        if position and (a['type'] != 'VEC3' or a['componentType'] != 5126):
            raise UnsupportedGeometry('Positions must be uncompressed float32 VEC3')
        if not position and a['type'] != 'SCALAR':
            raise UnsupportedGeometry('Triangle indices must be unsigned scalar integers')
        dtype = formats.get(a['componentType'])
        if dtype is None or (not position and dtype.kind != 'u'):
            raise UnsupportedGeometry('Unsupported geometry component type')
        view = doc['bufferViews'][a['bufferView']]
        if view['buffer'] != 0:
            raise UnsupportedGeometry('External geometry buffer')
        width = 3 if position else 1
        stride = int(view.get('byteStride', width*dtype.itemsize))
        start = int(view.get('byteOffset', 0))+int(a.get('byteOffset', 0))
        count = int(a['count'])
        end = start+max(0, count-1)*stride+width*dtype.itemsize
        if count < 0 or stride < width*dtype.itemsize or end > len(data):
            raise UnsupportedGeometry('Accessor outside geometry buffer')
        result = np.ndarray((count, width), dtype=dtype, buffer=data,
                            offset=start, strides=(stride, dtype.itemsize))
        return np.array(result, copy=True) if position else np.array(result[:, 0], dtype=np.int64)

    geometry = []
    for mesh in doc.get('meshes', []):
        if mesh.get('weights') or mesh.get('extensions'):
            raise UnsupportedGeometry('Morph or extended collision meshes are unsupported')
        primitives = []
        for primitive in mesh['primitives']:
            if primitive.get('mode', 4) != 4 or primitive.get('extensions') or primitive.get('targets'):
                raise UnsupportedGeometry('Only unextended triangle primitives are supported')
            vertices = accessor(primitive['attributes']['POSITION'], position=True)
            if 'indices' not in primitive:
                raise UnsupportedGeometry('Unindexed collision triangles are unsupported')
            indices = accessor(primitive['indices'])
            if len(indices) % 3 or (len(indices) and (indices.min() < 0 or indices.max() >= len(vertices))):
                raise UnsupportedGeometry('Malformed triangle indices')
            primitives.append(vertices[indices.reshape(-1, 3)])
        geometry.append(primitives)
    scenes = doc.get('scenes', [])
    if not scenes:
        raise UnsupportedGeometry('A GLB scene with explicit node hierarchy is required')
    scene_id = int(doc.get('scene', 0))
    if scene_id < 0 or scene_id >= len(scenes):
        raise UnsupportedGeometry('Invalid default GLB scene')
    triangles = []

    def visit(index, parent, ancestry):
        if index in ancestry:
            raise UnsupportedGeometry('Cyclic GLB node hierarchy')
        node = doc['nodes'][index]
        matrix = parent@_node_transform(node)
        if 'mesh' in node:
            for local in geometry[node['mesh']]:
                triangles.append(local.astype(float)@matrix[:3, :3].T+matrix[:3, 3])
        for child in node.get('children', []):
            visit(child, matrix, ancestry+(index,))
    for index in scenes[scene_id]['nodes']:
        visit(index, np.eye(4), ())
    if not triangles:
        raise UnsupportedGeometry('No collision triangles in default GLB scene')
    result = np.concatenate(triangles)
    if not np.isfinite(result).all():
        raise UnsupportedGeometry('Nonfinite collision vertices')
    return result, {'asset': str(path), 'asset_sha256': hashlib.sha256(raw).hexdigest(),
                    'asset_bytes': len(raw), 'asset_triangles': len(result),
                    'mesh_count': len(geometry), 'node_count': len(doc['nodes']),
                    'extensions_used': tuple(doc.get('extensionsUsed', []))}


def _frame_matrix(stage):
    # ResourceManager v0.3.1 builds CoordinateFrame(up,front,origin), mapping
    # source right/up/back to canonical +X/+Y/+Z. Origins other than zero are
    # refused until their translation convention has its own native parity test.
    up = np.asarray(stage.orient_up, dtype=float)
    front = np.asarray(stage.orient_front, dtype=float)
    origin = np.asarray(stage.origin, dtype=float)
    if not np.allclose(origin, 0., atol=0., rtol=0.):
        raise UnsupportedGeometry('Nonzero native stage origin needs an explicit validated transform')
    if abs(up@front) > 1e-7 or abs(np.linalg.norm(up)-1) > 1e-7 or abs(np.linalg.norm(front)-1) > 1e-7:
        raise UnsupportedGeometry('Native stage frame must be orthonormal')
    matrix = np.stack((np.cross(front, up), up, -front))
    return matrix


def _build_bvh(triangles, leaf_size):
    lower = triangles.min(1)
    upper = triangles.max(1)
    center = (lower+upper)/2
    indices = np.arange(len(triangles), dtype=np.int64)
    bounds, left, right, first, count = [], [], [], [], []

    def build(start, stop):
        current = len(bounds)
        ids = indices[start:stop]
        lo, hi = lower[ids].min(0), upper[ids].max(0)
        bounds.append(np.concatenate((lo, hi)))
        left.append(-1);right.append(-1);first.append(start);count.append(stop-start)
        if stop-start > leaf_size:
            axis = int(np.argmax(np.ptp(center[ids], axis=0)))
            middle = (start+stop)//2
            order = np.argpartition(center[ids, axis], middle-start)
            indices[start:stop] = ids[order]
            left[current] = build(start, middle)
            right[current] = build(middle, stop)
            count[current] = 0
        return current
    build(0, len(triangles))
    return tuple(_readonly(x, dtype) for x, dtype in (
        (bounds, np.float64), (left, np.int64), (right, np.int64),
        (first, np.int64), (count, np.int64), (indices, np.int64)))


@njit(nogil=True)
def _aabb_entry(origin, direction, bounds, distance):
    low, high = 0., distance
    for axis in range(3):
        if abs(direction[axis]) < 1e-30:
            if origin[axis] < bounds[axis] or origin[axis] > bounds[axis+3]:
                return np.inf
        else:
            a = (bounds[axis]-origin[axis])/direction[axis]
            b = (bounds[axis+3]-origin[axis])/direction[axis]
            low = max(low, min(a, b))
            high = min(high, max(a, b))
            if low > high:
                return np.inf
    return low


@njit(nogil=True)
def _ray_batch(triangles, bounds, left, right, first, count, indices,
               origins, directions, max_distances):
    distances = max_distances.copy()
    hit_ids = np.full(len(origins), -1, np.int64)
    edges = np.ones(len(origins))
    ambiguous = np.zeros(len(origins), np.bool_)
    stack = np.empty(128, np.int64)
    for ray in range(len(origins)):
        size = 1;stack[0] = 0
        origin, direction = origins[ray], directions[ray]
        ox, oy, oz = origin[0], origin[1], origin[2]
        dx, dy, dz = direction[0], direction[1], direction[2]
        while size:
            size -= 1;node = stack[size]
            if not math.isfinite(_aabb_entry(origin, direction, bounds[node], distances[ray])):
                continue
            if count[node] == 0:
                a, b = left[node], right[node]
                da = _aabb_entry(origin, direction, bounds[a], distances[ray])
                db = _aabb_entry(origin, direction, bounds[b], distances[ray])
                if da > db:
                    a, b = b, a;da, db = db, da
                if math.isfinite(db):
                    if size >= len(stack):
                        raise RuntimeError('BVH traversal stack exhausted')
                    stack[size] = b;size += 1
                if math.isfinite(da):
                    if size >= len(stack):
                        raise RuntimeError('BVH traversal stack exhausted')
                    stack[size] = a;size += 1
                continue
            for position in range(first[node], first[node]+count[node]):
                index = indices[position]
                tri = triangles[index]
                ax, ay, az = tri[0, 0], tri[0, 1], tri[0, 2]
                e1x, e1y, e1z = tri[1, 0]-ax, tri[1, 1]-ay, tri[1, 2]-az
                e2x, e2y, e2z = tri[2, 0]-ax, tri[2, 1]-ay, tri[2, 2]-az
                px, py, pz = dy*e2z-dz*e2y, dz*e2x-dx*e2z, dx*e2y-dy*e2x
                determinant = e1x*px+e1y*py+e1z*pz
                if abs(determinant) < 1e-16:
                    continue
                inverse = 1./determinant
                tx, ty, tz = ox-ax, oy-ay, oz-az
                u = (tx*px+ty*py+tz*pz)*inverse
                qx, qy, qz = ty*e1z-tz*e1y, tz*e1x-tx*e1z, tx*e1y-ty*e1x
                v = (dx*qx+dy*qy+dz*qz)*inverse
                distance = (e2x*qx+e2y*qy+e2z*qz)*inverse
                # Bullet's native triangle callback admits edge cross products
                # down to -0.0001*normal.length2(), i.e. a small negative
                # barycentric coordinate even in exact arithmetic. A rejected
                # triangle therefore MUST retain ambiguity; looking only at
                # the closest accepted triangle loses these native hits.
                # This larger envelope chooses actual native recovery only;
                # it is never an enlarged sensor/triangle intersection.
                scale = math.sqrt((e1x*e1x+e1y*e1y+e1z*e1z)
                                  *(e2x*e2x+e2y*e2y+e2z*e2z))
                guard = 0.0002+0.00001*scale/abs(determinant)
                edge = min(u, v, 1.-u-v)
                if (distance >= -0.0001 and distance <= max_distances[ray]+0.0001
                        and edge >= -guard and edge <= guard):
                    ambiguous[ray] = True
                if u < 0. or u > 1. or v < 0. or u+v > 1.:
                    continue
                if distance >= 0. and distance < distances[ray]:
                    distances[ray] = distance;hit_ids[ray] = index
                    edges[ray] = edge
    return distances, hit_ids, edges, ambiguous


@njit(nogil=True)
def _triangle_hull_intersection(triangle, planes, padding):
    # Sutherland-Hodgman clipping against every convex-hull halfspace.
    polygon = np.empty((64, 3));other = np.empty((64, 3))
    polygon[:3] = triangle
    size = 3
    for j in range(len(planes)):
        if size == 0:
            return False
        output = 0
        previous = polygon[size-1]
        old = np.dot(planes[j, :3], previous)+planes[j, 3]-padding
        for k in range(size):
            current = polygon[k]
            new = np.dot(planes[j, :3], current)+planes[j, 3]-padding
            if (new <= 0.) != (old <= 0.):
                if output >= 64:
                    return True  # conservative unknown, never false-clear
                other[output] = previous+(current-previous)*(old/(old-new));output += 1
            if new <= 0.:
                if output >= 64:
                    return True
                other[output] = current;output += 1
            previous = current;old = new
        polygon[:output] = other[:output];size = output
    return size > 0


@njit(nogil=True)
def _hull_has_triangle(triangles, bounds, left, right, first, count, indices,
                       planes, padding):
    stack = np.empty(128, np.int64);stack[0] = 0;size = 1
    while size:
        size -= 1;node = stack[size]
        excluded = False
        for j in range(len(planes)):
            distance = planes[j, 3]
            for axis in range(3):
                distance += planes[j, axis]*bounds[node, axis if planes[j, axis] >= 0 else axis+3]
            if distance > padding:
                excluded = True;break
        if excluded:
            continue
        if count[node] == 0:
            if size+2 > len(stack):
                return True
            stack[size] = left[node];stack[size+1] = right[node];size += 2
        else:
            for position in range(first[node], first[node]+count[node]):
                if _triangle_hull_intersection(triangles[indices[position]], planes, padding):
                    return True
    return False


@njit(nogil=True,cache=True)
def _point_frusta(points,origin,rotation,torso,max_range,horizontal_half,vertical_half):
    """Fuse ideal point fields; near-field scalar/native recovery is unchanged."""
    n=len(points);delta=np.empty((n,3));distance=np.empty(n)
    good=np.zeros(n,np.bool_);near=np.zeros(n,np.bool_)
    for i in range(n):
        dx=points[i,0]-origin[0];dy=points[i,1]+torso-origin[1];dz=points[i,2]-origin[2]
        delta[i,0]=dx;delta[i,1]=dy;delta[i,2]=dz
        d=math.sqrt(dx*dx+dy*dy+dz*dz);distance[i]=d
        lx=dx*rotation[0,0]+dy*rotation[1,0]+dz*rotation[2,0]
        ly=dx*rotation[0,1]+dy*rotation[1,1]+dz*rotation[2,1]
        lz=dx*rotation[0,2]+dy*rotation[1,2]+dz*rotation[2,2]
        h=math.degrees(math.atan2(lx,-lz));v=math.degrees(math.atan2(ly,-lz))
        good[i]=(d<=max_range and d>=1e-9 and lz<0 and abs(h)<=horizontal_half and abs(v)<=vertical_half)
        near[i]=(abs(abs(h)-horizontal_half)<1e-10 or abs(abs(v)-vertical_half)<1e-10
                 or abs(d-max_range)<1e-10 or abs(d-1e-9)<1e-12)
    return delta,distance,good,near


@dataclass(frozen=True)
class VisibilityBounds:
    lower: float
    upper: float
    reason: str
    boxes: int = 1
    unresolved_volume: float = 0.


@dataclass(frozen=True, eq=False)
class StaticVisibility:
    triangles: np.ndarray
    bvh: tuple
    metadata: dict
    native: object = field(default=None, repr=False, compare=False)
    numeric_padding_m: float = 2e-6
    ray_fallback_tolerance_m: float = 1e-4
    version: str = field(init=False)
    ray_bvh: tuple = field(init=False, repr=False)
    edge_triangles: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'triangles', _readonly(self.triangles, np.float64))
        object.__setattr__(self, 'bvh', tuple(_readonly(x) for x in self.bvh))
        # Separate candidate bounds for Bullet's nonzero edge acceptance.
        # A barycentric component as low as -e can extend a coordinate by
        # 2e*span. Overinclude that region for native recovery and CLEAR-hull
        # exclusion. Raw triangles remain the source of shadow certificates.
        bounds = np.asarray(self.bvh[0], np.float64)
        span = bounds[:, 3:]-bounds[:, :3]
        magnitude = np.maximum(1., np.maximum(abs(bounds[:, :3]), abs(bounds[:, 3:])))
        padding = 0.0004*span+0.000002*magnitude
        candidate = np.concatenate((np.nextafter(bounds[:, :3]-padding, -np.inf),
                                    np.nextafter(bounds[:, 3:]+padding, np.inf)), axis=1)
        object.__setattr__(self, 'ray_bvh', (_readonly(candidate),)+self.bvh[1:])
        # The native accepted barycentric edge envelope is a triangle
        # homothetic to the original. It is an OCCLUDER CANDIDATE only: use
        # it to refuse a clear-pyramid certificate, never to assert a hit.
        e = 0.0002
        envelope = (1.+3.*e)*self.triangles-e*self.triangles.sum(axis=1)[:, None, :]
        object.__setattr__(self, 'edge_triangles', _readonly(envelope))
        metadata = json.loads(json.dumps(self.metadata, sort_keys=True, allow_nan=False))
        object.__setattr__(self, 'metadata', MappingProxyType(metadata))
        if self.numeric_padding_m <= 0 or self.ray_fallback_tolerance_m <= 0:
            raise ValueError('Positive conservative numerical padding required')
        digest = hashlib.sha256(VERSION.encode()+self.triangles.tobytes())
        digest.update(self.edge_triangles.tobytes())
        digest.update(self.ray_bvh[0].tobytes())
        digest.update(repr(('native-edge-candidate', e, 'native-near-edge-recovery',
                            0.0002, 0.00001)).encode())
        digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        digest.update(repr((self.numeric_padding_m, self.ray_fallback_tolerance_m)).encode())
        for value in self.bvh:
            digest.update(value.tobytes())
        object.__setattr__(self, 'version', digest.hexdigest())

    @classmethod
    def from_native(cls, geometry, *, leaf_size=12, numeric_padding_m=2e-6,
                    ray_fallback_tolerance_m=1e-4):
        """Extract the actual installed static stage, refusing other colliders."""
        sim = geometry.sim
        if sim.get_rigid_object_manager().get_num_objects() or sim.get_articulated_object_manager().get_num_objects():
            raise UnsupportedGeometry('This stage-only index does not include additional scene objects')
        stage = sim.get_stage_initialization_template()
        if (Path(str(geometry.scene)).suffix.lower() != '.glb'
                or Path(str(geometry.scene)).resolve() != Path(stage.render_asset_handle).resolve()):
            raise UnsupportedGeometry('Scene-instance or extra stage-instance transforms need separate native extraction')
        if not stage.is_collidable or stage.collision_asset_is_primitive or not stage.use_mesh_for_collision:
            raise UnsupportedGeometry('Native stage must use a collidable triangle asset')
        triangles, metadata = load_glb_triangles(stage.collision_asset_handle)
        rotation = _frame_matrix(stage)
        scale = np.asarray(stage.scale, dtype=float)*float(stage.units_to_meters)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise UnsupportedGeometry('Positive finite native stage scale required')
        if not np.all(scale == scale[0]):
            raise UnsupportedGeometry('Nonuniform native scale needs separately validated transform order')
        if not isinstance(leaf_size, int) or leaf_size <= 0:
            raise ValueError('Positive integer BVH leaf size required')
        triangles = (triangles@rotation.T)*scale
        # Native collision storage is float32; retain those world vertices.
        triangles = triangles.astype(np.float32).astype(np.float64)
        metadata.update(version=VERSION, scene=str(geometry.scene),
                        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        native_stage_up=list(stage.orient_up),
                        native_stage_front=list(stage.orient_front),
                        native_scale=scale.tolist(), native_origin=list(stage.origin),
                        stage_frame_matrix=rotation.tolist(),
                        native_margin=float(stage.margin),
                        collision_scope='stage-only triangle rays, margin parity must be validated',
                        leaf_size=int(leaf_size), numeric_padding_m=float(numeric_padding_m),
                        ray_fallback_tolerance_m=float(ray_fallback_tolerance_m))
        metadata['world_triangles_sha256'] = hashlib.sha256(triangles.tobytes()).hexdigest()
        return cls(triangles, _build_bvh(triangles, int(leaf_size)), metadata,
                   geometry, float(numeric_padding_m), float(ray_fallback_tolerance_m))

    def ray_distances(self, origins, directions, max_distances, *, return_ambiguity=False):
        o = np.asarray(origins, dtype=np.float32).astype(np.float64).reshape(-1, 3)
        d = np.asarray(directions, dtype=np.float32).astype(np.float64).reshape(-1, 3)
        limits = np.broadcast_to(np.asarray(max_distances, float), (len(o),)).copy()
        if len(d) != len(o) or not np.isfinite(o).all() or not np.isfinite(d).all() or (limits <= 0).any():
            raise ValueError('Finite paired rays with positive limits required')
        result = _ray_batch(self.triangles, *self.ray_bvh, o, d, limits)
        return result if return_ambiguity else result[:3]

    def visible(self, points, coverage, *, native_boundary_fallback=True):
        """Existing exact sensor geometry, with native fallback near ray edges."""
        if coverage.get('kind') != 'native_static':
            raise UnsupportedGeometry('Native-static coverage required')
        p = np.asarray(points, float);shape = p.shape[:-1];flat = p.reshape(-1, 3)
        cfg = coverage['sensor'];origin = np.asarray(coverage['origin'], float)
        rotation = np.asarray(coverage['rotation'], float)
        delta,distance,good,near_field=_point_frusta(flat,origin,rotation,float(cfg['torso_height_m']),
            float(cfg['max_range_m']),float(cfg['horizontal_fov_deg']/2),float(cfg['vertical_fov_deg']/2))
        # Match the official scalar atan2/degrees comparisons at exact angular
        # boundaries; tan(45deg) rounding must not reject an included45deg ray.
        for index in np.flatnonzero(near_field):
            d = float(np.linalg.norm(delta[index]))
            local_point = np.asarray(coverage['rotation'], float).T@delta[index]
            h = math.degrees(math.atan2(local_point[0], -local_point[2]))
            v = math.degrees(math.atan2(local_point[1], -local_point[2]))
            distance[index] = d
            good[index] = (d <= cfg['max_range_m'] and d >= 1e-9 and local_point[2] < 0
                           and abs(h) <= cfg['horizontal_fov_deg']/2
                           and abs(v) <= cfg['vertical_fov_deg']/2)
        selected = np.flatnonzero(good)
        fallback_count = 0
        if len(selected):
            hit, ids, edges, ambiguous = self.ray_distances(np.broadcast_to(origin, (len(selected), 3)),
                delta[selected]/distance[selected, None], distance[selected], return_ambiguity=True)
            threshold = distance[selected]-cfg['los_tolerance_m']
            good[selected] = hit >= threshold
            # This guard selects exact native fallback, never changes the
            # sensor's LOS tolerance. It is separate from hull geometry padding:
            # grazing rays amplify world-coordinate error into distance error.
            near = ((np.abs(hit-threshold) <= self.ray_fallback_tolerance_m)
                    | ((ids >= 0) & (edges < 1e-6)) | ambiguous)
            if native_boundary_fallback and near.any():
                if self.native is None:
                    raise UnsupportedGeometry('Native boundary fallback handle unavailable')
                import habitat_sim
                for local_index in np.flatnonzero(near):
                    index = selected[local_index]
                    ray = habitat_sim.geo.Ray(origin, delta[index]/distance[index])
                    hits = self.native.sim.cast_ray(ray, max_distance=float(distance[index])).hits
                    good[index] = not any(h.ray_distance < distance[index]-cfg['los_tolerance_m'] for h in hits)
                    fallback_count += 1
        return good.reshape(shape), {'rays': len(selected), 'native_boundary_fallbacks': fallback_count}

    def _single_triangle_shadow(self, corners, origin, triangle, tolerance):
        """Whole-box inclusion in one triangle's convex shadow cone.

        The box must lie beyond its plane by at least LOS tolerance in normal
        distance, which also lower-bounds the remaining along-ray distance.
        Each cone-face inequality is affine, so corner extrema certify its
        entire convex box. This is not inference from corner ray outcomes.
        """
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        length = np.linalg.norm(normal)
        if length == 0:
            return False
        normal /= length
        if normal@(origin-triangle[0]) < 0:
            normal = -normal
        if normal@(origin-triangle[0]) <= self.numeric_padding_m:
            return False
        if np.max((corners-triangle[0])@normal) >= -tolerance-self.numeric_padding_m:
            return False
        vectors = triangle-origin
        for j in range(3):
            edge = np.cross(vectors[j], vectors[(j+1) % 3])
            norm = np.linalg.norm(edge)
            if norm == 0:
                return False
            edge /= norm
            if edge@vectors[(j+2) % 3] < 0:
                edge = -edge
            if np.min((corners-origin)@edge) <= self.numeric_padding_m:
                return False
        return True

    def box_bounds(self, lo, hi, coverage):
        """Bounds valid for any distribution supported inside this base box."""
        if coverage.get('kind') != 'native_static':
            raise UnsupportedGeometry('Native-static coverage required')
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if lo.shape != (3,) or hi.shape != (3,) or (hi < lo).any() or not np.isfinite([lo, hi]).all():
            raise ValueError('Finite ordered XYZ box required')
        cfg = coverage['sensor'];origin = np.asarray(coverage['origin'], float)
        torso_lo = lo+[0., cfg['torso_height_m'], 0.]
        torso_hi = hi+[0., cfg['torso_height_m'], 0.]
        corners = np.asarray(list(itertools.product(*zip(torso_lo, torso_hi))))
        delta = corners-origin
        rotation = np.asarray(coverage['rotation'], float)
        local = delta@rotation
        horizontal = math.tan(math.radians(cfg['horizontal_fov_deg']/2))
        vertical = math.tan(math.radians(cfg['vertical_fov_deg']/2))
        # Each column is a linear frustum inequality <=0 over the whole box.
        inequalities = np.stack((local[:, 2], local[:, 0]+horizontal*local[:, 2],
                                 -local[:, 0]+horizontal*local[:, 2],
                                 local[:, 1]+vertical*local[:, 2],
                                 -local[:, 1]+vertical*local[:, 2]), axis=1)
        normals = np.stack((rotation[:, 2], rotation[:, 0]+horizontal*rotation[:, 2],
                            -rotation[:, 0]+horizontal*rotation[:, 2],
                            rotation[:, 1]+vertical*rotation[:, 2],
                            -rotation[:, 1]+vertical*rotation[:, 2]))
        inequalities /= np.linalg.norm(normals, axis=1)
        padding = self.numeric_padding_m
        closest = np.maximum(np.maximum(torso_lo-origin, origin-torso_hi), 0.)
        min_distance = np.linalg.norm(closest)
        if np.any(inequalities.min(0) > padding) or min_distance > cfg['max_range_m']+padding:
            return VisibilityBounds(0., 0., 'entire-box-outside-range-or-frustum')
        inside = (np.all(inequalities.max(0) < -padding)
                  and np.linalg.norm(delta, axis=1).max() < cfg['max_range_m']-padding
                  and min_distance > 1e-9+padding)
        if not inside:
            return VisibilityBounds(0., 1., 'range-or-frustum-boundary', unresolved_volume=1.)
        center = (torso_lo+torso_hi)/2
        direction = center-origin
        distance = np.linalg.norm(direction)
        hit, ids, edges = self.ray_distances(origin[None], (direction/distance)[None], distance)
        if ids[0] >= 0 and self._single_triangle_shadow(
                corners, origin, self.triangles[ids[0]], cfg['los_tolerance_m']):
            return VisibilityBounds(0., 0., 'entire-box-inside-triangle-shadow')
        try:
            hull = ConvexHull(np.concatenate((origin[None], corners)))
        except QhullError:
            return VisibilityBounds(0., 1., 'degenerate-pyramid', unresolved_volume=1.)
        # The hull contains every entire camera-to-torso segment. A triangle
        # clipping test covers all its interior, not only the corner rays.
        intersects = _hull_has_triangle(self.edge_triangles, *self.ray_bvh,
                                        np.asarray(hull.equations), padding)
        if not intersects:
            return VisibilityBounds(1., 1., 'entire-ray-pyramid-clear')
        return VisibilityBounds(0., 1., 'native-edge-envelope-intersects-ray-pyramid', unresolved_volume=1.)

    def uniform_box_bounds(self, lo, hi, coverage, *, max_depth=4, max_boxes=127):
        """Certified volume interval for an explicitly uniform 3D base box.

        Unresolved subboxes contribute their full volume to upper but not lower.
        These volume weights are not a general nonuniform transition integral.
        """
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(hi <= lo):
            raise ValueError('Uniform volume bounds require a positive 3D box')
        stack = [(lo, hi, 0, 1.)]
        lower = upper = 0.;checked = 0
        while stack:
            a, b, depth, mass = stack.pop()
            bound = self.box_bounds(a, b, coverage);checked += 1
            if bound.lower == bound.upper:
                lower += mass*bound.lower;upper += mass*bound.upper
            elif depth >= max_depth or checked+len(stack)+2 > max_boxes:
                upper += mass
            else:
                axis = int(np.argmax(b-a));middle = (a[axis]+b[axis])/2
                first_hi = b.copy();first_hi[axis] = middle
                second_lo = a.copy();second_lo[axis] = middle
                stack.append((a, first_hi, depth+1, mass/2))
                stack.append((second_lo, b, depth+1, mass/2))
        return VisibilityBounds(float(lower), float(upper), 'uniform-box-conservative-subdivision',
                                checked, float(upper-lower))
