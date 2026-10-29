"""Fast environment-side range camera, numerically equivalent to the native ``CurrentRangeSensor``.

Native reference (``stage2_repair.current_range_sensor``): one Habitat/Bullet ``castRay`` per pixel (256 x 256 rays,
90 degree frustum, 10 m, all hits sorted, first hit that is not the robot or an inactive agent; ~1.5 s per frame).

This sensor splits every frame into
* rays that can only meet the static stage: cast on the GPU against the stage's collision triangles with an exact
  float32 emulation of Bullet's ``btTriangleRaycastCallback`` (same operations and order; NVRTC with --fmad=false),
  in the stage asset's local frame; the range is ``double(fraction) * 10`` as Habitat reports it;
* rays that come within an active human's collision capsules (URDF boxes + ``CAPSULE_MARGIN``), rays near an
  edge-acceptance or 10 m boundary decision, and any traversal overflow: re-cast with the native batch call itself
  (exact by construction).

Measured agreement with the native sensor (``validate``): identical hit / no-hit pattern; ranges equal or within a
few float32 ulps (Bullet's own ray transform into the stage frame is not reproduced bit for bit).  The stage index
holds only the static collision mesh of the loaded scene, which the environment owns; nothing here is given to the
world model except the finished range frame.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .ranges import runtime_view
from .context import RESOLUTION, RANGE_SIZE

VERSION = 'sandbox_fast_range_3'
MAX_RANGE = 10.
CAPSULE_MARGIN = 0.02      # m added to each humanoid collision box's capsule (box half-diagonal); the native hit points
                           # lie at most 0.010 m outside the bare capsules (Bullet's box margin), measured on 202,799
                           # human hits in 206 frames of 3 episodes; validate() re-checks every frame it compares
LEAF = 12
ROOT = Path(__file__).resolve().parents[2]

SOURCE = r'''
__device__ __forceinline__ bool slab(const float* b, float sx, float sy, float sz, float ix, float iy, float iz,
                                     float hi, float* enter){
 // conservative float32 slab test of the segment from + t (to - from), t in [0, hi]; i* = 1 / (to - from) per axis
 // (+-inf for an axis-parallel segment); node bounds are padded outward and the interval widened by 1e-5 (0.1 mm)
 float lo=0.f;
 for(int a=0;a<3;a++){
  const float s=a==0?sx:(a==1?sy:sz), inv=a==0?ix:(a==1?iy:iz);
  const float bl=b[a], bh=b[a+3];
  if(isinf(inv)){ if(s<bl||s>bh) return false; continue; }
  float t0=(bl-s)*inv, t1=(bh-s)*inv; if(t0>t1){float t=t0;t0=t1;t1=t;}
  if(t0>lo)lo=t0; if(t1<hi)hi=t1;
  if(lo>hi+1e-5f) return false;
 }
 *enter=lo; return true;
}
extern "C" __global__ void cast_static(const float* __restrict__ tri, const float* __restrict__ bounds,
  const int* __restrict__ left, const int* __restrict__ right, const int* __restrict__ first, const int* __restrict__ count,
  const int* __restrict__ index, const float* __restrict__ rays, int n, float maxd,
  const float* __restrict__ boxes, int nbox, const float* __restrict__ caps, const int* __restrict__ offsets,
  float* __restrict__ out_fr, unsigned char* __restrict__ out_flag){
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n) return;
 const float ox=rays[0], oy=rays[1], oz=rays[2];
 const float dx=rays[3+3*r], dy=rays[4+3*r], dz=rays[5+3*r];
 const float tx=ox+dx*maxd, ty=oy+dy*maxd, tz=oz+dz*maxd;           // Habitat: to = origin + direction * maxDistance
 const double sx=ox, sy=oy, sz=oz, ex=(double)tx-sx, ey=(double)ty-sy, ez=(double)tz-sz;
 const float ix=1.f/(tx-ox), iy=1.f/(ty-oy), iz=1.f/(tz-oz);
 float best=2.f, best_g=0.f; int flag=0;
 const float scale=fmaxf(fmaxf(fmaxf(fabsf(ox),fabsf(oy)),fmaxf(fabsf(oz),fabsf(tx))),fmaxf(fmaxf(fabsf(ty),fabsf(tz)),1.f));
 int stack[64]; int sp=0; float enter;
 if(slab(bounds, ox,oy,oz,ix,iy,iz, 1.f+1e-5f, &enter)) stack[sp++]=0;
 while(sp>0){
  int node=stack[--sp];
  const float hi=(best<1.f?best:1.f)+1e-5f;
  if(!slab(bounds+6*node, ox,oy,oz,ix,iy,iz, hi, &enter)) continue;
  if(count[node]==0){
   int a=left[node], b=right[node]; float ea, eb;
   bool ha=slab(bounds+6*a, ox,oy,oz,ix,iy,iz, hi, &ea), hb=slab(bounds+6*b, ox,oy,oz,ix,iy,iz, hi, &eb);
   if(sp+2>64){ flag|=4; break; }
   if(ha&&hb){ if(ea<=eb){ stack[sp++]=b; stack[sp++]=a; } else { stack[sp++]=a; stack[sp++]=b; } }
   else if(ha) stack[sp++]=a; else if(hb) stack[sp++]=b;
   continue;
  }
  for(int k=first[node]; k<first[node]+count[node]; k++){
   const float* t=tri+9*index[k];
   const float v0x=t[0],v0y=t[1],v0z=t[2],v1x=t[3],v1y=t[4],v1z=t[5],v2x=t[6],v2y=t[7],v2z=t[8];
   const float ax=v1x-v0x, ay=v1y-v0y, az=v1z-v0z, bx=v2x-v0x, by=v2y-v0y, bz=v2z-v0z;
   const float nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
   const float dist=(v0x*nx+v0y*ny)+v0z*nz;
   const float da=((nx*ox+ny*oy)+nz*oz)-dist;
   const float db=((nx*tx+ny*ty)+nz*tz)-dist;
   if(da*db>=0.f) continue;
   const float fr=da/(da-db);
   const float n2=(nx*nx+ny*ny)+nz*nz;
   const float et=n2*-0.0001f;
   const float s1=1.f-fr;
   const float px=s1*ox+fr*tx, py=s1*oy+fr*ty, pz=s1*oz+fr*tz;
   const float q0x=v0x-px, q0y=v0y-py, q0z=v0z-pz, q1x=v1x-px, q1y=v1y-py, q1z=v1z-pz, q2x=v2x-px, q2y=v2y-py, q2z=v2z-pz;
   float cx=q0y*q1z-q0z*q1y, cy=q0z*q1x-q0x*q1z, cz=q0x*q1y-q0y*q1x;
   const float e0=(cx*nx+cy*ny)+cz*nz;
   cx=q1y*q2z-q1z*q2y; cy=q1z*q2x-q1x*q2z; cz=q1x*q2y-q1y*q2x;
   const float e1=(cx*nx+cy*ny)+cz*nz;
   cx=q2y*q0z-q2z*q0y; cy=q2z*q0x-q2x*q0z; cz=q2x*q0y-q2y*q0x;
   const float e2=(cx*nx+cy*ny)+cz*nz;
   const bool acc=e0>=et && e1>=et && e2>=et;
   // Edge decisions within the float32 error bound of e_k = (q_a x q_b) . n are ambiguous: the hit point carries an
   // absolute error ~ulp(|coordinates|) (and Bullet's own ray differs from ours by ~1 ulp), so
   // |delta e_k| <~ eps_p * (|q_a| + |q_b|) * |n| with eps_p = 3e-7 * coordinate scale (generous; tiny triangles of the
   // scanned meshes make this the dominant term, see validate())
   const float la=sqrtf(q0x*q0x+q0y*q0y+q0z*q0z), lb=sqrtf(q1x*q1x+q1y*q1y+q1z*q1z), lc=sqrtf(q2x*q2x+q2y*q2y+q2z*q2z);
   const float nn=sqrtf(n2);
   const float ep=3e-7f*scale*nn;
   const bool amb=fabsf(e0-et)<ep*(la+lb)+n2*1e-5f || fabsf(e1-et)<ep*(lb+lc)+n2*1e-5f || fabsf(e2-et)<ep*(lc+la)+n2*1e-5f;
   if(fr<1.f){
    if(acc && fr<best){ best=fr; best_g=fr*maxd*nn/fabsf(da); }   // hit distance / perpendicular camera distance
    if(amb && fr<=best) flag|=1;                 // near Bullet's edge tolerance: native decides
    if((acc||amb) && fr>0.99999f) flag|=1;       // near the 10 m limit
   } else if(fr<1.00001f && (acc||amb)) flag|=1;
  }
 }
 if(best<1.f && best_g>50.f) flag|=8;            // grazing incidence (< ~1.1 deg): the float32 plane distances lose
                                                  // ~ulp(|coordinates|) * best_g, so the native call decides
 // active humans: the segment up to the static hit comes within a collision capsule -> native
 double reach=(best<1.f?(double)best:1.0)+1e-4;
 const double d1x=ex*reach, d1y=ey*reach, d1z=ez*reach, A=d1x*d1x+d1y*d1y+d1z*d1z;
 for(int h=0; h<nbox && !(flag&2); h++){
  float e2; if(!slab(boxes+6*h, ox,oy,oz,ix,iy,iz, (float)reach, &e2)) continue;
  for(int k=offsets[h]; k<offsets[h+1]; k++){
   const float* c=caps+7*k;
   const double d2x=(double)c[3]-c[0], d2y=(double)c[4]-c[1], d2z=(double)c[5]-c[2];
   const double rx=sx-c[0], ry=sy-c[1], rz=sz-c[2];
   const double E=d2x*d2x+d2y*d2y+d2z*d2z, F=d2x*rx+d2y*ry+d2z*rz, C=d1x*rx+d1y*ry+d1z*rz, B=d1x*d2x+d1y*d2y+d1z*d2z;
   double s_=0., t_=0.;                                             // closest points of two segments (Ericson 5.1.9)
   if(E<=1e-18){ s_=A>1e-18?fmin(fmax(-C/A,0.),1.):0.; t_=0.; }
   else{
    const double den=A*E-B*B;
    s_=den>1e-18?fmin(fmax((B*F-C*E)/den,0.),1.):0.;
    t_=(B*s_+F)/E;
    if(t_<0.){ t_=0.; s_=fmin(fmax(-C/A,0.),1.); } else if(t_>1.){ t_=1.; s_=fmin(fmax((B-C)/A,0.),1.); }
   }
   const double qx=rx+d1x*s_-d2x*t_, qy=ry+d1y*s_-d2y*t_, qz=rz+d1z*s_-d2z*t_;
   if(qx*qx+qy*qy+qz*qz<=(double)c[6]*(double)c[6]){ flag|=2; break; }
  }
 }
 out_fr[r]=best; out_flag[r]=(unsigned char)flag;
}
'''


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def _signed_permutation(R):
    return np.all(np.isin(R, (-1., 0., 1.))) and np.all(np.abs(R).sum(0) == 1) and np.all(np.abs(R).sum(1) == 1)


class StageIndex:
    """The scene's static collision triangles (float32, Bullet's storage precision) with a padded BVH.

    Triangles are kept in the asset's local frame when the stage frame is a signed permutation (HM3D / MP3D: z-up
    assets), so the GPU test sees the coordinates in Bullet's order; otherwise in the world frame.  Cached on disk
    per asset SHA-256 (``SANDBOX_CACHE``, default ``output/sandbox_cache``).
    """

    def __init__(self, sim, cache_dir=None):
        from eventwm_graph.l4.mesh_visibility import load_glb_triangles, _frame_matrix, _build_bvh
        stage = sim.get_stage_initialization_template()
        if sim.get_rigid_object_manager().get_num_objects():
            raise RuntimeError('fast range sensor: scenes with rigid objects need the native sensor')
        if not stage.is_collidable or stage.collision_asset_is_primitive or not stage.use_mesh_for_collision:
            raise RuntimeError('fast range sensor: the stage must collide with its triangle mesh')
        path = stage.collision_asset_handle
        if Path(path).suffix.lower() != '.glb':
            raise RuntimeError('fast range sensor: GLB stage assets only')
        R = _frame_matrix(stage)
        scale = np.asarray(stage.scale, float)*float(stage.units_to_meters)
        if not np.all(scale == scale[0]) or scale[0] <= 0:
            raise RuntimeError('fast range sensor: uniform positive stage scale required')
        self.local = bool(_signed_permutation(R))
        self.R = R
        digest = _sha256(path)
        cache = Path(cache_dir or os.environ.get('SANDBOX_CACHE', ROOT/'output/sandbox_cache'))/'stage_bvh'
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(('%s|%s|%r|%r|%d' % (VERSION, digest, R.tolist(), scale.tolist(), self.local)).encode()).hexdigest()[:24]
        f = cache/(key+'.npz')
        t0 = time.perf_counter()
        if f.exists():
            with np.load(f) as z:
                arrays = {k: z[k] for k in z.files}
        else:
            raw, meta = load_glb_triangles(path)
            tri = (raw*scale[0]) if self.local else (raw@R.T)*scale[0]
            tri32 = tri.astype(np.float32)
            bounds, left, right, first, count, indices = _build_bvh(tri32.astype(np.float64), LEAF)
            span = bounds[:, 3:]-bounds[:, :3]
            mag = np.maximum(1., np.maximum(np.abs(bounds[:, :3]), np.abs(bounds[:, 3:])))
            pad = 0.0004*span+0.000002*mag+1e-5
            lo = np.nextafter((bounds[:, :3]-pad).astype(np.float32), np.float32(-np.inf))
            hi = np.nextafter((bounds[:, 3:]+pad).astype(np.float32), np.float32(np.inf))
            arrays = dict(tri=tri32.reshape(-1, 9), bounds=np.concatenate([lo, hi], 1).astype(np.float32),
                          left=left.astype(np.int32), right=right.astype(np.int32), first=first.astype(np.int32),
                          count=count.astype(np.int32), index=indices.astype(np.int32),
                          meta=np.frombuffer(json.dumps(dict(meta, version=VERSION, local=self.local)).encode(), np.uint8))
            tmp = f.with_suffix('.tmp.npz')
            np.savez(tmp, **arrays)
            os.replace(tmp, f)
        self.arrays = arrays
        self.load_s = time.perf_counter()-t0
        self.triangles = len(arrays['tri'])
        self.asset_sha256 = digest

    def to_local(self, world):
        """World -> index frame (exact for signed permutations)."""
        w = np.asarray(world, np.float64)
        return w@self.R if self.local else w


def kernel_ptx(source, capability):
    """The PTX ``stage2_repair.cuda_module`` builds for ``source``: same NVRTC options, same on-disk cache entry."""
    import ctypes as ct
    import sys
    options = [('--gpu-architecture=compute_%d%d' % tuple(capability)).encode(), b'--fmad=false', b'--prec-div=true',
               b'--prec-sqrt=true', b'--std=c++11']
    identity = hashlib.sha256(source.encode()+b'\0'.join(options)).hexdigest()
    cache = Path(os.environ['NUMBA_CACHE_DIR'])/'geometry_cuda'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/(identity+'.ptx')
    if not path.exists():
        lib = ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'))

        def ck(name, types, *args):
            f = getattr(lib, name); f.argtypes = types; f.restype = ct.c_int
            if f(*args):
                raise RuntimeError('%s failed' % name)
        program = ct.c_void_p()
        ck('nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p],
           ct.byref(program), source.encode(), b'geometry.cu', 0, None, None)
        try:
            ck('nvrtcCompileProgram', [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)], program, len(options),
               (ct.c_char_p*len(options))(*options))
            size = ct.c_size_t()
            ck('nvrtcGetPTXSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
            buf = ct.create_string_buffer(size.value)
            ck('nvrtcGetPTX', [ct.c_void_p, ct.c_void_p], program, buf)
            import tempfile
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                stream.write(buf.raw)
                tmp = stream.name
            os.replace(tmp, path)
        finally:
            ck('nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)], ct.byref(program))
    return path.read_bytes()


class DriverCuda:
    """Torch-free CUDA runtime for the range kernel (``FastRangeSensor(backend='driver')``): the device's primary
    context through the driver API, the PTX of ``kernel_ptx`` and plain device buffers.  A worker with no other GPU work
    then holds ~0.3 GB of GPU memory instead of a PyTorch CUDA context's ~0.7 GB (torch 1.12 / CUDA 11.3: no lazy
    module loading).  Same kernel, same inputs, same launch shape: outputs identical to the torch backend (tested)."""

    _shared = {}           # (device, source) -> (driver functions, context, module, functions): one per process

    def __init__(self, source, device=0):
        import ctypes as ct
        self.ct = ct
        self.buffers = {}
        key = (int(device), hashlib.sha256(source.encode()).hexdigest())
        shared = DriverCuda._shared.get(key)
        if shared is None:
            shared = DriverCuda._shared[key] = self._open(source, int(device))
        self.f, self.ctx, self.module, self.functions = shared

    def _open(self, source, device):
        ct = self.ct
        cu = ct.CDLL('libcuda.so.1')
        P, U64, SZ, I, U = ct.c_void_p, ct.c_uint64, ct.c_size_t, ct.c_int, ct.c_uint
        sig = dict(cuInit=[U], cuDeviceGet=[ct.POINTER(I), I], cuDevicePrimaryCtxRetain=[ct.POINTER(P), I],
                   cuCtxSetCurrent=[P], cuDeviceGetAttribute=[ct.POINTER(I), I, I], cuModuleLoadData=[ct.POINTER(P), P],
                   cuModuleGetFunction=[ct.POINTER(P), P, ct.c_char_p], cuMemAlloc_v2=[ct.POINTER(U64), SZ],
                   cuMemFree_v2=[U64], cuMemcpyHtoD_v2=[U64, P, SZ], cuMemcpyDtoH_v2=[P, U64, SZ],
                   cuLaunchKernel=[P, U, U, U, U, U, U, U, P, ct.POINTER(P), P])
        self.f = {}
        for name, types in sig.items():
            fn = getattr(cu, name); fn.argtypes = types; fn.restype = I
            self.f[name] = fn
        self._ck('cuInit', 0)
        dev = I(); self._ck('cuDeviceGet', ct.byref(dev), device)
        self.ctx = P(); self._ck('cuDevicePrimaryCtxRetain', ct.byref(self.ctx), dev)
        self._ck('cuCtxSetCurrent', self.ctx)
        major, minor = I(), I()
        self._ck('cuDeviceGetAttribute', ct.byref(major), 75, dev)      # COMPUTE_CAPABILITY_MAJOR
        self._ck('cuDeviceGetAttribute', ct.byref(minor), 76, dev)      # COMPUTE_CAPABILITY_MINOR
        module = P()
        self._ck('cuModuleLoadData', ct.byref(module), ct.create_string_buffer(kernel_ptx(source, (major.value, minor.value))))
        return self.f, self.ctx, module, {}

    def close(self):
        """Free this sensor's device buffers (the shared context and module stay loaded for the process)."""
        buffers, self.buffers = self.buffers, {}
        for ptr, _ in buffers.values():
            try:
                self.current()
                self._ck('cuMemFree_v2', ptr)
            except Exception:
                pass

    def __del__(self):
        self.close()

    def _ck(self, name, *args):
        code = self.f[name](*args)
        if code:
            raise RuntimeError('%s returned CUDA status %d' % (name, code))

    def current(self):
        self._ck('cuCtxSetCurrent', self.ctx)

    def buffer(self, key, nbytes):
        """Device pointer of a reusable allocation of at least ``nbytes`` (grown on demand)."""
        cur = self.buffers.get(key)
        if cur is None or cur[1] < nbytes:
            if cur is not None:
                self._ck('cuMemFree_v2', self.ct.c_uint64(cur[0]))
            size = max(int(nbytes), 256)
            p = self.ct.c_uint64()
            self._ck('cuMemAlloc_v2', self.ct.byref(p), size)
            cur = self.buffers[key] = (p.value, size)
        return cur[0]

    def upload(self, key, array):
        a = np.ascontiguousarray(array)
        ptr = self.buffer(key, a.nbytes)
        if a.nbytes:
            self._ck('cuMemcpyHtoD_v2', ptr, a.ctypes.data, a.nbytes)
        return ptr

    def download(self, ptr, out):
        if out.nbytes:
            self._ck('cuMemcpyDtoH_v2', out.ctypes.data, ptr, out.nbytes)     # legacy default stream: after the kernel
        return out

    def launch(self, name, total, values, threads=128):
        ct = self.ct
        if not total:
            return
        fn = self.functions.get(name)
        if fn is None:
            fn = ct.c_void_p()
            self._ck('cuModuleGetFunction', ct.byref(fn), self.module, name.encode())
            self.functions[name] = fn
        args = [ct.c_void_p(v) if isinstance(v, int) else v for v in values]   # device pointers are ints here
        params = (ct.c_void_p*len(args))(*[ct.addressof(v) for v in args])
        self._ck('cuLaunchKernel', fn, (total+threads-1)//threads, 1, 1, threads, 1, 1, 0, None, params, None)


class FastRangeSensor:
    kind = 'fast'

    def __init__(self, task, observation_config, size=RANGE_SIZE, device='cuda:0', cache_dir=None, check_every=0,
                 backend='torch'):
        from eventwm_graph.stage2_repair.current_range_sensor import CurrentRangeSensor
        self.native = CurrentRangeSensor(runtime_view(task, observation_config), RESOLUTION, size=size)
        self.task = task
        self.size = int(size)
        self.index = StageIndex(task.sim, cache_dir)
        self.backend = str(backend)
        a = self.index.arrays
        n = self.size*self.size
        keys = ('tri', 'bounds', 'left', 'right', 'first', 'count', 'index')
        if self.backend == 'driver':
            self.cuda = DriverCuda(SOURCE, int(str(device).split(':')[-1]) if ':' in str(device) else 0)
            self.gpu = [self.cuda.upload('stage:'+k, np.ascontiguousarray(a[k])) for k in keys]
            self.out_fr = self.cuda.buffer('out_fr', 4*n)
            self.out_flag = self.cuda.buffer('out_flag', n)
        elif self.backend == 'torch':
            import torch
            from eventwm_graph.stage2_repair.cuda_module import CudaModule
            self.device = torch.device(device)
            self.module = CudaModule(SOURCE, self.device)
            self.gpu = [torch.from_numpy(np.ascontiguousarray(a[k])).to(self.device) for k in keys]
            self.out_fr = torch.empty(n, dtype=torch.float32, device=self.device)
            self.out_flag = torch.empty(n, dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('unknown fast range backend %r' % (backend,))
        obs = self.native.boundary
        self.ignored = sorted(obs.robot_ids | obs.inactive_ids)
        agents = [task.sim.get_agent_data(i).articulated_agent for i in range(1, task.human_count+1)]
        self.humans = [a.sim_obj for a in agents]
        self.geometry = [urdf_capsules(a.urdf_path) for a in agents]
        self.check_every = int(check_every)
        self.frames = 0
        self.stats = dict(frames=0, rays=0, native_rays=0, human_rays=0, ambiguous_rays=0, gpu_s=0., native_s=0., host_s=0.)

    def close(self):
        """Release the device buffers (driver backend; torch tensors are freed with the object)."""
        if self.backend == 'driver':
            self.cuda.close()

    def human_capsules(self):
        """Index-frame collision capsules of every active human (one per URDF collision box: the box's long axis,
        radius = half cross-section diagonal + CAPSULE_MARGIN), per-human AABBs and offsets.  Environment-side
        simulator state (link transforms); never part of the frame."""
        boxes, caps, offsets = [], [], [0]
        for so, geo in zip(self.humans, self.geometry):
            frames = {so.get_link_name(l): np.asarray(so.get_link_scene_node(l).transformation, np.float64)
                      for l in so.get_link_ids()}
            root = np.asarray(so.transformation, np.float64)
            pts, rad = [], []
            for name, (a, b), r in geo:
                T = frames.get(name, root)
                pts.append(T[:3, :3]@a+T[:3, 3]); pts.append(T[:3, :3]@b+T[:3, 3]); rad.append(r+CAPSULE_MARGIN)
            loc = self.index.to_local(np.asarray(pts)).reshape(-1, 2, 3)
            rad = np.asarray(rad)
            boxes.append(np.r_[loc.reshape(-1, 3).min(0)-rad.max(), loc.reshape(-1, 3).max(0)+rad.max()])
            caps.append(np.c_[loc[:, 0], loc[:, 1], rad])
            offsets.append(offsets[-1]+len(loc))
        if not boxes:
            return np.zeros((0, 6), np.float32), np.zeros((1, 7), np.float32), np.zeros(1, np.int32)
        return (np.asarray(boxes, np.float32), np.concatenate(caps).astype(np.float32), np.asarray(offsets, np.int32))

    def cast(self, origin, directions):
        """(fractions float32 [n], flags uint8 [n]) for float32 world rays (GPU)."""
        import ctypes as ct
        o = np.asarray(origin, np.float32); d = np.asarray(directions, np.float32)
        ol = self.index.to_local(o.astype(np.float64)).astype(np.float32)
        dl = self.index.to_local(d.astype(np.float64)).astype(np.float32)
        if self.backend == 'driver':
            return self._cast_driver(ol, dl, len(d))
        import torch
        rays = torch.from_numpy(np.ascontiguousarray(np.r_[ol, dl.reshape(-1)], np.float32)).to(self.device, non_blocking=False)
        boxes, caps, offsets = self.human_capsules()
        dev = self.device
        bt = torch.from_numpy(np.ascontiguousarray(boxes.reshape(-1) if len(boxes) else np.zeros(6, np.float32))).to(dev)
        st = torch.from_numpy(np.ascontiguousarray(caps.reshape(-1))).to(dev)
        ot = torch.from_numpy(np.ascontiguousarray(offsets)).to(dev)
        n = len(d)
        fr = self.out_fr[:n]; fl = self.out_flag[:n]
        self.module.launch('cast_static', n, [*self.gpu, rays, ct.c_int(n), ct.c_float(MAX_RANGE), bt, ct.c_int(len(boxes)),
                                              st, ot, fr, fl], 128)
        return fr.cpu().numpy(), fl.cpu().numpy()

    def _cast_driver(self, ol, dl, n):
        import ctypes as ct
        c = self.cuda
        c.current()
        rays = c.upload('rays', np.ascontiguousarray(np.r_[ol, dl.reshape(-1)], np.float32))
        boxes, caps, offsets = self.human_capsules()
        bt = c.upload('boxes', np.ascontiguousarray(boxes.reshape(-1) if len(boxes) else np.zeros(6, np.float32)))
        st = c.upload('caps', np.ascontiguousarray(caps.reshape(-1)))
        ot = c.upload('offsets', np.ascontiguousarray(offsets))
        c.launch('cast_static', n, [*self.gpu, rays, ct.c_int(n), ct.c_float(MAX_RANGE), bt, ct.c_int(len(boxes)), st, ot,
                                    self.out_fr, self.out_flag], 128)
        return c.download(self.out_fr, np.empty(n, np.float32)), c.download(self.out_flag, np.empty(n, np.uint8))

    def measure(self, timestamp=None):
        """One range frame at the simulator's current camera pose.  ``timestamp``: the observation clock when the
        caller's simulator does not keep the world time (querygen's branch simulator replays the tape clock);
        default the simulator world time, as the native sensor."""
        t0 = time.perf_counter()
        rs = self.native
        origin, rotation = rs.boundary.camera.camera_pose()
        directions = rs.local@rotation.T
        t1 = time.perf_counter()
        fr, flags = self.cast(origin, directions)
        t2 = time.perf_counter()
        valid = fr < 1.
        dist = fr.astype(np.float64)*MAX_RANGE
        fb = np.flatnonzero(flags)
        if len(fb):
            hits = rs.batch.nearest_hits(origin, directions[fb], np.full(len(fb), MAX_RANGE), self.ignored)
            valid[fb] = hits['valid']; dist[fb] = hits['distances']
        t3 = time.perf_counter()
        ranges = np.where(valid, dist, np.nan).astype(np.float32).reshape(self.size, self.size)
        s = self.stats
        s['frames'] += 1; s['rays'] += len(fr); s['native_rays'] += len(fb)
        s['human_rays'] += int(((flags & 2) > 0).sum()); s['ambiguous_rays'] += int(((flags & 13) > 0).sum())
        s['gpu_s'] += t2-t1; s['native_s'] += t3-t2; s['host_s'] += (t1-t0)+(time.perf_counter()-t3)
        self.frames += 1
        out = dict(timestamp=float(rs.runtime.sim.get_world_time()) if timestamp is None else float(timestamp),
                   camera_origin=origin.copy(), camera_rotation=rotation.copy(),
                   range_image=ranges, sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1', max_range_m=MAX_RANGE,
                   horizontal_fov_deg=90., vertical_fov_deg=90., fast=dict(native_rays=int(len(fb))))
        if self.check_every and self.frames % self.check_every == 0:
            ref = self.native.measure()['range_image']             # validation only: the slow native frame
            out['fast']['check'] = compare(ranges, ref)
        return out


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz@Ry@Rx


_URDF = {}


def urdf_capsules(path):
    """[(link name, (end a, end b), radius)] in link coordinates: every URDF collision box as a capsule along its
    longest axis (Falcon's humanoids collide with 55 thin bone boxes).  Other collision primitives are refused."""
    import xml.etree.ElementTree as ET
    if path in _URDF:
        return _URDF[path]
    out = []
    for link in ET.parse(path).getroot().findall('link'):
        for c in link.findall('collision'):
            g = c.find('geometry')
            box = g.find('box') if g is not None else None
            if box is None:
                raise RuntimeError('fast range sensor: humanoid collision geometry other than boxes in %s' % path)
            size = np.asarray([float(v) for v in box.get('size').split()])
            o = c.find('origin')
            xyz = np.asarray([float(v) for v in ((o.get('xyz') if o is not None else None) or '0 0 0').split()])
            rpy = [float(v) for v in ((o.get('rpy') if o is not None else None) or '0 0 0').split()]
            R = _rpy(*rpy)
            k = int(np.argmax(size)); half = size/2; ax = np.zeros(3); ax[k] = half[k]
            out.append((link.get('name'), (xyz+R@ax, xyz-R@ax), float(np.sqrt((half**2).sum()-half[k]**2))))
    _URDF[path] = out
    return out


def compare(fast, native):
    """Per-pixel agreement of two range frames."""
    a = np.asarray(fast, np.float32); b = np.asarray(native, np.float32)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.abs(a[both].astype(np.float64)-b[both])
    return dict(pixels=int(a.size), mask_equal=bool(np.array_equal(fa, fb)), mask_mismatch=int((fa != fb).sum()),
                bit_equal=float((a[both] == b[both]).mean()) if both.any() else 1., max_abs=float(d.max()) if len(d) else 0.,
                mean_abs=float(d.mean()) if len(d) else 0., ulp_le2=float((d <= 2*np.spacing(b[both]).astype(np.float64)).mean()) if len(d) else 1.)
