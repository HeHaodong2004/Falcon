"""The same 23 signed supplied-pose features, in one CUDA kernel."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from eventwm_graph.stage2_segment.cuda_kernel import checked

SOURCE = r'''
extern "C" __global__ void signed_pose(const int* keys,const double* poses,
 const float* x,double resolution,int n,float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double wx=(keys[2*i]+.5)*resolution,wz=(keys[2*i+1]+.5)*resolution;
 float* o=out+23*i;const float* f=x+88*i;
 for(int k=0;k<4;k++) {
  const double* p=poses+5*k;double dx=wx-p[0],dz=wz-p[2],d=sqrt(dx*dx+dz*dz),den=fmax(d,1.e-6);
  o[4*k]=(float)((dx*p[3]-dz*p[4])/den);
  o[4*k+1]=(float)((dx*p[4]+dz*p[3])/den);
  o[4*k+2]=(float)(d/10.);o[4*k+3]=(float)((double)f[24]+poses[1]-p[1]);
 }
 o[16]=(float)poses[3];o[17]=(float)poses[4];o[18]=(float)poses[18];o[19]=(float)poses[19];
 o[20]=(float)((double)o[0]-sqrt(.5));o[21]=(float)((double)f[32]-sqrt(.5));o[22]=f[32]-o[0];
}
'''


def compile_source(device):
    cap = torch.cuda.get_device_capability(device)
    options = [('--gpu-architecture=compute_%d%d'%cap).encode(), b'--fmad=false', b'--prec-div=true', b'--prec-sqrt=true', b'--std=c++11']
    identity = hashlib.sha256(SOURCE.encode()+b'\0'.join(options)).hexdigest()
    cache = Path(os.environ['NUMBA_CACHE_DIR'])/'repair_pose_cuda'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/(identity+'.ptx')
    if target.exists():
        return target.read_bytes(), identity
    lib = ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'))
    program = ct.c_void_p()
    checked(lib, 'nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p], ct.byref(program), SOURCE.encode(), b'signed_pose.cu', 0, None, None)
    try:
        args = (ct.c_char_p*len(options))(*options)
        try:
            checked(lib, 'nvrtcCompileProgram', [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)], program, len(options), args)
        except RuntimeError:
            size = ct.c_size_t()
            checked(lib, 'nvrtcGetProgramLogSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            checked(lib, 'nvrtcGetProgramLog', [ct.c_void_p, ct.c_void_p], program, log)
            raise RuntimeError(log.value.decode())
        size = ct.c_size_t()
        checked(lib, 'nvrtcGetPTXSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
        buf = ct.create_string_buffer(size.value)
        checked(lib, 'nvrtcGetPTX', [ct.c_void_p, ct.c_void_p], program, buf)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
            f.write(buf.raw)
            temporary = f.name
        os.replace(temporary, target)
        return buf.raw, identity
    finally:
        checked(lib, 'nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)], ct.byref(program))


class PoseKernel:
    def __init__(self, device):
        self.device = torch.device(device)
        self.driver = ct.CDLL('libcuda.so.1')
        with torch.cuda.device(self.device):
            torch.empty(1, device=self.device)
            ptx, self.identity = compile_source(self.device)
            self.module = ct.c_void_p()
            checked(self.driver, 'cuModuleLoadData', [ct.POINTER(ct.c_void_p), ct.c_void_p], ct.byref(self.module), ct.create_string_buffer(ptx))
            self.function = ct.c_void_p()
            checked(self.driver, 'cuModuleGetFunction', [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p], ct.byref(self.function), self.module, b'signed_pose')

    def __call__(self, keys, path, resolution, x):
        assert x.ndim == 2 and x.shape[1] == 88 and x.dtype == torch.float32 and x.is_contiguous()
        indices = [0, min(1, len(path)-1), len(path)//2, len(path)-1]
        p = np.asarray(path, np.float64)[indices]
        poses = np.column_stack([p[:, 1:4], np.cos(p[:, 4]), np.sin(p[:, 4])])
        kt = torch.from_numpy(np.ascontiguousarray(keys, dtype=np.int32)).to(self.device)
        pt = torch.from_numpy(poses).to(self.device)
        out = torch.empty((len(keys), 23), device=self.device)
        if len(keys):
            values = [ct.c_void_p(t.data_ptr()) for t in (kt, pt, x)]+[ct.c_double(resolution), ct.c_int(len(keys)), ct.c_void_p(out.data_ptr())]
            args = (ct.c_void_p*len(values))(*[ct.addressof(v) for v in values])
            with torch.cuda.device(self.device):
                stream = ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
                checked(self.driver, 'cuLaunchKernel', [ct.c_void_p, *([ct.c_uint]*7), ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p], self.function, (len(keys)+127)//128, 1, 1, 128, 1, 1, 0, stream, args, None)
        return out
