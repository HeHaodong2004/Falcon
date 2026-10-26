"""Optional hardware BVH traversal over the same measured-map leaf boxes.

Custom intersection programs keep the existing float Moller tolerance and
12-triangle leaf grouping. OptiX replaces only tree traversal. SDK headers
and CUDA headers are external build dependencies, never hidden-map inputs.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .diagnose import OUT
from .cuda_module import checked
from .geometry_cuda import SOURCE as GEOMETRY
from .range_fast import SOURCE as RANGE
from .range_exact_fast import source as exact_source,HELPERS


class MeshPointers(ct.Structure):
    _fields_=[(n,ct.c_void_p)for n in ['t','b','l','r','f','n','ids']]


class Parameters(ct.Structure):
    _fields_=[('mesh',MeshPointers),('handle',ct.c_ulonglong)]+[(n,ct.c_void_p)for n in ['keys','floors','cameras','planes','probe_offsets','unknown','frames','support','difference','origins','directions','limits','test_output']]+[('resolution',ct.c_double)]+[(n,ct.c_int)for n in ['nc','nv','lx','lz','nx','nz','mode','leaf_start','n','primitive_mode']]


PARAMETERS=r'''
struct Params {
 Mesh mesh;unsigned long long handle;
 const int* keys;const float* floors;const double* cameras;const double* planes;const double* probe_offsets;const unsigned char* unknown;int* frames;float* support;float* difference;
 const float* origins;const float* directions;const float* limits;float* test_output;
 double resolution;int nc,nv,lx,lz,nx,nz,mode,leaf_start,n,primitive_mode;
};
extern "C" __constant__ Params params;
__device__ float ray(Mesh m,float ox,float oy,float oz,float dx,float dy,float dz,float limit,bool anyhit){
 if(limit<=0.f)return limit;
 unsigned value=__float_as_uint(limit);float3 origin={ox,oy,oz},direction={dx,dy,dz};
 optixTrace(params.handle,origin,direction,0.f,limit,0.f,255,OPTIX_RAY_FLAG_DISABLE_ANYHIT|(anyhit?OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT:0),0,1,0,value);
 return __uint_as_float(value);
}
extern "C" __global__ void __miss__main(){}
extern "C" __global__ void __closesthit__main(){optixSetPayload_0(__float_as_uint(optixGetRayTmax()));}
extern "C" __global__ void __intersection__leaf(){
 Mesh m=params.mesh;int primitive=optixGetPrimitiveIndex(),node=params.leaf_start+(params.primitive_mode?primitive/12:primitive);float3 o=optixGetObjectRayOrigin(),d=optixGetObjectRayDirection();float ox=o.x,oy=o.y,oz=o.z,dx=d.x,dy=d.y,dz=d.z,best=optixGetRayTmax();
 if(params.primitive_mode){
  // Individual triangle boxes are conservative for the original barycentric
  // tolerance. Retain the original12-triangle leaf's additional slab gate.
  float low=0.f,high=__uint_as_float(optixGetPayload_0());
  for(int a=0;a<3;a++){float origin=a==0?ox:a==1?oy:oz,direction=a==0?dx:a==1?dy:dz,lo=m.b[node*6+a]-.00002f,hi=m.b[node*6+a+3]+.00002f;
   if(fabsf(direction)<1.e-30f){if(origin<lo||origin>hi)return;}
   else{float ta=(lo-origin)/direction,tb=(hi-origin)/direction;low=fmaxf(low,fminf(ta,tb));high=fminf(high,fmaxf(ta,tb));if(low>high)return;}
  }
 }
 int begin=params.primitive_mode?primitive:m.f[node],end=params.primitive_mode?primitive+1:m.f[node]+m.n[node];
 for(int k=begin;k<end;k++){
  const float* t=m.t+9*m.ids[k];float ax=t[0],ay=t[1],az=t[2],ex=t[3],ey=t[4],ez=t[5],fx=t[6],fy=t[7],fz=t[8];
  float px=dy*fz-dz*fy,py=dz*fx-dx*fz,pz=dx*fy-dy*fx,det=ex*px+ey*py+ez*pz;
  if(fabsf(det)<1.e-20f)continue;float inv=1.f/det,tx=ox-ax,ty=oy-ay,tz=oz-az;
  float u=(tx*px+ty*py+tz*pz)*inv;if(u<-.0001f||u>1.0001f)continue;
  float qx=ty*ez-tz*ey,qy=tz*ex-tx*ez,qz=tx*ey-ty*ex,v=(dx*qx+dy*qy+dz*qz)*inv;
  if(v<-.0001f||u+v>1.0001f)continue;float h=(fx*qx+fy*qy+fz*qz)*inv;
  if(h>=0.f&&h<best&&optixReportIntersection(h,0)){best=h;if(optixGetRayFlags()&OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)return;}
 }
}
'''


def device_source(normalization=False,native_triangles=False):
    mesh=GEOMETRY[GEOMETRY.index('struct Mesh'):GEOMETRY.index('__device__ float ray')]
    common=GEOMETRY[GEOMETRY.index('__device__ bool frustum'):GEOMETRY.index('extern "C" __global__ void certificate_views')]
    frustum=HELPERS[HELPERS.index('__device__ bool certificate_frustum'):HELPERS.index('extern "C" __global__ void test_certificate')]
    normalize_helpers=''
    if normalization:
        from .range_normalization import source as normalized_source,HELPERS as NORMALIZATION
        full=normalized_source();normalize_helpers=NORMALIZATION[:NORMALIZATION.index('extern "C" __global__ void test_normalization')]
    else:full=exact_source()
    views=full[full.index('extern "C" __global__ void generated_views'):full.index('extern "C" __global__ void square_support')]
    support=RANGE[RANGE.index('extern "C" __global__ void square_support'):RANGE.index('__device__ double view_margin')]
    wall=GEOMETRY[GEOMETRY.index('extern "C" __global__ void wall_views'):GEOMETRY.index('extern "C" __global__ void gather_increments')]
    functions=[]
    for name,text in [('generated_views',views),('square_support',support),('wall_views',wall)]:
        text=text.replace('extern "C" __global__ void '+name,'__device__ void '+name+'_impl').replace('blockIdx.x*blockDim.x+threadIdx.x','optixGetLaunchIndex().x');functions.append(text)
    entry=r'''
#define MESH_PASS params.mesh.t,params.mesh.b,params.mesh.l,params.mesh.r,params.mesh.f,params.mesh.n,params.mesh.ids
extern "C" __global__ void __raygen__main(){
 if(params.mode==0)generated_views_impl(params.keys,params.floors,params.nc,params.cameras,params.nv,params.planes,params.probe_offsets,params.unknown,params.lx,params.lz,params.nx,params.nz,params.resolution,params.frames,MESH_PASS);
 else if(params.mode==1)wall_views_impl(params.cameras,params.nv,params.unknown,params.lx,params.lz,params.nx,params.nz,params.resolution,params.frames,MESH_PASS);
 else if(params.mode==2)square_support_impl(params.keys,params.floors,params.nc,params.resolution,params.support,params.difference,MESH_PASS);
 else{int i=optixGetLaunchIndex().x;if(i<params.n)params.test_output[i]=ray(params.mesh,params.origins[i*3],params.origins[i*3+1],params.origins[i*3+2],params.directions[i*3],params.directions[i*3+1],params.directions[i*3+2],params.limits[i],false);}
}
'''
    parameters=PARAMETERS[:PARAMETERS.index('extern "C" __global__ void __intersection__leaf')]if native_triangles else PARAMETERS
    return '#include <optix_device.h>\n'+mesh+parameters+common+frustum+normalize_helpers+''.join(functions)+entry


def compile_device(source,device,optix_include,cuda_include):
    capability=torch.cuda.get_device_capability(device);options=[('--gpu-architecture=compute_%d%d'%capability).encode(),b'--fmad=false',b'--prec-div=true',b'--prec-sqrt=true',b'--std=c++11',('-I'+str(optix_include)).encode(),('-I'+str(cuda_include)).encode()]
    identity=hashlib.sha256(source.encode()+b'\0'.join(options)).hexdigest();path=OUT/'deps'/(identity+'.optix.ptx')
    if path.exists():return path.read_bytes()
    lib=ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'));program=ct.c_void_p();checked(lib,'nvrtcCreateProgram',[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_void_p],ct.byref(program),source.encode(),b'falcon_optix.cu',0,None,None)
    try:
        args=(ct.c_char_p*len(options))(*options)
        try:checked(lib,'nvrtcCompileProgram',[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)],program,len(options),args)
        except RuntimeError:
            size=ct.c_size_t();checked(lib,'nvrtcGetProgramLogSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size));buf=ct.create_string_buffer(size.value);checked(lib,'nvrtcGetProgramLog',[ct.c_void_p,ct.c_void_p],program,buf);raise RuntimeError(buf.value.decode())
        size=ct.c_size_t();checked(lib,'nvrtcGetPTXSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size));buf=ct.create_string_buffer(size.value);checked(lib,'nvrtcGetPTX',[ct.c_void_p,ct.c_void_p],program,buf);path.write_bytes(buf.raw);return buf.raw
    finally:checked(lib,'nvrtcDestroyProgram',[ct.POINTER(ct.c_void_p)],ct.byref(program))


class OptixBackend:
    def __init__(self,device='cuda:0',normalization=False,primitive_mode='leaf'):
        if primitive_mode not in ['leaf','triangle','native','native-fast']:raise ValueError('unsupported OptiX measured primitive mode')
        self.primitive_mode=primitive_mode
        self.device=torch.device(device);deps=OUT/'deps';oi=Path(os.environ.get('FALCON_OPTIX_INCLUDE',deps/'optix-dev/include'));ci=Path(os.environ.get('FALCON_CUDA_INCLUDE',deps/'cuda_runtime/nvidia/cuda_runtime/include'))
        native=primitive_mode.startswith('native');flags=['-DFALCON_OPTIX_FAST_BUILD=1']if primitive_mode=='native-fast'else[]
        cpp=Path(__file__).with_suffix('.cpp').with_name('optix_native_bridge.cpp'if native else'optix_bridge.cpp');identity=hashlib.sha256(cpp.read_bytes()+str(oi).encode()+str(ci).encode()+str(flags).encode()).hexdigest();binary=deps/(identity+'.optix_bridge.so')
        if not binary.exists():subprocess.run(['g++','-O3','-std=c++14','-fPIC','-shared',str(cpp),'-I',str(oi),'-I',str(ci),'-l:libcuda.so.1','-ldl','-o',str(binary)]+flags,check=True)
        self.lib=ct.CDLL(str(binary));self.state=ct.c_void_p();P=ct.c_void_p;U=ct.c_ulonglong;I=ct.c_uint;S=ct.c_size_t;C=ct.c_char_p;args={
            'create':[C,S,ct.POINTER(P),P,S], 'sizes':[P,U,I,ct.POINTER(U),ct.POINTER(U),P,S], 'build':[P,U,I,U,S,U,S,U,ct.POINTER(U),P,S], 'headers':[P,P,P,S], 'launch':[P,U,U,S,I,U,P,S]}
        for name,types in args.items():f=getattr(self.lib,'falcon_optix_'+name);f.argtypes=types;f.restype=ct.c_int
        self.lib.falcon_optix_destroy.argtypes=[P]
        with torch.cuda.device(self.device):
            torch.empty(1,device=self.device);ptx=compile_device(device_source(normalization,native_triangles=native),self.device,oi,ci);self.call('create',ptx,len(ptx),ct.byref(self.state));headers=np.zeros(96,np.uint8);self.call('headers',self.state,headers.ctypes.data);self.records=torch.from_numpy(headers).to(self.device)

    def call(self,name,*args):
        error=ct.create_string_buffer(16384)
        with torch.cuda.device(self.device):code=getattr(self.lib,'falcon_optix_'+name)(*args,error,len(error))
        if code:raise RuntimeError('OptiX%s:%d %s'%(name,code,error.value.decode(errors='replace')))

    def close(self):
        if self.state:
            with torch.cuda.device(self.device):self.lib.falcon_optix_destroy(self.state)
            self.state=ct.c_void_p()

    def __del__(self):
        try:self.close()
        except Exception:pass

    def prepare(self,mesh):
        if hasattr(mesh,'optix_accel')and mesh.optix_accel['backend']is self:return mesh.optix_accel
        # Same12-triangle leaf boxes and original2e-5padding, no geometry lost.
        leaves=(mesh.triangles+11)//12;k=mesh.kernel
        if self.primitive_mode.startswith('native'):
            t=k.static['triangles'];a=t[:,:3];boxes=torch.stack([a,a+t[:,3:6],a+t[:,6:9]],1).reshape(-1,3).contiguous();primitives=mesh.triangles
        elif self.primitive_mode=='leaf':
            boxes=k.static['bounds'][-leaves:].clone();boxes[:,:3]-=.00002;boxes[:,3:]+=.00002;primitives=leaves
        else:
            t=k.static['triangles'];a=t[:,:3];e=t[:,3:6];f=t[:,6:9];b=a+e;c=a+f;pad=.0002*(e.abs()+2*f.abs())+.00002
            boxes=torch.cat([torch.minimum(a,torch.minimum(b,c))-pad,torch.maximum(a,torch.maximum(b,c))+pad],1).contiguous();primitives=mesh.triangles
        output=ct.c_ulonglong();temporary=ct.c_ulonglong()
        self.call('sizes',self.state,boxes.data_ptr(),primitives,ct.byref(output),ct.byref(temporary));storage=torch.empty(output.value,dtype=torch.uint8,device=self.device);scratch=torch.empty(temporary.value,dtype=torch.uint8,device=self.device);handle=ct.c_ulonglong();stream=torch.cuda.current_stream(self.device).cuda_stream
        self.call('build',self.state,boxes.data_ptr(),primitives,scratch.data_ptr(),scratch.numel(),storage.data_ptr(),storage.numel(),stream,ct.byref(handle));result=dict(backend=self,boxes=boxes,storage=storage,scratch=scratch,handle=handle.value,leaf_start=0 if self.primitive_mode.startswith('native')else leaves-1);mesh.optix_accel=result;return result

    def parameters(self,mesh):
        acceleration=self.prepare(mesh);p=Parameters();p.mesh=MeshPointers(*[v.data_ptr()for v in mesh.kernel.mesh]);p.handle=acceleration['handle'];p.leaf_start=acceleration['leaf_start'];p.primitive_mode=2 if self.primitive_mode.startswith('native')else int(self.primitive_mode=='triangle');return p

    def launch(self,p,total):
        if not total:return
        # Pinned staging avoids a pageable H2D synchronization behind rays.
        raw=np.frombuffer(bytes(p),np.uint8).copy();host=torch.from_numpy(raw).pin_memory();parameters=host.to(self.device,non_blocking=True)
        self.call('launch',self.state,self.records.data_ptr(),parameters.data_ptr(),ct.sizeof(p),total,torch.cuda.current_stream(self.device).cuda_stream)

    def trace(self,mesh,origins,directions,limits):
        tensors=[torch.as_tensor(a,device=self.device,dtype=torch.float32).contiguous()for a in [origins,directions,limits]];p=self.parameters(mesh);p.mode=3;p.n=len(tensors[0]);p.origins,p.directions,p.limits=[t.data_ptr()for t in tensors];out=torch.empty(p.n,device=self.device);p.test_output=out.data_ptr();self.launch(p,p.n);return out


if __name__=='__main__':
    torch.cuda.set_device(1);backend=OptixBackend('cuda:1');print('OPTIX_BACKEND_CREATED',backend.state,flush=True)
