"""Merge fully measured planar pixel tiles; preserve holes and depth edges."""
import ctypes as ct
import time
import numpy as np
import torch
from .range_gpu import GpuObservedRangeMesh,SOURCE as BUILD
from .range_history import validate
from .geometry_cuda import GeometryKernel
from .cuda_module import CudaModule

SOURCE=BUILD+r'''
__device__ bool measured_point(int f,int u,int v,const float* ranges,const double* origins,const double* rotations,int h,int w,const double* tracks,const int* offsets,float* p){
 float d=ranges[(f*h+v)*w+u];if(!(d>.03f&&d<=10.00001f))return false;
 double x=(u+.5)*2/w-1.,y=1.-(v+.5)*2/h,z=-1.,len=sqrt(x*x+y*y+1.);x/=len;y/=len;z/=len;const double* r=rotations+f*9;const double* o=origins+f*3;
 double a=o[0]+d*(r[0]*x+r[1]*y+r[2]*z),b=o[1]+d*(r[3]*x+r[4]*y+r[5]*z),c=o[2]+d*(r[6]*x+r[7]*y+r[8]*z);
 for(int t=offsets[f];t<offsets[f+1];t++)if((a-tracks[t*3])*(a-tracks[t*3])+(c-tracks[t*3+2])*(c-tracks[t*3+2])<.25&&b>tracks[t*3+1]-.05&&b<tracks[t*3+1]+2.)return false;
 p[0]=a;p[1]=b;p[2]=c;return true;
}
__device__ bool small_triangle(const float* a,const float* b,const float* c){
 float aa=0,bb=0,cc=0;for(int j=0;j<3;j++){aa+=(a[j]-b[j])*(a[j]-b[j]);bb+=(a[j]-c[j])*(a[j]-c[j]);cc+=(b[j]-c[j])*(b[j]-c[j]);}return aa<.09f&&bb<.09f&&cc<.09f;
}
__device__ void write_triangle(const float* a,const float* b,const float* c,float* output,long long* keys,int slot){
 float ex=b[0]-a[0],ey=b[1]-a[1],ez=b[2]-a[2],fx=c[0]-a[0],fy=c[1]-a[1],fz=c[2]-a[2];float nx=ey*fz-ez*fy,ny=ez*fx-ex*fz,nz=ex*fy-ey*fx;if(nx*nx+ny*ny+nz*nz<=1.e-16f)return;
 for(int j=0;j<3;j++){output[slot*9+j]=a[j];output[slot*9+3+j]=b[j];output[slot*9+6+j]=c[j];}keys[slot]=0;
}
#define PLANAR_TILE 4
extern "C" __global__ void planar_tiles(const float* ranges,const double* origins,const double* rotations,int frames,int h,int w,const double* tracks,const int* offsets,float epsilon,float* output,long long* keys){
 const int side=PLANAR_TILE+1,slots=2*PLANAR_TILE*PLANAR_TILE;
 int tiles_x=(w-2)/PLANAR_TILE+1,tiles_y=(h-2)/PLANAR_TILE+1,per=tiles_x*tiles_y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=frames*per)return;int f=i/per,tx=i%per%tiles_x,ty=i%per/tiles_x,u=tx*PLANAR_TILE,v=ty*PLANAR_TILE,tw=min(PLANAR_TILE,w-1-u),th=min(PLANAR_TILE,h-1-v),base=i*slots;
 for(int j=0;j<slots;j++)keys[base+j]=9223372036854775807LL;
 float p[3*side*side];bool valid[side*side];bool planar=true;
 for(int y=0;y<=th;y++)for(int x=0;x<=tw;x++){int at=y*side+x;valid[at]=measured_point(f,u+x,v+y,ranges,origins,rotations,h,w,tracks,offsets,p+at*3);if(!valid[at])planar=false;}
 if(planar){
  const float* a=p;const float* b=p+th*side*3;const float* c=p+tw*3;float ex=b[0]-a[0],ey=b[1]-a[1],ez=b[2]-a[2],fx=c[0]-a[0],fy=c[1]-a[1],fz=c[2]-a[2],nx=ey*fz-ez*fy,ny=ez*fx-ex*fz,nz=ex*fy-ey*fx,len=sqrtf(nx*nx+ny*ny+nz*nz);
  if(len<1.e-8f)planar=false;else{
   nx/=len;ny/=len;nz/=len;
   for(int y=0;y<=th;y++)for(int x=0;x<=tw;x++){const float* q=p+(y*side+x)*3;if(fabsf((q[0]-a[0])*nx+(q[1]-a[1])*ny+(q[2]-a[2])*nz)>epsilon)planar=false;}
   for(int y=0;y<th;y++)for(int x=0;x<tw;x++)if(!small_triangle(p+(y*side+x)*3,p+((y+1)*side+x)*3,p+(y*side+x+1)*3)||!small_triangle(p+((y+1)*side+x+1)*3,p+(y*side+x+1)*3,p+((y+1)*side+x)*3))planar=false;
  }
 }
 if(planar){write_triangle(p,p+th*side*3,p+tw*3,output,keys,base);write_triangle(p+(th*side+tw)*3,p+tw*3,p+th*side*3,output,keys,base+1);return;}
 int slot=base;
 for(int y=0;y<th;y++)for(int x=0;x<tw;x++){
  int a=y*side+x,b=(y+1)*side+x,c=y*side+x+1,d=(y+1)*side+x+1;
  if(valid[a]&&valid[b]&&valid[c]&&small_triangle(p+a*3,p+b*3,p+c*3))write_triangle(p+a*3,p+b*3,p+c*3,output,keys,slot);slot++;
  if(valid[d]&&valid[c]&&valid[b]&&small_triangle(p+d*3,p+c*3,p+b*3))write_triangle(p+d*3,p+c*3,p+b*3,output,keys,slot);slot++;
 }
}
'''


def select_keyframes(origins,rotations,distance=.07,rotation=.07):
    """Motion approximation only; zero thresholds retain all legal observations.

    Identical camera poses can contain different later measurements. This rule
    does not establish that the skipped frames contain no new information.
    """
    if distance<0 or rotation<0:raise ValueError('keyframe thresholds must be nonnegative')
    selected=[];last=None
    for i in range(len(origins)):
        if last is not None and i!=len(origins)-1 and np.linalg.norm(origins[i]-origins[last])<distance and np.max(abs(rotations[i]-rotations[last]))<rotation:continue
        selected.append(i);last=i
    return selected


class PlanarRangeMesh(GpuObservedRangeMesh):
    def __init__(self,ranges,origins,rotations,timestamps,now,resolution,device='cuda:0',stride=1,tracks=None,epsilon=.00005,bvh='balanced',keyframe_distance=.07,keyframe_rotation=.07,tile_size=4,input_buffers=None):
        tick=time.perf_counter();validate(ranges,origins,rotations,timestamps,now)
        if stride!=1:raise ValueError('planar merging preserves full original range pixels')
        if tile_size not in(4,8):raise ValueError('supported measured planar tile sizes4/8')
        self.keyframes=select_keyframes(origins,rotations,keyframe_distance,keyframe_rotation)
        self.latest_sensor_time=float(timestamps[-1]);self.planes=[];self.tile_size=tile_size;device=torch.device(device);mod=CudaModule(SOURCE.replace('#define PLANAR_TILE 4','#define PLANAR_TILE %d'%tile_size),device)
        def tensor(x,dtype):
            if input_buffers is not None:return input_buffers.tensor(x,dtype)
            return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
        active=[];offsets=[0]
        for i in self.keyframes:
            for track in tracks[i]if tracks is not None else[]:
                if not track.get('valid',True):continue
                p=np.asarray(track['position'],np.float64)
                if track.get('velocity_valid')and track.get('estimated_velocity')is not None:p=p+np.asarray(track['estimated_velocity'])*min(float(track.get('observation_age',0)),.5)
                active.append(p)
            offsets.append(len(active))
        nf=len(self.keyframes);_,h,w=ranges.shape;tiles=((w-2)//tile_size+1)*((h-2)//tile_size+1)*nf;total=tiles*2*tile_size*tile_size;vertices=torch.empty((total,9),device=device);keys=torch.empty(total,device=device,dtype=torch.int64)
        selected=[tensor(a[self.keyframes],dt)for a,dt in [(ranges,np.float32),(origins,np.float64),(rotations,np.float64)]]if input_buffers is None else input_buffers.selected(self.keyframes)
        mod.launch('planar_tiles',tiles,[*selected,ct.c_int(nf),ct.c_int(h),ct.c_int(w),tensor(np.array(active).reshape(-1,3),np.float64),tensor(offsets,np.int32),ct.c_float(epsilon),vertices,keys])
        vertices=vertices[keys!=torch.iinfo(torch.int64).max];self.triangles=int(len(vertices))
        if not self.triangles:raise ValueError('no observed planar surface')
        if bvh=='native':
            # Hardware triangle traversal needs no software hierarchy or
            # Morton sort. All measured triangles stay in original order.
            packed=torch.cat([vertices[:,:3],vertices[:,3:6]-vertices[:,:3],vertices[:,6:9]-vertices[:,:3]],1)
            arrays=dict(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3),np.float64),triangles=packed,bounds=torch.empty((0,6),device=device))
            for name in ['left','right','first','count','indices']:arrays[name]=torch.empty(0,device=device,dtype=torch.int32)
            self.builder=mod;self.obstacle_cache={};self.kernel=GeometryKernel(arrays,resolution,device);self.native_only=True
            torch.cuda.synchronize(device);self.prepare_seconds=time.perf_counter()-tick;return
        if bvh=='radix':
            from .radix_bvh import build_arrays
            packed=torch.cat([vertices[:,:3],vertices[:,3:6]-vertices[:,:3],vertices[:,6:9]-vertices[:,:3]],1)
            started=time.perf_counter();arrays=build_arrays(packed)
            arrays.update(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3),np.float64))
            self.builder=mod;self.obstacle_cache={};self.kernel=GeometryKernel(arrays,resolution,device)
            torch.cuda.synchronize(device);self.radix_build_seconds=time.perf_counter()-started;self.prepare_seconds=time.perf_counter()-tick
            return
        if bvh!='balanced':raise ValueError('unsupported measured-surface hierarchy')
        view=vertices.view(-1,3,3);low=view.amin((0,1));extent=(view.amax((0,1))-low).clamp_min(.001);code=torch.empty(self.triangles,dtype=torch.int64,device=device);mod.launch('morton',self.triangles,[vertices,ct.c_int(self.triangles),low,extent,code]);vertices=vertices[torch.argsort(code)];group=12;leaves=1<<((self.triangles+group-1)//group-1).bit_length();nodes=leaves*2-1
        arrays=dict(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3),np.float64),triangles=torch.empty_like(vertices),bounds=torch.empty((nodes,6),device=device),indices=torch.empty(self.triangles,dtype=torch.int32,device=device))
        for name in ['left','right','first','count']:arrays[name]=torch.empty(nodes,dtype=torch.int32,device=device)
        args=[arrays[k]for k in ['bounds','left','right','first','count']];mod.launch('leaves',leaves,[vertices,ct.c_int(self.triangles),ct.c_int(group),ct.c_int(leaves),*args,arrays['triangles'],arrays['indices']]);width=leaves//2
        while width:mod.launch('parents',width,[ct.c_int(width-1),ct.c_int(width),*args]);width//=2
        self.builder=mod;self.obstacle_cache={};self.kernel=GeometryKernel(arrays,resolution,device);torch.cuda.synchronize(device);self.prepare_seconds=time.perf_counter()-tick
