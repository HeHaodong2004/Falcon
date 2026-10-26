"""Batched static certificate visibility and wall witnesses on Torch's stream.

Float32 mesh intersections approximate the native Bullet ray calculation;
the measured native/GPU differences are reported separately. Frustum and
certificate coordinates remain float64. No target labels enter this module.
"""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule

SOURCE = r'''
struct Mesh {const float* t;const float* b;const int* l;const int* r;const int* f;const int* n;const int* ids;};
__device__ float ray(Mesh m,float ox,float oy,float oz,float dx,float dy,float dz,float limit,bool anyhit){
 int stack[64],size=1;stack[0]=0;float best=limit;
 while(size){int node=stack[--size];if(m.n[node]<0)continue;float low=0.f,high=best;bool intersects=true;
  for(int a=0;a<3;a++){float o=a==0?ox:a==1?oy:oz,d=a==0?dx:a==1?dy:dz;
   float lo=m.b[node*6+a]-.00002f,hi=m.b[node*6+a+3]+.00002f;
   if(fabsf(d)<1.e-30f){if(o<lo||o>hi){intersects=false;break;}}
   else{float ta=(lo-o)/d,tb=(hi-o)/d;low=fmaxf(low,fminf(ta,tb));high=fminf(high,fmaxf(ta,tb));if(low>high){intersects=false;break;}}
  }
  if(!intersects)continue;
  if(m.n[node]==0){if(size+2>64)return -2.f;stack[size++]=m.l[node];stack[size++]=m.r[node];continue;}
  for(int k=m.f[node];k<m.f[node]+m.n[node];k++){
   const float* t=m.t+9*m.ids[k];float ax=t[0],ay=t[1],az=t[2],ex=t[3],ey=t[4],ez=t[5],fx=t[6],fy=t[7],fz=t[8];
   float px=dy*fz-dz*fy,py=dz*fx-dx*fz,pz=dx*fy-dy*fx,det=ex*px+ey*py+ez*pz;
   if(fabsf(det)<1.e-20f)continue;float inv=1.f/det,tx=ox-ax,ty=oy-ay,tz=oz-az;
   float u=(tx*px+ty*py+tz*pz)*inv;if(u<-.0001f||u>1.0001f)continue;
   float qx=ty*ez-tz*ey,qy=tz*ex-tx*ez,qz=tx*ey-ty*ex,v=(dx*qx+dy*qy+dz*qz)*inv;
   if(v<-.0001f||u+v>1.0001f)continue;
   float h=(fx*qx+fy*qy+fz*qz)*inv;
   if(h>=0.f&&h<best){if(anyhit)return h;best=h;}
  }
 }
 return best;
}
__device__ bool frustum(double x,double y,double z,const double* c){
 double dx=x-c[0],dy=y-c[1],dz=z-c[2];
 double a=dx*c[3]+dy*c[6]+dz*c[9],b=dx*c[4]+dy*c[7]+dz*c[10],f=-(dx*c[5]+dy*c[8]+dz*c[11]);
 return f>1.e-6&&fabs(a)<=f&&fabs(b)<=f&&dx*dx+dy*dy+dz*dz<=100.;
}
#define MESH_ARGS const float* tri,const float* bounds,const int* left,const int* right,const int* first,const int* count,const int* indices
#define MESH_VALUE Mesh m={tri,bounds,left,right,first,count,indices}
extern "C" __global__ void certificate_views(const int* keys,const double* targets,int nc,const double* cameras,int nv,
 const unsigned char* unknown,int lx,int lz,int nx,int nz,double resolution,int* output,MESH_ARGS){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=nc*nv)return;int ci=tid%nc,cam=tid/nc;
 int a=keys[2*ci]-lx,b=keys[2*ci+1]-lz;if(a<0||b<0||a>=nx||b>=nz)return;int cell=a*nz+b;if(!unknown[cell])return;
 const double* c=cameras+13*cam;const double* p=targets+ci*189;double floor_y=p[1]-.025;
 if(fabs(floor_y-c[12])>.18)return;
 if(!frustum((keys[2*ci]+.5)*resolution,c[12]+.55,(keys[2*ci+1]+.5)*resolution,c))return;
 for(int j=0;j<63;j++)if(!frustum(p[3*j],p[3*j+1],p[3*j+2],c))return;
 MESH_VALUE;
 for(int j=0;j<63;j++){
  double dx=p[3*j]-c[0],dy=p[3*j+1]-c[1],dz=p[3*j+2]-c[2],length=sqrt(dx*dx+dy*dy+dz*dz);
  if(length<1.e-7)continue;float limit=(float)(length-.015);
  float hit=ray(m,(float)c[0],(float)c[1],(float)c[2],(float)(dx/length),(float)(dy/length),(float)(dz/length),limit,true);
  if(hit<0.f){output[cam*nx*nz+cell]=-2147483647;return;}
  if(hit<limit)return;
 }
 output[cam*nx*nz+cell]=ci+1;
}
extern "C" __global__ void wall_views(const double* cameras,int nv,const unsigned char* unknown,int lx,int lz,int nx,int nz,double resolution,int* output,MESH_ARGS){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=nv*105)return;int cam=tid/105,j=tid%105;
 double x=-.95+1.9*(j/3)/34.,y=(j%3-1)*.5;const double* c=cameras+13*cam;
 double dx=c[3]*x+c[4]*y-c[5],dy=c[6]*x+c[7]*y-c[8],dz=c[9]*x+c[10]*y-c[11],length=sqrt(dx*dx+dy*dy+dz*dz);
 float vx=(float)(dx/length),vy=(float)(dy/length),vz=(float)(dz/length),ox=(float)c[0],oy=(float)c[1],oz=(float)c[2];
 MESH_VALUE;float hit=ray(m,ox,oy,oz,vx,vy,vz,10.f,false);if(hit>=10.f)return;
 if(hit<0.f){output[cam*nx*nz]=-2147483647;return;}
 float px=ox+vx*hit,py=oy+vy*hit,pz=oz+vz*hit;if(!(c[12]+.2<(double)py&&(double)py<c[12]+1.4))return;
 int ix=(int)floor((double)px/resolution),iz=(int)floor((double)pz/resolution),span=(int)ceil(.25/resolution)+1;
 for(int da=-span;da<=span;da++)for(int db=-span;db<=span;db++){
  int a=ix+da-lx,b=iz+db-lz;if(a<0||b<0||a>=nx||b>=nz)continue;
  double ex=(ix+da+.5)*resolution-(double)px,ez=(iz+db+.5)*resolution-(double)pz;
  if(ex*ex+ez*ez<=.0625&&unknown[a*nz+b])output[cam*nx*nz+a*nz+b]=-cam-1;
 }
}
extern "C" __global__ void gather_increments(const int* frames,int area,const int* cells,const int* queries,int n,
 const int* steps,const int* offsets,int* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int q=queries[i],cell=cells[i],value=0;
 for(int j=offsets[q+1]-1;j>=offsets[q];j--){int v=frames[steps[j]*area+cell];if(v){value=v;break;}}
 output[i]=value;
}
extern "C" __global__ void test_rays(const float* origins,const float* directions,const float* limits,int n,float* out,MESH_ARGS){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;MESH_VALUE;
 out[i]=ray(m,origins[3*i],origins[3*i+1],origins[3*i+2],directions[3*i],directions[3*i+1],directions[3*i+2],limits[i],false);
}
'''


class GeometryKernel:
    def __init__(self, arrays, resolution, device):
        self.device = torch.device(device)
        self.resolution = float(resolution)
        self.module = CudaModule(SOURCE, device)
        self.arrays = arrays
        self.static = {k: (v.to(device) if torch.is_tensor(v) else torch.from_numpy(np.ascontiguousarray(v)).to(device)) for k, v in arrays.items()}
        self.mesh = [self.static[k] for k in ['triangles', 'bounds', 'left', 'right', 'first', 'count', 'indices']]

    def tensor(self, value, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(value, dtype=dtype)).to(self.device)

    def compute(self, unknown, lo, cameras, candidate_sets, steps):
        if not self.static['bounds'].numel():raise RuntimeError('hardware-only measured mesh has no software traversal tree; use its hardware range kernel')
        nx, nz = unknown.shape
        nviews = len(cameras)
        if nviews*nx*nz*4 > 512*1024*1024:
            raise ValueError('geometry frame grid exceeds 512 MiB; chunked dispatch required')
        frame = torch.zeros((nviews, nx*nz), device=self.device, dtype=torch.int32)
        cam = self.tensor(cameras, np.float64)
        u = self.tensor(unknown, np.uint8)
        nc = len(self.arrays['keys'])
        spec = [u, ct.c_int(int(lo[0])), ct.c_int(int(lo[1])), ct.c_int(nx), ct.c_int(nz), ct.c_double(self.resolution), frame, *self.mesh]
        self.module.launch('certificate_views', nc*nviews, [self.static['keys'], self.static['targets'], ct.c_int(nc), cam, ct.c_int(nviews), *spec])
        self.module.launch('wall_views', 105*nviews, [cam, ct.c_int(nviews), *spec])
        cells = np.concatenate([(k[:, 0]-lo[0])*nz+k[:, 1]-lo[1] for k in candidate_sets]).astype(np.int32)
        queries = np.repeat(np.arange(len(candidate_sets), dtype=np.int32), [len(k) for k in candidate_sets])
        offsets = np.r_[0, np.cumsum([len(s) for s in steps])].astype(np.int32)
        flat = np.concatenate(steps).astype(np.int32)
        ci, qi, st, off = [self.tensor(a) for a in [cells, queries, flat, offsets]]
        output = torch.empty(len(cells), device=self.device, dtype=torch.int32)
        self.module.launch('gather_increments', len(cells), [frame, ct.c_int(nx*nz), ci, qi, ct.c_int(len(cells)), st, off, output])
        codes = output.cpu().numpy()
        if np.any(codes <= -2147483647):
            raise RuntimeError('GPU BVH traversal stack exhausted')
        return np.split(codes, np.cumsum([len(k) for k in candidate_sets])[:-1])
