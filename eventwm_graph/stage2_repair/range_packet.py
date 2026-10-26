"""Warp-coherent exact63-ray occlusion: one shared stack per certificate.

All rays share an origin. Per-ray ancestor masks retain the original BVH
pruning semantics; the result is obstructed if any required probe is hit.
The persistent grid bounds scratch memory independently of query count.
"""
import torch
from .range_fast import RangeFeatureKernel,SOURCE
from .range_exact_fast import HELPERS,PreparedModule
from .range_map_reuse import SupportReuse
from .camera_vectorized import parameters

PACKET=r'''
struct ProbeRay {float x,y,z,limit;bool valid;};
__device__ bool probe_box(Mesh m,int node,float ox,float oy,float oz,ProbeRay d){
 float low=0.f,high=d.limit;
 for(int a=0;a<3;a++){
  float o=a==0?ox:a==1?oy:oz,v=a==0?d.x:a==1?d.y:d.z,lo=m.b[node*6+a]-.00002f,hi=m.b[node*6+a+3]+.00002f;
  if(fabsf(v)<1.e-30f){if(o<lo||o>hi)return false;}
  else{float ta=(lo-o)/v,tb=(hi-o)/v;low=fmaxf(low,fminf(ta,tb));high=fminf(high,fmaxf(ta,tb));if(low>high)return false;}
 }
 return true;
}
__device__ bool probe_triangle(const float* t,float ox,float oy,float oz,ProbeRay d){
 float ax=t[0],ay=t[1],az=t[2],ex=t[3],ey=t[4],ez=t[5],fx=t[6],fy=t[7],fz=t[8];
 float px=d.y*fz-d.z*fy,py=d.z*fx-d.x*fz,pz=d.x*fy-d.y*fx,det=ex*px+ey*py+ez*pz;
 if(fabsf(det)<1.e-20f)return false;float inv=1.f/det,tx=ox-ax,ty=oy-ay,tz=oz-az;
 float u=(tx*px+ty*py+tz*pz)*inv;if(u<-.0001f||u>1.0001f)return false;
 float qx=ty*ez-tz*ey,qy=tz*ex-tx*ez,qz=tx*ey-ty*ex,v=(d.x*qx+d.y*qy+d.z*qz)*inv;
 if(v<-.0001f||u+v>1.0001f)return false;float h=(fx*qx+fy*qy+fz*qz)*inv;return h>=0.f&&h<d.limit;
}
__device__ int probe_packet(Mesh m,float ox,float oy,float oz,ProbeRay a,ProbeRay b,int* stack,unsigned int* ma,unsigned int* mb){
 const unsigned int full=0xffffffffu;int lane=threadIdx.x&31,size=1;
 unsigned int initial_a=__ballot_sync(full,a.valid),initial_b=__ballot_sync(full,b.valid);
 if(lane==0){stack[0]=0;ma[0]=initial_a;mb[0]=initial_b;}
 while(__shfl_sync(full,size,0)){
  int node=0;unsigned int mask_a=0,mask_b=0;
  if(lane==0){node=stack[--size];mask_a=ma[size];mask_b=mb[size];}
  node=__shfl_sync(full,node,0);mask_a=__shfl_sync(full,mask_a,0);mask_b=__shfl_sync(full,mask_b,0);
  if(m.n[node]<0)continue;
  bool inside_a=((mask_a>>lane)&1)&&probe_box(m,node,ox,oy,oz,a);
  bool inside_b=((mask_b>>lane)&1)&&probe_box(m,node,ox,oy,oz,b);
  mask_a=__ballot_sync(full,inside_a);mask_b=__ballot_sync(full,inside_b);
  if(!(mask_a|mask_b))continue;
  if(m.n[node]==0){
   if(__shfl_sync(full,size,0)+2>64)return -1;
   if(lane==0){stack[size]=m.l[node];ma[size]=mask_a;mb[size++]=mask_b;stack[size]=m.r[node];ma[size]=mask_a;mb[size++]=mask_b;}
   continue;
  }
  for(int k=m.f[node];k<m.f[node]+m.n[node];k++){
   const float* t=m.t+9*m.ids[k];bool hit=(inside_a&&probe_triangle(t,ox,oy,oz,a))||(inside_b&&probe_triangle(t,ox,oy,oz,b));
   if(__any_sync(full,hit))return 1;
  }
 }
 return 0;
}
__device__ ProbeRay make_probe(int j,double x,double y,double z,const double* c,const double* offsets){
 ProbeRay r={0.f,0.f,0.f,0.f,false};r.valid=j<63;
 if(r.valid){double dx=x+offsets[j*3]-c[0],dy=y+offsets[j*3+1]-c[1],dz=z+offsets[j*3+2]-c[2],len=sqrt(dx*dx+dy*dy+dz*dz);r.valid=len>=1.e-7;r.x=dx/len;r.y=dy/len;r.z=dz/len;r.limit=len-.015;}
 return r;
}
extern "C" __global__ void generated_views(const int* keys,const float* floors,int nc,const double* cameras,int nv,
 const double* planes,const double* offsets,const unsigned char* unknown,int lx,int lz,int nx,int nz,double resolution,int* output,MESH_ARGS){
 __shared__ int stacks[256];__shared__ unsigned int masks_a[256],masks_b[256];
 int lane=threadIdx.x&31,local_warp=threadIdx.x/32,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,stride=gridDim.x*blockDim.x/32;MESH_VALUE;
 for(int tid=warp;tid<nc*nv;tid+=stride){
  int ci=tid%nc,cam=tid/nc,a=keys[ci*2]-lx,b=keys[ci*2+1]-lz,cell=a*nz+b;const double* c=cameras+cam*13;
  double x=(keys[ci*2]+.5)*resolution,y=floors[ci],z=(keys[ci*2+1]+.5)*resolution;int valid=0;
  if(lane==0)valid=unknown[cell]&&fabs(y-c[12])<=.18&&frustum(x,c[12]+.55,z,c)&&certificate_frustum(x,y,z,c,planes+cam*20,offsets,.25+resolution/sqrt(2.));
  if(!__shfl_sync(0xffffffffu,valid,0))continue;
  int j0=lane<21?lane+21:lane-21,j1=lane<10?lane+11:lane<31?lane+32:63;
  ProbeRay r0=make_probe(j0,x,y,z,c,offsets),r1=make_probe(j1,x,y,z,c,offsets);
  int result=probe_packet(m,(float)c[0],(float)c[1],(float)c[2],r0,r1,stacks+local_warp*64,masks_a+local_warp*64,masks_b+local_warp*64);
  if(lane==0){if(result<0)output[cam*nx*nz+cell]=-2147483647;else if(result==0)output[cam*nx*nz+cell]=ci+1;}
 }
}
extern "C" __global__ void test_packets(const float* origins,const float* directions,const float* limits,int n,int* out,MESH_ARGS){
 __shared__ int stacks[256];__shared__ unsigned int masks_a[256],masks_b[256];int lane=threadIdx.x&31,local_warp=threadIdx.x/32,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,stride=gridDim.x*blockDim.x/32;MESH_VALUE;
 for(int i=warp;i<n;i+=stride){
  int j0=lane,j1=lane+32;ProbeRay a={0.f,0.f,0.f,0.f,false},b={0.f,0.f,0.f,0.f,false};a.valid=true;b.valid=j1<63;
  a.x=directions[(i*63+j0)*3];a.y=directions[(i*63+j0)*3+1];a.z=directions[(i*63+j0)*3+2];a.limit=limits[i*63+j0];
  if(b.valid){b.x=directions[(i*63+j1)*3];b.y=directions[(i*63+j1)*3+1];b.z=directions[(i*63+j1)*3+2];b.limit=limits[i*63+j1];}
  int result=probe_packet(m,origins[i*3],origins[i*3+1],origins[i*3+2],a,b,stacks+local_warp*64,masks_a+local_warp*64,masks_b+local_warp*64);
  if(lane==0)out[i]=result;
 }
}
'''


class PacketModule(PreparedModule):
    def launch(self,name,total,values,threads=128):
        if name=='generated_views':
            if threads!=128:raise ValueError('packet kernel requires four32lane warps per block')
            total=min(total*32,262144)
        return super().launch(name,total,values,threads)


class PacketExactKernel(SupportReuse,RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device):
        self.device=torch.device(device);begin=SOURCE.index('extern "C" __global__ void generated_views');end=SOURCE.index('extern "C" __global__ void square_support')
        self.module=PacketModule(SOURCE[:begin]+HELPERS+PACKET+SOURCE[end:],device)
