"""Shared top-down context from legal graph and measured past range only."""
import ctypes as ct
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from .cuda_module import CudaModule
from .range_history import validate

SIZE=192
CHANNELS=14
SOURCE=r'''
extern "C" __global__ void measured_context(const float* ranges,const double* origins,const double* rotations,int frames,int h,int w,
 const double* tracks,const int* offsets,int lx,int lz,int size,double resolution,double floor_y,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;
 double x=(lx+2*(i/size)+1.)*resolution,z=(lz+2*(i%size)+1.)*resolution;
 for(int level=0;level<3;level++){
  double y=floor_y+(level==0?.025:level==1?.7:1.4);bool clear=false,hit=false;
  for(int f=0;f<frames;f++){
   const double* o=origins+3*f;const double* r=rotations+9*f;double dx=x-o[0],dy=y-o[1],dz=z-o[2];
   double a=dx*r[0]+dy*r[3]+dz*r[6],b=dx*r[1]+dy*r[4]+dz*r[7],forward=-(dx*r[2]+dy*r[5]+dz*r[8]);
   double distance=sqrt(dx*dx+dy*dy+dz*dz);if(forward<=1.e-6||fabs(a)>forward||fabs(b)>forward||distance>10.)continue;
   int u=min(w-1,max(0,(int)((a/forward+1)*w/2))),v=min(h-1,max(0,(int)((1-b/forward)*h/2)));float d=ranges[(f*h+v)*w+u];if(!(d>0.f))d=10.f;
   if(distance<d-.03)clear=true;
   if(fabs(distance-d)<.08){bool person=false;for(int t=offsets[f];t<offsets[f+1];t++)if((x-tracks[t*3])*(x-tracks[t*3])+(z-tracks[t*3+2])*(z-tracks[t*3+2])<.25&&y>tracks[t*3+1]-.05&&y<tracks[t*3+1]+2.)person=true;if(!person)hit=true;}
  }
  output[(level*size*size)+i]=clear;output[((level+3)*size*size)+i]=hit;
 }
}
'''


class RangeCanvas:
    def __init__(self,device='cuda:0'):
        self.device=torch.device(device);self.module=CudaModule(SOURCE,device)

    def tensor(self,x,dtype):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(self.device)

    def build(self,keys,states,heights,ego,resolution,ranges,origins,rotations,timestamps,tracks):
        validate(ranges,origins,rotations,timestamps,timestamps[-1])
        if len(tracks)!=len(ranges):raise ValueError('track/sensor alignment')
        lo=(np.floor(np.asarray(ego)[[0,2]]/resolution).astype(np.int64)//32)*32-SIZE
        n=SIZE*2;s=np.zeros((n,n),np.uint8);height=np.zeros((n,n),np.float32);ij=np.asarray(keys)-lo;valid=np.all((ij>=0)&(ij<n),1);at=ij[valid];s[at[:,0],at[:,1]]=np.asarray(states)[valid];height[at[:,0],at[:,1]]=np.asarray(heights)[valid]-ego[1]
        def pool(a):return a.reshape(SIZE,2,SIZE,2).mean((1,3))
        free=pool((s==1).astype(np.float32));blocked=pool((s==2).astype(np.float32));known=free+blocked
        distances=[pool(distance_transform_edt(s!=state)*resolution/10)if np.any(s==state)else np.full_like(free,1.1)for state in [1,2]]
        graph=np.stack([free,blocked,pool(height)/np.maximum(known,.25),*distances,known]).astype(np.float32)
        positions=[];offsets=[0]
        for rows in tracks:
            for track in rows:
                if not track.get('valid',True):continue
                pos=np.asarray(track['position'],np.float64)
                if track.get('velocity_valid')and track.get('estimated_velocity')is not None:pos=pos+np.asarray(track['estimated_velocity'])*min(float(track.get('observation_age',0)),.5)
                positions.append(pos)
            offsets.append(len(positions))
        nf,h,w=ranges.shape;context=torch.empty((6,SIZE,SIZE),device=self.device)
        self.module.launch('measured_context',SIZE*SIZE,[self.tensor(ranges,np.float32),self.tensor(origins,np.float64),self.tensor(rotations,np.float64),ct.c_int(nf),ct.c_int(h),ct.c_int(w),self.tensor(np.array(positions).reshape(-1,3),np.float64),self.tensor(offsets,np.int32),ct.c_int(int(lo[0])),ct.c_int(int(lo[1])),ct.c_int(SIZE),ct.c_double(resolution),ct.c_double(float(ego[1])),context])
        xy=(np.arange(SIZE)*2+1)*resolution
        anchors=np.stack([np.broadcast_to((xy+lo[0]*resolution-ego[0])[:,None]/10,(SIZE,SIZE)),np.broadcast_to((xy+lo[1]*resolution-ego[2])[None,:]/10,(SIZE,SIZE))]).astype(np.float32)
        return np.concatenate([graph,context.cpu().numpy(),anchors]),lo
