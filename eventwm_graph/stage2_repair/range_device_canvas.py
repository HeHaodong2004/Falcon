"""Legal14-channel canvas without GPU-to-host-to-GPU round trips."""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule
from .range_bev import SIZE
from .range_canvas_fast import SOURCE
from .range_history import validate
from .range_gpu_graph import DeviceGraphSpatialPredictor

POOL=r'''
extern "C" __global__ void pool_graph(const unsigned char* states,const float* height,const int* squared,int size,double resolution,int has_free,int has_blocked,int lx,int lz,double ex,double ez,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;int x=i/size,z=i%size,n=size*2,area=size*size;
 int ids[4]={2*x*n+2*z,2*x*n+2*z+1,(2*x+1)*n+2*z,(2*x+1)*n+2*z+1};
 float free=0,blocked=0,h=0;for(int j=0;j<4;j++){free+=states[ids[j]]==1;blocked+=states[ids[j]]==2;h+=height[ids[j]];}
 free*=.25f;blocked*=.25f;float known=free+blocked;
 output[i]=free;output[area+i]=blocked;output[2*area+i]=(h*.25f)/fmaxf(known,.25f);output[5*area+i]=known;
 for(int c=0;c<2;c++){double sum=0.;for(int j=0;j<4;j++)sum+=sqrt((double)squared[c*n*n+ids[j]])*resolution/10.;output[(3+c)*area+i]=(c==0?has_free:has_blocked)?sum*.25:1.1;}
 double ax=(2*x+1)*resolution,az=(2*z+1)*resolution;output[12*area+i]=(ax+lx*resolution-ex)/10.;output[13*area+i]=(az+lz*resolution-ez)/10.;
}
'''


class DeviceRangeCanvas:
    def __init__(self,edt):
        self.edt=edt;self.device=edt.device;self.module=CudaModule(SOURCE+POOL,self.device)

    def tensor(self,a,dtype):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(self.device,non_blocking=True)

    def build(self,keys,states,heights,ego,resolution,ranges,origins,rotations,timestamps,tracks):
        validate(ranges,origins,rotations,timestamps,timestamps[-1])
        if len(tracks)!=len(ranges):raise ValueError('track/sensor alignment')
        low=(np.floor(np.asarray(ego)[[0,2]]/resolution).astype(np.int64)//32)*32-SIZE;n=SIZE*2
        state=np.zeros((n,n),np.uint8);height=np.zeros((n,n),np.float32);ij=np.asarray(keys)-low;valid=np.all((ij>=0)&(ij<n),1);at=ij[valid];state[at[:,0],at[:,1]]=np.asarray(states)[valid];height[at[:,0],at[:,1]]=np.asarray(heights)[valid]-ego[1]
        positions=[];offsets=[0]
        for rows in tracks:
            for t in rows:
                if not t.get('valid',True):continue
                p=np.asarray(t['position'],np.float64)
                if t.get('velocity_valid')and t.get('estimated_velocity')is not None:p=p+np.asarray(t['estimated_velocity'])*min(float(t.get('observation_age',0)),.5)
                positions.append(p)
            offsets.append(len(positions))
        # All pageable inputs are staged ahead of the map's device work.
        st=self.tensor(state,np.uint8);ht=self.tensor(height,np.float32)
        sensor=[self.tensor(ranges,np.float32),self.tensor(origins,np.float64),self.tensor(rotations,np.float64)];people=self.tensor(np.array(positions).reshape(-1,3),np.float64);off=self.tensor(offsets,np.int32)
        _,_,squared=self.edt.transform(st,ht,resolution,nearest_height=False);canvas=torch.empty((14,SIZE,SIZE),device=self.device)
        self.module.launch('pool_graph',SIZE*SIZE,[st,ht,squared,ct.c_int(SIZE),ct.c_double(resolution),ct.c_int(bool(np.any(state==1))),ct.c_int(bool(np.any(state==2))),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_double(float(ego[0])),ct.c_double(float(ego[2])),canvas])
        nf,h,w=ranges.shape
        self.module.launch('measured_context',SIZE*SIZE,[*sensor,ct.c_int(nf),ct.c_int(h),ct.c_int(w),people,off,ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(SIZE),ct.c_double(resolution),ct.c_double(float(ego[1])),canvas[6:12]])
        return canvas.to(torch.float16),low


class DeviceCanvasSpatialPredictor(DeviceGraphSpatialPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.canvas=DeviceRangeCanvas(self.edt)

    def _encoded_map(self,details):
        history=next(reversed(self.mesh_cache));assert self.mesh_cache[history]is self.active_mesh;key=(self.active_graph.identity,history,tuple(details[0][2][0,1:4]))
        if key not in self.spatial_cache:
            p=self.current_range_observations;g=self.active_graph;ego=details[0][2][0,1:4]
            canvas,low=self.canvas.build(g.keys,g.values,g.heights,ego,g.resolution,np.asarray(p['ranges'],np.float32),np.asarray(p['origins'],np.float64),np.asarray(p['rotations'],np.float64),np.asarray(p['timestamps'],np.float64),self.current_track_history)
            with torch.cuda.amp.autocast():encoded=self.model.encode(canvas[None])
            self.spatial_cache[key]=(encoded,low)
            while len(self.spatial_cache)>2:self.spatial_cache.popitem(last=False)
        self.spatial_cache.move_to_end(key);return self.spatial_cache[key]
