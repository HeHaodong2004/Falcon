"""Only analytic legal-camera covariates; no triangle mesh or private map."""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule
from .range_fast import SOURCE
from .camera_vectorized import parameters
from .legal_camera import current_extrinsic

MARGIN=SOURCE[SOURCE.index('__device__ double view_margin'):SOURCE.index('extern "C" __global__ void pack_evidence')]
CAMERA_SOURCE=MARGIN+r'''
extern "C" __global__ void analytic_camera(const int* keys,const float* x,int n,const double* params,const double* times,int poses,
 double y0,double resolution,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float* out=output+i*16;
 for(int j=0;j<8;j++)out[j]=0.f;
 double px=(keys[2*i]+.5)*resolution,pz=(keys[2*i+1]+.5)*resolution,py=(double)x[i*88+24]+y0,radius=.25+resolution/sqrt(2.);
 double initial=view_margin(params,px,py,pz,radius),best=-10.,first=1.,last=0.,count=0.,weighted=0.,positive=0.;
 double duration=fmax(times[poses-1]-times[0],.0001);
 for(int j=1;j<poses;j++){
  double margin=view_margin(params+j*23,px,py,pz,radius);best=fmax(best,margin);
  if(margin>=0.){count+=1.;weighted+=times[j]-times[j-1];positive+=margin;double when=(times[j]-times[0])/duration;first=fmin(first,when);last=fmax(last,when);}
 }
 double den=max(1,poses-1);out[8]=fmax(-2.,fmin(2.,initial/2.));out[9]=fmax(-2.,fmin(2.,best/2.));out[10]=count/den;out[11]=weighted/duration;out[12]=positive/(2*den);out[13]=first;out[14]=last;out[15]=fmax(-2.,fmin(2.,(best-initial)/2.));
}
'''


class AnalyticCameraKernel:
    def __init__(self,device):self.device=torch.device(device);self.module=CudaModule(CAMERA_SOURCE,device)
    def tensor(self,x,dtype=None):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(self.device,non_blocking=True)
    def compute(self,keys,x,path,origin,rotation,resolution):
        out=torch.empty((len(keys),16),device=self.device)
        if not len(keys):return out
        translation,extrinsic=current_extrinsic(path[:1],origin,rotation);params,_=parameters(path,translation,extrinsic,resolution)
        self.module.launch('analytic_camera',len(keys),[self.tensor(keys,np.int32),x,ct.c_int(len(keys)),self.tensor(params,np.float64),self.tensor(path[:,0],np.float64),ct.c_int(len(path)),ct.c_double(path[0,2]),ct.c_double(resolution),out])
        return out
