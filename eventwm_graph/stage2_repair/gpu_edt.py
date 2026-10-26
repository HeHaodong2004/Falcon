"""Exact integer-grid Euclidean transforms with SciPy's coordinate tie order.

Binary sites first minimize x (ties lower x); the second pass minimizes z
(ties lower z). Integer parabola crossing points preserve that tie ordering.
No grid truncation or spatial approximation is introduced.
"""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule

SOURCE=r'''
extern "C" __global__ void binary_x(const unsigned char* states,int nx,int nz,int channels,int* squared,int* nearest){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nz*channels)return;int c=t/nz,z=t%nz,n=nx*nz,previous=-1000000;
 for(int x=0;x<nx;x++){int state=states[x*nz+z];bool site=c<2?state==c+1:state>0;if(site)previous=x;int i=c*n+x*nz+z;squared[i]=previous<0?1073741823:(x-previous)*(x-previous);nearest[i]=previous;}
 int next=1000000;for(int x=nx-1;x>=0;x--){int state=states[x*nz+z];bool site=c<2?state==c+1:state>0;if(site)next=x;int i=c*n+x*nz+z;int d=next>=nx?1073741823:(next-x)*(next-x);if(d<squared[i]){squared[i]=d;nearest[i]=next;}}
}
__device__ long long floordiv(long long a,long long b){long long q=a/b,r=a%b;return q-(r<0);}
extern "C" __global__ void parabola_z(const int* first,const int* nearest,const float* height,int nx,int nz,int channels,double resolution,float default_height,
 int* vertices,int* starts,int* final_squared,float* distances,float* baseline){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nx*channels)return;int c=t/nx,x=t%nx,n=nx*nz,base=c*n+x*nz,k=-1;
 for(int q=0;q<nz;q++){
  int f=first[base+q];if(f>=1073741823)continue;long long crossing=-2147483647;
  while(k>=0){int p=vertices[base+k];long long numerator=(long long)f+q*q-first[base+p]-p*p;crossing=floordiv(numerator,2LL*(q-p))+1;if(crossing>starts[base+k])break;k--;}
  k++;vertices[base+k]=q;starts[base+k]=k?max(-2147483647LL,min(2147483647LL,crossing)):-2147483647;
 }
 int active=0;for(int z=0;z<nz;z++){
  int i=base+z;if(k<0){final_squared[i]=1073741823;if(c<2)distances[i]=11.f;else baseline[x*nz+z]=default_height;continue;}
  while(active<k&&z>=starts[base+active+1])active++;
  int q=vertices[base+active],d=first[base+q]+(q-z)*(q-z);final_squared[i]=d;
  if(c<2)distances[i]=(float)sqrt((double)d)*(float)resolution;
  else baseline[x*nz+z]=height[nearest[base+q]*nz+q];
 }
}
'''


class GridEDT:
    def __init__(self,device='cuda:0'):
        self.device=torch.device(device);self.module=CudaModule(SOURCE,device)

    def transform(self,states,heights,resolution,default_height=0.,nearest_height=True,channel_count=None):
        states=torch.as_tensor(states,device=self.device,dtype=torch.uint8).contiguous();heights=torch.as_tensor(heights,device=self.device,dtype=torch.float32).contiguous();nx,nz=states.shape
        if nx>8192 or nz>8192 or nx*nz>2000000:raise ValueError('GPU EDT bounded at8192axes/2millioncells; caller must retain CPU fallback')
        channels=3 if nearest_height else 2
        if channel_count is not None:
            if nearest_height or channel_count not in [1,2]:raise ValueError('single-channel transform excludes nearest-height output')
            channels=channel_count
        shape=(channels,nx,nz)
        first=torch.empty(shape,dtype=torch.int32,device=self.device);nearest=torch.empty_like(first);vertices=torch.empty_like(first);starts=torch.empty_like(first);squared=torch.empty_like(first)
        distance=torch.empty((min(2,channels),nx,nz),device=self.device);baseline=torch.empty((nx,nz),device=self.device)
        self.module.launch('binary_x',nz*channels,[states,ct.c_int(nx),ct.c_int(nz),ct.c_int(channels),first,nearest])
        self.module.launch('parabola_z',nx*channels,[first,nearest,heights,ct.c_int(nx),ct.c_int(nz),ct.c_int(channels),ct.c_double(resolution),ct.c_float(default_height),vertices,starts,squared,distance,baseline])
        return distance,baseline,squared
