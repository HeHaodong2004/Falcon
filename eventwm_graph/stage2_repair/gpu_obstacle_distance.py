"""Keep the exact measured-body distance field on GPU, including2m padding."""
from collections import OrderedDict
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule
from .gpu_edt import GridEDT
from .range_optix import OptixRangeKernel
from .gpu_candidates import GpuCandidateSpatialPredictor

SOURCE=r'''
extern "C" __global__ void gather_obstacle(const int* squared,const int* keys,int n,int lx,int lz,int nz,double resolution,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int cell=(keys[2*i]-lx)*nz+keys[2*i+1]-lz,d=squared[cell];out[i]=d>=1073741823?100.f:(float)(sqrt((double)d)*resolution);
}
'''


class DeviceObstacleMixin:
    def obstacle_request(self,mesh,union,base_height,uk):
        r=mesh.kernel.resolution;padding=getattr(mesh,'obstacle_padding_cells',int(np.ceil(2./r))+2);low=np.array([union[:,0].min(),union[:,1].min()])-padding;shape=np.array([union[:,0].max(),union[:,1].max()])-low+padding+1
        if max(shape)>8192 or int(np.prod(shape))>2000000:return('cpu',super().obstacle_request(mesh,union,base_height,uk))
        if not hasattr(self,'body_module'):self.body_module=CudaModule(SOURCE,self.device);self.body_edt=GridEDT(self.device)
        if not hasattr(mesh,'gpu_body_cache'):mesh.gpu_body_cache=OrderedDict()
        key=tuple(low)+tuple(shape)+(float(base_height),)
        if key not in mesh.gpu_body_cache:
            nx,nz=map(int,shape);field=torch.zeros((nx,nz),dtype=torch.int32,device=self.device)
            mesh.builder.launch('observed_obstacles',mesh.triangles,[mesh.kernel.static['triangles'],ct.c_int(mesh.triangles),ct.c_float(base_height+.2),ct.c_float(base_height+1.4),ct.c_float(r),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),field])
            states=(field>0).to(torch.uint8);_,_,squared=self.body_edt.transform(states,torch.empty_like(field,dtype=torch.float32),r,nearest_height=False,channel_count=1);mesh.gpu_body_cache[key]=squared[0]
            while len(mesh.gpu_body_cache)>8:mesh.gpu_body_cache.popitem(last=False)
        mesh.gpu_body_cache.move_to_end(key)
        return('gpu',mesh.gpu_body_cache[key],low,shape,uk,len(union),r)

    def obstacle_result(self,mesh,request):
        if request[0]=='cpu':return super().obstacle_result(mesh,request[1])
        _,squared,low,shape,keys,n,r=request;out=torch.empty(n,device=self.device)
        self.body_module.launch('gather_obstacle',n,[squared,keys,ct.c_int(n),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(int(shape[1])),ct.c_double(r),out]);return out


class DeviceObstacleKernel(DeviceObstacleMixin,OptixRangeKernel):pass


class FullyDeviceSpatialPredictor(GpuCandidateSpatialPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=DeviceObstacleKernel(self.device,primitive_mode='triangle')
