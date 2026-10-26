"""Original-cell query construction without intermediate host key/height copies."""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule

SOURCE=r'''
extern "C" __global__ void query_masks(const unsigned char* states,int gx,int gz,int glx,int glz,int lx,int lz,int nx,int nz,
 const int* bounds,const float* poses,const int* offsets,int queries,double resolution,unsigned char* mask,unsigned char* unknown){
 int i=blockIdx.x*blockDim.x+threadIdx.x,area=nx*nz;if(i>=area*queries)return;int q=i/area,cell=i%area,kx=cell/nz+lx,kz=cell%nz+lz;bool legal=states[(kx-glx)*gz+kz-glz]==0;
 if(q==0)unknown[cell]=legal;
 if(!legal||kx<bounds[q*4]||kx>=bounds[q*4+2]||kz<bounds[q*4+1]||kz>=bounds[q*4+3]){mask[i]=0;return;}
 float x=__fmul_rn(__fadd_rn((float)kx,.5f),(float)resolution),z=__fmul_rn(__fadd_rn((float)kz,.5f),(float)resolution);bool keep=false;
 for(int j=offsets[q];j<offsets[q+1];j++){double dx=(double)x-poses[j*2],dz=(double)z-poses[j*2+1];if(sqrt(dx*dx+dz*dz)<=11.){keep=true;break;}}
 mask[i]=keep;
}
extern "C" __global__ void union_keys(const long long* indices,int n,int lx,int lz,int nz,int* keys,int* ids,const float* baseline,int gz,int glx,int glz,double base_y,float* floors){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;long long cell=indices[i];int x=cell/nz+lx,z=cell%nz+lz;keys[2*i]=x;keys[2*i+1]=z;ids[cell]=i;float h=baseline[(x-glx)*gz+z-glz];float d=__fsub_rn(h,(float)base_y);floors[i]=(double)d+base_y;
}
extern "C" __global__ void row_keys(const long long* rows,int n,int lx,int lz,int nz,const int* ids,int* keys,int* cells,int* queries){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int q=rows[2*i],cell=rows[2*i+1];keys[2*i]=cell/nz+lx;keys[2*i+1]=cell%nz+lz;cells[i]=cell;queries[i]=q;
}
'''


class DeviceQuery:
    def __init__(self,device):self.device=torch.device(device);self.module=CudaModule(SOURCE,device)
    def tensor(self,a,dtype):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(self.device,non_blocking=True)

    def build(self,prepared,contexts):
        bounds=np.array([np.r_[c.candidate_lo,c.candidate_lo+c.candidate_shape]for c in contexts],np.int32);low=bounds[:,:2].min(0)-4;high=bounds[:,2:].max(0)+4;shape=high-low;nx,nz=map(int,shape);area=nx*nz
        # The caller checks this expanded rectangle against the global grid.
        state,baseline,distance=prepared.global_fields
        if not hasattr(prepared,'pipeline_fields'):
            states=self.tensor(state,np.uint8);images=[]
            for cls in [1,2]:images.append(torch.nn.functional.pad((states==cls).to(torch.int64).cumsum(0).cumsum(1),(1,0,1,0)))
            prepared.pipeline_fields=(states,baseline,*distance,*images)
        states,baseline,df,db,free,blocked=prepared.pipeline_fields;gx,gz=state.shape;glx,glz=map(int,prepared.global_lo)
        positions=[np.unique(c.xy,axis=0)for c in contexts];poses=self.tensor(np.concatenate(positions),np.float32);offset=self.tensor(np.r_[0,np.cumsum([len(p)for p in positions])],np.int32);spec=self.tensor(bounds,np.int32);mask=torch.empty((len(contexts),area),dtype=torch.uint8,device=self.device);unknown=torch.empty(area,dtype=torch.uint8,device=self.device)
        self.module.launch('query_masks',mask.numel(),[states,ct.c_int(gx),ct.c_int(gz),ct.c_int(glx),ct.c_int(glz),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),spec,poses,offset,ct.c_int(len(contexts)),ct.c_double(prepared.resolution),mask,unknown])
        # CUDA nonzero can return a column-major[N,2] view. Raw-pointer
        # kernels require row-major pairs; counts alone do not expose this.
        counts=mask.sum(1).cpu().numpy().astype(np.int64);indices=mask.any(0).nonzero().flatten().contiguous();rows=mask.nonzero().contiguous();n=len(rows);nu=len(indices)
        keys=torch.empty((n,2),dtype=torch.int32,device=self.device);cells=torch.empty(n,dtype=torch.int32,device=self.device);queries=torch.empty_like(cells);uk=torch.empty((nu,2),dtype=torch.int32,device=self.device);ids=torch.full((area,),-1,dtype=torch.int32,device=self.device);floors=torch.empty(nu,device=self.device)
        self.module.launch('union_keys',nu,[indices,ct.c_int(nu),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nz),uk,ids,baseline,ct.c_int(gz),ct.c_int(glx),ct.c_int(glz),ct.c_double(contexts[0].path[0,2]),floors])
        self.module.launch('row_keys',n,[rows,ct.c_int(n),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nz),ids,keys,cells,queries])
        return dict(keys=keys,cells=cells,queries=queries,union=uk,ids=ids,floors=floors,low=low,shape=shape,area=area,unknown=unknown,counts=counts,offsets=np.r_[0,np.cumsum(counts)],positions=positions,fields=prepared.pipeline_fields)
