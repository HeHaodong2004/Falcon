"""Exact original11m candidate domain, shared GPU mask construction."""
from collections import OrderedDict
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule
from .range_optix import OptixTriangleSpatialPredictor

SOURCE=r'''
extern "C" __global__ void candidates(const int* keys,const int* queries,int n,const float* poses,const int* offsets,double resolution,unsigned char* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int q=queries[i];
 float x=__fmul_rn(__fadd_rn((float)keys[2*i],.5f),(float)resolution),z=__fmul_rn(__fadd_rn((float)keys[2*i+1],.5f),(float)resolution);bool keep=false;
 for(int j=offsets[q];j<offsets[q+1];j++){double dx=(double)x-(double)poses[2*j],dz=(double)z-(double)poses[2*j+1];if(sqrt(dx*dx+dz*dz)<=11.){keep=true;break;}}
 out[i]=keep;
}
'''


class CandidateKernel:
    def __init__(self,device):self.device=torch.device(device);self.module=CudaModule(SOURCE,device)
    def compute(self,keysets,poses,resolution):
        lengths=[len(k)for k in keysets]
        if not any(lengths):return [np.empty(0,bool)for k in keysets]
        def tensor(a,dtype):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(self.device,non_blocking=True)
        keys=tensor(np.concatenate(keysets),np.int32);queries=tensor(np.repeat(np.arange(len(keysets)),lengths),np.int32);xy=tensor(np.concatenate(poses),np.float32);offset=tensor(np.r_[0,np.cumsum([len(p)for p in poses])],np.int32);out=torch.empty(sum(lengths),device=self.device,dtype=torch.uint8)
        self.module.launch('candidates',len(out),[keys,queries,ct.c_int(len(out)),xy,offset,ct.c_double(resolution),out]);flat=out.cpu().numpy().astype(bool);splits=np.r_[0,np.cumsum(lengths)];return [flat[splits[i]:splits[i+1]]for i in range(len(lengths))]


class GpuCandidateMixin:
    def _candidate_keys(self,prepared,contexts):
        if not contexts:return[]
        if not hasattr(self,'candidate_kernel'):self.candidate_kernel=CandidateKernel(self.device)
        if not hasattr(prepared,'unknown_grids'):prepared.unknown_grids=OrderedDict()
        domains={};domain_indices=[];unknown_sets=[];positions=[]
        for ctx in contexts:
            lo=getattr(ctx,'candidate_lo',ctx.lo);shape=getattr(ctx,'candidate_shape',ctx.shape);states=getattr(ctx,'candidate_states',ctx.states);gridkey=tuple(lo)+tuple(shape)
            if gridkey not in prepared.unknown_grids:
                prepared.unknown_grids[gridkey]=np.argwhere(states==0).astype(np.int32)+lo
                while len(prepared.unknown_grids)>8:prepared.unknown_grids.popitem(last=False)
            xy=np.unique(ctx.xy,axis=0);domainkey=(gridkey,xy.tobytes())
            if domainkey not in domains:domains[domainkey]=len(unknown_sets);unknown_sets.append(prepared.unknown_grids[gridkey]);positions.append(xy)
            domain_indices.append(domains[domainkey])
        masks=self.candidate_kernel.compute(unknown_sets,positions,prepared.resolution);selected=[keys[mask]for keys,mask in zip(unknown_sets,masks)];result=[selected[i]for i in domain_indices]
        for ctx,keys in zip(contexts,result):ctx._candidate_keys=keys
        return result


class GpuCandidateSpatialPredictor(GpuCandidateMixin,OptixTriangleSpatialPredictor):pass
