"""Read already computed START-view render codes; never alter certificates."""
import ctypes as ct
import torch
from .cuda_module import CudaModule
from .range_map_reuse import ReusedExactKernel
from .range_optix import OptixRangeKernel

SOURCE=r'''
extern "C" __global__ void initial_code(const int* frames,int area,const int* cells,const int* queries,const int* initial,int n,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int code=frames[initial[queries[i]]*area+cells[i]];output[3*i]=code==0;output[3*i+1]=code>0;output[3*i+2]=code<0;
}
'''


class InitialMixin:
    output_features=19
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.initial_module=CudaModule(SOURCE,self.device)

    def pack(self,total,values):
        super().pack(total,values);self.initial=torch.empty((total,3),device=self.device)
        self.initial_module.launch('initial_code',total,[values[0],values[1],values[4],values[5],values[9],ct.c_int(total),self.initial])

    def transform_evidence(self,values):
        if len(self.initial)!=len(values):raise ValueError('initial render row alignment changed')
        return torch.cat([values,self.initial],1)


class TrainingInitialKernel(InitialMixin,ReusedExactKernel):pass
class OptixInitialKernel(InitialMixin,OptixRangeKernel):pass
