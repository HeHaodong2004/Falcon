"""Isolated visibility cost: count true traversal, then suppress it for timing.

The suppressed-ray variant is NEVER a predictor candidate or accuracy result.
It forces every normalized ray's directions to remain live in the compiler.
"""
import ctypes as ct
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_batched_spatial import BatchedSpatialPredictor
from .range_exact_fast import source,PreparedModule
from .range_fast import RangeFeatureKernel
from .camera_vectorized import parameters

COUNTERS=r'''
__device__ unsigned long long* counters;
extern "C" __global__ void set_counts(unsigned long long* p){counters=p;}
__device__ void count_ray(bool anyhit,int nodes,int triangles,bool hit){int offset=anyhit?0:4;atomicAdd(counters+offset,1ULL);atomicAdd(counters+offset+1,(unsigned long long)nodes);atomicAdd(counters+offset+2,(unsigned long long)triangles);atomicAdd(counters+offset+3,(unsigned long long)hit);}
'''


def code(suppressed):
    s=source();start=s.index('__device__ float ray(');end=s.index('__device__ bool frustum',start);body=s[start:end]
    if suppressed:
        body='__device__ float ray(Mesh m,float ox,float oy,float oz,float dx,float dy,float dz,float limit,bool anyhit){return dx*dy+dy*dz+dz*dx>4.f?0.f:limit;}\n'
    else:
        body=body.replace('int stack[64],size=1;','int visited=0,triangles=0;int stack[64],size=1;').replace('int node=stack[--size];','int node=stack[--size];visited++;').replace('if(size+2>64)return -2.f;','if(size+2>64){count_ray(anyhit,visited,triangles,false);return -2.f;}').replace('const float* t=m.t+9*m.ids[k];','triangles++;const float* t=m.t+9*m.ids[k];').replace('if(anyhit)return h;','if(anyhit){count_ray(anyhit,visited,triangles,true);return h;}').replace('return best;','count_ray(anyhit,visited,triangles,best<limit);return best;')
    return s[:start]+COUNTERS+body+s[end:]


class DiagnosticKernel(RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device,suppressed):
        self.device=torch.device(device);self.module=PreparedModule(code(suppressed),device);self.counts=torch.zeros(8,device=device,dtype=torch.int64);self.module.launch('set_counts',1,[self.counts])


def run():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;m=BatchedSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1');ex,packet,tracks=workloads()[30];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    call();records=[]
    for suppressed in [False,True]:
        kernel=DiagnosticKernel(m.device,suppressed);m.range_kernel=kernel;events=[];launch=kernel.module.launch
        def measured(name,*args,**kwargs):
            if name!='generated_views':return launch(name,*args,**kwargs)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);stream=torch.cuda.current_stream(m.device);a.record(stream);r=launch(name,*args,**kwargs);b.record(stream);events.append((a,b));return r
        kernel.module.launch=measured;call();torch.cuda.synchronize(m.device);counts=kernel.counts.cpu().numpy().tolist();row=dict(role='NO-RAY TIMING UPPER BOUND; NOT VALID PREDICTIONS'if suppressed else'Actual traversal counts only; atomics invalidate latency measurement',suppressed=suppressed,generated_gpu_ms=sum(a.elapsed_time(b)for a,b in events),counts_anyhit_and_closest=counts);records.append(row);print(row,flush=True)
    save(OUT/'ray_cost_diagnosis.json',records)


if __name__=='__main__':run()
