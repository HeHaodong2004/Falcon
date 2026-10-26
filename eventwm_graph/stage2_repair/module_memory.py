"""Bounded driver-module lifetime diagnosis; no predictor or data changes."""
import argparse
import gc
import torch
from .cuda_module import CudaModule
from .diagnose import OUT,save
from .quiet_owned_jobs import quiet_jobs

SOURCE='extern "C" __global__ void fill(float* out){out[threadIdx.x]=3.f;}'


def run(count=256,device='cuda:0',name='module_memory_before'):
    torch.set_num_threads(1);out=torch.empty(32,device=device);module=CudaModule(SOURCE,device);module.launch('fill',32,[out]);torch.cuda.synchronize(device);del module;gc.collect();torch.cuda.empty_cache();records=[]
    for i in range(count+1):
        if i in [0,16,64,128,256,count]:
            torch.cuda.synchronize(device);gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info(device);records.append(dict(modules_created=i,free_bytes=free,total_bytes=total,torch_allocated=torch.cuda.memory_allocated(device)));print(records[-1],flush=True)
        if i<count:
            module=CudaModule(SOURCE,device);module.launch('fill',32,[out]);del module
    assert torch.equal(out,torch.full_like(out,3.))
    save(OUT/(name+'.json'),dict(role='Synthetic module-lifetime test; owned jobs paused, unrelated jobs unchanged.',device=device,count=count,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=256);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='module_memory_before');a=p.parse_args()
    with quiet_jobs(a.name,pause_primary=True):run(**vars(a))
