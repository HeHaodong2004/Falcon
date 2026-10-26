"""Bounded driver-module lifetime diagnosis; no predictor or data changes."""
import argparse
import gc
import ctypes as ct
from pathlib import Path
import torch
from .cuda_module import CudaModule,checked
from .diagnose import OUT,save
from .quiet_owned_jobs import quiet_jobs

SOURCE='extern "C" __global__ void fill(float* out){out[threadIdx.x]=3.f;}'


def run(count=256,device='cuda:0',name='module_memory_before',source='noop'):
    code=SOURCE
    if source=='real':
        from .range_planar import SOURCE as code
    def load():
        if source=='combo':
            from .range_planar import SOURCE as planar
            from .radix_bvh import SOURCE as radix
            from .geometry_cuda import SOURCE as geometry
            specifications=[(planar,['planar_tiles','observed_obstacles']),(radix,['triangle_codes','make_leaves','radix_tree','reduce_bounds']),(geometry,[])];modules=[]
            with torch.cuda.device(device):
                for text,functions in specifications:
                    m=CudaModule(text,device);modules.append(m)
                    for name in functions:
                        function=ct.c_void_p();checked(m.driver,'cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(function),m.module,name.encode())
            return modules
        m=CudaModule(code,device)
        if source=='noop':m.launch('fill',32,[out])
        else:
            with torch.cuda.device(device):
                function=ct.c_void_p();checked(m.driver,'cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(function),m.module,b'observed_obstacles')
        return m
    torch.set_num_threads(1);out=torch.empty(32,device=device);module=load();torch.cuda.synchronize(device);del module;gc.collect();torch.cuda.empty_cache();records=[]
    for i in range(count+1):
        if i in [0,16,64,128,256,count]:
            torch.cuda.synchronize(device);gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info(device);rss=int(Path('/proc/self/statm').read_text().split()[1])*4096;records.append(dict(modules_created=i,free_bytes=free,total_bytes=total,torch_allocated=torch.cuda.memory_allocated(device),resident_bytes=rss));print(records[-1],flush=True)
        if i<count:
            module=load();del module
    if source=='noop':assert torch.equal(out,torch.full_like(out,3.))
    save(OUT/(name+'.json'),dict(role='Synthetic module-lifetime test; owned jobs paused, unrelated jobs unchanged.',source=source,device=device,count=count,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=256);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='module_memory_before');p.add_argument('--source',choices=['noop','real','combo'],default='noop');a=p.parse_args()
    with quiet_jobs(a.name,pause_primary=a.device=='cuda:0'):run(**vars(a))
