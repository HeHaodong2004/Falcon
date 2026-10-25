"""CUDA-event diagnosis on the measured worst legal-range workload."""
from collections import defaultdict
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_fast_proxy import FastRangePredictor
from .cuda_module import CudaModule


def run():
    torch.set_num_threads(1);m=FastRangePredictor(OUT/'training/range_evidence64/selected.pt',device='cuda:1');ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    call();events=[];original=CudaModule.launch
    def measured(self,name,*args,**kwargs):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record(torch.cuda.current_stream(self.device));result=original(self,name,*args,**kwargs);end.record(torch.cuda.current_stream(self.device));events.append((name,start,end));return result
    CudaModule.launch=measured
    try:tick=time.perf_counter();call();torch.cuda.synchronize(m.device);elapsed=time.perf_counter()-tick
    finally:CudaModule.launch=original
    totals=defaultdict(float)
    for name,start,end in events:totals[name]+=start.elapsed_time(end)
    result=dict(group=20,complete_K6_ms=elapsed*1000,kernel_ms=dict(totals),profile=m.last_batch_profile,triangles=m.active_mesh.triangles,history_frames=len(packet['ranges']),distinct_base_poses=len({tuple(p[1:])for path in paths for p in path}));save(OUT/'range_kernel_profile.json',result);print(result,flush=True)


if __name__=='__main__':run()
