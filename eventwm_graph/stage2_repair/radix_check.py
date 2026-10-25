"""Matched whole API outputs and CUDA timings before/after GPU radix build."""
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_fast_proxy import FastRangePredictor
from .radix_bvh import rebuild
from .cuda_module import CudaModule


def run():
    torch.set_num_threads(1);m=FastRangePredictor(OUT/'training/range_evidence64/selected.pt',device='cuda:1');ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    before=call();rebuild(m.active_mesh);events=[];original=CudaModule.launch
    def measured(self,name,*args,**kwargs):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record(torch.cuda.current_stream(self.device));value=original(self,name,*args,**kwargs);b.record(torch.cuda.current_stream(self.device));events.append((name,a,b));return value
    CudaModule.launch=measured
    try:tick=time.perf_counter();after=call();torch.cuda.synchronize(m.device);ms=(time.perf_counter()-tick)*1000
    finally:CudaModule.launch=original
    for a,b in zip(before,after):np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=2.e-6,rtol=2.e-5)
    result=dict(group=20,complete_K6_ms=ms,gpu_radix_build_seconds=m.active_mesh.radix_build_seconds,kernel_ms={name:a.elapsed_time(b)for name,a,b in events},identical_hard_predictions=True);save(OUT/'radix_check.json',result);print(result,flush=True)


if __name__=='__main__':run()
