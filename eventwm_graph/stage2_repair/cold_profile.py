"""Complete cold API cost including input context, with GPU event attribution."""
from collections import defaultdict
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_radix_api import RadixExactSpatialPredictor
from .range_exact_api import ExactFastSpatialPredictor
from .cuda_module import CudaModule


def run():
    torch.set_num_threads(1);ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]];rows=[];reference=None
    for cls in [RadixExactSpatialPredictor,ExactFastSpatialPredictor]:
        model=cls(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1');model.config['observed_obstacle_schema']='capped_2m_complete'
        def call():return model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
        call();model.clear_cache();events=[];original=CudaModule.launch
        def measured(self,name,*args,**kwargs):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record(torch.cuda.current_stream(self.device));result=original(self,name,*args,**kwargs);b.record(torch.cuda.current_stream(self.device));events.append((name,a,b));return result
        CudaModule.launch=measured
        try:tick=time.perf_counter();result=call();torch.cuda.synchronize(model.device);milliseconds=(time.perf_counter()-tick)*1000
        finally:CudaModule.launch=original
        if reference is None:reference=result
        for a,b in zip(reference,result):np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5)
        kernels=defaultdict(float)
        for name,a,b in events:kernels[name]+=a.elapsed_time(b)
        rows.append(dict(model=cls.__name__,cold_ms=milliseconds,kernel_ms=dict(kernels),profile=model.last_batch_profile));print(rows[-1],flush=True)
    save(OUT/'cold_profile.json',dict(role='Worst-workload cost diagnosis, concurrent preparation load; not clean acceptance',rows=rows))


if __name__=='__main__':run()
