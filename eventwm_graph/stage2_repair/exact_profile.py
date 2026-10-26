"""Matched exact-evidence kernel/whole-call timing; concurrent load recorded."""
from collections import defaultdict
import os
import subprocess
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_radix_api import RadixExactSpatialPredictor
from .range_exact_fast import FastExactKernel
from .range_packet import PacketExactKernel
from .cuda_module import CudaModule


def run():
    torch.set_num_threads(1);model=RadixExactSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1')
    model.config['observed_obstacle_schema']='capped_2m_complete'
    ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    variants=[('old63',model.range_kernel),('constant63',FastExactKernel(model.device,False,False)),('ordered63',FastExactKernel(model.device,True,False)),('closed_form63',FastExactKernel(model.device,True,True)),('packet63',PacketExactKernel(model.device))]
    rows=[];reference=None
    for name,kernel in variants:
        model.range_kernel=kernel;call();events=[];original=CudaModule.launch
        def measured(self,label,*args,**kwargs):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record(torch.cuda.current_stream(self.device));result=original(self,label,*args,**kwargs);b.record(torch.cuda.current_stream(self.device));events.append((label,a,b));return result
        CudaModule.launch=measured
        try:tick=time.perf_counter();result=call();torch.cuda.synchronize(model.device);milliseconds=(time.perf_counter()-tick)*1000
        finally:CudaModule.launch=original
        if reference is None:reference=result
        for before,after in zip(reference,result):np.testing.assert_array_equal(before['states'],after['states']);np.testing.assert_allclose(before['probabilities'],after['probabilities'],atol=3.e-6,rtol=3.e-5)
        kernels=defaultdict(float)
        for label,a,b in events:kernels[label]+=a.elapsed_time(b)
        rows.append(dict(variant=name,complete_K6_warm_ms=milliseconds,kernel_ms=dict(kernels),profile=model.last_batch_profile));print(rows[-1],flush=True)
    model.clear_cache();tick=time.perf_counter();call();torch.cuda.synchronize(model.device)
    result=dict(role='DIAGNOSTIC, not clean latency acceptance while preparation jobs run',group=20,rows=rows,last_variant_complete_cold_ms=(time.perf_counter()-tick)*1000,concurrent_load=subprocess.check_output(['ps','-eo','pid,etimes,pcpu,rss,args','--sort=-pcpu'],text=True).splitlines()[:26],load_average=os.getloadavg(),all_hard_predictions_match=True)
    save(OUT/'exact_profile.json',result)


if __name__=='__main__':run()
