"""Offline tree-quality diagnosis using the SAME legally observed triangles."""
import ctypes as ct
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_fast_proxy import FastRangePredictor
from .cuda_module import CudaModule
from eventwm_graph.l4.mesh_visibility import _build_bvh


def run():
    torch.set_num_threads(1);m=FastRangePredictor(OUT/'training/range_evidence64/selected.pt',device='cuda:1');ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    before=call();k=m.active_mesh.kernel;t=k.static['triangles'].cpu().numpy();vertices=np.stack([t[:,:3],t[:,:3]+t[:,3:6],t[:,:3]+t[:,6:9]],1);tick=time.perf_counter();bvh=_build_bvh(vertices.astype(np.float64),12);build_seconds=time.perf_counter()-tick
    for name,a in zip(['bounds','left','right','first','count','indices'],bvh):k.static[name]=k.tensor(a,np.float32 if name=='bounds'else np.int32)
    k.mesh=[k.static[name]for name in ['triangles','bounds','left','right','first','count','indices']];events=[];original=CudaModule.launch
    def measured(self,name,*args,**kwargs):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record(torch.cuda.current_stream(self.device));value=original(self,name,*args,**kwargs);end.record(torch.cuda.current_stream(self.device));events.append((name,start,end));return value
    CudaModule.launch=measured
    try:tick=time.perf_counter();after=call();torch.cuda.synchronize(m.device);ms=(time.perf_counter()-tick)*1000
    finally:CudaModule.launch=original
    for a,b in zip(before,after):np.testing.assert_array_equal(a['states'],b['states'])
    result=dict(role='OFFLINE accelerator diagnosis; CPU BVH build cost NOT excluded from any deployment acceptance.',group=20,triangles=len(t),cpu_tree_build_seconds=build_seconds,complete_K6_after_build_ms=ms,kernel_ms={name:start.elapsed_time(end)for name,start,end in events},identical_hard_predictions=True);save(OUT/'bvh_quality_check.json',result);print(result,flush=True)


if __name__=='__main__':run()
