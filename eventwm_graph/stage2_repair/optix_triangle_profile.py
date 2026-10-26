"""Attribute actual worst complete workloads after shared GPU preprocessing."""
from collections import defaultdict
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_optix import OptixTriangleSpatialPredictor as BatchedSpatialPredictor
from .optix_backend import OptixBackend
from .cuda_module import CudaModule


def run():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;m=BatchedSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1');m.config['observed_obstacle_schema']='capped_2m_complete';work=workloads();records=[]
    for index in [30,18,20]:
        ex,packet,tracks=work[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]]
        def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
        call();events=[];old_optix=OptixBackend.launch;old=CudaModule.launch;prob=m.model.probabilities;feature=m.cuda.feature_kernel
        def measure(name,function,*args,**kwargs):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);stream=torch.cuda.current_stream(m.device);a.record(stream);result=function(*args,**kwargs);b.record(stream);events.append((name,a,b));return result
        def launch(self,name,*args,**kwargs):return measure(name,old,self,name,*args,**kwargs)
        OptixBackend.launch=lambda self,p,total:measure('optix_mode%d'%p.mode,old_optix,self,p,total)
        CudaModule.launch=launch;m.model.probabilities=lambda *a,**k:measure('neural',prob,*a,**k)
        original_feature=feature.launch;original_assembly=feature.launch_assembly;feature.launch=lambda *a,**k:measure('x88_views_lines',original_feature,*a,**k);feature.launch_assembly=lambda *a,**k:measure('x88_assembly',original_assembly,*a,**k)
        try:
            for cache in ['warm','cold']:
                if cache=='cold':m.clear_cache()
                events.clear();tick=time.perf_counter();result=call();torch.cuda.synchronize(m.device);elapsed=(time.perf_counter()-tick)*1000;kernels=defaultdict(float)
                for name,a,b in events:kernels[name]+=a.elapsed_time(b)
                record=dict(group=index,cache=cache,complete_ms=elapsed,kernels_ms=dict(kernels),profile=m.last_batch_profile,mesh_triangles=m.active_mesh.triangles,views=[len(p)for p in paths],range_frames=len(packet['ranges']));records.append(record);print(record,flush=True)
        finally:OptixBackend.launch=old_optix;CudaModule.launch=old;m.model.probabilities=prob;feature.launch=original_feature;feature.launch_assembly=original_assembly
    save(OUT/'optix_triangle_profile.json',records)


if __name__=='__main__':run()
