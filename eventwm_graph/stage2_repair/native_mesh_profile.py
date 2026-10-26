"""Diagnostic profiler on the two slowest original sensor256 cold workloads."""
import numpy as np
import torch
from .diagnose import OUT,save
from .native_triangle_api import NativeTrianglePredictor
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import workloads


def run():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;m=NativeTrianglePredictor(OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt',device='cuda:0');records=[]
    for index in [3,16]:
        e,p,t=workloads('range_benchmark_256')[index];paths=[np.asarray(v['trajectory'],np.float64)for v in e['paths'][:6]];kw=dict(partial_graph=e['graph'],trajectories=paths,range_observations=p,track_history=t);m.predict_segments(**kw);m.clear_cache();torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as profile:m.predict_segments(**kw);torch.cuda.synchronize()
        table=profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=35);(OUT/('native_mesh_profile_%02d.txt'%index)).write_text(table);records.append(dict(group=index,range_frames=len(p['ranges']),triangles=m.active_mesh.triangles,measured_mesh_prepare_ms=m.active_mesh.prepare_seconds*1000,profile=m.last_batch_profile,table=table));print('PROFILE',index,'triangles',m.active_mesh.triangles,'mesh_prepare_ms',m.active_mesh.prepare_seconds*1000,flush=True)
    save(OUT/'native_mesh_profile.json',records)


if __name__=='__main__':
    with quiet_jobs('native_mesh_profile',pause_primary=True,extra_training_names=['range_online16384_streamed','online_sensor256_roots256','legal_range_online16384_epoch01_columns']):run()
