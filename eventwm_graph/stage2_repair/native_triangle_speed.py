"""Matched original32sensor256 workloads: custom versus native triangles."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run,workloads


if __name__=='__main__':
    work=workloads('range_benchmark_256');names=['range_online16384_streamed','online_sensor256_roots256','legal_range_online16384_epoch01_columns']
    with quiet_jobs('native_triangle_speed_gpu0',pause_primary=True,extra_training_names=names):
        for implementation in ['fast-columns','native-triangle']:
            run(OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation=implementation,map_only_warm=True,name='benchmark_online16384_sensor256_native_comparison_'+implementation,_workloads=work)
