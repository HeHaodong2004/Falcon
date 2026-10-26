"""Matched original32 maps/queries:128versus256legal pixels, all outputs."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run,workloads


if __name__=='__main__':
    cases=[('128','fast-columns',workloads()),('256','fast-columns',workloads('range_benchmark_256')),('256','fast-point',workloads('range_benchmark_256')),('256','tile8',workloads('range_benchmark_256'))]
    names=['range_online16384_streamed','online_joint_range_head256','legal_range_online16384_epoch01_columns','online_sensor256_roots256']
    with quiet_jobs('sensor_resolution_speed_gpu0',pause_primary=True,extra_training_names=names):
        for sensor,implementation,work in cases:
            run(OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation=implementation,map_only_warm=True,name='benchmark_online16384_sensor'+sensor+'_'+implementation,_workloads=work)
