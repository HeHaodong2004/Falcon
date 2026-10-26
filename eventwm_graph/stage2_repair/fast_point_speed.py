"""Matched complete costs, frozen F32 point versus explicit FP16 approximation."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run


if __name__=='__main__':
    names=['range_online16384_streamed','online_micro_dice256','legal_range_online4096_epoch02_columns']
    with quiet_jobs('fast_point_speed_epoch02_gpu0',pause_primary=True,extra_training_names=names):
        for implementation in ['fast-columns','fast-point']:
            run(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation=implementation,map_only_warm=True,name='benchmark_epoch02_point_precision_'+implementation)
