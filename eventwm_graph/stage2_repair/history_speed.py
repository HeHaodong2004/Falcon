"""Matched current implementation versus all legal frames; includes new queries."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run


if __name__=='__main__':
    with quiet_jobs('history_speed_epoch02_gpu0',pause_primary=True,extra_training_names=['range_online16384_streamed','online_initial_visibility256','online_micro_dice256']):
        for implementation in ['fast-columns','full-history']:
            run(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation=implementation,map_only_warm=True,name='benchmark_epoch02_lifetime_'+implementation)
