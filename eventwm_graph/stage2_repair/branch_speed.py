"""Additional intermediate-graph end-to-end cost; preserves original benchmark."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run
from .branch_workloads import workloads


if __name__=='__main__':
    work=workloads()
    with quiet_jobs('branch_speed_epoch02_gpu0',pause_primary=True,extra_training_names=['range_online16384_streamed','online_initial_visibility256','online_micro_dice256']):
        run(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation='fast-columns',map_only_warm=True,name='benchmark_epoch02_intermediate_graphs',_workloads=work)
