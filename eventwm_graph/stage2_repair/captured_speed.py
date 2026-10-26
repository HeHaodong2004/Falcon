"""Complete same-checkpoint32-workload CUDA graph cost, startup separate."""
from .diagnose import OUT
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run


if __name__=='__main__':
    with quiet_jobs('captured_pipeline_epoch02_gpu0',pause_primary=True):
        run(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device='cuda:0',repetitions=3,spatial=True,implementation='captured-pipeline',name='benchmark_epoch02_gpu0_captured-pipeline')
