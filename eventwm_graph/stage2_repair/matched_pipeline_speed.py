"""Matched complete OptiX versus fused query pipeline cost on the same GPU."""
from .diagnose import OUT,read,save
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run


def main():
    checkpoint=OUT/'training/range_online4096/selected_epoch02_argmax.pt';names=['optix-triangle','fast-pipeline'];reports={}
    with quiet_jobs('matched_pipeline_epoch02_gpu0',pause_primary=True)as isolation:
        for implementation in names:
            name='benchmark_epoch02_gpu0_'+implementation
            run(checkpoint,spatial=True,implementation=implementation,device='cuda:0',repetitions=3,name=name)
            reports[implementation]=read(OUT/(name+'.json'))
    save(OUT/'matched_pipeline_epoch02_gpu0.json',dict(role='Matched frozen epoch02 checkpoint, same RTX3090 GPU0, precision,32 original workloads and K1/4/6. Full cold and warm hypothetical materialization; no target acceptance claim.',load_isolation=isolation,models={name:{k:r[k]for k in ['checkpoint_sha256','implementation','precision','summary','startup','scope','excluded']}for name,r in reports.items()}))


if __name__=='__main__':main()
