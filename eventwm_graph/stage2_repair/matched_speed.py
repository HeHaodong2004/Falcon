"""Matched complete cost; same selected weights and full32originalworkloads."""
from .diagnose import OUT,read,save
from .quiet_owned_jobs import quiet_jobs
from .range_benchmark import run


def main():
    checkpoint=OUT/'training/range_online4096/selected_epoch01_argmax.pt'
    with quiet_jobs('matched_speed_online4096_epoch01')as isolation:
        run(checkpoint,spatial=True,batched=True,device='cuda:1',repetitions=3,name='benchmark_online4096_epoch01_cuda')
        run(checkpoint,spatial=True,implementation='optix-triangle',device='cuda:1',repetitions=3,name='benchmark_online4096_epoch01_optix_triangle')
    reports={name:read(OUT/('benchmark_online4096_epoch01_'+name+'.json'))for name in ['cuda','optix_triangle']}
    result=dict(role='Matched checkpoint, hardware, precision,32originalworkloads, K1/4/6,cold/warm,normalhypotheticalgraphs. Acquisition/serializationexcludedexplicitly. No accuracy acceptanceclaim.',load_isolation=isolation,models={name:{k:r[k]for k in ['checkpoint_sha256','implementation','precision','summary','startup','scope','excluded']}for name,r in reports.items()})
    save(OUT/'matched_speed_online4096_epoch01.json',result)


if __name__=='__main__':main()
