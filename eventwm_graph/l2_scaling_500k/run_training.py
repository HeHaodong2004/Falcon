"""Bounded experiment orchestration; scientific fitting source is unmodified."""
import json
import os
import subprocess
import sys
import time
from .common import BASE,OUT,PRIOR,sha,dump
from .prepare import RUNTIME

def run():
    started=time.perf_counter();env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')})
    while not ((OUT/'snapshot/manifest.json').exists() and (OUT/'snapshot').stat().st_mode&0o777==0o555):
        if time.perf_counter()-started>900:raise RuntimeError('Snapshot did not finish sealing; no fitting started')
        time.sleep(1)
    def prep(*args):
        return subprocess.run([sys.executable,'-m','eventwm_graph.l2_scaling_500k.prepare',*map(str,args)],cwd=BASE,env=env,check=True)
    if not (OUT/'preparation_compatibility.json').exists():
        prep('shard','--start',0,'--stop',30,'--tags','compatibility')
        prep('compatibility')
    manifest=json.loads((OUT/'snapshot/manifest.json').read_text());n=len(manifest['rows']);begin=manifest['original_training_roots']
    bounds=[begin+(n-begin)*i//4 for i in range(5)];tags=['added_%d'%i for i in range(4)]
    workers=[]
    for start,stop,tag in zip(bounds[:-1],bounds[1:],tags):
        log=(OUT/(tag+'.log')).open('x')
        command=[sys.executable,'-m','eventwm_graph.l2_scaling_500k.prepare','shard','--start',str(start),'--stop',str(stop),'--tags',tag]
        p=subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT);workers.append((p,log))
    for p,log in workers:
        code=p.wait();log.close()
        if code:raise RuntimeError('Frozen preparation failed; do not alter pipeline')
    prep('assemble','--tags',*tags)
    env['PYTHONPATH']=str(RUNTIME)
    with (OUT/'fit.log').open('x') as log:
        subprocess.run([sys.executable,'-m','eventwm_graph.l2_intermittent.fit','train','--epochs','80'],cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import torch
    old=torch.load(PRIOR/'training/epoch0.pt',map_location='cpu');new=torch.load(OUT/'training/epoch0.pt',map_location='cpu')
    assert old['state_dict'].keys()==new['state_dict'].keys()
    assert all(torch.equal(old['state_dict'][k],new['state_dict'][k]) for k in old['state_dict'])
    assert old['dev_path_es']==new['dev_path_es']
    frozen=json.loads((OUT/'runtime_setup.json').read_text())
    assert all(sha(BASE/p)==h for p,h in frozen['untouched_runtime_sources'].items())
    assert sha(OUT/'motion.json')==sha(PRIOR/'motion.json') and sha(OUT/'dev.npy')==sha(PRIOR/'dev.npy')
    dump(OUT/'training_control.json',dict(epoch0_all_tensors_bitwise_equal=True,epoch0_dev_es_exactly_equal=new['dev_path_es'],
        runtime_unchanged=True,law_unchanged=True,dev_cache_unchanged=True,seconds=time.perf_counter()-started,
        fit_source_sha256=sha(RUNTIME/'eventwm_graph/l2_intermittent/fit.py'),checkpoint_sha256=sha(OUT/'training/checkpoint.pt')))

if __name__=='__main__':run()
