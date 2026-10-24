"""Sequential controlled evaluation after training; never controls production."""
import json
import os
import subprocess
import sys
import time
from .common import BASE,OUT,dump

def run():
    start=time.perf_counter();env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')})
    while not (OUT/'training_control.json').exists():
        if time.perf_counter()-start>28800:raise RuntimeError('Training did not complete; do not evaluate an unfinished candidate')
        time.sleep(1)
    audit=json.loads((OUT/'data_integrity.json').read_text());assert not audit['errors']
    def command(module,*args,log):
        with (OUT/log).open('x') as f:
            subprocess.run([sys.executable,'-m','eventwm_graph.l2_scaling_500k.'+module,*args],cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    command('evaluation','setup','--arm','new',log='evaluation_setup.log')
    bounds=[0,528,1056,1583,2110];workers=[]
    for index,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
        if index==2:
            # At most6 single-thread evaluation jobs, leaving2 logical CPUs
            # beyond the28 normal-priority collector workers. Training is done.
            while not (OUT/'original_baseline_complete.json').exists() and len(list((OUT/'evaluation_arm_original_full').glob('evaluation_shard_*/execution_guard.json')))<2:
                if time.perf_counter()-start>28800:raise RuntimeError('Original matched baseline did not advance')
                time.sleep(1)
        f=(OUT/('evaluate_%d_%d.log'%(a,b))).open('x')
        p=subprocess.Popen([sys.executable,'-m','eventwm_graph.l2_scaling_500k.evaluation','run','--arm','new','--start',str(a),'--stop',str(b)],cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT);workers.append((p,f))
    for p,f in workers:
        code=p.wait();f.close()
        if code:raise RuntimeError('Frozen evaluation failed; preserve evidence')
    while not (OUT/'original_baseline_complete.json').exists():
        if time.perf_counter()-start>28800:raise RuntimeError('Original matched baseline did not complete')
        time.sleep(1)
    command('baseline','--reconcile-only',log='baseline_reconciliation.log')
    command('merge',log='merge.log')
    command('evaluation','physical','--arm','new',log='physical_wrapper.log')
    command('compare',log='comparison.log')
    alias=BASE/'output/l2_intermittent_wm_20261022'
    assert not alias.exists();alias.symlink_to(OUT,target_is_directory=True)
    # Timings follow all heavy research fitting/evaluation, with collector live.
    for arm in ('original','new'):command('benchmark','--arm',arm,log='benchmark_'+arm+'.log')
    validation=json.loads((OUT/'scaling_regressions/validation.json').read_text())
    assert validation['total_tests']==399 and validation['inherited_sources_unchanged'] and validation['optimized_sources_unchanged']
    from .common import sha
    assert all(sha(BASE/p)==h for p,h in validation['optimized_sources'].items())
    dump(OUT/'evaluation_complete.json',dict(complete=True,seconds=time.perf_counter()-start,
        stages=['exact_frozen_cases','unchanged_annotation','native_physics','paired_comparison','old_new_complete_latency','regressions']))

if __name__=='__main__':run()
