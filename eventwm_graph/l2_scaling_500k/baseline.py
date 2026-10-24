"""Rerun the frozen checkpoint and MemoryCV under the identical deployed code."""
import json
import argparse
import collections
import gzip
import os
import subprocess
import sys
import time
from .common import BASE,OUT,dump,sha

def reconcile():
    historical=BASE/'docs/l2/intermittent_wm_20261022/artifacts/evaluation'
    fresh=OUT/'evaluation_arm_original_full/evaluation';reports=[]
    for kind in ('forecasts','reobservation','update_gains'):
        def load(path):
            result={}
            with gzip.open(path,'rt') as f:
                for line in f:
                    r=json.loads(line)
                    if r['model']!='D_StopMove':continue
                    if kind=='forecasts' and r['status']=='joint_supported':continue
                    key=tuple(r.get(k) for k in ('root_id','cohort','decision','stream','physical','H'))
                    assert key not in result;result[key]=r
            return result
        a=load(historical/(kind+'.jsonl.gz'));b=load(fresh/(kind+'.jsonl.gz'))
        assert a.keys()==b.keys(),'The original complete target/censor ledger must be preserved'
        different=collections.Counter();maximum=collections.defaultdict(float);different_rows=0
        for key in a:
            fields=[]
            for field in set(a[key])|set(b[key]):
                x,y=a[key].get(field),b[key].get(field)
                if x!=y:
                    fields.append(field);different[field]+=1
                    if isinstance(x,(int,float)) and isinstance(y,(int,float)):maximum[field]=max(maximum[field],abs(x-y))
            different_rows+=bool(fields)
        reports.append(dict(kind=kind,cases=len(a),same_complete_case_ledger=True,different_rows=different_rows,
            different_fields=dict(different),maximum_numeric_difference=dict(maximum),
            historical_sha256=sha(historical/(kind+'.jsonl.gz')),fresh_sha256=sha(fresh/(kind+'.jsonl.gz'))))
    dump(OUT/'baseline_historical_reconciliation.json',dict(reports=reports,
        semantics='Fresh original-checkpoint inference uses the exact same deployed code as the scaled arm. Historical artifacts and complete target/censor denominators remain unchanged and preserved; numerical/runtime differences are explicit.'))

def complete_existing():
    assert (OUT/'evaluation_arm_original_full/evaluation/merged_integrity.json').exists()
    from .evaluation import physical
    physical('original_full')
    inventory=OUT/'evaluation_runtime_original_full/source_manifest.json'
    dump(OUT/'original_baseline_complete.json',dict(complete=True,seconds=time.time()-inventory.stat().st_mtime,
        source_manifest_sha256=sha(inventory),resumed_postprocessing_only=True,
        note='All original predictions preserved; annotation lookup resolved to the archived byte-identical rule, then native physics completed.'))

def run():
    start=time.perf_counter();env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')})
    def command(module,*args,log):
        with (OUT/log).open('x') as f:subprocess.run([sys.executable,'-m','eventwm_graph.l2_scaling_500k.'+module,*args],cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    command('evaluation','setup','--arm','original_full',log='baseline_setup.log')
    bounds=[0,528,1056,1583,2110];workers=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        f=(OUT/('baseline_%d_%d.log'%(a,b))).open('x')
        p=subprocess.Popen([sys.executable,'-m','eventwm_graph.l2_scaling_500k.evaluation','run','--arm','original_full','--start',str(a),'--stop',str(b)],cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT);workers.append((p,f))
    for p,f in workers:
        result=p.wait();f.close()
        if result:raise RuntimeError('Original-checkpoint matched evaluation failed')
    command('merge','--arm','original_full',log='baseline_merge.log')
    command('evaluation','physical','--arm','original_full',log='baseline_physical_wrapper.log')
    dump(OUT/'original_baseline_complete.json',dict(complete=True,seconds=time.perf_counter()-start,
        source_manifest_sha256=sha(OUT/'evaluation_runtime_original_full/source_manifest.json'),
        purpose='Full original D and fixed MemoryCV rerun on identical2110 frozen roots, actual deployed inference bytes and root streams. Historical artifacts preserved separately.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reconcile-only',action='store_true');p.add_argument('--complete-existing',action='store_true');a=p.parse_args()
    if a.reconcile_only:reconcile()
    elif a.complete_existing:complete_existing()
    else:run()
