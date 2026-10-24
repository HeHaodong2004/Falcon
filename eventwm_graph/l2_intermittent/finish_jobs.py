"""Finish the already declared sealed research jobs; no production operations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import OUT, BASE, dump
from .package_release import copy_once


def wait_guard(path):
    while not path.exists():time.sleep(15)
    record=json.loads(path.read_text())
    if record['error'] or not record['sources_stable'] or not record['inputs_stable']:
        raise RuntimeError('failed scientific execution: '+str(path))


def call(arguments,name):
    with (OUT/(name+'.log')).open('x') as output:
        result=subprocess.run([sys.executable,*arguments],cwd=BASE,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('research command failed: '+name)


def run():
    os.setpriority(os.PRIO_PROCESS,0,19)
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONPATH']=str(BASE)
    runtime=OUT/'sealed_runtime';source=runtime/'output/l2_intermittent_wm_20261022'
    ranges=((0,528),(528,1055),(1055,1583),(1583,2110))
    for begin,end in ranges:
        wait_guard(source/('evaluation_shard_%04d_%04d'%(begin,end))/'forecast_execution_guard.json')
    call([str(runtime/'eventwm_graph/l2_intermittent/execution.py'),'merge','--runtime',str(runtime)],'sealed_merge')
    wait_guard(source/'evaluation/merge_execution_guard.json')
    for name in ['evaluation']+['evaluation_shard_%04d_%04d'%r for r in ranges]:
        shutil.copytree(source/name,OUT/name)
    dump(OUT/'NATIVE_SCORES_READY_FOR_REVIEW.json',dict(timestamp=time.time(),evaluation=str(OUT/'evaluation'),
        seal=str(runtime/'manifest.json'),scope='All four disjoint source/input-stable shards merged; no doubled physical roots.'))
    call(['-m','eventwm_graph.l2_intermittent.gap_context'],'gap_context')
    call(['-m','eventwm_graph.l2_intermittent.figures'],'figures')
    for operation,name in (('old_physical','old_physical_reproduction.json'),('physical','physical.json'),('optical','optical_validation.json')):
        wait_guard(source/(operation+'_execution_guard.json'))
        copy_once(source/name,OUT/name)
    while not (OUT/'validation/validation.json').exists():time.sleep(15)
    dump(OUT/'ALL_SCIENCE_EXPORTED.json',dict(timestamp=time.time(),
        scope='Measured forecast, native context, physical, optical, systems and regression artifacts complete. Final independent review/report/package/commit remain.'))
    print('Scientific jobs merged/exported; independent final review remains.',flush=True)


if __name__=='__main__':run()
