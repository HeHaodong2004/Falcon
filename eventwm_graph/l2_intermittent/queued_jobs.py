"""Low-priority research queues; no production-process control operations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import OUT, BASE


def wait_for_research_file(path):
    while not path.exists():time.sleep(15)


def call(module,*arguments):
    name=module.rsplit('.',1)[-1]+('_'+str(arguments[0]).lstrip('-') if arguments else '')
    log=OUT/('queued_'+name+'.log')
    with log.open('x') as output:
        result=subprocess.run([sys.executable,'-m',module,*arguments],cwd=BASE,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('research command failed; see '+str(log))


def run(queue):
    os.setpriority(os.PRIO_PROCESS,0,19)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONPATH']=str(BASE)
    if queue=='sealed_diagnostics':
        wait_for_research_file(OUT/'primary_visible_masked_frozen_1/summary.json')
        runtime=OUT/'sealed_runtime'
        for operation in ('old_physical','physical','optical'):
            with (OUT/('sealed_'+operation+'.log')).open('x') as output:
                result=subprocess.run([sys.executable,str(runtime/'eventwm_graph/l2_intermittent/execution.py'),
                    operation,'--runtime',str(runtime)],cwd=BASE,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('sealed diagnostics failed: '+operation)
        call('eventwm_graph.l2_intermittent.validate')
    elif queue=='second_shard':
        wait_for_research_file(OUT/'data_and_gaps.json')
        call('eventwm_graph.l2_intermittent.evaluate','--start','1055')
    else:
        wait_for_research_file(OUT/'evaluation_shard_0000_1055/summary.json')
        call('eventwm_graph.l2_intermittent.physical','--reproduce-old')
        call('eventwm_graph.l2_intermittent.physical')
        call('eventwm_graph.l2_intermittent.optical_validation')
        call('eventwm_graph.l2_intermittent.validate')
    (OUT/('QUEUE_'+queue+'_COMPLETE')).write_text(json.dumps(dict(timestamp=time.time())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('queue',choices=('second_shard','diagnostics','sealed_diagnostics'));run(parser.parse_args().queue)
