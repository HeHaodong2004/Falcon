import json, sys, time
from pathlib import Path
code = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(code/'scripts'), str(code)]
from eventwm_pipeline_pool import PipelineWorker
from eventwm_execution_pool import stop_checkpoint
pids=set(); checkpoints={}; worker=None
job=dict(root_id='v3m_e7c6987aa64211d2f63c', scene='hm3d_KJxdMPgweZG',
         output=str(output/'case'), worker_id='root', query_id=None, limit_steps=None)
try:
    worker=PipelineWorker(0,code,output,output/'logs',pids,checkpoints,time.perf_counter())
    result=worker.execute(job)
    (output/'root_result.json').write_text(json.dumps(result,indent=2))
    print('ROOT',result['error'],flush=True)
    assert result['error'] is None
    checkpoints[job['root_id']]=result['checkpoint']
    job.update(worker_id='b027',query_id='q_b85b8f97f044')
    result=worker.execute(job)
    (output/'cow_result.json').write_text(json.dumps(result,indent=2))
    print('COW',result['error'],result['label'],flush=True)
    stop_checkpoint(checkpoints.pop(job['root_id']),pids)
    job['worker_id']='fresh_b027'
    result=worker.request(job)
    (output/'fresh_result.json').write_text(json.dumps(result,indent=2))
    print('FRESH',result['error'],result['label'],flush=True)
finally:
    for cp in checkpoints.values(): stop_checkpoint(cp,pids)
    if worker: worker.close()
