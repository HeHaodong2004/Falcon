"""Detached operations restricted to versioned forecasting research outputs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/haodong/workspace/projects/eventwm-falcon')
PYTHON='/home/haodong/miniforge3/envs/eventwm-falcon/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','stop','resume','_run']);p.add_argument('config')
    a=p.parse_args();config=Path(a.config).resolve();cfg=json.loads(config.read_text());run=Path(cfg['run']).resolve()
    if PROJECT/'output' not in run.parents or 'eventwm_forecast_' not in str(run):raise ValueError('Research path restriction')
    run.mkdir(parents=True,exist_ok=True)
    session='wmforecast_'+hashlib.sha256(str(run).encode()).hexdigest()[:12]
    active=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
    if a.action=='status':
        result=dict(run=str(run),session=session,active=active)
        for name in ('progress','result','failure','guard'):
            path=run/(name+'.json')
            if path.exists():
                value=json.loads(path.read_text())
                if name=='guard':value={k:v for k,v in value.items() if k!='samples'}
                result[name]=value
        print(json.dumps(result,indent=2));return
    if a.action=='stop':
        if not (run/'result.json').exists() and not (run/'manifest.json').exists():
            (run/'STOP_RESEARCH').write_text('Cooperative research-only stop\n')
        return
    if a.action in ('launch','resume'):
        if active:raise RuntimeError('Research session active')
        shared_guard=run.parent/'cache_full/guard.json'
        # The lightweight queue only monitors during a resource stop. Every
        # child extraction/training/benchmark launch still passes this gate.
        monitor_only=cfg['module']=='eventforecastwm.workflow'
        if run.name!='cache_full' and not monitor_only and shared_guard.exists():
            evidence=json.loads(shared_guard.read_text())
            if evidence['stopped']:
                raise RuntimeError('Full-pool collector guard stopped: new jobs cannot reset or bypass it')
            if evidence['samples'] and time.time()-evidence['samples'][-1]['timestamp']>180:
                raise RuntimeError('Shared collector guard evidence stale; run recovery monitor first')
        if (run/'result.json').exists() or (run/'manifest.json').exists():raise FileExistsError('Completed artifact')
        if a.action=='resume':(run/'STOP_RESEARCH').unlink(missing_ok=True)
        elif (run/'STOP_RESEARCH').exists():raise RuntimeError('Explicit resume required')
        frozen=run/'launch_config.json'
        if frozen.exists() and json.loads(frozen.read_text())!=cfg:raise ValueError('Changed run config')
        if not frozen.exists():frozen.write_text(json.dumps(cfg,indent=2)+'\n')
        # A running tmux server does not inherit a new client's environment.
        # Pin research-only runtime libraries explicitly, including on resume.
        runtime=cfg.get('environment',{})
        if set(runtime)-{'LD_LIBRARY_PATH'}:raise ValueError('Unsupported runtime environment override')
        cmd=shlex.join(['env',*[f'{k}={v}' for k,v in runtime.items()],
            PYTHON,'-B',str(Path(__file__).resolve()),'_run',str(config)])
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
        print(json.dumps(dict(session=session,run=str(run),action=a.action)));return
    with (run/'research.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='0,1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
        runtime=cfg.get('environment',{})
        if set(runtime)-{'LD_LIBRARY_PATH'}:raise ValueError('Unsupported runtime environment override')
        env.update(runtime)
        worker=['-m',cfg['module'],*cfg['args']]
        if 'eventwm_forecast_performance_v2_20261015' in run.parts:
            worker=['-m','eventforecastwm.performance_v2.policy_worker','--module',cfg['module'],
                    '--run',str(run),'--',*cfg['args']]
        cmd=['nice','-n','19','ionice','-c','3',PYTHON,'-B',*worker]
        # Append attempt history; the latest-exit receipt alone cannot account
        # for compute spent before a resource pause. This does not change guards
        # or model/optimizer state. RUSAGE_CHILDREN includes the single worker.
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        attempt=dict(timestamp=time.time(),controller_pid=os.getpid(),module=cfg['module'],
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),source_commit=commit,
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        def event(value):
            with (run/'lifecycle.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(attempt=attempt,**value))+'\n');ledger.flush();os.fsync(ledger.fileno())
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic()
        with (run/'job.log').open('a') as f:
            worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            event(dict(event='started',worker_pid=worker.pid));returncode=worker.wait()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        event(dict(event='exited',worker_pid=worker.pid,code=returncode,ended_timestamp=time.time(),
            wall_s=time.monotonic()-started,cpu_user_s=after.ru_utime-before.ru_utime,
            cpu_system_s=after.ru_stime-before.ru_stime,largest_child_peak_rss_bytes=after.ru_maxrss*1024))
        (run/'launcher_exit.json').write_text(json.dumps(dict(code=returncode,timestamp=time.time()))+'\n')
        if returncode:
            tail=(run/'job.log').read_text()[-6000:].rsplit('Traceback',1)[-1]
            (run/'failure.json').write_text(json.dumps(dict(error=tail,exit_code=returncode,timestamp=time.time()))+'\n')
        sys.exit(returncode)


if __name__=='__main__':main()
