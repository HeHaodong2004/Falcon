"""Execute an explicit frozen-data research queue, respecting shared recovery.

This watches only the specified immutable manifest dependencies. It never reads
new live queries into a training set and never creates a new dataset snapshot.
Resource pauses resume the same run and checkpoint, not a new experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.guard import Guard

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/haodong/miniforge3/envs/eventwm-falcon/bin/python'


def active(run):
    session='wmforecast_'+hashlib.sha256(str(Path(run).resolve()).encode()).hexdigest()[:12]
    return subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0


def command(action,path):
    return subprocess.run(['bash',str(ROOT/'scripts/run_eventwm_forecast.sh'),action,str(path)],cwd=ROOT,check=True)


def retry_ready(job,now=None):
    """Bound retries independently of the effective guard's recovery policy."""
    run=Path(read(job['config'])['run']);path=run/'launcher_exit.json'
    if not path.exists():return True
    last=read(path)
    return (time.time() if now is None else now)-last['timestamp']>=65.


def capacity_available(gpu,occupied,config):
    gpu_limit=config.get('jobs_per_gpu',1);cpu_limit=config.get('cpu_jobs',1)
    if gpu_limit not in (1,2) or cpu_limit not in (1,2):
        raise ValueError('Research queue permits one or two workers per resource slot')
    return occupied.get(gpu,0)<(cpu_limit if gpu is None else gpu_limit)


def preflight(directory):
    stamp=int(time.time());path=directory/f'preflight_{stamp}.json'
    subprocess.run([PYTHON,'-B',str(ROOT/'scripts/belief_resource_probe.py'),'--cuda','--output',str(path)],
        cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    probe=read(path);memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts=line.split();memory[parts[0].rstrip(':')]=int(parts[1])*1024
    if memory['MemAvailable']<48*1024**3:raise RuntimeError('Research RAM reserve below48GiB')
    # CPU load alone is not exhaustion. Low-priority jobs are deferred by the
    # sharing policy; RAM, disk, collector health and GPU reserves remain hard.
    if not probe['collector']['pid_exists'] or probe['collector']['state']!='running':raise RuntimeError('Collector health/ownership')
    if probe['disk_free_bytes']<50*1024**3:raise RuntimeError('Disk reserve')
    return probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    directory=Path(cfg['run']);directory.mkdir(parents=True,exist_ok=True)
    capacity_available(None,{},cfg)  # Validate both limits before starting workers.
    identity=dict(config=cfg,configs={j['config']:sha256(j['config']) for j in [cfg['cache'],*cfg['jobs']]},
                  workflow_sha256=sha256(__file__))
    frozen=directory/'identity.json'
    if frozen.exists() and read(frozen)!=identity:raise ValueError('Workflow queue/configs changed')
    save_json(frozen,identity)
    cache_cfg=read(cfg['cache']['config']);cache=Path(cache_cfg['run']);last_sample=0.;last_launch=0.
    started=time.time()
    while time.time()-started<cfg.get('max_hours',12)*3600:
        if (directory/'STOP_RESEARCH').exists():
            for j in [cfg['cache'],*cfg['jobs']]:
                run=read(j['config'])['run']
                if active(run):command('stop',j['config'])
            save_json(directory/'progress.json',dict(status='cooperative_stop_requested',production_actions=[]));return
        running_cache=active(cache)
        if not running_cache and time.time()-last_sample>=60:
            last_sample=time.time()
            try:Guard(cache)
            except RuntimeError:pass  # This monitor performs no training or raw extraction.
        evidence=read(cache/'guard.json') if (cache/'guard.json').exists() else None
        healthy=evidence is None or not evidence['stopped']
        running=[];pending=[];completed=[];blocked=[];occupied={}
        for j in cfg['jobs']:
            run=Path(read(j['config'])['run'])
            if active(run):
                running.append(str(run));gpu=j.get('gpu');occupied[gpu]=occupied.get(gpu,0)+1
            elif (run/'result.json').exists():completed.append(str(run))
            elif (run/'failure.json').exists() and 'Research stopped:' not in read(run/'failure.json').get('error',''):
                blocked.append(dict(run=str(run),failure=read(run/'failure.json')))
            else:pending.append(j)
        priority_runs=set(cfg.get('training_priority_runs',[]))
        priority_pending=any(str(Path(read(j['config'])['run'])) in priority_runs for j in pending)
        priority_active=bool(priority_runs&set(running))
        # Soft throughput degradation is handled by bounded duty throttling in
        # every worker. Once physical training finishes, one CPU calibration or
        # evaluation job must remain eligible even when all remaining jobs are
        # CPU-only; otherwise a healthy slow collector could stall all research.
        defer_auxiliary=priority_active or priority_pending
        if defer_auxiliary:
            for j in cfg['jobs']:
                run=str(Path(read(j['config'])['run']))
                if run in running and run not in priority_runs and j.get('gpu') is None:
                    command('stop',j['config'])
        status=dict(timestamp=time.time(),collector_guard_healthy=healthy,cache_active=running_cache,gpu_job_counts=occupied,
            cache_complete=(cache/'manifest.json').exists(),running=running,pending=len(pending),completed=completed,blocked=blocked)
        status.update(resource_policy=evidence.get('policy_version') if evidence else None,
            recent_collector_qph=evidence.get('latest_valid_qph') if evidence else None,
            throttle_level=evidence.get('throttle_level',0) if evidence else 0,
            training_priority_active=priority_active,auxiliary_jobs_deferred=defer_auxiliary,
            cpu_job_limit=cfg.get('cpu_jobs',1),gpu_job_limit=cfg.get('jobs_per_gpu',1))
        save_json(directory/'progress.json',status)
        if not pending and not running and (cache/'manifest.json').exists():
            save_json(directory/'result.json',dict(**status,status='queue_blocked' if blocked else 'explicit_queue_complete'));return
        ready=[j for j in pending if capacity_available(j.get('gpu'),occupied,cfg) and retry_ready(j)
               and all(Path(path).exists() for path in j.get('requires',[]))]
        if priority_active or priority_pending:
            ready=[j for j in ready if str(Path(read(j['config'])['run'])) in priority_runs]
        cache_ready=not running_cache and not (cache/'manifest.json').exists()
        if healthy and (cache_ready or ready) and time.time()-last_launch>15:
            try:
                probe=preflight(directory)
                if not running_cache and not (cache/'manifest.json').exists():
                    # Non-guard extraction errors must be inspected, never blind retried.
                    log=(cache/'job.log').read_text()[-4000:] if (cache/'job.log').exists() else ''
                    if 'Traceback' in log and not any(reason in log for reason in ('Research stopped:', 'Research-only stop requested')):
                        raise RuntimeError('Non-resource cache failure needs inspection')
                    command('resume' if (cache/'config.json').exists() else 'launch',cfg['cache']['config'])
                    last_launch=time.time()
                else:
                    for j in ready:
                        gpu=j.get('gpu')
                        if gpu is not None and probe['gpus'][gpu]['free_total_bytes'][0]<cfg.get('gpu_launch_free_gib',4)*1024**3:continue
                        run=Path(read(j['config'])['run'])
                        action='resume' if (run/'last.pt').exists() or (run/'STOP_RESEARCH').exists() else 'launch'
                        command(action,j['config']);last_launch=time.time();break
            except (RuntimeError,subprocess.CalledProcessError) as error:
                with (directory/'resource_waits.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=time.time(),reason=str(error)))+'\n')
                last_launch=time.time()
        time.sleep(15.)
    save_json(directory/'progress.json',dict(status='bounded_workflow_monitor_expired',running_jobs_may_continue=True))


if __name__=='__main__':main()
