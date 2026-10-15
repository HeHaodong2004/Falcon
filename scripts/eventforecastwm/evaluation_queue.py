"""One CPU evaluation at a time, gated on completed frozen training artifacts.

This queue does not sample or reset the shared production guard. The owning
training workflow supplies that evidence; each evaluation checks it independently.
It never selects final models or opens reserved confirmation data.
"""
import argparse
from pathlib import Path
import time
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.workflow import active,command,retry_ready,preflight


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    identity=dict(config=cfg,configs={j['config']:sha256(j['config']) for j in cfg['jobs']},source_sha256=sha256(__file__))
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Evaluation queue changed')
    save_json(out/'identity.json',identity);started=time.time()
    while time.time()-started<cfg.get('max_hours',12)*3600:
        running=[];pending=[];completed=[];blocked=[]
        for j in cfg['jobs']:
            run=Path(read(j['config'])['run'])
            if active(run):running.append(j)
            elif (run/'result.json').exists():completed.append(str(run))
            elif (run/'failure.json').exists() and 'Research stopped:' not in read(run/'failure.json').get('error',''):
                blocked.append(str(run))
            else:pending.append(j)
        if (out/'STOP_RESEARCH').exists():
            for j in running:command('stop',j['config'])
            save_json(out/'progress.json',dict(status='cooperative_stop_requested'));return
        status=dict(timestamp=time.time(),running=[j['config'] for j in running],pending=len(pending),completed=completed,blocked=blocked)
        save_json(out/'progress.json',status)
        if not running and not pending:
            save_json(out/'result.json',dict(**status,status='queue_blocked' if blocked else 'evaluation_queue_complete'));return
        if not running:
            ready=[j for j in pending if retry_ready(j) and all(Path(p).exists() for p in j.get('requires',[]))]
            shared=out.parent/'cache_full/guard.json'
            if shared.exists():
                evidence=read(shared)
                if evidence['stopped'] or not evidence['samples'] or time.time()-evidence['samples'][-1]['timestamp']>180:ready=[]
            if ready:
                try:
                    preflight(out);j=ready[0];run=Path(read(j['config'])['run'])
                    command('resume' if (run/'STOP_RESEARCH').exists() else 'launch',j['config'])
                except Exception as error:
                    save_json(out/'last_wait.json',dict(timestamp=time.time(),reason=str(error)))
        time.sleep(30.)
    save_json(out/'progress.json',dict(status='bounded_evaluation_queue_expired'))


if __name__=='__main__':main()
