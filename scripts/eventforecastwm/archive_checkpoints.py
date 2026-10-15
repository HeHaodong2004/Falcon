"""Bounded research-only checkpoint observer for fixed-compute comparisons.

It reads atomically published checkpoints and copies small model/optimizer files,
never raw datasets. It does not start, stop, or modify any training process.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import torch
from eventbeliefwm.data import read
from eventbeliefwm.experiment import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[4096,8192,16384,24576,32768])
    p.add_argument('--output',required=True);p.add_argument('--max-hours',type=float,default=12.);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed checkpoint archive')
    identity=dict(runs=a.run,steps=sorted(a.steps),purpose='Fixed-compute checkpoint preservation; no training actions')
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Archive membership changed')
    save_json(out/'identity.json',identity);started=time.time()
    while time.time()-started<a.max_hours*3600:
        if (out/'STOP_RESEARCH').exists():return
        pending=0
        for path in a.run:
            run=Path(path);directory=out/run.name;directory.mkdir(exist_ok=True)
            due=[step for step in a.steps if not (directory/f'target{step:06d}.json').exists()]
            pending+=len(due)
            progress=run/'progress.json'
            if not due or not progress.exists() or read(progress)['step']<min(due):continue
            if shutil.disk_usage(out).free<50*1024**3:raise RuntimeError('Disk reserve; checkpoints remain in their training runs')
            payload=(run/'last.pt').read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu');step=state['step']
            reached=[target for target in due if target<=step]
            if not reached:continue
            digest=hashlib.sha256(payload).hexdigest();destination=directory/f'step{step:06d}_{digest[:12]}.pt'
            if not destination.exists():
                with destination.open('xb') as handle:handle.write(payload)
                destination.chmod(0o444)
            for target in reached:
                save_json(directory/f'target{target:06d}.json',dict(target_step=target,actual_step=step,
                    exact_step=target==step,checkpoint=str(destination),sha256=digest,source_run=str(run),timestamp=time.time()))
        save_json(out/'progress.json',dict(pending_targets=pending,seconds=time.time()-started,training_actions=[]))
        if not pending:save_json(out/'result.json',dict(status='archived',**identity));return
        time.sleep(5.)
    save_json(out/'progress.json',dict(status='bounded_observer_expired',training_actions=[]))


if __name__=='__main__':main()
