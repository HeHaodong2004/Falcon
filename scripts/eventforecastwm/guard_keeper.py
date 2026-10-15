"""Maintain existing collector evidence after the training monitor exits.

Never samples concurrently with the specified owning workflow. This process
does not launch jobs, reset guard history, or send production signals.
"""
import argparse
from pathlib import Path
import time
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.guard import Guard
from eventforecastwm.workflow import active


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True);identity=dict(config=cfg,source_sha256=sha256(__file__))
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Keeper identity changed')
    save_json(out/'identity.json',identity);started=time.time();last=0.
    while time.time()-started<cfg.get('max_hours',12)*3600:
        if (out/'STOP_RESEARCH').exists():save_json(out/'progress.json',dict(status='stopped'));return
        owner_active=any(active(run) for run in cfg['owners'])
        dependents_active=any(active(run) for run in cfg['dependents'])
        if not owner_active and dependents_active and time.time()-last>=60:
            last=time.time()
            try:Guard(cfg['guard_directory'])
            except RuntimeError:pass
        save_json(out/'progress.json',dict(timestamp=time.time(),owner_active=owner_active,
            dependents_active=dependents_active,last_guard_sample=last,production_actions=[]))
        if not owner_active and not dependents_active:
            save_json(out/'result.json',dict(status='research_monitors_finished',production_actions=[]));return
        time.sleep(15.)
    save_json(out/'progress.json',dict(status='bounded_guard_keeper_expired'))


if __name__=='__main__':main()
