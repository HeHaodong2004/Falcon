#!/usr/bin/env python3
"""Read-only live counts, measured windows and deadline projections."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from eventwm_pipeline_io import read_status
from eventwm_mass_collection import hardware
from eventwm_300k_storage import capacity

PROJECT=Path('/home/haodong/workspace/projects/eventwm-falcon')
OUTPUT=Path('/tmp/eventwm_v4_300k')
LOG=PROJECT/'output/eventwm_300k_logs/progress.jsonl'
DEADLINE=datetime.fromisoformat('2026-10-12T09:20:00+01:00').timestamp()
START=datetime.fromisoformat('2026-10-09T09:20:00+01:00').timestamp()

def snapshot():
    if not (OUTPUT/'pipeline.sqlite3').exists():return {'state':'not_initialized','new_retained':0,'hours_remaining':(DEADLINE-time.time())/3600}
    s=read_status(OUTPUT);p=s['progress'];now=time.time();q=p.get('primary_queries',0)
    oldfile=PROJECT/'output/eventwm_72h_legacy_verified.json'
    old=json.loads(oldfile.read_text()) if oldfile.exists() else {}
    credit=old.get('retained',0) if old.get('all_seals_verified') else 0
    remaining=max(0,300000-q-credit);hours=(DEADLINE-now)/3600
    samples=[]
    if LOG.exists():
        with LOG.open('rb') as f:
            f.seek(max(0,LOG.stat().st_size-1024*1024));data=f.read().decode(errors='replace')
        for line in data.splitlines():
            try:
                row=json.loads(line)
                if row.get('timestamp',0)>=now-900:samples.append(row)
            except ValueError:pass
    previous=next((r for r in samples if now-r['timestamp']>=120),None)
    rate=(q-previous['new_retained'])/(now-previous['timestamp'])*3600 if previous else None
    cumulative=q/s['collection_seconds']*3600 if s['collection_seconds'] else 0
    projection=now+remaining/rate*3600 if rate and rate>0 else None
    terminal=p.get('state_failed',0)+p.get('completed_roots',0)
    return dict(timestamp=now,now_london=datetime.fromtimestamp(now,ZoneInfo('Europe/London')).isoformat(),
                state=s['state'],runner=s['runner'],source_commit=s['source_commit'],target_new=s['target'],
                new_retained=q,new_train=p.get('role_train',0),new_natural_reference=p.get('role_natural_reference',0),
                new_unseen_evaluation=p.get('role_unseen_evaluation',0),verified_legacy=credit,
                total_retained=q+credit,total_train=p.get('role_train',0)+(old.get('role_queries',{}).get('train',0) if credit else 0),
                proposals=p.get('proposals',0),completed_roots=p.get('completed_roots',0),terminal_proposals=terminal,
                terminal_root_acceptance=p.get('completed_roots',0)/terminal if terminal else None,
                production_seconds=s['collection_seconds'],deadline_elapsed_hours=(now-START)/3600,hours_remaining=hours,
                required_qph=remaining/hours if hours>0 else None,measured_window_qph=rate,cumulative_production_qph=cumulative,
                rate_is_sustained=bool(previous and q-previous['new_retained']>=100 and now-previous['timestamp']>=600),
                window_seconds=now-previous['timestamp'] if previous else None,
                projected_finish_london=datetime.fromtimestamp(projection,ZoneInfo('Europe/London')).isoformat() if projection else None,
                deadline_pace_met=rate>=remaining/hours if rate is not None and hours>0 else None,
                free_capacity_gb=capacity(OUTPUT)/1e9,counts=p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=float,default=90);args=parser.parse_args();previous=None
    while True:
        row=snapshot()
        if not args.once:
            h=hardware(previous);previous=h;row['resources']=h
            LOG.parent.mkdir(parents=True,exist_ok=True)
            with LOG.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row,indent=2 if args.once else None),flush=True)
        if args.once:return
        time.sleep(args.interval)
if __name__=='__main__':main()
