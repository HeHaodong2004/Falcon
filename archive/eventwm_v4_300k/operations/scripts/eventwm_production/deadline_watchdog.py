from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Durable deadline accounting; stopping time is distinct from target time."""
import json,pathlib,subprocess,sys,time
from datetime import datetime
from zoneinfo import ZoneInfo
P=PROJECT_PATH;R=SCIENCE_PATH;sys.path.insert(0,str(R/'scripts'))
from monitor import snapshot,DEADLINE,START,LOG
from eventwm_validate_staged import durable_json
from eventwm_pipeline_io import read_status
from active_release import drain
D=DATA_PATH;target_receipt=P/'output/eventwm_72h_TARGET_REACHED.json';misses=0;stopping=target_receipt.exists()
while True:
    row=snapshot();now=time.time();history=[]
    if LOG.exists():
        with LOG.open('rb') as f:f.seek(max(0,LOG.stat().st_size-1024*1024));lines=f.read().decode(errors='replace').splitlines()
        for line in lines:
            try:v=json.loads(line)
            except ValueError:continue
            if now-3600<=v.get('timestamp',0)<=now-1200:history.append(v)
    previous=history[0] if history else None
    rate=(row['new_retained']-previous['new_retained'])/(now-previous['timestamp'])*3600 if previous else None
    row['watchdog_qph_20_to_60_min']=rate
    if rate is not None and row['state']=='running' and row.get('required_qph'):
        misses=misses+1 if rate<row['required_qph'] else 0
        if misses>=3:
            alert=dict(time=now,total_retained=row['total_retained'],required_qph=row['required_qph'],measured_qph=rate,deficit_qph=row['required_qph']-rate,
                       remaining_hours=row['hours_remaining'],action='Inspect native slot profiles, root acceptance and validation backlog; second workstation unavailable')
            durable_json(P/'output/eventwm_72h_DEADLINE_ALERT.json',alert);print('DEADLINE_ALERT',json.dumps(alert),flush=True)
    if row['state'] in ('interrupted_error','stopped_resource_guard'):
        durable_json(P/'output/eventwm_72h_COLLECTOR_ALERT.json',row)
    # Milestones explicitly account for the compatible old TOTAL, not TRAIN.
    for hour in (6,24,48):
        file=P/'output'/f'eventwm_72h_milestone_{hour}h.json'
        if now>=START+hour*3600 and not file.exists():
            expected=row.get('verified_legacy',0)+(TARGET_QUERIES-row.get('verified_legacy',0))*hour/72
            durable_json(file,dict(hour=hour,total_retained=row['total_retained'],train_retained=row['total_train'],expected_linear_total=expected,
                                   linear_checkpoint_met=row['total_retained']>=expected,timestamp=now,current_required_qph=row['required_qph']))
    if now>=DEADLINE and row.get('total_retained',0)<TARGET_QUERIES:durable_json(P/'output/eventwm_72h_DEADLINE_MISSED.json',row)
    if row.get('total_retained',0)>=TARGET_QUERIES and row.get('verified_legacy',0) and not stopping:
        receipt=dict(timestamp=now,now_london=row['now_london'],total_retained=row['total_retained'],train_retained=row['total_train'],within_deadline=now<=DEADLINE)
        durable_json(target_receipt,receipt)
        drain();stopping=True
        print('300K TOTAL REACHED; graceful drain requested',json.dumps(receipt),flush=True)
    if stopping and read_status(D)['state'] in ('stopped_by_user','target_reached'):
        integrity_file=P/'output/eventwm_300k_combined_index/integrity.json'
        integrity=json.loads(integrity_file.read_text()) if integrity_file.exists() else {}
        if integrity.get('passed') and integrity.get('total_verified',0)>=TARGET_QUERIES:
            final=snapshot();target=json.loads(target_receipt.read_text());final.update(mission_target_reached=True,within_deadline=target['within_deadline'],
                    first_verified_target_observed_london=target['now_london'],deep_verified_total=integrity['total_verified'],integrity_report=str(integrity_file))
            durable_json(P/'output/eventwm_72h_RETENTION_TARGET_CONFIRMED.json',final);print('RETENTION_TARGET_CONFIRMED',json.dumps(final),flush=True);break
    durable_json(P/'output/eventwm_72h_live_status.json',row);print(json.dumps(row),flush=True);time.sleep(90)
