from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
from settings import publication_epoch, MEASUREMENT_PREFIX
"""Measure retained production, never infer q/h from root-only experiments."""
import json,pathlib,sys,time
from datetime import datetime
from zoneinfo import ZoneInfo
P=PROJECT_PATH;R=SCIENCE_PATH;sys.path.insert(0,str(R/'scripts'))
from monitor import snapshot
from eventwm_validate_staged import durable_json
D=DATA_PATH
checkpoint=publication_epoch();initial=checkpoint['retained_before_resume'];started=checkpoint['time'];rows=[];done=set()
while True:
    row=snapshot();elapsed=time.time()-started;q=row['new_retained']-initial
    rate=q/elapsed*3600 if elapsed else 0
    report=dict(timestamp=time.time(),source_commit=row['source_commit'],state=row['state'],seconds=elapsed,retained_since_checkpoint=q,actual_publication_qph_including_resume_startup=rate,total_retained=row['total_retained'],total_train=row['total_train'],required_qph=row.get('required_qph'),remaining_hours=row['hours_remaining'],deadline_pace_met=rate>=row.get('required_qph',1e99),projected_finish_london=datetime.fromtimestamp(time.time()+(TARGET_QUERIES-row['total_retained'])/rate*3600,ZoneInfo('Europe/London')).isoformat() if rate else None)
    rows.append(report);durable_json(P/'output'/f'{MEASUREMENT_PREFIX}_production_measurement.json',dict(latest=report,rows=rows))
    print(json.dumps(report),flush=True)
    for minute in (10,30,60):
        if elapsed>=minute*60 and minute not in done:
            durable_json(P/'output'/f'{MEASUREMENT_PREFIX}_{minute}min.json',report);done.add(minute)
    if row['state'] in ('interrupted_error','stopped_resource_guard'):durable_json(P/'output'/f'{MEASUREMENT_PREFIX}_ALERT.json',report)
    if row.get('total_retained',0)>=TARGET_QUERIES:break
    time.sleep(90)
