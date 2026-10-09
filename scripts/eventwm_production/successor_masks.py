from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
from control import durable
"""Versioned supervision-only horizon artifacts derived from sealed native truth.

Never imported by any collector, sampler or model-input builder. Retention and
SR remain the frozen collector's decisions. Linear velocity is derived from
actual timestamp/position differences; native zero velocity reports are unused.
"""
import bisect,gzip,hashlib,json,math,pathlib,sqlite3,time,os
PROJECT=PROJECT_PATH;DATA=DATA_PATH;DEST=PROJECT/'output/eventwm_300k_supervision';HORIZONS=(.25,.5,1.,2.,3.)

def targets(branch,label):
 rows=branch['truth'];times=[r['timestamp'] for r in rows];assert all(b>a for a,b in zip(times,times[1:]));t0=times[0]
 observations=branch['legal_observations'];obs_times=[r['timestamp'] for r in observations]
 out=[]
 for h in HORIZONS:
  target=t0+h;valid=times[-1]>=target-1e-8
  pose=None;velocity=None;sample_time=None;obs=None
  if valid:
   i=bisect.bisect_left(times,target);i=min(i,len(rows)-1)
   if i>0 and abs(times[i-1]-target)<abs(times[i]-target):i-=1
   sample_time=times[i];pose=[dict(object_id=p['object_id'],position=p['position'],yaw=p['yaw']) for p in rows[i]['humans']]
   a=max(0,i-1);b=min(len(rows)-1,i+1);dt=times[b]-times[a]
   velocity=[]
   for p0,p1 in zip(rows[a]['humans'],rows[b]['humans']):
    assert p0['object_id']==p1['object_id'];v=[(y-x)/dt for x,y in zip(p0['position'],p1['position'])] if dt>0 else None
    if v is not None:assert all(math.isfinite(x) for x in v)
    velocity.append(dict(object_id=p0['object_id'],position_difference_velocity=v,source_timestamps=[times[a],times[b]]))
   j=bisect.bisect_left(obs_times,target);j=min(j,len(observations)-1)
   if j>0 and abs(obs_times[j-1]-target)<abs(obs_times[j]-target):j-=1
   obs=observations[j] if observations else None
  events={k:label.get(k) for k in ('first_native_event_time','first_static_event_time','first_social_violation_time')}
  physical=any(events[k] is not None and events[k]<=h+1e-8 for k in ('first_native_event_time','first_static_event_time'))
  event_valid=bool(valid or physical);horizon_sr=0 if physical else 1 if valid else None
  out.append(dict(horizon_seconds=h,target_timestamp=target,actual_sample_timestamp=sample_time,
      actual_human_poses=pose,derived_human_velocities=velocity,future_arrived_legal_observation=obs,
      censoring_masks=dict(pose=valid,velocity=valid and bool(velocity is not None),future_observation=valid and obs is not None,physical_event=event_valid),
      horizon_physical_collision=physical if event_valid else None,horizon_sr=horizon_sr,event_times_relative_to_query_start=events))
 return dict(schema='eventwm-successor-horizons-v1',supervision_only=True,root_id=branch['root_id'],query_id=branch['query_id'],branch_id=branch['branch_id'],
    sr_label=label['sr_label'],sr_valid=label['sr_valid'],velocity_method='timestamp-position symmetric finite difference; native linear_velocity ignored',
    sampling_method='nearest actually recorded native sample within the observed horizon; exact timestamp retained',horizons=out,
    end_state=dict(timestamp=times[-1],robot_position=rows[-1]['robot']['position'],robot_yaw=rows[-1]['robot']['yaw'],
      actual_human_poses=[dict(object_id=p['object_id'],position=p['position'],yaw=p['yaw']) for p in rows[-1]['humans']],
      termination_reason=branch['termination_reason'],completed=branch['completed'],observed_duration=branch['observed_duration'],planned_duration=branch['planned_duration']))

def process(root):
 DEST.mkdir(parents=True,exist_ok=True)
 path=DATA/root['path'];labels={r['query_id']:r for r in json.load(open(DATA/'primary_indices'/f"{root['root_id']}.json"))['samples']};dest=DEST/f"{root['root_id']}.jsonl.gz";tmp=dest.with_suffix('.gz.tmp');counts={str(h):dict(valid=0,censored=0,determined_early_collision=0) for h in HORIZONS};retained=0
 with gzip.open(path/'branch_truth/branches.jsonl.gz','rt') as source,gzip.open(tmp,'wt',compresslevel=1) as target:
  for line in source:
   b=json.loads(line)
   if not b['branch_id'].startswith('b') or b['query_id'] not in labels:continue
   row=targets(b,labels[b['query_id']]);target.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');retained+=1
   for x in row['horizons']:
    c=counts[str(x['horizon_seconds'])];c['valid']+=int(x['censoring_masks']['pose']);c['censored']+=int(not x['censoring_masks']['pose']);c['determined_early_collision']+=int(x['horizon_physical_collision'] is True and not x['censoring_masks']['pose'])
 assert retained==root['stats']['primary_queries']
 with tmp.open('rb') as f:os.fsync(f.fileno())
 tmp.replace(dest)
 proof=dict(root_id=root['root_id'],role=root['role'],source_dataset=str(DATA),sealed_root_path=root['path'],source_seal_file=root['seal_file'],samples=retained,horizon_counts=counts,artifact_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),supervision_only=True)
 durable(DEST/f"{root['root_id']}.provenance.json",proof);return proof

def main():
 last=-1
 while True:
  with sqlite3.connect((DATA/'pipeline.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
   rows=[(ordinal,json.loads(v)) for ordinal,v in db.execute("SELECT ordinal,value FROM roots WHERE ordinal>? ORDER BY ordinal LIMIT 256",(last,))]
  advanced=False
  for ordinal,r in rows:
   if r['state'] not in ('completed','failed'):break
   if r['state']=='failed' and len(r['attempts'])<2 and r['attempts'][-1].get('reason') in ('worker_timeout','root_timeout','interrupted_before_commit'):break
   if r['state']=='completed' and not (DEST/f"{r['root_id']}.provenance.json").exists():
    proof=process(r);print(json.dumps(proof),flush=True)
   last=ordinal;advanced=True
  time.sleep(1 if advanced and len(rows)==256 else 90)
if __name__=='__main__':main()
