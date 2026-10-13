from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
import collections,json,pathlib,time,sys,os
project=PROJECT_PATH;repo=SCIENCE_PATH;sys.path.insert(0,str(repo/'scripts'))
from monitor import snapshot
output=DATA_PATH;totals=collections.Counter();offsets={};last_queue=None;busy_integral=capacity_integral=0.;prep=[];validation=[];first_event_time=None;previous_epoch=None
log=project/'output/eventwm_300k_logs/phase_progress.jsonl'
while True:
 current=snapshot()
 for directory in sorted((output/'logs').glob('*')):
  path=directory/'scheduler_events.jsonl'
  if not path.exists():continue
  with path.open() as f:
   f.seek(offsets.get(str(path),0))
   while True:
    before=f.tell();line=f.readline()
    if not line:break
    try:e=json.loads(line)
    except ValueError:f.seek(before);break
    if previous_epoch!=str(directory):last_queue=None;previous_epoch=str(directory)
    kind=e['kind']
    if kind=='queue':
     if last_queue:
      dt=max(0.,e['time']-last_queue['time']);busy_integral+=last_queue['busy']*dt;capacity_integral+=last_queue.get('native_limit',36)*dt
     last_queue=e
    if kind=='result':
     totals['native_jobs']+=1;totals['slot_seconds']+=e.get('slot_seconds',0.);totals['native_wall_seconds']+=e.get('native_wall_seconds',0.);totals['native_cpu_seconds']+=e.get('native_cpu_seconds',0.)
     for k,v in e.get('metrics',{}).items():
      if isinstance(v,(int,float)):totals['phase_'+k]+=v
    if kind=='preparation_end':prep.append(e.get('seconds',0.))
    if kind=='validation_end':validation.append(e.get('seconds',0.))
   offsets[str(path)]=f.tell()
 current.update(native_slot_busy_fraction=busy_integral/capacity_integral if capacity_integral else None,
                average_completed_job_slot_seconds_per_published_query=totals['slot_seconds']/current['new_retained'] if current.get('new_retained') else None,
                last_queue=last_queue,totals=dict(totals),preparation_mean_seconds=sum(prep)/len(prep) if prep else None,
                validation_mean_seconds=sum(validation)/len(validation) if validation else None,
                phase_note='Nested phase times overlap; job slot sums are not elapsed wall time. Published denominator is provisional while validated backlog remains.')
 with log.open('a') as f:f.write(json.dumps(current)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(current),flush=True);time.sleep(90)
