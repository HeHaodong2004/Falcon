from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Native phase accounting restricted to the geometry production epoch."""
import collections,json,pathlib,sqlite3,time
P=PROJECT_PATH;D=DATA_PATH;C=P/'output/eventwm_72h_v4_resume_checkpoint.json'
checkpoint=json.loads(C.read_text());offsets={};last={};totals=collections.Counter();busy=capacity=0.;prep=[];validation=[]
while True:
    with sqlite3.connect((DATA_PATH/'pipeline.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        meta=json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0]);progress=json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
    latest=None
    for directory in sorted(D.glob('logs/*')):
        file=directory/'scheduler_events.jsonl'
        if not file.exists() or file.stat().st_mtime<checkpoint['time']:continue
        # Exclude the fully drained epoch even if its final fsync was recent.
        if directory.name.endswith('_2399600'):continue
        with file.open() as f:
            f.seek(offsets.get(str(file),0))
            while True:
                position=f.tell();line=f.readline()
                if not line:break
                try:e=json.loads(line)
                except ValueError:f.seek(position);break
                kind=e['kind']
                if kind=='queue':
                    previous=last.get(str(file))
                    if previous:
                        dt=max(0,e['time']-previous['time']);busy+=dt*previous['busy'];capacity+=dt*previous['native_limit']
                    last[str(file)]=e
                if kind=='result':
                    totals['native_jobs']+=1
                    totals['root_jobs' if e['worker_id']=='root' else 'branch_audit_jobs']+=1
                    totals['failed_root_jobs']+=int(e['worker_id']=='root' and e.get('failed',False))
                    for key in ('slot_seconds','native_wall_seconds','native_cpu_seconds'):totals[key]+=e.get(key,0)
                    for key,value in e.get('metrics',{}).items():
                        if isinstance(value,(int,float)):totals['phase_'+key]+=value
                if kind=='preparation_end':prep.append(e.get('seconds',0))
                if kind=='validation_end':validation.append(e.get('seconds',0))
            offsets[str(file)]=f.tell()
        latest=last.get(str(file))
    retained=progress['primary_queries']-checkpoint['retained_before_resume']
    result=dict(timestamp=time.time(),source_commit=meta['frozen_commit'],retained_since_resume=retained,native_slot_busy_fraction=busy/capacity if capacity else None,completed_job_slot_seconds_per_retained_query=totals['slot_seconds']/retained if retained else None,totals=dict(totals),last_queue=latest,preparation_mean_seconds=sum(prep)/len(prep) if prep else None,validation_mean_seconds=sum(validation)/len(validation) if validation else None,scope='New geometry source epoch only; phase timers overlap; unfinished and unpublished costs affect denominator')
    target=P/'output/eventwm_72h_v4_native_metrics.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(target)
    print(json.dumps(result),flush=True);time.sleep(90)
