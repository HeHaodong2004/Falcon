"""Durable collection-only campaign: owned workers, sealed receipts, safe resume.

There is no model, optimizer, trainer, or historical supervisor dependency.
SQLite transactions record each attempted seed; immutable roots are the final
authority after a crash between publication and a database commit.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sqlite3
import time
import traceback
from ..config import REPO,digest,file_digest,read_json,write_json
from .collect import verify_gate

_WORKER_CACHE={}
EXPECTED_REJECTIONS={'prefix_human_collision','insufficient_legal_compilable_queries',
                     'unsupported_encounter_initialization','encounter_spawn_too_close','no_private_bootstrap_start'}


def ready_scenes():
    result={}
    for path in sorted((REPO/'output/stage0_benchmark/corpus/scenes').glob('*/ready.json')):
        r=read_json(path)
        if r.get('ready') and Path(r['manifest']).exists():result[path.parent.name]=r['manifest']
    for path in sorted((REPO/'output/stage0_benchmark/certified').glob('*/*/manifest.json')):
        result.setdefault(path.parent.parent.name,str(path.resolve()))
    return result


def worker_batch(jobs,configuration):
    from ..benchmark.map import load_certified_map
    from .collect import collect_root
    from .report import distribution
    from .sampling import choose_start
    from ..stage1.storage import scene_split
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
    try:os.nice(10)
    except OSError:pass
    verify_gate(configuration['stage1_gate'],'stage1')
    results=[]
    for job in jobs:
        began=time.time()
        try:
            scene=job['scene']
            if scene not in _WORKER_CACHE:
                if len(_WORKER_CACHE)>=2:_WORKER_CACHE.pop(next(iter(_WORKER_CACHE)))
                t=time.perf_counter();h=load_certified_map(job['manifest'])
                _WORKER_CACHE[scene]=(h,{},time.perf_counter()-t,{})
            h,private_cache,cache_seconds,prior_cache=_WORKER_CACHE[scene]
            prior_started=time.perf_counter()
            node_override,prior=choose_start(h,job['seed'],configuration.get('private_root_curriculum_probability',0.),prior_cache)
            prior_seconds=time.perf_counter()-prior_started
            record=collect_root(h,configuration['dataset'],job['seed'],configuration['stage0_gate'],
                configuration['queries_per_root'],configuration['max_query_steps'],node_index=node_override,private_cache=private_cache,
                safe_margin=configuration.get('safe_margin_m',.2),intrusion_radius=configuration.get('intrusion_radius_m',.96))
            prior.update(root_id=record['root_id'],seed=job['seed'],scene=scene,
                         sampler_sha256=file_digest(Path(__file__).with_name('sampling.py')),
                         private_geometry_manifest_sha256=file_digest(job['manifest']))
            sampling_file=Path(configuration['dataset'])/'sampling'/(record['root_id']+'.json')
            if sampling_file.exists() and read_json(sampling_file)!=prior:raise ValueError('Root sampling receipt changed')
            write_json(sampling_file,prior)
            with sampling_file.open('rb') as file:os.fsync(file.fileno())
            fd=os.open(sampling_file.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
            result={'seed':job['seed'],'status':'sealed','root_id':record['root_id'],
                'folder':record['folder'],'outcomes':record['queries'],'statistics':record['statistics'],
                'bytes':sum(p.stat().st_size for p in Path(record['folder']).iterdir() if p.is_file()),
                'wall_seconds':time.time()-began,
                'geometry_load_seconds':cache_seconds,'performance':record['performance']}
            result['private_initialization_sampling_seconds']=prior_seconds
            if scene_split(scene)=='train':result['distributions']=distribution(record['folder'])
            result['root_prior']=prior;result['sampling_receipt_sha256']=file_digest(sampling_file)
            results.append(result)
        except Exception as error:
            expected=isinstance(error,ValueError) and str(error) in EXPECTED_REJECTIONS
            results.append({'seed':job['seed'],'status':'rejected' if expected else 'error',
                'error':repr(error),'wall_seconds':time.time()-began,
                'traceback':None if expected else traceback.format_exc()})
            if not expected:break
    return results


def database(path):
    db=sqlite3.connect(path,timeout=60)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(seed INTEGER PRIMARY KEY, scene TEXT NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL, result TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs(state,seed)')
    db.commit();return db


def available_memory():
    with open('/proc/meminfo') as file:
        for line in file:
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux MemAvailable resource guard unavailable')


def statistics(db):
    counts=dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    totals={'roots':0,'outcomes':0,'bytes':0,'human_collisions':0,'psc_violations':0,
            'intrusive_without_collision':0,'proposed_safe':0,'proposed_intrusion':0,
            'query_completions':0,'query_timeouts':0}
    scenes=set();split_counts={}
    distributions={key:{} for key in ('spatial_proposal_attempts','geometrically_accepted_orientations',
                                      'selected_and_executed_queries','actual_outcomes')}
    from ..stage1.storage import scene_split
    for scene,result in db.execute("SELECT scene,result FROM jobs WHERE state='sealed'"):
        row=json.loads(result);totals['roots']+=1;totals['outcomes']+=row['outcomes'];totals['bytes']+=row['bytes'];scenes.add(scene)
        split={'train':'train','validation':'dev','test':'confirmation'}[scene_split(scene)]
        split_counts[split]=split_counts.get(split,0)+row['outcomes']
        # Outcome diagnostics used for development come only from train scenes.
        if split=='train':
            for k in totals:
                if k not in ('roots','outcomes','bytes'):totals[k]+=row['statistics'].get(k,0)
            for category,values in distributions.items():
                for key,value in row.get('distributions',{}).get(category,{}).items():values[key]=values.get(key,0)+value
    return {'attempts':counts,**totals,'scenes':len(scenes),'outcomes_by_preserved_scene_split':split_counts,
            'train_distributions':distributions,
            'diagnostic_scope':'train scene outcomes only; confirmation contents not used for development'}


def run(configuration_path):
    import fcntl
    import shutil
    cfg=read_json(configuration_path);output=Path(cfg['output']).resolve();output.mkdir(parents=True,exist_ok=True)
    owner=(output/'owner.lock').open('a')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bundle=Path(__file__).resolve().parents[2]
    if (bundle/'manifest.json').exists():
        frozen=read_json(bundle/'manifest.json')
        for name,sha in frozen.get('frozen_sources',{}).items():
            if file_digest(bundle/name)!=sha:raise ValueError('Frozen source checksum changed:'+name)
        for name,sha in frozen.get('external_sources',{}).items():
            if file_digest(REPO/name)!=sha:raise ValueError('External native source changed:'+name)
    verify_gate(cfg['stage0_gate']);verify_gate(cfg['stage1_gate'],'stage1')
    if cfg.get('automatic_training') is not False:raise ValueError('Training must be explicitly disabled')
    if not 1<=cfg['workers']<=max(1,(os.cpu_count() or 1)-4):raise ValueError('worker_reserve_violated')
    identity=digest({k:v for k,v in cfg.items() if k not in ('workers','target_outcomes','report_interval_s')})
    db=database(output/'campaign.sqlite3')
    existing=db.execute("SELECT value FROM metadata WHERE key='configuration_id'").fetchone()
    if existing and existing[0]!=identity:raise ValueError('Resume configuration changed; use a new campaign directory')
    db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',('configuration_id',identity))
    db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',('campaign_started_at',str(time.time())))
    # A SIGKILL can leave running jobs. Immutable publication handles retries.
    db.execute("UPDATE jobs SET state='queued' WHERE state='running'");db.commit()
    write_json(output/'configuration.json',cfg)
    scenes=ready_scenes()
    if cfg.get('scenes'):scenes={s:scenes[s] for s in cfg['scenes']}
    names=sorted(scenes);random.Random(cfg['campaign_seed']).shuffle(names)
    if not names:raise ValueError('No previously certified scene caches')
    stats=statistics(db);campaign_started=float(db.execute("SELECT value FROM metadata WHERE key='campaign_started_at'").fetchone()[0])
    stopping=False;started=time.time();initial=stats['outcomes'];last_report=0.;inflight={};fatal=None
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write_json(output/'owner.json',{'pid':os.getpid(),'started':started,'configuration_id':identity,
        'source_file':str(Path(__file__).resolve()),'no_training':True})
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            now=time.time()
            target=stats['outcomes']>=cfg['target_outcomes']
            if (output/'STOP').exists() or (output/'HOLD.json').exists():stopping=True
            free=shutil.disk_usage(output).free;available=available_memory()
            if free<cfg['disk_reserve_bytes'] or stats['bytes']>=cfg['max_dataset_bytes']:
                stopping=True;fatal={'reason':'storage_limit','free_bytes':free}
            if available<cfg['memory_reserve_bytes']:
                stopping=True;fatal={'reason':'memory_reserve','available_bytes':available}
            if now-last_report>=cfg.get('report_interval_s',20):
                stats=statistics(db)
                report={**stats,'status':'draining' if stopping or target else 'collecting',
                    'workers':cfg['workers'],'inflight_batches':len(inflight),'elapsed_since_resume_s':now-started,
                    'validated_outcomes_per_second_since_resume':(stats['outcomes']-initial)/max(1.,now-started),
                    'campaign_started_at':campaign_started,
                    'validated_outcomes_per_second_lifetime':stats['outcomes']/max(1.,now-campaign_started),
                    'disk_free_bytes':free,'memory_available_bytes':available,'configuration_id':identity,
                    'updated':now,'automatic_training':False}
                write_json(output/'status.json',report);print(json.dumps(report),flush=True);last_report=now
            while not stopping and not target and len(inflight)<cfg['workers']:
                rows=db.execute("SELECT seed,scene,manifest FROM jobs WHERE state='queued' ORDER BY seed LIMIT ?",(cfg['roots_per_batch'],)).fetchall()
                if not rows:
                    previous=db.execute('SELECT MAX(seed) FROM jobs').fetchone()[0]
                    first=cfg['seed_base'] if previous is None else previous+1
                    index=(first-cfg['seed_base'])//cfg['roots_per_batch'];scene=names[index%len(names)]
                    rows=[(first+i,scene,scenes[scene]) for i in range(cfg['roots_per_batch'])]
                    db.executemany('INSERT INTO jobs VALUES(?,?,?,?,NULL)',[(s,c,m,'queued') for s,c,m in rows]);db.commit()
                db.executemany("UPDATE jobs SET state='running' WHERE seed=?",[(r[0],) for r in rows]);db.commit()
                jobs=[dict(seed=r[0],scene=r[1],manifest=r[2]) for r in rows]
                inflight[pool.submit(worker_batch,jobs,cfg)]=jobs
            if not inflight:
                if stopping or target:break
                continue
            done,_=concurrent.futures.wait(inflight,timeout=1.,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                jobs=inflight.pop(future)
                try:results=future.result()
                except Exception as error:
                    results=[{'seed':jobs[0]['seed'],'status':'error','error':repr(error),'traceback':traceback.format_exc()}]
                completed={r['seed'] for r in results}
                for r in results:
                    db.execute('UPDATE jobs SET state=?,result=? WHERE seed=?',(r['status'],json.dumps(r,allow_nan=False),r['seed']))
                    if r['status']=='sealed':
                        stats['outcomes']+=r['outcomes'];stats['bytes']+=r['bytes'];stats['roots']+=1
                    if r['status']=='error':fatal={'reason':'correctness_or_worker_error','result':r};stopping=True
                for job in jobs:
                    if job['seed'] not in completed:db.execute("UPDATE jobs SET state='queued' WHERE seed=?",(job['seed'],))
                db.commit()
            if fatal:write_json(output/'HOLD.json',fatal)
    final={**statistics(db),'status':'held' if fatal or (output/'HOLD.json').exists() else
           'complete' if statistics(db)['outcomes']>=cfg['target_outcomes'] else 'stopped',
           'updated':time.time(),'automatic_training':False,'configuration_id':identity,
           'validated_outcomes_per_second_since_resume':(statistics(db)['outcomes']-initial)/max(1.,time.time()-started),
           'campaign_started_at':campaign_started,
           'validated_outcomes_per_second_lifetime':statistics(db)['outcomes']/max(1.,time.time()-campaign_started),
           'elapsed_since_resume_s':time.time()-started}
    write_json(output/'status.json',final);db.close();return final


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    print(json.dumps(run(parser.parse_args().config),indent=2),flush=True)


if __name__=='__main__':main()
