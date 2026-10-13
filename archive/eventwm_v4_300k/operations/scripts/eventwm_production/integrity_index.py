from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Continuously verify new seals and expose a role-preserving combined index."""
import collections,hashlib,json,pathlib,sqlite3,subprocess,sys,time
PROJECT=PROJECT_PATH
REPO=SCIENCE_PATH;sys.path.insert(0,str(REPO/'scripts'))
from collector.core import read
from active_release import drain
from eventwm_scale import verify_root
from eventwm_validate_staged import durable_json
NEW=DATA_PATH;DEST=PROJECT/'output/eventwm_300k_combined_index'
DEST.mkdir(exist_ok=True);DB=DEST/'verified.sqlite3';REPORT=DEST/'integrity.json'

def add(db,base,r,deep):
    rid=r['root_id'];seal=read(base/r['seal_file'])
    verify_root(base,dict(r,seals=seal),deep=deep)
    idx=read(base/'primary_indices'/f'{rid}.json')
    assert idx['role']==r['role'] and idx['physical_scene_id']==r['physical_scene_id']
    assert idx['split_group']==rid and len(idx['samples'])==r['stats']['primary_queries']
    assert r['stats']['spatial_paths']==8 and r['stats']['variants']=={'immediate_normal':8,'wait_0.5':8,'wait_1.0':8,'half_speed':8,'WAIT':1}
    assert r['role'] in ('train','natural_reference','unseen_evaluation')
    assert (r['role']=='unseen_evaluation')==(r['scene_role']=='unseen_evaluation')
    previous=db.execute('SELECT scene_role FROM scenes WHERE physical_scene=?',(r['physical_scene_id'],)).fetchone()
    assert not previous or previous[0]==r['scene_role']
    stats=collections.Counter()
    for s in idx['samples']:
        assert s['sr_valid'] and s['sr_label'] in (0,1) and s['social_outcome_valid']
        assert not s['static_feasibility_violation']
        assert s['sr_label']==int(not(s['native_human_collision'] or s['static_collision']))
        if s['sr_label']==1:assert s['query_completed']
        stats['sr_'+str(s['sr_label'])]+=1
        stats['early_collision_retained']+=int(s['sr_label']==0 and not s['query_completed'])
        stats['social_'+str(s['social_outcome_class'])]+=1
    with db:
        db.execute('INSERT OR IGNORE INTO scenes VALUES (?,?)',(r['physical_scene_id'],r['scene_role']))
        db.execute('INSERT INTO roots VALUES (?,?,?,?,?,?,?,?)',(rid,str(base),r['path'],r['role'],r['physical_scene_id'],r['seed'],len(idx['samples']),json.dumps(stats)))
        db.executemany('INSERT INTO examples VALUES (?,?,?,?,?)',[(s['query_id'],rid,s['branch_id'],s['sr_label'],s['social_outcome_class']) for s in idx['samples']])
    return stats

def main():
    db=sqlite3.connect(DB);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('CREATE TABLE IF NOT EXISTS scenes(physical_scene TEXT PRIMARY KEY,scene_role TEXT NOT NULL); CREATE TABLE IF NOT EXISTS roots(root_id TEXT PRIMARY KEY,dataset TEXT NOT NULL,path TEXT NOT NULL,role TEXT NOT NULL,physical_scene TEXT NOT NULL,seed INTEGER UNIQUE NOT NULL,queries INTEGER NOT NULL,stats TEXT NOT NULL); CREATE TABLE IF NOT EXISTS examples(query_id TEXT PRIMARY KEY,root_id TEXT NOT NULL,branch_id TEXT NOT NULL,sr INTEGER NOT NULL,social INTEGER NOT NULL); CREATE INDEX IF NOT EXISTS examples_root ON examples(root_id); CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);')
    proof=read(PROJECT/'output/eventwm_72h_legacy_verified.json');assert proof['all_seals_verified'] and not proof['errors']
    old=pathlib.Path(proof['source']);assert hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest()==proof['manifest_sha256']
    oldmanifest=read(old/'manifest.json')
    for r in oldmanifest['roots']:
        if r['state']=='completed' and not db.execute('SELECT 1 FROM roots WHERE root_id=?',(r['root_id'],)).fetchone():add(db,old,r,False)
    row=db.execute("SELECT value FROM meta WHERE key='new_cursor'").fetchone();cursor=int(row[0]) if row else -1
    while True:
        with sqlite3.connect((DATA_PATH/'pipeline.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as source:
            rows=[(n,json.loads(v)) for n,v in source.execute('SELECT ordinal,value FROM roots WHERE ordinal>? ORDER BY ordinal LIMIT 256',(cursor,))]
        for n,r in rows:
            if r['state'] not in ('completed','failed'):break
            if r['state']=='failed' and len(r['attempts'])<2 and r['attempts'][-1].get('reason') in ('worker_timeout','root_timeout','interrupted_before_commit'):break
            if r['state']=='completed' and not db.execute('SELECT 1 FROM roots WHERE root_id=?',(r['root_id'],)).fetchone():
                stats=add(db,NEW,r,True);print('VERIFIED',r['root_id'],dict(stats),flush=True)
            cursor=n
            with db:db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('new_cursor',str(cursor)))
        totals=collections.Counter()
        for value, in db.execute('SELECT stats FROM roots'):totals.update(json.loads(value))
        counts={role:q for role,q in db.execute('SELECT role,SUM(queries) FROM roots GROUP BY role')}
        datasets={dataset:q for dataset,q in db.execute('SELECT dataset,SUM(queries) FROM roots GROUP BY dataset')}
        report=dict(timestamp=time.time(),passed=True,total_verified=sum(counts.values()),roles=counts,datasets=datasets,stats=dict(totals),
                    duplicate_query_ids=0,duplicate_root_ids=0,duplicate_accepted_root_seeds=0,physical_scene_role_conflicts=0,
                    root_count=db.execute('SELECT COUNT(*) FROM roots').fetchone()[0],scene_count=db.execute('SELECT COUNT(*) FROM scenes').fetchone()[0],
                    legacy_deep_seal_proof=str(PROJECT/'output/eventwm_72h_legacy_verified.json'),new_cursor=cursor,
                    combined_index=str(DB),raw_data_copied=False,raw_mutated=False,note='Index includes verified sealed roots only; preserves natural/unseen roles; counters lag live collection during verification.')
        durable_json(REPORT,report);print('INTEGRITY',json.dumps(report),flush=True)
        time.sleep(1 if len(rows)==256 and cursor==rows[-1][0] else 90)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        durable_json(DEST/'FAILURE.json',dict(timestamp=time.time(),error=repr(e),action='Graceful production drain requested; inspect integrity failure'))
        drain()
        raise
