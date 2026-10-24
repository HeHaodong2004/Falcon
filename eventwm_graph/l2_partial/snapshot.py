"""Read-only SQLite backup + checksum-verified copies of atomically sealed roots."""
import argparse, collections, datetime, hashlib, json, os, pathlib, shutil, sqlite3

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def snapshot(source, destination):
    source=pathlib.Path(source).resolve(); destination=pathlib.Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect('file:'+str(source/'campaign.sqlite3')+'?mode=ro',uri=True)
    backup=sqlite3.connect(destination/'journal.sqlite3')
    db.backup(backup,pages=256,sleep=.05); db.close()
    assert backup.execute('pragma integrity_check').fetchone()[0]=='ok'
    jobs=backup.execute("select scene,result from jobs where state='sealed' order by seed").fetchall();backup.close()
    rows=[]; excluded=collections.Counter(); stats=collections.defaultdict(collections.Counter)
    for scene,result in jobs:
        record=json.loads(result); src=pathlib.Path(record['folder'])
        m=json.loads((src/'manifest.json').read_text())
        if m['split']!='train':
            excluded[m['split']+'_roots']+=1;excluded[m['split']+'_outcomes']+=m['queries'];continue
        assert m['sealed'] and m['root_id']==record['root_id'] and m['scene_id']==scene
        assert m['schema']=='observation_limited_query_conditioned_root_1'
        split_hash=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
        split='train' if split_hash<70 else ('dev' if split_hash<85 else 'holdout')
        dest=destination/'roots'/m['root_id'];dest.mkdir(parents=True)
        hashes={}
        for name,want in dict(m['files'],**{'manifest.json':sha(src/'manifest.json')}).items():
            shutil.copyfile(src/name,dest/name)
            got=sha(dest/name)
            if got!=want:raise ValueError('checksum mismatch '+str(src/name))
            hashes[name]=got;os.chmod(dest/name,0o444)
        receipt=source/'dataset'/'sampling'/(m['root_id']+'.json')
        if receipt.exists():
            shutil.copyfile(receipt,dest/'sampling.json');hashes['sampling.json']=sha(dest/'sampling.json');os.chmod(dest/'sampling.json',0o444)
        row={'root_id':m['root_id'],'scene_id':scene,'split':split,'queries':m['queries'],
             'query_ids':m['query_ids'],'source':str(src),'files':hashes,'schema':m['schema'],
             'versions':m['version'],'collection_config':m['collection_config']}
        rows.append(row);stats[split].update({'roots':1,'queries':m['queries'],**{k:v for k,v in m['statistics'].items() if k in ('human_collisions','psc_violations','static_obstacle_collisions','query_timeouts')}})
        if len(rows)%500==0:print('copied',len(rows),flush=True)
    for split in stats:stats[split]['scenes']=len({r['scene_id'] for r in rows if r['split']==split})
    manifest={'snapshot_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_revision':'93f41d5dde3b94d4399ffbaa16f16d5f8eea8e08','source':str(source),
              'journal_sha256':sha(destination/'journal.sqlite3'),'source_roots':len(jobs),
              'source_outcomes':sum(json.loads(v)['outcomes'] for _,v in jobs),
              'source_scenes':len({s for s,_ in jobs}),'excluded':dict(excluded),'splits':dict(stats),
              'split_rule':'sha256(l2-explore-20261021:scene) first8 modulo100: train<70 dev<85 holdout otherwise; source train scenes only',
              'rows':rows}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2));os.chmod(destination/'manifest.json',0o444);os.chmod(destination/'journal.sqlite3',0o444)
    print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();snapshot(a.source,a.destination)
