"""Read-only collector cut, protected exclusions, independent frozen copies."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from .common import BASE,OUT,DOC,OLD,PRIOR,SOURCE,CORE,sha,dump,copy_frozen


def freeze():
    destination=OUT/'snapshot';destination.mkdir(exist_ok=False)
    started=time.perf_counter();source_guard={str(p):sha(p) for p in
        (Path(__file__),Path(__file__).with_name('common.py'),DOC/'PROTOCOL.md',
         OLD/'snapshot/manifest.json',PRIOR/'train.npy',PRIOR/'dev.npy',PRIOR/'motion.json')}
    before=json.loads((SOURCE/'status.json').read_text())
    db=sqlite3.connect('file:'+str(SOURCE/'campaign.sqlite3')+'?mode=ro',uri=True)
    backup=sqlite3.connect(destination/'journal.sqlite3')
    db.backup(backup,pages=256,sleep=.05);db.close()
    if backup.execute('pragma integrity_check').fetchone()[0]!='ok':raise ValueError('invalid journal backup')
    jobs=backup.execute("select seed,scene,json_extract(result,'$.root_id'),json_extract(result,'$.folder'),json_extract(result,'$.outcomes') from jobs where state='sealed' order by seed").fetchall()
    backup.close();os.chmod(destination/'journal.sqlite3',0o444)
    total=sum(r[4] for r in jobs)
    if not 475000<=total<=525000:raise ValueError('outside predeclared approximately500k nominal cut')
    original=json.loads((OLD/'snapshot/manifest.json').read_text())
    old_train=[r for r in original['rows'] if r['split']=='train']
    old_ids={r['root_id'] for r in old_train};old_by_id={r['root_id']:r for r in old_train}
    dev_scenes={r['scene_id'] for r in original['rows'] if r['split']=='dev'}
    opened={r['scene_id'] for r in original['protected'] if r['exclusion']=='previously_opened_fresh_scene'}
    reference=old_train[0]['versions'];eligible={};excluded=[];counts=collections.defaultdict(collections.Counter)
    versions={};all_scenes=set();source_counts=collections.Counter()
    for i,(seed,scene,rid,folder,n) in enumerate(jobs):
        src=Path(folder).resolve()
        if not src.is_relative_to(SOURCE/'dataset/roots'):raise ValueError('noncanonical root path')
        mh=sha(src/'manifest.json');m=json.loads((src/'manifest.json').read_text())
        if not (m['sealed'] is True and m['root_id']==rid and m['future_id']==rid and
                m['scene_id']==scene and m['schema']=='observation_limited_query_conditioned_root_1' and
                m['independent_futures']==1 and m['queries']==n==len(m['query_ids'])):
            raise ValueError('sealed identity/schema incompatibility')
        if any(m['version'][k]!=reference[k] for k in CORE):
            raise ValueError('incompatible frozen graph/sensor/query/executor contract: '+rid)
        if m['collection_config']!=old_train[0]['collection_config']:
            raise ValueError('collection configuration differs: '+rid)
        all_scenes.add(scene);source_counts[m['split']]+=n
        h=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
        reason=('protected_source_'+m['split'] if m['split']!='train' else
                'previously_opened_fresh_scene' if scene in opened else
                'protected_research_holdout' if h>=85 else
                'frozen_research_dev_scene' if h>=70 or scene in dev_scenes else None)
        if reason:
            counts[reason].update(roots=1,outcomes=n)
            excluded.append(dict(root_id=rid,scene_id=scene,source_split=m['split'],queries=n,reason=reason))
        else:
            if rid in eligible:raise ValueError('duplicate physical root')
            vh=hashlib.sha256(json.dumps(m['version'],sort_keys=True).encode()).hexdigest();versions[vh]=m['version']
            eligible[rid]=dict(root_id=rid,scene_id=scene,split='train',source_split='train',
                seed=seed,queries=n,query_ids=m['query_ids'],schema=m['schema'],
                independent_futures=1,version_set=vh,source=str(src),
                manifest_sha256=mh,expected_files={**m['files'],'manifest.json':mh},
                original_cohort=rid in old_ids)
        if (i+1)%10000==0:print(json.dumps(dict(manifests=i+1,eligible=len(eligible),seconds=round(time.perf_counter()-started,1))),flush=True)
    if old_ids-set(eligible):raise ValueError('original training cohort missing/incompatible; do not silently change experiment')
    ordered=[eligible[r['root_id']] for r in old_train]+[v for k,v in eligible.items() if k not in old_ids]
    rows=[]
    for i,row in enumerate(ordered):
        rid=row['root_id'];target=destination/'roots'/rid;target.mkdir(parents=True)
        src=Path(row['source'])
        if row['original_cohort']:
            old=old_by_id[rid]
            if old['files']!=row['expected_files']:raise ValueError('original root bytes differ from frozen baseline')
            src=OLD/'snapshot/roots'/rid
        hashes={}
        for name,value in row['expected_files'].items():
            if Path(name).name!=name:raise ValueError('invalid payload name')
            hashes[name]=copy_frozen(src/name,target/name,value)
        os.chmod(target,0o555)
        rows.append({k:v for k,v in row.items() if k!='expected_files'}|dict(files=hashes))
        if (i+1)%2000==0:print(json.dumps(dict(copied=i+1,seconds=round(time.perf_counter()-started,1))),flush=True)
    assert not ({r['scene_id'] for r in rows}&dev_scenes)
    if any(sha(Path(p))!=v for p,v in source_guard.items()):raise ValueError('freeze inputs changed')
    after=json.loads((SOURCE/'status.json').read_text())
    if any(before[k]!=after[k] for k in ('configuration_identity','collector_restart_count','automatic_training','expected_worker_count')):
        raise ValueError('collector configuration changed during observation')
    manifest=dict(schema='l2_intermittent_scaling_snapshot_1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(SOURCE),nominal_target=500000,nominal_tolerance=[475000,525000],source_outcomes=total,
        source_roots=len(jobs),source_scenes=len(all_scenes),source_split_outcomes=dict(source_counts),
        training_outcomes=sum(r['queries'] for r in rows),training_roots=len(rows),training_scenes=len({r['scene_id'] for r in rows}),
        original_training_roots=len(old_train),added_training_roots=len(rows)-len(old_train),
        baseline_snapshot_sha256=sha(OLD/'snapshot/manifest.json'),journal_sha256=sha(destination/'journal.sqlite3'),
        split_rule=original['split_rule'],exclusion_counts={k:dict(v) for k,v in counts.items()},
        frozen_development_roots=2110,frozen_development_scenes=108,version_sets=versions,
        core_reference={k:reference[k] for k in CORE},rows=rows,excluded_metadata=excluded,
        immutable_payload_policy='Independent copied inodes, checksum checked before/after, files0444 and root directories0555; protected payloads never copied',
        source_guard=source_guard,collector_before=before,collector_after=after,seconds=time.perf_counter()-started)
    dump(destination/'manifest.json',manifest);os.chmod(destination/'manifest.json',0o444)
    os.chmod(destination/'roots',0o555);os.chmod(destination,0o555)
    print(json.dumps({k:manifest[k] for k in ('source_outcomes','source_roots','training_outcomes','training_roots','training_scenes','original_training_roots','added_training_roots','seconds')}),flush=True)


if __name__=='__main__':freeze()
