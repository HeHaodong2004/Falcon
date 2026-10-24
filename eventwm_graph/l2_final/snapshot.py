"""Read-only WAL snapshot; independent immutable copies of sealed TRAIN data."""
import collections
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import time
from .common import OUT,OLD,SOURCE,CORE,sha,dump


def run():
    dest=OUT/'snapshot';dest.mkdir(exist_ok=False)
    before=json.loads((SOURCE/'status.json').read_text());start=time.perf_counter()
    cutoff=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source=sqlite3.connect('file:'+str(SOURCE/'campaign.sqlite3')+'?mode=ro',uri=True)
    source.execute('PRAGMA query_only=ON');backup=sqlite3.connect(dest/'journal.sqlite3')
    source.backup(backup,pages=256,sleep=.02);source.close()
    assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    jobs=backup.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed' ORDER BY seed").fetchall()
    backup.close();os.chmod(dest/'journal.sqlite3',0o444)
    journal_cut=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=json.loads((OLD/'snapshot/manifest.json').read_text())
    train=[r for r in old['rows'] if r['split']=='train'];reference=train[0]
    dev_scenes={r['scene_id'] for r in old['rows'] if r['split']=='dev'}
    opened={r['scene_id'] for r in old['protected'] if r['exclusion']=='previously_opened_fresh_scene'}
    rows=[];excluded=[];counts=collections.defaultdict(collections.Counter);versions={};scenes=set();splits=collections.Counter();total=0;seen=set()
    for index,(seed,scene,result) in enumerate(jobs):
        result=json.loads(result);rid=result['root_id'];folder=__import__('pathlib').Path(result['folder']).resolve()
        assert folder.is_relative_to(SOURCE/'dataset/roots') and rid not in seen
        seen.add(rid);m=json.loads((folder/'manifest.json').read_text());mh=sha(folder/'manifest.json')
        assert m['sealed'] is True and m['schema']=='observation_limited_query_conditioned_root_1'
        assert m['root_id']==m['future_id']==rid and m['scene_id']==scene and m['independent_futures']==1
        assert m['queries']==result['outcomes']==len(m['query_ids']) and len(set(m['query_ids']))==m['queries']
        assert m['collection_config']==reference['collection_config'],rid
        assert all(m['version'][key]==reference['versions'][key] for key in CORE),rid
        scenes.add(scene);total+=m['queries'];splits[m['split']]+=m['queries']
        h=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
        reason=('protected_source_'+m['split'] if m['split']!='train' else 'previously_opened_fresh_scene' if scene in opened else
                'protected_research_holdout' if h>=85 else 'frozen_research_dev_scene' if h>=70 or scene in dev_scenes else None)
        if reason:
            counts[reason].update(roots=1,outcomes=m['queries']);excluded.append(dict(root_id=rid,scene_id=scene,queries=m['queries'],source_split=m['split'],reason=reason));continue
        vh=hashlib.sha256(json.dumps(m['version'],sort_keys=True).encode()).hexdigest();versions[vh]=m['version']
        target=dest/'roots'/rid;target.mkdir(parents=True)
        files={**m['files'],'manifest.json':mh}
        for name,expected in files.items():
            assert __import__('pathlib').Path(name).name==name
            source_file=folder/name;target_file=target/name
            assert not source_file.is_symlink() and sha(source_file)==expected
            shutil.copyfile(source_file,target_file)
            assert sha(target_file)==expected and sha(source_file)==expected and source_file.stat().st_ino!=target_file.stat().st_ino
            os.chmod(target_file,0o444)
        os.chmod(target,0o555)
        rows.append(dict(root_id=rid,scene_id=scene,split='train',source_split='train',seed=seed,queries=m['queries'],
                         independent_futures=1,source=str(folder),version_set=vh,files=files))
        if len(rows)%5000==0:print(json.dumps(dict(scanned=index+1,copied=len(rows),seconds=time.perf_counter()-start)),flush=True)
    assert {r['root_id'] for r in train}.issubset({r['root_id'] for r in rows})
    after=json.loads((SOURCE/'status.json').read_text())
    for key in ('configuration_identity','collector_restart_count','automatic_training','expected_worker_count'):
        assert before[key]==after[key],key
    assert after['collection_status']=='RUNNING' and not after['automatic_training']
    manifest=dict(schema='l2_final_maximum_sealed_snapshot_1',cutoff_started_utc=cutoff,journal_cut_completed_utc=journal_cut,
                  source=str(SOURCE),source_outcomes=total,source_roots=len(jobs),source_scenes=len(scenes),source_split_outcomes=dict(splits),
                  training_outcomes=sum(r['queries'] for r in rows),training_roots=len(rows),training_scenes=len({r['scene_id'] for r in rows}),
                  rows=rows,excluded_metadata=excluded,exclusion_counts={k:dict(v) for k,v in counts.items()},version_sets=versions,
                  core_reference={k:reference['versions'][k] for k in CORE},collection_config=reference['collection_config'],
                  original_snapshot_sha256=sha(OLD/'snapshot/manifest.json'),journal_sha256=sha(dest/'journal.sqlite3'),
                  split_rule=old['split_rule'],collector_before=before,collector_after=after,seconds=time.perf_counter()-start,
                  integrity='all manifest and payload bytes verified before/after independent copy; no hard links; protected metadata only')
    dump(dest/'manifest.json',manifest);os.chmod(dest/'manifest.json',0o444);os.chmod(dest/'roots',0o555);os.chmod(dest,0o555)
    print(json.dumps({k:manifest[k] for k in ('source_outcomes','source_roots','training_outcomes','training_roots','training_scenes','seconds')}),flush=True)


if __name__=='__main__':run()
