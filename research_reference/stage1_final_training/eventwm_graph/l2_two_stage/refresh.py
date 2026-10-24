"""Refresh the promising-candidate cutoff by independently copying new sealed roots.

The original snapshot is preserved. Previously copied bytes are referenced by
hash; new production bytes get independent, read-only copies. Protected roots
are metadata only. This module has no collector control operation.
"""
import collections
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from .common import OUT,SOURCE,OLD,CORE,dump,sha


def run(name='latest_snapshot',parent='snapshot'):
    if Path(name).name!=name or Path(parent).name!=parent:raise ValueError('snapshot names must be path components')
    dest=OUT/name;dest.mkdir(exist_ok=False)
    before=json.loads((SOURCE/'status.json').read_text());start=time.perf_counter()
    begun=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source=sqlite3.connect('file:'+str(SOURCE/'campaign.sqlite3')+'?mode=ro',uri=True);source.execute('PRAGMA query_only=ON')
    backup=sqlite3.connect(dest/'journal.sqlite3');source.backup(backup,pages=256,sleep=.02);source.close()
    assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    jobs=backup.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed' ORDER BY seed").fetchall();backup.close()
    cutoff=datetime.datetime.now(datetime.timezone.utc).isoformat();os.chmod(dest/'journal.sqlite3',0o444)
    original=json.loads((OUT/'snapshot/manifest.json').read_text())
    prior=json.loads((OUT/parent/'manifest.json').read_text());old={r['root_id']:r for r in prior['rows']}
    exclusions=collections.Counter();rows=[];source_outcomes=0;physical=set();source_scenes=set();new_roots=0
    original_boundaries=json.loads((OLD/'snapshot/manifest.json').read_text())
    dev={r['scene_id'] for r in original_boundaries['rows'] if r['split']=='dev'}
    opened={r['scene_id'] for r in original_boundaries['protected'] if r['exclusion']=='previously_opened_fresh_scene'}
    for seed,scene,result in jobs:
        result=json.loads(result);folder=Path(result['folder']).resolve();m=json.loads((folder/'manifest.json').read_text());rid=m['root_id']
        assert m['sealed'] and m['schema']=='observation_limited_query_conditioned_root_1' and rid==m['future_id']==result['root_id']
        assert m['scene_id']==scene and m['independent_futures']==1 and m['queries']==result['outcomes']==len(m['query_ids'])
        source_outcomes+=m['queries'];source_scenes.add(scene)
        h=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
        reason=('protected_source_'+m['split'] if m['split']!='train' else 'previously_opened_fresh_scene' if scene in opened else
                'protected_research_holdout' if h>=85 else 'frozen_research_dev_scene' if h>=70 or scene in dev else None)
        if reason:exclusions[reason]+=1;continue
        assert m['collection_config']==original['collection_config'] and all(m['version'][k]==original['core_reference'][k] for k in CORE)
        files={**m['files'],'manifest.json':sha(folder/'manifest.json')}
        if rid in old:
            r=old[rid];assert r['files']==files
            target=Path(r['folder']) if 'folder' in r else OUT/parent/'roots'/rid
        else:
            target=dest/'roots'/rid;target.mkdir(parents=True)
            for name,expected in files.items():
                assert Path(name).name==name and not (folder/name).is_symlink() and sha(folder/name)==expected
                shutil.copyfile(folder/name,target/name)
                assert sha(target/name)==expected and sha(folder/name)==expected and (target/name).stat().st_ino!=(folder/name).stat().st_ino
                os.chmod(target/name,0o444)
            os.chmod(target,0o555);new_roots+=1
            r=dict(root_id=rid,scene_id=scene,seed=seed,queries=m['queries'],split='train',source_split='train',independent_futures=1,files=files)
        assert (scene,seed) not in physical;physical.add((scene,seed));rows.append(dict(r,folder=str(target)))
    assert {rid for rid,r in old.items() if 'campaign' not in r}.issubset({r['root_id'] for r in rows})
    supplement=json.loads((OUT/'supplement/manifest.json').read_text());supplement_rows=0
    for r in supplement['rows']:
        unit=(r['scene_id'],r['seed'])
        if unit in physical:exclusions['supplement_duplicate_at_new_cutoff']+=1;continue
        physical.add(unit);rows.append(dict(r,folder=str(OUT/'supplement/roots'/r['root_id'])));supplement_rows+=1
    after=json.loads((SOURCE/'status.json').read_text())
    for k in ('configuration_identity','collector_restart_count','automatic_training','expected_worker_count'):assert before[k]==after[k]
    assert not after['automatic_training'] and after['collection_status'] in ('RUNNING','DEGRADED')
    manifest=dict(schema='two_stage_latest_immutable_training_union_1',cutoff_started_utc=begun,journal_cut_completed_utc=cutoff,
        source_roots=len(jobs),source_outcomes=source_outcomes,source_scenes=len(source_scenes),new_roots_copied=new_roots,
        training_roots=len(rows),training_outcomes=sum(r['queries'] for r in rows),training_scenes=len({r['scene_id'] for r in rows}),
        supplement_roots=supplement_rows,rows=rows,exclusions=dict(exclusions),collector_before=before,collector_after=after,
        original_snapshot_sha256=sha(OUT/'snapshot/manifest.json'),supplement_sha256=sha(OUT/'supplement/manifest.json'),
        parent_snapshot=str(OUT/parent/'manifest.json'),parent_snapshot_sha256=sha(OUT/parent/'manifest.json'),
        journal_sha256=sha(dest/'journal.sqlite3'),source_sha256=sha(__file__),seconds=time.perf_counter()-start)
    dump(dest/'manifest.json',manifest);os.chmod(dest/'manifest.json',0o444);os.chmod(dest,0o555)
    print(json.dumps({k:manifest[k] for k in ('source_outcomes','training_roots','training_outcomes','new_roots_copied','seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='latest_snapshot');parser.add_argument('--parent',default='snapshot');args=parser.parse_args();run(args.name,args.parent)
