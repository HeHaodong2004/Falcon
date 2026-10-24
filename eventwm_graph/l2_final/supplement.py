"""Inventory all older campaign journals; freeze additional compatible futures.

Repeated benchmark/recovery executions sharing scene+seed are conservatively
one physical-future unit even when code-addressed root IDs differ. These frozen
older campaigns cannot enlarge the already fixed live-collector cutoff.
"""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from .common import OUT,OLD,SOURCE,CORE,sha,dump


def run():
    dest=OUT/'supplement';dest.mkdir(exist_ok=False)
    main=json.loads((OUT/'snapshot/manifest.json').read_text());old=json.loads((OLD/'snapshot/manifest.json').read_text())
    db=sqlite3.connect('file:'+str(OUT/'snapshot/journal.sqlite3')+'?mode=ro',uri=True)
    physical={(scene,seed) for seed,scene in db.execute("SELECT seed,scene FROM jobs WHERE state='sealed'")};db.close()
    dev={r['scene_id'] for r in old['rows'] if r['split']=='dev'}
    opened={r['scene_id'] for r in old['protected'] if r['exclusion']=='previously_opened_fresh_scene'}
    inventory=json.loads((OUT/'campaign_inventory_initial.json').read_text());rows=[];excluded=[];counts=collections.Counter();campaigns=[]
    # Prefer production over its repeated performance/smoke executions.
    inventory.sort(key=lambda v:('production' not in v['path'],v['path']))
    for item in inventory:
        path=Path(item['path'])
        if path==SOURCE/'campaign.sqlite3':continue
        c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.execute('pragma query_only=ON')
        jobs=c.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed' ORDER BY seed").fetchall();c.close()
        info=dict(journal=str(path),sealed_roots=len(jobs),eligible_added=0,disposition=collections.Counter())
        for seed,scene,result in jobs:
            result=json.loads(result);folder=Path(result['folder']).resolve();m=json.loads((folder/'manifest.json').read_text())
            assert m['sealed'] is True and m['root_id']==m['future_id']==result['root_id'] and m['scene_id']==scene and m['independent_futures']==1
            assert m['queries']==result['outcomes']==len(m['query_ids'])
            h=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
            reason=('protected_source_'+m['split'] if m['split']!='train' else 'previously_opened_fresh_scene' if scene in opened else
                    'protected_research_holdout' if h>=85 else 'frozen_research_dev_scene' if h>=70 or scene in dev else None)
            if reason is None and (scene,seed) in physical:reason='duplicate_scene_seed_physical_future'
            if reason is None and (m['schema']!='observation_limited_query_conditioned_root_1' or any(m['version'].get(k)!=main['core_reference'][k] for k in CORE)):
                reason='incompatible_core_contract'
            configuration=dict(m['collection_config']);configuration.pop('private_graph_curriculum',None)
            if reason is None and configuration!=main['collection_config']:reason='incompatible_collection_semantics'
            if reason is not None:
                counts[reason]+=1;info['disposition'][reason]+=1
                excluded.append(dict(root_id=m['root_id'],scene_id=scene,seed=seed,queries=m['queries'],source_split=m['split'],reason=reason,campaign=str(path)))
                continue
            # Different private root-sampling priors are compatible conditional
            # observations/dynamics, and are explicitly retained as provenance.
            physical.add((scene,seed));target=dest/'roots'/m['root_id'];target.mkdir(parents=True)
            mh=sha(folder/'manifest.json');files={**m['files'],'manifest.json':mh}
            if result.get('root_manifest_sha256'):assert result['root_manifest_sha256']==mh
            for name,expected in files.items():
                assert Path(name).name==name and not (folder/name).is_symlink() and sha(folder/name)==expected
                shutil.copyfile(folder/name,target/name)
                assert sha(target/name)==expected and sha(folder/name)==expected and (folder/name).stat().st_ino!=(target/name).stat().st_ino
                os.chmod(target/name,0o444)
            os.chmod(target,0o555)
            rows.append(dict(root_id=m['root_id'],scene_id=scene,seed=seed,split='train',source_split='train',independent_futures=1,
                             queries=m['queries'],source=str(folder),campaign=str(path),version=m['version'],collection_config=m['collection_config'],files=files))
            info['eligible_added']+=1
        info['disposition']=dict(info['disposition']);campaigns.append(info)
    manifest=dict(schema='l2_final_compatible_older_campaign_supplement_1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  rows=rows,excluded_metadata=excluded,campaigns=campaigns,exclusion_roots=dict(counts),
                  training_roots=len(rows),training_outcomes=sum(r['queries'] for r in rows),training_scenes=len({r['scene_id'] for r in rows}),
                  main_snapshot_sha256=sha(OUT/'snapshot/manifest.json'),deduplication='conservative scene+seed across all sealed campaigns; main cut wins',
                  compatibility='all core observation/graph/executor/native label hashes identical; base collection config identical; optional private initialization sampling prior retained and disclosed')
    dump(dest/'manifest.json',manifest);os.chmod(dest/'manifest.json',0o444);os.chmod(dest/'roots',0o555);os.chmod(dest,0o555)
    dump(OUT/'final_snapshot_manifest.json',dict(main_snapshot_sha256=sha(OUT/'snapshot/manifest.json'),supplement_sha256=sha(dest/'manifest.json'),
        live_cutoff=main['journal_cut_completed_utc'],training_roots=main['training_roots']+len(rows),
        training_outcomes=main['training_outcomes']+manifest['training_outcomes'],
        training_scenes=len({r['scene_id'] for r in main['rows']+rows}),source_live_outcomes=main['source_outcomes'],
        additional_older_compatible_roots=len(rows),additional_older_compatible_outcomes=manifest['training_outcomes']))
    print(json.dumps({k:manifest[k] for k in ('training_roots','training_outcomes','training_scenes')}),flush=True)


if __name__=='__main__':run()
