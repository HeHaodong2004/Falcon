#!/usr/bin/env python3
"""Root/trajectory-disjoint large-data study, following the revised user scope."""
import argparse
from collections import Counter, defaultdict
from copy import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
import eventwm_v4_research as r
import eventwm_v4_evaluate as e
from eventquerywm.v2.labels import inspect_social_root

DEFAULT=r.PROJECT/'output/eventwm_v4_full_scaling_20261012'
OLD=r.DEFAULT_OUT


def fixture(dataset,root):
    catalogue=dataset/'catalogue/episodes'/f"{root['root_id']}.json"
    if catalogue.exists():episode=r.read(catalogue)
    else:
        path=Path(root['path'])
        if not path.is_absolute():path=dataset/path
        name='audits/workers/b000/episode.json.gz'
        seals=r.read(dataset/root['seal_file'])
        assert r.sha(path/name)==seals[name]
        with gzip.open(path/name,'rt') as f:episode=json.load(f)['episodes'][0]
    count=root['human_count']
    waypoints={k:v for k,v in episode['info'].items() if any(k.startswith(f'human_{i}_waypoint_') for i in range(count))}
    if count and not waypoints:raise ValueError('No physical trajectory fixture metadata')
    body=dict(scene=root['physical_scene_id'],human_count=count,waypoints=waypoints)
    h=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    return dict(episode_id=root['episode_sha256'],fixture_id=h(dict(body,policy_seed=episode['info'].get('policy_seed'))),
                trajectory_id=h(body) if count else None)


def block(dataset,root,commit):
    rows=inspect_social_root(dataset,root)
    assert len(rows)==33
    path=(dataset/root['path']).resolve()
    root['source_commit']=root.get('source_commit',commit)
    seals=r.read(dataset/root['seal_file'])
    for row in rows:row.update(source_commit=root['source_commit'],root_path=str(path))
    return dict(root=root,rows=rows,evidence={n:seals[n] for n in ('VALIDATED.json','legal_observations/history.json','queries/queries.json','labels/labels.json','branch_truth/branches.jsonl.gz') if n in seals},
                archive_path=str(path),complete_sha256=r.sha(path/'COMPLETE.json'),seal_sha256=r.sha(dataset/root['seal_file']))


def snapshot(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'snapshot.json').exists()
    before=r.production_sample();r.save(out/'production_before.json',before)
    old=r.read(OLD/'snapshot.json')
    assert r.sha(OLD/'snapshot.json')==r.read(OLD/'snapshot_identity.json')['sha256']
    old_blocks={b['root']['root_id']:b for b in old['blocks']}
    fixed_val={b['root']['root_id'] for b in old['blocks'] if b['split']=='validation'}
    with r.ro(r.VERIFIED) as db:
        db.execute('BEGIN')
        verified=db.execute('SELECT root_id,path,role,physical_scene,queries FROM roots WHERE dataset=? ORDER BY root_id',(str(r.DATA),)).fetchall()
    with r.ro(r.DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        meta=json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        progress=json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
        fresh={v[0]:json.loads(db.execute('SELECT value FROM roots WHERE id=?',(v[0],)).fetchone()[0]) for v in verified if v[0] not in old_blocks}
    blocks=[];excluded=[];started=time.time()
    for v in verified:
        try:
            b=old_blocks.get(v[0])
            if b is None:b=block(r.DATA,fresh[v[0]],meta['frozen_commit'])
            root=b['root']
            assert root['state']=='completed' and root['path']==v[1] and root['role']==v[2] and root['physical_scene_id']==v[3] and len(b['rows'])==v[4]
            b['cohort']='V4'
            b['split']='validation' if root['root_id'] in fixed_val else root['role']
            for row in b['rows']:row['split']=b['split'];row['cohort']='V4'
            blocks.append(b)
        except (AssertionError,ValueError,OSError,KeyError) as exc:
            excluded.append(dict(root_id=v[0],queries=v[4],reason=str(exc)))
    proof=r.read(r.PROJECT/'output/eventwm_72h_legacy_verified.json')
    legacy=Path(proof['source']);assert r.sha(legacy/'manifest.json')==proof['manifest_sha256']
    lm=r.read(legacy/'manifest.json')
    assert lm['state']=='target_reached' and lm['schema_version']==meta['schema_version']
    assert lm['protocol_version']==meta['protocol_version'] and lm['sr_contract_version']==meta['sr_contract_version']
    bank_index=r.read(r.BANK_INDEX)
    bank_roots={row['root_id'] for row in bank_index['rows'] if row['split']=='validation'}
    protected=[];legacy_added=0
    for root in lm['roots']:
        if root['state']!='completed':continue
        f=fixture(legacy,root)
        if root['role']!='train' or root['root_id'] in bank_roots:
            protected.append(dict(root_id=root['root_id'],**f));continue
        try:
            b=block(legacy,root,lm['frozen_commit']);b['cohort']='V3';b['split']='train'
            for row in b['rows']:row.update(**f,split='train',cohort='V3')
            blocks.append(b);legacy_added+=len(b['rows'])
        except (AssertionError,ValueError,OSError,KeyError) as exc:
            excluded.append(dict(root_id=root['root_id'],queries=33,reason='legacy '+str(exc)))
    # Physical-scene normalized waypoint signatures catch scene-path aliases.
    for i,b in enumerate(blocks):
        if b['cohort']=='V4':
            f=fixture(r.DATA,b['root'])
            for row in b['rows']:row.update(**f)
        if (i+1)%1000==0:print(json.dumps(dict(stage='root_fixture_audit',roots=i+1,total=len(blocks),seconds=time.time()-started)),flush=True)
    groups=defaultdict(set)
    for b in blocks:
        for field in ('episode_id','fixture_id','trajectory_id'):
            value=b['rows'][0].get(field)
            if value:groups[field,value].add(b['root']['root_id'])
    protected_keys={(field,p[field]) for p in protected for field in ('episode_id','fixture_id','trajectory_id') if p.get(field)}
    held={b['root']['root_id'] for b in blocks if b['split']!='train'}|bank_roots
    for key,ids in groups.items():
        if key in protected_keys:held.update(ids)
    # Propagate connected fixture/episode groups so transitive sharing is held out.
    while True:
        previous=len(held)
        for ids in groups.values():
            if ids&held:held.update(ids)
        if len(held)==previous:break
    for b in blocks:
        if b['split']=='train' and b['root']['root_id'] in held:
            b['split']='quarantine_trajectory_overlap'
        for row in b['rows']:row['split']=b['split']
    active=('train','validation','natural_reference','unseen_evaluation')
    leakage={}
    for field in ('root_id','query_id','episode_id','fixture_id','trajectory_id'):
        values={s:{row[field] for b in blocks for row in b['rows'] if b['split']==s and row.get(field)} for s in active}
        intersections={a+'/'+c:len(values[a]&values[c]) for ai,a in enumerate(active) for c in active[ai+1:]}
        assert not any(intersections.values()),(field,intersections)
        leakage[field]=intersections
    train=[b for b in blocks if b['split']=='train']
    assert not bank_roots&{b['root']['root_id'] for b in train}
    unseen={b['root']['physical_scene_id'] for b in blocks if b['split']=='unseen_evaluation'}
    assert not unseen&{b['root']['physical_scene_id'] for b in train}
    ids=[row['query_id'] for b in blocks for row in b['rows']]
    assert len(ids)==len(set(ids))
    counts={s:dict(roots=sum(b['split']==s for b in blocks),queries=sum(len(b['rows']) for b in blocks if b['split']==s),scenes=len({b['root']['physical_scene_id'] for b in blocks if b['split']==s})) for s in sorted({b['split'] for b in blocks})}
    ordered=sorted([b['root']['root_id'] for b in train],key=r.key)
    subsets={name:ordered[:min(len(ordered),round(n/33))] for name,n in [('30k',30000),('100k',100000),('150k',150000),('all',33*len(ordered))]}
    audit=dict(revised_scope='Scene overlap allowed; strict root/query/episode/physical-waypoint fixture separation, frozen benchmark protected, unseen scenes independent',
               cutoff_unix=before['timestamp'],validated_v4_queries=sum(v[4] for v in verified),validated_v4_roots=len(verified),
               eligible_v3_queries=legacy_added,legacy_scientific_compatibility=dict(passed=True,manifest_sha256=proof['manifest_sha256'],
                  evidence='Matching schema4/social-v3/SR contract, immutable deep-seal proof plus individual legal/query/label digests, 33 query content digests and exact authoritative labels; sealed episode metadata supplies waypoint grouping',
                  heldout_legacy_roots=len(protected),source_commit=lm['frozen_commit']),
               splits=counts,leakage=leakage,invalid_excluded_queries=sum(v['queries'] for v in excluded),exclusions=excluded,
               reserved_selector_bank=dict(index=str(r.BANK_INDEX),sha256=r.sha(r.BANK_INDEX),roots=len(bank_roots),scenes_excluded=False,episodes_protected=True),
               parent_v4_snapshot_sha256=r.sha(OLD/'snapshot.json'),production_progress_at_cutoff=progress,
               scene_overlap_train_validation=len({b['root']['physical_scene_id'] for b in train}&{b['root']['physical_scene_id'] for b in blocks if b['split']=='validation'}),
               scene_overlap_train_natural=len({b['root']['physical_scene_id'] for b in train}&{b['root']['physical_scene_id'] for b in blocks if b['split']=='natural_reference'}),
               unseen_scene_overlap=0,raw_copied=False)
    r.save(out/'snapshot.json',dict(version='eventwm-v4-research-snapshot-v1',audit=audit,subsets=subsets,blocks=blocks))
    r.save(out/'audit.json',audit);r.save(out/'snapshot_identity.json',dict(sha256=r.sha(out/'snapshot.json'),cutoff=before['timestamp']))
    print(json.dumps(dict(stage='large_snapshot_complete',splits=counts,legacy_queries=legacy_added,excluded=excluded)),flush=True)


def prepare(args):
    args.reuse_cache=OLD
    r.prepare(args)


def suite(args):
    planned=[(size,seed) for size in ('30k','100k','150k','all') for seed in (20261012,20261013)]
    r.save(args.output/'suite_configuration.json',dict(runs=planned,fixed_steps=1024,matched_exposure_epochs=2,
            maximum_steps='max(1024,2*ceil(unique_train_queries/64))',early_stopping=False,selection='Best full validation BCE also reported',
            resources='Sequential GPU1, CPU1, loader workers0, nice15/ionice3, no concurrent CPU-heavy research'))
    for size,seed in planned:
        name=f'{size}_full_seed{seed}'
        if (args.output/'runs'/name/'results.json').exists():continue
        a=copy(args);a.subset=size;a.seed=seed;a.mode='full';a.exposure_epochs=2
        r.train(a)


def evaluate(args):
    # Existing CV/selector equations; every trained model and checkpoint is scored.
    if not (args.output/'cv_fast/results.json').exists():e.cv_fast(args)
    cache=r.Cache(args.output)
    for path in sorted((args.output/'runs').glob('*/results.json')):
        name=path.parent.name
        for stage in ('fixed','exposure','best'):
            tag=f'planning/{name}_{stage}'
            if (args.output/tag/'results.json').exists():continue
            a=copy(args);a.cache=cache;a.checkpoint=path.parent/(stage+'.pt');a.planning_tag=tag
            e.planning(a)
            print(json.dumps(dict(stage='planning_complete',model=name,checkpoint=stage)),flush=True)


def report(args):
    from eventwm_v4_scaling_report import main as export_report
    export_report(['--output',str(args.output)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('snapshot','prepare','suite','evaluate','report'))
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();a.output=a.output.resolve();a.fixed_steps=1024;a.steps=2048;a.eval_every=256;a.patience=4
    torch.set_num_threads(1)
    if '/production/' in str(a.output) or a.output==r.DATA or r.DATA in a.output.parents:raise ValueError('Protected production output')
    globals()[a.command](a)


if __name__=='__main__':main()
