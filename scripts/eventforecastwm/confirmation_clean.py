"""Exclude every historical full-V4 evaluation root, not only the WM pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.confirmation import PROJECT,OLD,select,build,recovery
from eventforecastwm.data import SNAPSHOT_HASH,SOURCE_HASH
from eventforecastwm.guard import Guard

OLD_INDEX=PROJECT/'output/eventwm_full_v4_wm_20261013/verified.sqlite3'
OLD_INDEX_SHA='348b609887f9288403f48d1aadba790ff52920745952469770ec587ce33cce81'


def prepare(snapshot,output,count=128):
    out=Path(output)
    if out.exists():raise FileExistsError('Use a new explicitly versioned confirmation membership')
    if sha256(snapshot)!=SNAPSHOT_HASH or sha256(OLD_INDEX)!=OLD_INDEX_SHA:raise ValueError('Frozen source identity changed')
    con=sqlite3.connect('file:'+str(OLD_INDEX)+'?mode=ro',uri=True)
    parent={r[0] for r in con.execute('select root_id from roots')};con.close()
    source=read(snapshot);_,_,old_pilot=select(source['blocks'],read(OLD/'config.json'),1)
    if not parent<={b['root']['root_id'] for b in source['blocks']}:raise ValueError('Historical frozen index not fully inherited')
    keys=('episode_id','fixture_id','trajectory_id')
    physical={k:{b['rows'][0].get(k) for b in source['blocks'] if b['root']['root_id'] in parent and b['rows'][0].get(k) is not None} for k in keys}
    shared={b['root']['root_id'] for b in source['blocks'] if b['root']['root_id'] not in parent and
        any(b['rows'][0].get(k) is not None and b['rows'][0].get(k) in physical[k] for k in keys)}
    excluded=parent|old_pilot|shared;banks={};chosen=[];available={}
    for split in ('natural_reference','unseen_evaluation'):
        pool=[b for b in source['blocks'] if b['split']==split and b['root']['root_id'] not in excluded]
        pool.sort(key=lambda b:hashlib.sha256(('forecast-clean-confirmation-v2/'+b['root']['root_id']).encode()).hexdigest())
        available[split]=len(pool)
        if len(pool)<count:raise ValueError('Insufficient historically unused roots: '+str(available))
        chosen+=pool[:count];banks[split]=[b['root']['root_id'] for b in pool[:count]]
    blocks=[]
    for b in chosen:
        r=b['rows'][0];root=b['root'];blocks.append(dict(root_id=root['root_id'],split=b['split'],scene=r['scene'],category=r['category'],
            episode_id=r['episode_id'],trajectory_id=r['trajectory_id'],archive_path=b['archive_path'],
            seal_path=str(Path(r['dataset'])/root['seal_file']),seal_sha256=b['seal_sha256'],complete_sha256=b['complete_sha256'],evidence=b['evidence'],
            rows=[{k:r[k] for k in ('branch_id','query_id','sr_valid','sr_label','observed_duration_s','human_collision_time_valid','human_collision_time_s')} for r in b['rows']]))
    out.mkdir(parents=True);save_json(out/'blocks.json',blocks)
    previous=out.parent/'confirmation_protocol/membership.json';prior=read(previous)
    membership=dict(snapshot_sha256=SNAPSHOT_HASH,blocks_sha256=sha256(out/'blocks.json'),banks=banks,available_unused_roots=available,
        excluded_historical_roots=sorted(excluded),old_full_v4_index_sha256=OLD_INDEX_SHA,old_pilot_config_sha256=sha256(OLD/'config.json'),
        additional_historical_physical_overlap_roots=sorted(shared),
        scenes={k:len({b['scene'] for b in blocks if b['split']==k}) for k in banks},
        supersedes_confirmation_claim_only=sha256(previous),previous_bank_unchanged=True,
        previous_bank_historical_scalar_overlap={k:len(set(v)&parent) for k,v in prior['banks'].items()},
        selection='fixed root hash after excluding complete old full-V4 index and complete previous pilot/extension plan; no forecast scores',
        policy='Freeze checkpoint panel before reserved forecast metrics; historical scene familiarity is distinct from root leakage',created_at=time.time())
    save_json(out/'membership.json',membership)
    for p in out.iterdir():p.chmod(0o444)
    print(json.dumps({k:v for k,v in membership.items() if k not in ('banks','excluded_historical_roots')}))


def build_reuse(protocol_dir,output,reuse):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);guard=Guard(out)
    identity=dict(clean_builder_sha256=sha256(__file__),membership_sha256=sha256(Path(protocol_dir)/'membership.json'),reuse=str(reuse))
    if (out/'clean_identity.json').exists() and read(out/'clean_identity.json')!=identity:raise ValueError('Clean confirmation recovery identity changed')
    save_json(out/'clean_identity.json',identity)
    recovery(out.parent/'cache_full/guard.json');blocks=read(Path(protocol_dir)/'blocks.json');linked=[]
    for block in blocks:
        guard.check();source=Path(reuse)/(block['root_id']+'.pt');dest=out/source.name
        if source.exists() and not dest.exists():
            value=torch.load(source,map_location='cpu');p=value['provenance']
            if (value['source_hash']!=SOURCE_HASH or value['root_id']!=block['root_id'] or
                p['seal_sha256']!=block['seal_sha256'] or p['complete_sha256']!=block['complete_sha256']):raise ValueError('Compact reuse provenance mismatch')
            os.link(source,dest);linked.append(dict(root_id=block['root_id'],sha256=sha256(dest)))
    save_json(out/f'compact_reuse_{time.time_ns()}.json',dict(source=str(reuse),linked=linked,raw_data_copied=False))
    build(protocol_dir,output)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--snapshot',required=True);a.add_argument('--output',required=True);a.add_argument('--count',type=int,default=128)
    a=sub.add_parser('build');a.add_argument('--protocol-dir',required=True);a.add_argument('--output',required=True);a.add_argument('--reuse',required=True)
    a=p.parse_args();kw=vars(a);command=kw.pop('command');(prepare if command=='prepare' else build_reuse)(**kw)


if __name__=='__main__':main()
