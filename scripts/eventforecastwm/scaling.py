"""Explicit audited-snapshot scaling with exact inherited evaluation/subset IDs.

This command never freezes production, changes collection, or launches training.
It accepts a separately audited snapshot and creates a NEW research protocol.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.data import VERSION,SOURCE_HASH,order_key


def canonical(block):
    return (block['split'],block['seal_sha256'],block['complete_sha256'],
            tuple(sorted((r['branch_id'],r['query_id'],r['sr_valid'],r['sr_label'],
                          r['scene'],r['episode_id'],str(r['trajectory_id']),
                          r['observed_duration_s'],r['human_collision_time_valid'],r['human_collision_time_s'])
                         for r in block['rows'])))


def validate_partitions(blocks):
    """Recheck the new pool, not only unchanged inherited records."""
    roots=[b['root']['root_id'] for b in blocks]
    queries=[r['query_id'] for b in blocks for r in b['rows']]
    if len(set(roots))!=len(roots) or len(set(queries))!=len(queries):raise ValueError('Duplicate root/query identity')
    train=[b for b in blocks if b['split']=='train'];held=[b for b in blocks if b['split']!='train']
    unseen={b['rows'][0]['scene'] for b in blocks if b['split']=='unseen_evaluation'}
    if unseen&{b['rows'][0]['scene'] for b in train}:raise ValueError('New training roots overlap reserved unseen scenes')
    for key in ('episode_id','fixture_id','trajectory_id'):
        fit=[b['rows'][0].get(key) for b in train if b['rows'][0].get(key) is not None]
        reserved={b['rows'][0].get(key) for b in held if b['rows'][0].get(key) is not None}
        if reserved&set(fit):raise ValueError('New cross-reservation '+key+' overlap')
        if len(fit)!=len(set(fit)):
            raise ValueError('Repeated training '+key+' requires explicit grouped sampling; roots cannot be counted as independent')
    return dict(training_episodes=len({b['rows'][0]['episode_id'] for b in train}),
        nonnull_training_trajectories=len({b['rows'][0]['trajectory_id'] for b in train if b['rows'][0].get('trajectory_id') is not None}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-protocol',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--snapshot-sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--parent-cache');p.add_argument('--new-cache');p.add_argument('--minimum-queries',type=int,default=495000)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a separate versioned protocol directory')
    parent=read(a.parent_protocol);snapshot=Path(a.snapshot)
    identity=read(snapshot.parent/'snapshot_identity.json');status=read(snapshot.parent/'status.json')
    digest=sha256(snapshot)
    if digest!=a.snapshot_sha256 or identity['sha256']!=digest or status['stage']!='frozen':raise ValueError('Snapshot not frozen/verified')
    if identity['parent_snapshot_sha256']!=parent['snapshot_sha256']:raise ValueError('Explicit direct-parent snapshot required')
    old=read(parent['snapshot']);new=read(snapshot)
    independence=validate_partitions(new['blocks'])
    old_by={b['root']['root_id']:b for b in old['blocks']};new_by={b['root']['root_id']:b for b in new['blocks']}
    if sum(len(b['rows']) for b in new['blocks'])<a.minimum_queries:raise ValueError('Research scaling milestone not reached')
    for rid,block in old_by.items():
        if rid not in new_by or canonical(block)!=canonical(new_by[rid]):raise ValueError('Parent membership/provenance changed: '+rid)
    pool=[b for b in new['blocks'] if b['split']=='train'];eligible={b['root']['root_id'] for b in pool}
    dev=set(parent['development']);old_fit=parent['subsets']['full']
    if not (set(old_fit)|dev)<=eligible:raise ValueError('Original fit/dev identities missing')
    extra=sorted(eligible-set(old_fit)-dev,key=order_key)
    blocks=[]
    for b in pool:
        root=b['root'];r=b['rows'][0];rid=root['root_id']
        blocks.append(dict(root_id=rid,split='dev' if rid in dev else 'fit',scene=r['scene'],category=r['category'],
            episode_id=r['episode_id'],trajectory_id=r['trajectory_id'],archive_path=b['archive_path'],
            seal_path=str(Path(r['dataset'])/root['seal_file']),seal_sha256=b['seal_sha256'],complete_sha256=b['complete_sha256'],
            evidence=b['evidence'],rows=[{k:r[k] for k in ('branch_id','query_id','sr_valid','sr_label',
                'observed_duration_s','human_collision_time_valid','human_collision_time_s')} for r in b['rows']]))
    out.mkdir(parents=True)
    save_json(out/'blocks.json',blocks)
    subsets={k:list(v) for k,v in parent['subsets'].items()}
    subsets['full350matched']=list(old_fit);subsets['full']=list(old_fit)+extra
    subsets['refit_all']=subsets['full']+sorted(dev)
    protocol=dict(parent,version=VERSION,snapshot=str(snapshot.resolve()),snapshot_sha256=digest,
        parent_protocol_sha256=sha256(a.parent_protocol),parent_snapshot_sha256=parent['snapshot_sha256'],
        blocks_sha256=sha256(out/'blocks.json'),eligible_roots=len(pool),queries=sum(len(b['rows']) for b in pool),
        scenes=len({b['scene'] for b in blocks}),subsets=subsets,additional_independent_fit_roots=len(extra),
        independence_audit=independence,
        evaluation='exact inherited development IDs; use the same separately frozen confirmation caches',
        scaling='new immutable experiment only; no automatic training restart')
    save_json(out/'protocol.json',protocol)
    for path in (out/'blocks.json',out/'protocol.json'):path.chmod(0o444)
    reused=0
    if a.parent_cache or a.new_cache:
        if not (a.parent_cache and a.new_cache):raise ValueError('Both cache paths required')
        source=Path(a.parent_cache);destination=Path(a.new_cache)
        if destination.exists():raise FileExistsError('New cache must be separate')
        destination.mkdir(parents=True)
        for row in read(source/'manifest.json')['roots']:
            rid=row['root_id']
            if rid not in eligible:continue
            path=source/row['file']
            if sha256(path)!=row['sha256']:raise ValueError('Parent compact cache changed')
            value=torch.load(path,map_location='cpu')
            if value['source_hash']!=SOURCE_HASH:raise ValueError('Explicit extractor migration required')
            block=new_by[rid]
            if value['provenance']['seal_sha256']!=block['seal_sha256']:raise ValueError('Root raw provenance changed')
            # Only compact payloads are linked. Original raw datasets stay in place.
            os.link(path,destination/(rid+'.pt'));reused+=1
        save_json(destination/'incremental_reuse.json',dict(parent_cache=str(source),roots_linked=reused,
            parent_manifest_sha256=sha256(source/'manifest.json'),raw_data_copied=False))
    print(json.dumps(dict(protocol=str(out/'protocol.json'),eligible_roots=len(pool),new_fit_roots=len(extra),
        inherited_development_roots=len(dev),reused_compact_roots=reused)))


if __name__=='__main__':main()
