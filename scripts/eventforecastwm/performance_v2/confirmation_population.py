"""Audited incremental snapshot and new confirmation membership, never training.

This uses the established freeze without changing production or its query target.
A sub-495k snapshot is explicitly confirmation-only, never a 500k training claim.
No model inference, hyperparameter selection or reserved score is computed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark_reuse import CooperativeGuard


def freeze(cfg):
    import eventwm_belief_freeze as audited
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    frozen=Path(cfg.get('frozen_directory',out));frozen.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed snapshot receipt is immutable')
    original=audited.full.WMGuard
    class CombinedGuard(original):
        def __init__(self,directory):
            self.shared=CooperativeGuard(directory);super().__init__(directory)
        def check(self):
            self.shared.check()
            try:super().check()
            except RuntimeError as error:raise RuntimeError('Research stopped: '+str(error)) from error
    audited.full.WMGuard=CombinedGuard
    if not (frozen/'snapshot.json').exists():
        CooperativeGuard(out).check(True)
        audited.freeze(SimpleNamespace(output=frozen,parent=Path(cfg['parent']),index=audited.full.INDEX))
    elif read(frozen/'status.json').get('stage')!='frozen':raise ValueError('Snapshot exists without completed freeze receipt')
    identity=read(frozen/'snapshot_identity.json')
    if sha256(frozen/'snapshot.json')!=identity['sha256']:raise ValueError('Frozen snapshot changed')
    audit=read(frozen/'audit.json')
    save_json(out/'result.json',dict(status='confirmation_snapshot_complete',snapshot_identity=identity,
        validated_queries=audit['retained_queries'],validated_roots=audit['retained_roots'],
        purpose='New untouched confirmation membership; not used to fit models',
        is_500k_training=False,frozen_directory=str(frozen),production_actions=[],audited_freeze_source_sha256=sha256(audited.__file__),
        wrapper_source_sha256=sha256(__file__),config=cfg))


def select(cfg):
    out=Path(cfg['run'])
    if any((out/name).exists() for name in ('blocks.json','membership.json','result.json')):
        raise FileExistsError('Confirmation membership must be a new population')
    parent=read(cfg['parent_snapshot']);source=read(cfg['snapshot'])
    identity=read(Path(cfg['snapshot']).with_name('snapshot_identity.json'))
    if sha256(cfg['snapshot'])!=identity['sha256'] or identity['parent_snapshot_sha256']!=sha256(cfg['parent_snapshot']):
        raise ValueError('New population must inherit the exact 350k snapshot')
    old_roots={b['root']['root_id'] for b in parent['blocks']}
    fields=('episode_id','fixture_id','trajectory_id')
    old_ids={key:{row[key] for b in parent['blocks'] for row in b['rows'] if row.get(key)} for key in fields}
    candidates={};excluded={}
    for bank in ('natural_reference','unseen_evaluation'):
        pool=[b for b in source['blocks'] if b['split']==bank and b['root']['root_id'] not in old_roots]
        excluded[bank]=[];chosen=[]
        for block in pool:
            overlap=[key for key in fields if any(row.get(key) in old_ids[key] for row in block['rows'])]
            if overlap:excluded[bank].append(dict(root_id=block['root']['root_id'],fields=overlap))
            else:chosen.append(block)
        chosen.sort(key=lambda b:hashlib.sha256(('performance-v2-confirmation-20261015/'+b['root']['root_id']).encode()).hexdigest())
        candidates[bank]=chosen
    # The fallback is fixed before any forecasts are evaluated.
    available=min(map(len,candidates.values()));count=128 if available>=128 else 64 if available>=64 else 0
    if not count:raise ValueError('Fewer than 64 independent new roots per bank; do not claim clean confirmation')
    blocks=[];banks={}
    for bank,pool in candidates.items():
        banks[bank]=[b['root']['root_id'] for b in pool[:count]]
        for block in pool[:count]:
            row=block['rows'][0];root=block['root']
            blocks.append(dict(root_id=root['root_id'],split=bank,scene=row['scene'],category=row['category'],
                episode_id=row['episode_id'],trajectory_id=row['trajectory_id'],archive_path=block['archive_path'],
                seal_path=str(Path(row['dataset'])/root['seal_file']),seal_sha256=block['seal_sha256'],
                complete_sha256=block['complete_sha256'],evidence=block['evidence'],
                rows=[{k:r[k] for k in ('branch_id','query_id','sr_valid','sr_label','observed_duration_s',
                                      'human_collision_time_valid','human_collision_time_s')} for r in block['rows']]))
    fitting_scenes={b['root']['physical_scene_id'] for b in parent['blocks'] if b['split']=='train'}
    if fitting_scenes & {b['scene'] for b in blocks if b['split']=='unseen_evaluation'}:raise ValueError('Unseen scene overlaps fitting pool')
    out.mkdir(parents=True,exist_ok=True);save_json(out/'blocks.json',blocks)
    membership=dict(snapshot_sha256=identity['sha256'],parent_snapshot_sha256=sha256(cfg['parent_snapshot']),
        blocks_sha256=sha256(out/'blocks.json'),banks=banks,count_per_bank=count,
        available_new_independent_roots={k:len(v) for k,v in candidates.items()},excluded_physical_overlap=excluded,
        all_parent_roots_excluded=True,all_parent_episode_fixture_trajectory_ids_excluded=True,
        selection='Fixed root hash, 128 per bank or predeclared 64 fallback; never model scores, ages, difficulty or outcomes',
        scenes={k:len({b['scene'] for b in blocks if b['split']==k}) for k in banks},
        policy='Checkpoint family, calibration, seeds and hashes must be committed before any reserved metric',
        source_sha256=sha256(__file__),config=cfg)
    save_json(out/'membership.json',membership)
    save_json(out/'result.json',dict(status='confirmation_membership_frozen',membership_sha256=sha256(out/'membership.json'),
                                   roots_per_bank=count,forecast_metrics_opened=False))
    # Seal membership artifacts; the launcher still needs to append its exit
    # record and retain a writable cooperative guard in the enclosing run.
    for name in ('blocks.json','membership.json','result.json'):(out/name).chmod(0o444)


def build(cfg):
    """Run the established target extractor after development selection freezes."""
    from eventforecastwm import confirmation
    selected=read(cfg['selection']);out=Path(cfg['run'])
    if selected.get('status')!='development_selection_frozen':
        raise ValueError('Freeze development selection before opening new targets')
    if selected.get('membership_sha256')!=sha256(Path(cfg['protocol_dir'])/'membership.json'):
        raise ValueError('Selected confirmation population changed')
    for spec in selected['models'].values():
        if sha256(spec['checkpoint'])!=spec['sha256']:raise ValueError('Selected model changed')
    confirmation.Guard=CooperativeGuard
    try:confirmation.build(cfg['protocol_dir'],out)
    except RuntimeError as error:
        if str(error).startswith('Reserved raw extraction requires'):
            raise RuntimeError('Research stopped: '+str(error)) from error
        raise
    manifest=read(out/'manifest.json')
    save_json(out/'result.json',dict(status='confirmation_cache_complete',roots=len(manifest['roots']),
        manifest_sha256=sha256(out/'manifest.json'),selection_sha256=sha256(cfg['selection']),
        production_actions=[],wrapper_sha256=sha256(__file__)))


def geometry(cfg):
    """Read permitted static meshes with the same cooperative shared guard."""
    from eventforecastwm import geometry as extractor
    out=Path(cfg['run']);extractor.Guard=CooperativeGuard
    sys.argv=[sys.argv[0],'--manifest',cfg['manifest'],'--output',str(out)]
    extractor.main()
    manifest=read(out/'manifest.json')
    save_json(out/'result.json',dict(status='confirmation_geometry_complete',roots=len(manifest['roots']),
        manifest_sha256=sha256(out/'manifest.json'),production_actions=[],wrapper_sha256=sha256(__file__)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','select','build','geometry'));parser.add_argument('--config',required=True)
    args=parser.parse_args();dict(freeze=freeze,select=select,build=build,geometry=geometry)[args.action](read(args.config))


if __name__=='__main__':main()
