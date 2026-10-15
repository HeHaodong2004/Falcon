"""Seal and independently verify a completed research release.

Packaging never substitutes for training, testing, or confirmation. All required
completion receipts and checkpoint identities are checked before copying bytes.
The verifier uses the Python standard library and reads no training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def relative_file(root,name):
    relative=Path(name)
    if relative.is_absolute() or not relative.parts or any(p in ('.','..') for p in relative.parts):
        raise ValueError('Invalid release member: '+str(name))
    path=root/relative
    if root.resolve() not in path.resolve().parents:raise ValueError('Release member escapes directory')
    return path


def verify(directory):
    root=Path(directory).resolve();manifest=read(root/'release_manifest.json')
    if manifest.get('status')!='verified_release':raise ValueError('Release is not sealed')
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Release must not contain symlinks')
    expected=set(manifest['files'])|{'release_manifest.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Release file set differs from manifest')
    for name,spec in manifest['files'].items():
        path=relative_file(root,name)
        if path.is_symlink() or not path.is_file():raise ValueError('Release member is not a regular copied file')
        if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:
            raise ValueError('Release bytes changed: '+name)
    for role in ('primary','replicate'):
        if manifest['models'][role]['sha256']!=digest(root/(role+'.pt')):
            raise ValueError('Checkpoint role identity mismatch')
    return dict(status='release_integrity_verified',release=str(root),
        release_manifest_sha256=digest(root/'release_manifest.json'),
        files_verified=len(manifest['files']),source_commit=manifest['source_commit'],
        checkpoints={k:v['sha256'] for k,v in manifest['models'].items()})


def validate_release_choice(selected,review,models):
    """Confirmation may reject a family, never pick another test-winning seed."""
    if review.get('status')!='confirmation_review_complete' or not review.get('reason'):
        raise ValueError('A documented completed confirmation review is required')
    decision=review.get('decision')
    if decision=='promote_frozen_candidate':roles=selected['models']
    elif decision=='preserve_frozen_incumbent':roles=selected.get('fallback_models',{})
    else:raise ValueError('Unknown confirmation release decision')
    if not {'primary','replicate'}<=set(roles):raise ValueError('Both release roles must have been frozen before confirmation')
    if any(models[role]['sha256']!=roles[role]['sha256'] for role in ('primary','replicate')):
        raise ValueError('Release changes the frozen primary/replicate roles')
    return decision


def build(config_path):
    cfg=read(config_path);out=Path(cfg['output']).resolve()
    if out.exists():raise FileExistsError('Release destination must be new')
    source=Path(__file__).resolve().parents[3]
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Commit source and reports before sealing the release')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    tests=read(cfg['test_receipt'])
    if not tests.get('passed') or tests.get('status')!='forecast_tests_complete':
        raise ValueError('Completed passing forecasting suite is required')
    if any(digest(source/name)!=value for name,value in tests['sources'].items()):
        raise ValueError('Source changed since completed forecasting tests')
    selected=read(cfg['selection']);decision=read(cfg['confirmation_decision'])
    confirmation=read(cfg['confirmation_receipt'])
    identity_path=Path(cfg['confirmation_receipt']).parent/'identity.json';identity=read(identity_path)
    if selected.get('status')!='development_selection_frozen':raise ValueError('Missing frozen development selection')
    if confirmation['identity_sha256']!=digest(identity_path):raise ValueError('Confirmation identity changed')
    for field,path in [('decision_sha256',cfg['confirmation_decision']),('membership_sha256',cfg['membership']),
                       ('manifest_sha256',cfg['confirmation_manifest'])]:
        if identity[field]!=digest(path):raise ValueError('Confirmation population/decision changed')
    if selected['membership_sha256']!=digest(cfg['membership']):raise ValueError('Selection membership changed')
    for name,spec in decision['models'].items():
        if digest(spec['checkpoint'])!=spec['sha256']:raise ValueError('Frozen panel checkpoint changed')
        if set(confirmation['models'].get(name,{}))!=set(read(cfg['membership'])['banks']):
            raise ValueError('Confirmation panel is incomplete')
        for result in confirmation['models'][name].values():
            if digest(result['path'])!=result['sha256']:raise ValueError('Confirmation score receipt changed')
    if not {'primary','replicate'}<=set(cfg['models']):
        raise ValueError('Primary and independent replicate are required')
    review=read(cfg['confirmation_review'])
    if (review.get('selection_sha256')!=digest(cfg['selection']) or
        review.get('confirmation_receipt_sha256')!=digest(cfg['confirmation_receipt'])):
        raise ValueError('Release review does not match the frozen selection and completed confirmation')
    release_choice=validate_release_choice(selected,review,cfg['models'])
    panel_hashes={v['sha256'] for v in decision['models'].values()}
    selected_hashes={v['sha256'] for v in (*selected['models'].values(),*selected.get('fallback_models',{}).values())}
    for role,spec in cfg['models'].items():
        if digest(spec['checkpoint'])!=spec['sha256'] or spec['sha256'] not in panel_hashes&selected_hashes:
            raise ValueError('Released model was not frozen and confirmed: '+role)
    for spec in cfg['dataset'].values():
        if digest(spec['path'])!=spec['sha256']:raise ValueError('Dataset identity changed')
    training=[]
    for directory in cfg['full_training_runs']:
        run=Path(directory);receipt=read(run/'result.json');run_identity=read(run/'identity.json')
        if (receipt.get('status')!='training_complete' or receipt['steps']<24576 or
            receipt['training_roots']<8965):raise ValueError('Required full-data budget is incomplete')
        for name,key in [('best.pt','best_sha256'),('last.pt','last_sha256')]:
            if digest(run/name)!=receipt[key]:raise ValueError('Training checkpoint changed')
        training.append(dict(run=str(run),seed=run_identity['config']['seed'],
            steps=receipt['steps'],roots=receipt['training_roots'],presentations=receipt['presentations'],
            receipt_sha256=digest(run/'result.json')))
    if len({r['seed'] for r in training})<2:raise ValueError('Two completed independent physical seeds are required')
    for role in ('primary','replicate'):
        proof=read(cfg['standalone_receipts'][role]);recurrent=read(cfg['recurrent_receipts'][role])
        checkpoint_hash=cfg['models'][role]['sha256']
        if (proof.get('status')!='standalone_verification_complete' or not proof.get('identical_seed_outputs_equal') or
            not proof.get('all_numeric_outputs_finite') or proof.get('rejected_accesses') or
            proof['identity']['inputs']['checkpoint']['sha256']!=checkpoint_hash):
            raise ValueError('Missing standalone proof for '+role)
        if any(digest(source/'scripts'/name)!=value for name,value in proof['identity']['sources'].items()):
            raise ValueError('Inference source changed after standalone proof')
        if (recurrent.get('status')!='recurrent_audit_complete' or recurrent['checkpoint_sha256']!=checkpoint_hash or
            recurrent['supported_window_pairs']<1 or
            recurrent['maximum_composition_position_difference']>1e-5 or
            recurrent['maximum_persistent_path_difference']>1e-5 or
            (recurrent.get('maximum_anonymous_occupancy_composition_difference') or 0)>1e-5):
            raise ValueError('Recurrent verification failed for '+role)
    # All prerequisites are validated before creating the new release directory.
    copies=dict(cfg['files'])
    required={'selection.json':cfg['selection'],'confirmation_decision.json':cfg['confirmation_decision'],
        'confirmation.json':cfg['confirmation_receipt'],'confirmation_identity.json':str(identity_path),
        'confirmation_review.json':cfg['confirmation_review'],
        'membership.json':cfg['membership'],'test_receipt.json':cfg['test_receipt'],'package_config.json':config_path}
    for role,spec in cfg['models'].items():required[role+'.pt']=spec['checkpoint']
    if set(copies)&set(required):raise ValueError('Duplicate required release filename')
    copies.update(required)
    if not {'README.md','MODEL.md','TRAINING.md','CONFIRMATION.md','REPRODUCE.md'}<=set(copies):
        raise ValueError('Required release reports are missing')
    if set(copies)&{'validation.json','source.tar.gz','release_manifest.json'}:
        raise ValueError('Reserved release filename')
    for name,path in copies.items():
        relative_file(out,name)
        if not Path(path).is_file():raise FileNotFoundError(path)
    out.mkdir(parents=True)
    for name,path in copies.items():
        target=relative_file(out,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        if digest(target)!=digest(path):raise ValueError('Copy changed source bytes')
    archive=out/'source.tar.gz'
    if archive.exists():raise ValueError('source.tar.gz is reserved')
    subprocess.run(['git','archive','--format=tar.gz','--prefix=eventwm-source/',
        '--output='+str(archive),commit,'--',*cfg['source_paths']],cwd=source,check=True)
    write(out/'validation.json',dict(status='research_prerequisites_verified',source_commit=commit,
        full_training=training,test_count=tests['tests_run'],models=cfg['models'],release_choice=release_choice,
        dataset=cfg['dataset'],confirmation_identity_sha256=digest(identity_path),
        production_actions=[],scope='Integrity and completed receipt verification; measured quality remains in the reports.'))
    files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'release_manifest.json',dict(status='verified_release',schema_version=1,
        source_commit=commit,models=cfg['models'],files=files,
        integrity='SHA256 detects changed bytes relative to this manifest; this is not a digital signature.'))
    result=verify(out)
    for path in out.rglob('*'):
        if path.is_file():path.chmod(0o444)
    return result


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('build').add_argument('--config',required=True)
    sub.add_parser('verify').add_argument('--release',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.config) if args.action=='build' else verify(args.release),indent=2))


if __name__=='__main__':main()
