"""Standalone release verification, public smoke, full fitting and frozen scoring.

Copy is installed as RELEASE/reproduce.py. No operation contacts the collector.
Run with the software versions documented in TRAINING_REPORT.md.
"""
import argparse,gzip,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def cp(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def tree(source,target):
    shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))


def link(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(Path(source).resolve(),target_is_directory=Path(source).is_dir())


def verify(full=False):
    manifest=json.loads((ROOT/'INTEGRITY.json').read_text());errors=[]
    for name,expected in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or sha(p)!=expected:errors.append(name)
    for name,expected in manifest.get('data_anchor_files',{}).items():
        p=ROOT/name
        if not p.is_file() or sha(p)!=expected:errors.append(name)
    prep=json.loads((ROOT/'artifacts/audit/preparation.json').read_text())
    for r in prep['shards']:
        p=ROOT/'data/training'/r['path']
        if not p.is_file() or sha(p)!=r['sha256']:errors.append(str(p.relative_to(ROOT)))
    if full:
        for cohort in ('snapshot','supplement'):
            base=ROOT/'data'/cohort;m=json.loads((base/'manifest.json').read_text())
            for row in m['rows']:
                for filename,expected in row['files'].items():
                    p=base/'roots'/row['root_id']/filename
                    if not p.is_file() or sha(p)!=expected:errors.append(str(p.relative_to(ROOT)))
        for cohort in ('frozen_evaluation','reobservation_supplement'):
            base=ROOT/'data'/cohort
            manifest_path=base/('snapshot/manifest.json' if cohort=='frozen_evaluation' else 'manifest.json')
            data_manifest=json.loads(manifest_path.read_text())
            for row in data_manifest['rows']:
                if cohort=='frozen_evaluation' and row['split']!='dev':continue
                folder=base/('snapshot/roots' if cohort=='frozen_evaluation' else 'roots')/row['root_id']
                for filename,expected in row['files'].items():
                    p=folder/filename
                    if not p.is_file() or sha(p)!=expected:errors.append(str(p.relative_to(ROOT)))
    if errors:raise ValueError(errors)
    print(json.dumps(dict(verified_files=len(manifest['files']),training_shards=len(prep['shards']),full_training_payload_check=full,errors=errors)))


def smoke():
    runtime=ROOT/'runtime/final';sys.path.insert(0,str(runtime));os.chdir(runtime)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
    from eventwm_graph.l2_intermittent.api_example import inputs
    model=IntermittentWorldModel.load(ROOT/'artifacts/checkpoints/C/checkpoint.pt')
    history,graph,row=inputs();b=model.initialize(history,graph)
    b=model.observe(b,row(.144,2.06));tracks=model.get_previously_observed_tracks(b)
    f=model.predict_shared(b,.5,16,41,times=[0.,.5]);restored=model.deserialize_actual_belief(model.serialize_actual_belief(b))
    assert b.fingerprint()==restored.fingerprint()
    assert abs(float(f.weights.sum())-1)<1e-10
    assert b.diagnostics['association_history_search_complete'] is False
    print(json.dumps(dict(model_version=model.model_version,checkpoint_sha256=sha(ROOT/'artifacts/checkpoints/C/checkpoint.pt'),
                         tracks=len(tracks),worlds=len(f.weights),bootstrap=b.diagnostics['bootstrap_support_status'],roundtrip=True)))


def fit(destination,execute=True):
    dest=Path(destination).resolve();dest.mkdir(exist_ok=False)
    tree(ROOT/'runtime/final/eventwm_graph',dest/'eventwm_graph')
    archived=ROOT/'artifacts/training/sources'
    tree(archived,dest)
    # Remove later unused campaign modules from the four fitting packages;
    # retain the shared base package needed by eventwm_graph.__init__ imports.
    for folder in ('l2_final','l2_intermittent','l2_120k','l2_partial'):
        for path in (dest/'eventwm_graph'/folder).glob('*.py'):
            if not (archived/path.relative_to(dest)).exists():path.unlink()
    tree(ROOT/'runtime/final/docs',dest/'docs')
    cp(ROOT/'PROTOCOL.md',dest/'docs/l2/final_800k_20261023/PROTOCOL.md')
    out=dest/'output/l2_final_800k_20261023'
    for name in ('preparation.json','preparation_contract.json','final_snapshot_manifest.json','environment.json'):
        cp(ROOT/'artifacts/audit'/name,out/name)
    for cohort in ('snapshot','supplement'):cp(ROOT/'data'/cohort/'manifest.json',out/cohort/'manifest.json')
    prep=json.loads((out/'preparation.json').read_text())
    for shard in prep['shards']:link(ROOT/'data/training'/shard['path'],out/shard['path'])
    prior=dest/'inputs/prior';previous=dest/'inputs/previous'
    for name in ('dev.npy','motion.json'):link(ROOT/'data/training'/name,prior/name)
    link(ROOT/'data/training/reference_epoch0.pt',previous/'training/epoch0.pt')
    sys.path.insert(0,str(dest));os.chdir(dest)
    from eventwm_graph.l2_final import train
    train.BASE=dest;train.OUT=out;train.PRIOR=prior;train.PREVIOUS=previous
    if not execute:
        contract=json.loads((out/'preparation_contract.json').read_text())
        assert sha(out/'snapshot/manifest.json')==contract['snapshot_sha256']
        assert sha(out/'final_snapshot_manifest.json')==contract['final_snapshot_sha256']
        assert all(sha(dest/name)==value for name,value in contract['estimator_sources'].items())
        assert all(sha(out/r['path'])==r['sha256'] for r in prep['shards'])
        print('Full fitting inputs and archived estimator source closure verified; no gradients:',dest)
        return
    train.run()
    print('Completed full fit:',out/'training/training.json')


def evaluate(destination,arm,start,stop):
    dest=Path(destination).resolve();dest.mkdir(exist_ok=False)
    source=ROOT/'runtime'/('control_'+arm);tree(source,dest)
    old=dest/'output/l2_graph_wm_120k_20261021';old.mkdir(parents=True)
    for entry in (ROOT/'data/frozen_evaluation').iterdir():link(entry,old/entry.name)
    out=dest/'output/l2_intermittent_wm_20261022';out.mkdir(parents=True)
    weights='A' if arm.startswith(('A','B')) else 'C'
    for name in ('checkpoint.pt','epoch0.pt','training.json'):cp(ROOT/'artifacts/checkpoints'/weights/name,out/'training'/name)
    cp(ROOT/'artifacts/checkpoints'/weights/'motion.json',out/'motion.json')
    source_before={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.py')}
    inputs_before={str(p):sha(p) for p in [old/'dev.npy',old/'snapshot/manifest.json',out/'training/checkpoint.pt']}
    env=os.environ.copy();env['PYTHONPATH']=str(dest);env['L2_REPRO_OUTPUT']=str(old)
    subprocess.run([sys.executable,'-m','eventwm_graph.l2_intermittent.evaluate','--start',str(start),'--stop',str(stop)],cwd=dest,env=env,check=True)
    assert all(sha(dest/p)==h for p,h in source_before.items()) and all(sha(p)==h for p,h in inputs_before.items())
    output=out/('evaluation' if start==0 and stop==2110 else f'evaluation_shard_{start:04d}_{stop:04d}')
    (output/'forecast_execution_guard.json').write_text(json.dumps(dict(error=None,sources_stable=True,inputs_stable=True,inputs_before=inputs_before)))
    sys.path.insert(0,str(dest));os.chdir(dest);os.environ['L2_REPRO_OUTPUT']=str(old)
    path=ROOT/'artifacts/evaluation_tools/annotation_accounting.py'
    spec=importlib.util.spec_from_file_location('release_annotation',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.apply(output,ROOT/'artifacts/evaluation_tools/native_annotation_validity.json')
    print('Completed source-bound frozen scoring:',output)


def regressions(destination):
    dest=Path(destination).resolve();dest.mkdir(exist_ok=False)
    tree(ROOT/'runtime/final',dest);link(ROOT/'runtime/tests',dest/'tests')
    (dest/'output/l2_final_800k_20261023').mkdir(parents=True)
    runner=dest/'scripts/l2_final_tests.py';cp(ROOT/'runtime/l2_final_tests.py',runner)
    env=os.environ.copy();env['PYTHONPATH']=str(dest);env['L2_REPRO_OUTPUT']=str(dest/'output/l2_graph_wm_120k_20261021')
    subprocess.run([sys.executable,str(runner)],cwd=dest,env=env,check=True)
    print('Completed isolated packaged regression suites:',dest/'output/l2_final_800k_20261023/regressions.json')


def benchmark(destination,arm):
    import inspect
    dest=Path(destination).resolve();dest.mkdir(exist_ok=False)
    tree(ROOT/'runtime'/('control_A' if arm=='A' else 'final'),dest)
    old=dest/'output/l2_graph_wm_120k_20261021';old.mkdir(parents=True)
    for entry in (ROOT/'data/frozen_evaluation').iterdir():link(entry,old/entry.name)
    weights='A' if arm.startswith(('A','B')) else 'C'
    out=dest/'output/l2_intermittent_wm_20261022'
    for name in ('checkpoint.pt','epoch0.pt','training.json'):cp(ROOT/'artifacts/checkpoints'/weights/name,out/'training'/name)
    path=dest/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py'
    cp(ROOT/'artifacts/evaluation_tools/reviewer_c_benchmark.py',path)
    sys.path.insert(0,str(dest));os.chdir(dest);os.environ['L2_REPRO_OUTPUT']=str(old)
    spec=importlib.util.spec_from_file_location('portable_benchmark',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_sources=module.source_paths
    module.source_paths=lambda:sorted(set(original_sources()+list((dest/'eventwm_graph/l2_intermittent').glob('*.py'))))
    if arm!='A':
        source=inspect.getsource(module.functional_checks)
        assert 'mix=b.fork(components=' in source
        source=source.replace('mix=b.fork(components=','mix=b.fork(next_slot=2,components=')
        exec(compile(source,str(path),'exec'),module.__dict__)
    module.run(argparse.Namespace(mode='benchmark',runtime_frozen=True,checkpoint=str(out/'training/checkpoint.pt'),tag='complete',limit=60,warm_repeats=5,cold_repeats=2,sensitivity_repeats=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('verify','smoke','benchmark','regressions','prepare-fit','fit','evaluate'))
    p.add_argument('--full',action='store_true');p.add_argument('--destination');p.add_argument('--arm',choices=('A','B','C','B_public','C_public'),default='C')
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=2110);a=p.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    if a.operation=='verify':verify(a.full)
    elif a.operation=='smoke':smoke()
    elif not a.destination:p.error('--destination is required for a new isolated run')
    elif a.operation=='regressions':regressions(a.destination)
    elif a.operation=='benchmark':
        if a.arm not in ('A','B','C'):p.error('benchmark uses A, B or C')
        benchmark(a.destination,a.arm)
    elif a.operation in ('fit','prepare-fit'):fit(a.destination,execute=a.operation=='fit')
    else:evaluate(a.destination,a.arm,a.start,a.stop)
