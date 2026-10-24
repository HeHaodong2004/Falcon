"""Data-only extension using the baseline's unchanged executable preparation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from .common import BASE,OUT,DOC,OLD,PRIOR,BASELINE,sha,dump,copy_frozen

RUNTIME=OUT/'training_runtime'

def bootstrap():
    assert BASE!=BASELINE
    OUT.mkdir(parents=True,exist_ok=True)
    local=BASE/'output/l2_graph_wm_120k_20261021'
    if local.is_symlink():
        assert local.resolve()==OLD.resolve();local.unlink()
    local.mkdir(exist_ok=True)
    for name in ('snapshot','dev.npy','train.npy','motion.json','population_prior.json','training'):
        p=local/name
        if p.exists():assert p.resolve()==(OLD/name).resolve()
        else:p.symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())

def setup():
    RUNTIME.mkdir(exist_ok=False)
    closure=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources'
    for p in closure.rglob('*'):
        if p.is_file():
            target=RUNTIME/p.relative_to(closure);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
    archive=BASE/'docs/l2/intermittent_wm_20261022/training_sources'
    m=json.loads((archive/'manifest.json').read_text());checks={}
    for row in m['sources']+m['inherited']:
        if 'preserved_at' not in row:continue
        src=BASE/row['preserved_at'];target=RUNTIME/row['path']
        expected=row.get('expected_sha256',row.get('sha256'))
        assert sha(src)==expected
        shutil.copyfile(src,target);checks[row['path']]=sha(target)
    # Fit never imports model.py; its historical metadata-only hash is disclosed.
    for folder in (BASE/'docs/l2').iterdir():
        target=RUNTIME/'docs/l2'/folder.name
        if folder.is_dir() and not target.exists():target.symlink_to(folder,target_is_directory=True)
    legacy=RUNTIME/'output/l2_graph_wm_120k_20261021';legacy.mkdir(parents=True)
    (legacy/'snapshot').symlink_to(OUT/'snapshot',target_is_directory=True)
    for name in ('population_prior.json','motion.json'):
        copy_frozen(OLD/name,legacy/name)
    (RUNTIME/'output/l2_intermittent_wm_20261022').symlink_to(OUT,target_is_directory=True)
    for name in ('dev.npy','motion.json'):
        copy_frozen(PRIOR/name,OUT/name)
    dump(OUT/'runtime_setup.json',dict(exact_training_sources=checks,
        untouched_runtime_sources={str(p.relative_to(BASE)):sha(p) for directory in ('l2_intermittent','l2_120k','l2_partial') for p in (BASE/'eventwm_graph'/directory).glob('*.py')},
        copied_inputs={name:sha(OUT/name) for name in ('dev.npy','motion.json')},
        caveat='Historical training model.py hash unavailable; module is metadata-only and not imported by fit. All executed fitting dependencies hash matched.'))

def activate():
    os.chdir(RUNTIME);sys.path.insert(0,str(RUNTIME))
    # Parent package was imported before changing sys.path; explicitly select closure.
    import eventwm_graph
    eventwm_graph.__path__=[str(RUNTIME/'eventwm_graph')]

def shard(start,stop,tag):
    activate()
    from eventwm_graph.l2_120k.data import prepare
    prepare('train',start_index=start,stop_index=stop,tag=tag)

def assemble(tags):
    activate();started=time.perf_counter()
    original=list(np.load(PRIOR/'train.npy',allow_pickle=True));result=original.copy();reports=[]
    snapshot=OUT/'snapshot';seen={r['root_id'] for r in original}
    for tag in tags:
        path=RUNTIME/'output/l2_graph_wm_120k_20261021'/('train_'+tag+'.npy')
        cached=np.load(path,allow_pickle=True);added=0
        for row in cached:
            if not row['components']:continue
            if row['root_id'] in seen:raise ValueError('duplicate supervised physical root')
            with np.load(snapshot/'roots'/row['root_id']/'privileged_future.npz') as f:
                times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
            keep=times<=times[0]+5.+1e-8;times=times[keep];truth=truth[keep];cs=[]
            for c in row['components']:
                ids=[]
                for p in c['truth'][0]:
                    d=np.linalg.norm(truth[0]-p,axis=-1);j=int(d.argmin())
                    if d[j]>2e-4:raise ValueError('lost frozen training correspondence')
                    ids.append(j)
                cs.append({**c,'truth':truth[:,ids].astype(np.float32)})
            result.append({**row,'components':cs,'times':times});seen.add(row['root_id']);added+=1
        reports.append(dict(tag=tag,cache_sha256=sha(path),built=len(cached),added_supervised_roots=added,
            preparation=json.loads(path.with_name('train_'+tag+'_prepare.json').read_text())))
    np.save(OUT/'train.npy',np.array(result,dtype=object),allow_pickle=True)
    # Verify the entire inherited row prefix, not just selected tensor fields.
    import pickle
    loaded=np.load(OUT/'train.npy',allow_pickle=True)
    assert all(pickle.dumps(a,protocol=4)==pickle.dumps(b,protocol=4) for a,b in zip(original,loaded))
    dump(OUT/'preparation.json',dict(original_supervised_roots=len(original),added_supervised_roots=len(result)-len(original),
        supervised_roots=len(result),scenes=len({r['scene_id'] for r in result}),
        train_sha256=sha(OUT/'train.npy'),dev_sha256=sha(OUT/'dev.npy'),motion_sha256=sha(OUT/'motion.json'),
        snapshot_sha256=sha(snapshot/'manifest.json'),prefix_bitwise_equal=True,shards=reports,seconds=time.perf_counter()-started))
    print(json.dumps(dict(supervised_roots=len(result),added=len(result)-len(original))),flush=True)

def compatibility():
    activate()
    prior={r['root_id']:r for r in np.load(OLD/'train.npy',allow_pickle=True)}
    rows=np.load(RUNTIME/'output/l2_graph_wm_120k_20261021/train_compatibility.npy',allow_pickle=True)
    count=0
    for row in rows:
        old=prior[row['root_id']]
        assert np.array_equal(row['times'],old['times'])
        assert len(row['components'])==len(old['components'])
        for a,b in zip(row['components'],old['components']):
            assert set(a)==set(b)
            for k in a:
                if a[k] is None:assert b[k] is None
                else:assert np.array_equal(a[k],b[k]),(row['root_id'],k)
            count+=1
    dump(OUT/'preparation_compatibility.json',dict(roots=len(rows),components=count,all_training_features_bitwise_equal=True,
        note='First30 original training roots; unchanged 3-second preparation inputs/features/targets; extended5-second original rows retained verbatim.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bootstrap','setup','shard','assemble','compatibility']);p.add_argument('--start',type=int);p.add_argument('--stop',type=int);p.add_argument('--tags',nargs='+');a=p.parse_args()
    if a.command=='bootstrap':bootstrap()
    elif a.command=='setup':setup()
    elif a.command=='shard':shard(a.start,a.stop,a.tags[0])
    elif a.command=='assemble':assemble(a.tags)
    else:compatibility()
