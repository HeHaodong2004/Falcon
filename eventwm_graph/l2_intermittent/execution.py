"""Seal research sources, then verify them BEFORE importing a scientific run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def seal(base,target):
    if target.exists():raise ValueError('preserve the original sealed execution')
    paths=sorted((base/'eventwm_graph').rglob('*.py'))
    before={str(p.relative_to(base)):digest(p) for p in paths}
    target.mkdir(parents=True)
    shutil.copytree(base/'eventwm_graph',target/'eventwm_graph',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if before!={str(p.relative_to(base)):digest(p) for p in paths}:raise ValueError('source changed while sealing')
    if before!={name:digest(target/name) for name in before}:raise ValueError('copied source mismatch')
    (target/'output').mkdir()
    os.symlink(base/'output/l2_graph_wm_120k_20261021',target/'output/l2_graph_wm_120k_20261021')
    output=target/'output/l2_intermittent_wm_20261022';output.mkdir()
    for name in ('training','motion.json','motion_fit.json','train.npy','dev.npy'):
        os.symlink(base/'output/l2_intermittent_wm_20261022'/name,output/name)
    (target/'output/read_only_import_guard').mkdir()
    (target/'docs/l2').mkdir(parents=True)
    for source in (base/'docs/l2').iterdir():
        if source.name!='intermittent_wm_20261022':os.symlink(source,target/'docs/l2'/source.name)
    documents=target/'docs/l2/intermittent_wm_20261022';documents.mkdir()
    for name in ('PROTOCOL.md','EVALUATION_FREEZE.md'):
        shutil.copyfile(base/'docs/l2/intermittent_wm_20261022'/name,documents/name)
        before[str((documents/name).relative_to(target))]=digest(documents/name)
    inputs=[base/'output/l2_graph_wm_120k_20261021'/name for name in
        ('snapshot/manifest.json','dev.npy','train.npy','motion.json','population_prior.json','training/checkpoint.pt')]
    inputs += [base/'output/l2_intermittent_wm_20261022'/name for name in
        ('motion.json','train.npy','dev.npy','training/checkpoint.pt','training/epoch0.pt','training/last.pt','training/training.json')]
    inputs += [base/'docs/l2/graph_wm_120k_20261021/artifacts'/name for name in
        ('motion.json','population_prior.json','checkpoints/frozen_observation_fit.json')]
    inputs += [base/'docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json']
    manifest=dict(source_root=str(base),sealed_sources=before,inputs={str(p):digest(p) for p in inputs},
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        scope='Owned source copy verified before scientific imports; immutable historical data reused read-only; no production writes.')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in before:(target/name).chmod(0o444)
    (target/'manifest.json').chmod(0o444)
    print(json.dumps(dict(runtime=str(target),manifest_sha256=digest(target/'manifest.json'),sealed_files=len(before))),flush=True)


def run(target,operation,start,stop,limit):
    manifest=json.loads((target/'manifest.json').read_text())
    sources=manifest['sealed_sources'];inputs=manifest['inputs']
    if sources!={name:digest(target/name) for name in sources}:raise ValueError('sealed source changed before import')
    if inputs!={name:digest(name) for name in inputs}:raise ValueError('sealed input changed before import')
    sys.path.insert(0,str(target));os.environ['L2_REPRO_OUTPUT']=str(target/'output/read_only_import_guard')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    started=time.perf_counter();error=None
    output=target/'output/l2_intermittent_wm_20261022'
    if operation=='forecast':
        destination=output/('evaluation_smoke' if limit else 'evaluation_shard_%04d_%04d'%(start,stop))
    elif operation=='merge':destination=output/'evaluation'
    else:destination=output
    guard_name=operation+'_execution_guard.json'
    if (destination/guard_name).exists():raise ValueError('preserve the previous execution guard')
    try:
        from eventwm_graph.l2_intermittent import model
        if not str(Path(model.__file__).resolve()).startswith(str(target.resolve())+os.sep):raise ValueError('wrong imported runtime')
        if operation in ('forecast','merge'):
            from eventwm_graph.l2_intermittent import evaluate
            if operation=='forecast':evaluate.run(limit,start,stop)
            else:
                shards=sorted(output.glob('evaluation_shard_*'))
                for shard in shards:
                    record=json.loads((shard/'forecast_execution_guard.json').read_text())
                    if record['error'] or not record['sources_stable'] or not record['inputs_stable'] or record['manifest_sha256']!=digest(target/'manifest.json'):
                        raise ValueError('incompatible sealed shard')
                evaluate.merge_shards()
        elif operation in ('physical','old_physical'):
            from eventwm_graph.l2_intermittent import physical
            physical.reproduce_old() if operation=='old_physical' else physical.run(limit)
        else:
            from eventwm_graph.l2_intermittent.optical_validation import run as optical
            optical(limit)
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        destination.mkdir(exist_ok=True)
        guard=dict(manifest_sha256=digest(target/'manifest.json'),sources=sources,inputs=inputs,
            sources_stable=sources=={name:digest(target/name) for name in sources},
            inputs_stable=inputs=={name:digest(name) for name in inputs},
            operation=operation,start=start,stop=stop,limit=limit,error=error,seconds=time.perf_counter()-started)
        (destination/guard_name).write_text(json.dumps(guard,indent=2))
        if not guard['sources_stable'] or not guard['inputs_stable']:raise RuntimeError('unstable sealed scientific execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('seal','forecast','merge','physical','old_physical','optical'))
    parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--base',type=Path,default=Path.cwd())
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=2110);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();target=args.runtime.resolve()
    seal(args.base.resolve(),target) if args.operation=='seal' else run(target,args.operation,args.start,args.stop,args.limit)
