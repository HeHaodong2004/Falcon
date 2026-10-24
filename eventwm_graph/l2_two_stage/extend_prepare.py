"""Append every new eligible sealed Root to an immutable prepared parent.

Parent payload shards are referenced by hash, not rewritten. Source identities
of both preprocessing generations remain explicit. No collector controls.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import inspect
import hashlib
import json
import gzip
import os
from pathlib import Path
import time
import numpy as np
from .common import OUT,dump,sha
from .data import one_root,prepared_root_ledger

JOBS=None
DEST=None
IDENTITY=None


def chunk(bounds):
    begin,end=bounds;tag='delta_%06d_%06d'%(begin,end);report=DEST/(tag+'.json')
    if report.exists():
        result=json.loads(report.read_text())
        assert result['identity']==IDENTITY and sha(OUT/result['path'])==result['sha256']
        return result
    rows=[];counts=collections.Counter();roots=[]
    for job in JOBS[begin:end]:
        records,ledger=one_root(job);rows.extend(records);counts.update(ledger)
        roots.append(dict(root_id=job[1]['root_id'],scene_id=job[1]['scene_id'],counts=ledger,anchors=len(records)))
    path=DEST/(tag+'.npy');temporary=DEST/(tag+'.tmp.npy');np.save(temporary,np.array(rows,object),allow_pickle=True);temporary.replace(path)
    result=dict(identity=IDENTITY,roots=roots,counts=dict(counts),anchors=len(rows),path=str(path.relative_to(OUT)),sha256=sha(path))
    dump(report,result);return result


def run(snapshot='final_snapshot',parent='train',name='train_final',workers=2):
    global JOBS,DEST,IDENTITY
    for item in (snapshot,parent,name):
        if Path(item).name!=item:raise ValueError('names must be path components')
    p=OUT/'prepared'/parent/'manifest.json';original=json.loads(p.read_text())
    if original['schema'] not in ('two_stage_full_native_anchor_cache_1','two_stage_extended_native_anchor_cache_1'):
        raise ValueError('unsupported prepared parent schema')
    root_reports=prepared_root_ledger(p)
    prior_roots={r['root_id'] for r in root_reports};assert len(prior_roots)==original['roots']
    manifest=OUT/snapshot/'manifest.json';latest=json.loads(manifest.read_text());current_roots={r['root_id'] for r in latest['rows']}
    if not prior_roots.issubset(current_roots):raise ValueError('parent physical-unit replacement needs explicit reweighting; no silent shard reuse')
    JOBS=[(r['folder'],r) for r in latest['rows'] if r['root_id'] not in prior_roots]
    DEST=OUT/'prepared'/name;DEST.mkdir(exist_ok=True)
    IDENTITY=dict(parent_manifest_sha256=sha(p),snapshot_sha256=sha(manifest),
        source_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        one_root_function_sha256=hashlib.sha256(inspect.getsource(one_root).encode()).hexdigest())
    identity_path=DEST/'preparation_identity.json'
    if identity_path.exists():assert json.loads(identity_path.read_text())==IDENTITY
    else:dump(identity_path,IDENTITY)
    counts=collections.Counter(original['counts']);reports=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(chunk,[(i,min(i+100,len(JOBS))) for i in range(0,len(JOBS),100)]):
            reports.append(result);counts.update(result['counts'])
            dump(DEST/'progress.json',dict(parent_roots=len(prior_roots),new_roots=sum(len(r['roots']) for r in reports),total=latest['training_roots']))
            print(json.dumps(dict(new_roots=sum(len(r['roots']) for r in reports),new_total=len(JOBS),seconds=time.perf_counter()-start)),flush=True)
    new_reports=[r for report in reports for r in report['roots']]
    if len(new_reports)!=len(JOBS) or len({r['root_id'] for r in new_reports})!=len(new_reports):
        raise ValueError('delta preparation omitted or duplicated a Root')
    all_roots=prior_roots|{r['root_id'] for r in new_reports};assert all_roots==current_roots
    ledger_path=DEST/'root_ledger.json.gz'
    with ledger_path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            stream.write(json.dumps(root_reports+new_reports,separators=(',',':'),allow_nan=False).encode())
    ledger=dict(path=str(ledger_path.relative_to(OUT)),sha256=sha(ledger_path))
    dump(DEST/'manifest.json',dict(schema='two_stage_extended_native_anchor_cache_1',split='train',roots=len(all_roots),scenes=latest['training_scenes'],
        counts=dict(counts),shards=original['shards']+[{k:r[k] for k in ('path','sha256','anchors')} for r in reports],
        identity=IDENTITY,parent_identity=original['identity'],parent_manifest=str(p),snapshot=str(manifest),
        root_ledger=ledger,failures=[],seconds=time.perf_counter()-start))
    for path in DEST.iterdir():
        if path.is_file():os.chmod(path,0o444)
    os.chmod(DEST,0o555)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='final_snapshot');parser.add_argument('--parent',default='train');parser.add_argument('--name',default='train_final');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();run(args.snapshot,args.parent,args.name,args.workers)
