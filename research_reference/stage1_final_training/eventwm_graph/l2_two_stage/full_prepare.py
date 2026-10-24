"""Resumable, full latest-root preparation in independently committed shards."""
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import BASE,OUT,dump,sha
from .data import jobs_for,one_root

JOBS=None
IDENTITY=None


def chunk(bounds):
    begin,end=bounds;dest=OUT/'prepared/train';tag='%06d_%06d'%(begin,end);report=dest/(tag+'.json')
    if report.exists():
        old=json.loads(report.read_text())
        assert old['identity']==IDENTITY and sha(dest/(tag+'.npy'))==old['sha256']
        return old
    records=[];counts=collections.Counter();roots=[];start=time.perf_counter()
    for job in JOBS[begin:end]:
        rows,ledger=one_root(job);records.extend(rows);counts.update(ledger)
        roots.append(dict(root_id=job[1]['root_id'],scene_id=job[1]['scene_id'],anchors=len(rows),counts=ledger))
        for r in rows:
            assert len(r['set_positions'])+int(np.sum(r['age']<1e-7))>=r['minimum_count']
    tmp=dest/(tag+'.tmp.npy');np.save(tmp,np.asarray(records,object),allow_pickle=True);tmp.replace(dest/(tag+'.npy'))
    result=dict(begin=begin,end=end,roots=roots,counts=dict(counts),anchors=len(records),
                path=str((dest/(tag+'.npy')).relative_to(OUT)),sha256=sha(dest/(tag+'.npy')),identity=IDENTITY,seconds=time.perf_counter()-start)
    dump(report,result);return result


def run(workers=2,chunk_size=200):
    global JOBS,IDENTITY
    JOBS=jobs_for('train');dest=OUT/'prepared/train';dest.mkdir(parents=True,exist_ok=True)
    IDENTITY=dict(snapshot_sha256=sha(OUT/'latest_snapshot/manifest.json'),
                  source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    dump(dest/'preparation_identity.json',IDENTITY)
    for p in Path(__file__).parent.glob('*.py'):
        target=dest/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    start=time.perf_counter();reports=[];counts=collections.Counter();completed=0
    work=[(i,min(i+chunk_size,len(JOBS))) for i in range(0,len(JOBS),chunk_size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(chunk,work,chunksize=1):
            reports.append(result);counts.update(result['counts']);completed+=len(result['roots'])
            print(json.dumps(dict(roots=completed,total=len(JOBS),anchors=counts['anchors'],seconds=time.perf_counter()-start)),flush=True)
            dump(dest/'progress.json',dict(roots=completed,total=len(JOBS),anchors=counts['anchors'],seconds=time.perf_counter()-start,
                completed_shards=len(reports),identity=IDENTITY))
    roots=[r for report in reports for r in report['roots']]
    assert len(roots)==len(JOBS)==len({r['root_id'] for r in roots})
    dump(dest/'manifest.json',dict(schema='two_stage_full_native_anchor_cache_1',split='train',roots=len(JOBS),
        scenes=len({r['scene_id'] for r in roots}),counts=dict(counts),shards=[{k:r[k] for k in ('path','sha256','anchors')} for r in reports],
        identity=IDENTITY,seconds=time.perf_counter()-start,failures=[],sampling='All eligible independent roots; sibling-free canonical branch anchors with root-normalized weights.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--chunk-size',type=int,default=200);a=p.parse_args();run(a.workers,a.chunk_size)
