"""Reuse completed live artifacts read-only; prepare missing shards in isolation.

This does not touch the existing preparation jobs or their output directories.
Every reused graph, timestamp sidecar and sampled shard is hash-bound.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
from .diagnose import OUT, HF, PREP, read, save, sha, source_rows
from eventwm_graph.stage2_hf import temporal_data as td
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from eventwm_graph.stage2_hf.time_features import moment_matrix, basis_numpy

ROWS=None


def annotate(item):
    index,shard=item;dest=OUT/'time_full';dest.mkdir(exist_ok=True)
    live=HF/'time_prepared/full'/('%06d.json'%index)
    owned=dest/('%06d.json'%index)
    for p in [live,owned]:
        if p.exists():
            r=read(p)
            if r['source_shard_sha256']!=shard['sha256'] or sha(r['path'])!=r['sha256']:raise ValueError('stale time-preparation shard')
            return r
    assert sha(shard['path'])==shard['sha256']
    with np.load(shard['path'],allow_pickle=False)as z:x=z['x']
    extra=np.empty((len(x),32),np.float16);groups=defaultdict(list);used=[]
    for e in shard['segments']:groups[e['root_id']].append(e)
    td.DEST=OUT/'temporal_full';td.DEST.mkdir(exist_ok=True);td.SPLIT='full'
    for rid,segments in groups.items():
        packed=read(HF/'packed/full'/(rid+'.json'));a=load_arrays(packed)
        lp=HF/'temporal/full'/(rid+'.json')
        tr=read(lp) if lp.exists() else td.root_times_fast(ROWS[rid])
        if tr['packed_sha256']!=packed['sha256'] or sha(tr['path'])!=tr['sha256']:raise ValueError('stale temporal graph')
        with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
        lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])};cache={}
        for e in segments:
            i=lookup[(e['start_node'],e['end_node'])];gi=int(a['graph_index'][i]);v=get_segment(a,i);p=v[3];key=(gi,*p[0,[1,3]])
            if key not in cache:
                gs=slice(*a['goffsets'][gi:gi+2]);cache[key]=moment_matrix(v[0],v[1],last[gs],p[0,0],p[0,[1,3]],v[-1][0])
            sl=slice(e['start'],e['end']);extra[sl]=basis_numpy(x[sl,36:38].astype(np.float32))@cache[key]
        used.append(dict(root_id=rid,temporal_sha256=tr['sha256'],packed_sha256=packed['sha256']))
    path=owned.with_suffix('.npz');np.savez(path,extra=extra)
    r=dict(index=index,path=str(path),sha256=sha(path),source_shard_sha256=shard['sha256'],roots=used,rows=len(extra),bytes=path.stat().st_size)
    save(owned,r);return r


def main(workers=12):
    global ROWS
    rows,source=source_rows('train');ROWS={r['root_id']:r for r in rows}
    path=PREP/'full_mixed_ray/manifest.json';m=read(path);dest=OUT/'time_full';dest.mkdir(exist_ok=True)
    assert len(ROWS)==m['roots'];tick=time.perf_counter();done=[]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        jobs=[pool.submit(annotate,item)for item in enumerate(m['shards'])]
        for f in as_completed(jobs):
            done.append(f.result())
            if len(done)%32==0 or len(done)==len(jobs):
                r=dict(shards=len(done),total=len(jobs),seconds=time.perf_counter()-tick);save(dest/'progress.json',r);print(r,flush=True)
    save(dest/'manifest.json',dict(shards=sorted(done,key=lambda r:r['index']),rows=sum(r['rows']for r in done),source_manifest_sha256=sha(path),training_roots=len(rows),source_sha256=sha(__file__),seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);main(**vars(p.parse_args()))
