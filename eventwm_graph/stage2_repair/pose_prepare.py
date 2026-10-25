"""Append signed supplied-pose bearings to valid original sampled rows."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
import numpy as np
from .diagnose import OUT,HF,PREP,read,save,sha
from .pose_features import signed_pose_features
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment

DEST=None;SPLIT=None


def annotate(item):
    index,shard=item;path=DEST/('%06d.npz'%index);receipt=path.with_suffix('.json')
    if receipt.exists():
        r=read(receipt)
        if r['source_shard_sha256']!=shard['sha256']or sha(path)!=r['sha256']:raise ValueError('stale signed-pose sidecar')
        return r
    with np.load(shard['path'],allow_pickle=False)as z:xx=z['x']
    extra=np.empty((len(xx),23),np.float32);groups=defaultdict(list);bindings=[];max_error=0.
    for e in shard['segments']:groups[e['root_id']].append(e)
    for rid,entries in groups.items():
        row=read(HF/'packed'/SPLIT/(rid+'.json'));a=load_arrays(row);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(row['segments'])}
        for e in entries:
            v=get_segment(a,lookup[(e['start_node'],e['end_node'])]);p=v[3];sl=slice(e['start'],e['end']);x=xx[sl,16:].astype(np.float32);centers=x[:,20:22].astype(np.float64)*10+p[0,[1,3]];keys=np.floor(centers/v[-1][0]).astype(np.int32)
            # The original stored F16 center error is at most a small fraction
            # of the actual half-cell width. Guard reconstruction itself.
            err=float(np.max(abs(centers-(keys+.5)*v[-1][0])))if len(keys)else 0.;max_error=max(max_error,err)
            if err>=v[-1][0]*.1:raise ValueError('quantized center cannot certify recovered key')
            extra[sl]=signed_pose_features(keys,p,float(v[-1][0]),x)
        bindings.append(dict(root_id=rid,packed_sha256=row['sha256']))
    np.savez(path,extra=extra);r=dict(index=index,path=str(path),sha256=sha(path),source_shard_sha256=shard['sha256'],rows=len(extra),packed=bindings,max_recovered_center_error_m=max_error);save(receipt,r);return r


def prepare(split='pilot_fit',workers=2):
    global DEST,SPLIT
    SPLIT=split;path=PREP/(split+'_mixed_ray/manifest.json');m=read(path);DEST=OUT/'pose_prepared'/split;DEST.mkdir(parents=True,exist_ok=True);identity=dict(schema='signed_supplied_pose23_v1',source_manifest_sha256=sha(path),source_feature_sha256=sha(Path(__file__).with_name('pose_features.py')),split=split)
    ip=DEST/'identity.json'
    if ip.exists()and read(ip)!=identity:raise ValueError('signed pose feature cache invalidated')
    save(ip,identity);tick=time.perf_counter();done=[]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        jobs=[pool.submit(annotate,item)for item in enumerate(m['shards'])]
        for f in as_completed(jobs):
            done.append(f.result())
            if len(done)%16==0 or len(done)==len(jobs):print(dict(shards=len(done),total=len(jobs),seconds=time.perf_counter()-tick),flush=True)
    save(DEST/'manifest.json',dict(identity=identity,shards=sorted(done,key=lambda r:r['index']),rows=sum(r['rows']for r in done),seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='pilot_fit');p.add_argument('--workers',type=int,default=2);prepare(**vars(p.parse_args()))
