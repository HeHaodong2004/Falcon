"""Immutable-source extraction; same legal targets, sampling and exact weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from eventwm_graph.stage2.data import branches,source_rows
from .features import PreparedGraph,Features,opportunity
from .common import OUT,dump,sha,seed,identity

JOBS=None
DEST=None
SOURCE=None

def order(value):return hashlib.sha256(('quality-pilot-2420:'+value).encode()).hexdigest()

def protocol():
    path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    rows,manifest=source_rows('train');by_scene=collections.defaultdict(list)
    for row in rows:by_scene[row['scene_id']].append(row)
    scenes=sorted(by_scene,key=order);hold_scenes=scenes[:80]
    hold=[]
    for scene in hold_scenes:hold.extend(sorted(by_scene[scene],key=lambda r:order(r['root_id']))[:4])
    eligible=[r for r in rows if r['scene_id'] not in set(hold_scenes)]
    fitting=sorted(eligible,key=lambda r:order(r['root_id']))[:4096]
    value=dict(schema='stage2_quality_bounded_protocol_1',source_manifest=str(manifest),source_manifest_sha256=sha(manifest),
        fitting_roots=len(rows),fitting_scenes=len(by_scene),pilot_fit_roots=[r['root_id'] for r in fitting],
        pilot_holdout_roots=[r['root_id'] for r in hold],pilot_holdout_scenes=hold_scenes,
        independence='scene-disjoint fitting-only holdout; all Query siblings; no original dev selection',
        experiments=['baseline','trajectory','map','both'],epochs=6,seed=2410,width=48,batch_size=8192,
        objective='original inverse-inclusion Root/Query weighted CE +.2 observed height smoothL1; lr.001 wd.0001 clip5',
        selection='Brier and logscore <=1.02 times baseline and both classF1 >= baseline; among eligible prefer lowest logscore. If no candidate meets all, preserve lowestlogscore candidate but report tradeoffs. No additional model/loss sweep.',
        feature_modes='fixed92inputs/48/32/4; A replaces16nearestpose cols with20fullpath; B adds25legalmap; C combines; inactive columns zero',
        sampling='same64positive+64UNKNOWN and original stable Root/Query seed; identical targets/inclusion weights',
        pilot_metrics='inverse-inclusion sampled estimates, scene-balanced confusion and scene-bootstrap2000seed2411; original dev is fullcandidate exact',
        full_data_fits_max=1,original_dev_tuning=False,source_at_protocol=identity())
    dump(path,value);return value


def sample(example,context):
    keys=context.candidates(view=False);n=len(keys)
    if not n:return None,None
    target=example['target'];tk=np.asarray(list(target),np.int32).reshape(-1,2)
    codes=(keys[:,0]-context.lo[0])*int(context.shape[1])+keys[:,1]-context.lo[1]
    tc=(tk[:,0]-context.lo[0])*int(context.shape[1])+tk[:,1]-context.lo[1]
    at=np.searchsorted(codes,tc);safe=np.minimum(at,n-1);valid=(at<n)&np.all(keys[safe]==tk,axis=1)
    positive=at[valid];mask=np.zeros(n,bool);mask[positive]=True;negative=np.flatnonzero(~mask)
    rng=np.random.default_rng(seed(example['root_id'],example['query_id']));selected=[];weights=[]
    for stratum in (positive,negative):
        count=min(64,len(stratum))
        if count:selected.extend(rng.choice(stratum,count,replace=False));weights.extend([len(stratum)/(count*n*example['queries_in_root'])]*count)
    chosen=keys[selected]
    x,baseline=context.transform(chosen,'baseline')
    _,view=opportunity(chosen,context.path,context.resolution);x[:,47:67]=view;x[:,67:92]=context.prepared.map_features(chosen)
    y=np.asarray([target.get(tuple(k),(0,0.))[0] for k in chosen],np.uint8)
    height=np.asarray([target.get(tuple(k),(0,float(b)))[1]-float(b) for k,b in zip(chosen,baseline)],np.float32)
    p=example['trajectory'];distance=float(np.linalg.norm(np.diff(p[:,[1,3]],axis=0),axis=1).sum())
    angles=np.diff(p[:,4]);rotation=float(np.abs(np.arctan2(np.sin(angles),np.cos(angles))).sum())
    meta=dict(root_id=example['root_id'],scene_id=example['scene_id'],query_id=example['query_id'],candidate_cells=n,
        queries_in_root=example['queries_in_root'],new_free=sum(v[0]==1 for v in target.values()),new_blocked=sum(v[0]==2 for v in target.values()),
        out_of_domain_positive=len(target)-int(valid.sum()),known_changes=example['known_changes'],known_cells=len(example['graph']['cells']),
        duration=float(p[-1,0]-p[0,0]),distance=distance,rotation=rotation,poses=len(p),termination=example['termination'])
    return dict(x=x.astype(np.float16),y=y,height=height.astype(np.float16),weight=np.asarray(weights,np.float32)),meta


def chunk(bounds):
    begin,end=bounds;path=DEST/('%06d_%06d.npz'%(begin,end));report=path.with_suffix('.json')
    if report.exists():
        old=json.loads(report.read_text());assert old['source']==SOURCE and sha(path)==old['sha256'];return old
    arrays=collections.defaultdict(list);metas=[];cursor=0
    for row in JOBS[begin:end]:
        prepared=None
        for ex in branches(row):
            if prepared is None:prepared=PreparedGraph(ex['graph'])
            value,meta=sample(ex,Features(ex['graph'],ex['trajectory'],prepared))
            if value is None:continue
            meta.update(start=cursor,end=cursor+len(value['y']));cursor=meta['end'];metas.append(meta)
            for key,item in value.items():arrays[key].append(item)
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**{k:np.concatenate(v) for k,v in arrays.items()});tmp.replace(path)
    value=dict(path=str(path),sha256=sha(path),source=SOURCE,rows=cursor,roots=[r['root_id'] for r in JOBS[begin:end]],queries=metas)
    dump(report,value);return value


def prepare(split,workers=12):
    global JOBS,DEST,SOURCE
    p=protocol();rows,manifest=source_rows('train')
    if split!='full':
        ids=set(p['pilot_fit_roots' if split=='pilot_fit' else 'pilot_holdout_roots']);rows=[r for r in rows if r['root_id'] in ids]
    JOBS=rows;DEST=OUT/'prepared'/split;DEST.mkdir(parents=True,exist_ok=True)
    SOURCE=dict(manifest_sha256=sha(manifest),sources=identity(),split=split)
    start=time.perf_counter();reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(chunk,[(i,min(i+32,len(JOBS))) for i in range(0,len(JOBS),32)]):
            reports.append(r)
            print(json.dumps(dict(split=split,roots=sum(len(x['roots']) for x in reports),total=len(rows),seconds=time.perf_counter()-start)),flush=True)
    assert sum(len(r['roots']) for r in reports)==len(rows)
    dump(DEST/'manifest.json',dict(source=SOURCE,roots=len(rows),scenes=len({r['scene_id'] for r in rows}),shards=reports,
        sampled_cells=sum(r['rows'] for r in reports),seconds=time.perf_counter()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot_fit','pilot_holdout','full'],required=True);p.add_argument('--workers',type=int,default=12)
    prepare(**vars(p.parse_args()))
