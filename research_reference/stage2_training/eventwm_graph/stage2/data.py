"""All eligible Root/Query branches; legal observations are the only targets.

UNKNOWN negatives and newly observed positives are sampled by independent
strata. Exact inverse-inclusion weights retain the full candidate-cell proper
score and equal Root weighting. Complete sparse targets remain reproducible
from the immutable legal DAG; no privileged file is opened by this module.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import FINAL,OLD,OUT,SCHEMA,dump,seed,sha,source_identity
from .features import SceneFeatures

JOBS=None
DEST=None
IDENTITY=None
SAMPLES_PER_STRATUM=64


def read_member(folder,name,expected=None):
    data=(Path(folder)/name).read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('source checksum mismatch: '+name)
    return json.loads(gzip.decompress(data))


def branches(row):
    """Replay actual executed ancestry with the unchanged cell-update rules.

    We retain final cell values only. Robot support at an occupied cell is
    ignored, occupied beats free within a frame, subsequent free may clear it.
    Full PartialGraph replay equivalence is separately tested and audited.
    """
    folder=Path(row['folder']);files=row['files']
    legal=read_member(folder,'legal_root.json.gz',files['legal_root.json.gz'])
    queries=read_member(folder,'queries.json.gz',files['queries.json.gz'])
    outcomes=read_member(folder,'outcomes.json.gz',files['outcomes.json.gz'])
    nodes=read_member(folder,'future_observations.json.gz',files['future_observations.json.gz'])
    graph=legal['partial_graph'];initial={tuple(c['cell']):c for c in graph['cells']}
    if len(queries)!=len(outcomes) or len(queries)!=row['queries']:raise ValueError('Query/outcome cardinality mismatch')
    if float(legal['sensor']['max_range_m'])>10.00001:raise ValueError('unreviewed sensor range')
    for query,outcome in zip(queries,outcomes):
        if query['query_id']!=outcome['query_id']:raise ValueError('Query/outcome identity mismatch')
        start=legal['history'][-1];trajectory=[[float(start['timestamp']),*start['robot_pose']]]
        cells={};parent=None;known_changes=set()
        for node_id in outcome['node_ids']:
            node=nodes[node_id];ob=node['observation'];now=float(ob['timestamp'])
            if node['node_id']!=node_id or node['parent']!=parent or now<=trajectory[-1][0]:raise ValueError('invalid executed ancestry/clock')
            parent=node_id;trajectory.append([now,*ob['robot_pose']])
            occupied={tuple(u['cell']) for u in ob['static_updates'] if u['state']==2}
            for update in ob['static_updates']:
                if update['timestamp']!=now:raise ValueError('noncausal static update')
                key=tuple(update['cell']);state=int(update['state'])
                if update['source']=='occupied_ego_base_support' or (state==1 and key in occupied):continue
                if state not in (1,2) or not np.isfinite(update['height']):raise ValueError('invalid observed cell')
                cells[key]=(state,float(update['height']))
                if key in initial and (initial[key]['state']!=state or abs(initial[key]['height']-update['height'])>1e-8):known_changes.add(key)
        novel={key:value for key,value in cells.items() if key not in initial}
        yield dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=query['query_id'],graph=graph,
            trajectory=np.asarray(trajectory,np.float64),target=novel,known_changes=len(known_changes),
            termination=outcome['summary']['termination'],queries_in_root=len(queries),executed_nodes=len(outcome['node_ids']))


def sampling(example):
    context=SceneFeatures(example['graph'],example['trajectory']);keys=context.candidates();n=len(keys)
    if not n:return None,dict(empty_domains=1)
    target=example['target'];target_keys=np.asarray(list(target),np.int32).reshape(-1,2)
    # Candidate keys retain row-major order. Vectorized membership avoids a
    # Python dictionary for tens of thousands of UNKNOWN cells per Query.
    codes=(keys[:,0]-context.lo[0])*int(context.shape[1])+keys[:,1]-context.lo[1]
    target_codes=(target_keys[:,0]-context.lo[0])*int(context.shape[1])+target_keys[:,1]-context.lo[1]
    positions=np.searchsorted(codes,target_codes);safe=np.minimum(positions,n-1)
    valid=(positions<n)&np.all(keys[safe]==target_keys,axis=1)
    positive=positions[valid];is_positive=np.zeros(n,bool);is_positive[positive]=True
    negative=np.flatnonzero(~is_positive);rng=np.random.default_rng(seed(example['root_id'],example['query_id']))
    selections=[];weights=[]
    for stratum in (positive,negative):
        count=min(SAMPLES_PER_STRATUM,len(stratum))
        if not count:continue
        selections.extend(rng.choice(stratum,count,replace=False))
        weights.extend([len(stratum)/(count*n*example['queries_in_root'])]*count)
    chosen=keys[selections];x,baseline=context.transform(chosen)
    y=np.asarray([target.get(tuple(key),(0,0.))[0] for key in chosen],np.uint8)
    height=np.asarray([target.get(tuple(key),(0,float(b)))[1]-float(b) for key,b in zip(chosen,baseline)],np.float32)
    if np.any(abs(x)>np.finfo(np.float16).max) or np.any(abs(height)>np.finfo(np.float16).max):raise ValueError('sample encoding range exceeded')
    return dict(x=x.astype(np.float16),y=y,height=height.astype(np.float16),weight=np.asarray(weights,np.float32)),dict(
        queries=1,executed_nodes=example['executed_nodes'],candidate_cells=n,new_free=sum(v[0]==1 for v in target.values()),
        new_blocked=sum(v[0]==2 for v in target.values()),out_of_domain_positive=len(target)-len(positive),
        changed_known_cells=example['known_changes'],sampled_cells=len(chosen),
        zero_execution=int(example['executed_nodes']==0),substantial_information=int(len(target)>=100))


def chunk(bounds):
    begin,end=bounds;tag='%06d_%06d'%(begin,end);report=DEST/(tag+'.json')
    if report.exists():
        old=json.loads(report.read_text())
        if old['identity']!=IDENTITY or sha(DEST/(tag+'.npz'))!=old['sha256']:raise ValueError('Stage2 resume identity changed')
        return old
    arrays=collections.defaultdict(list);counts=collections.Counter();roots=[];start=time.perf_counter()
    for row in JOBS[begin:end]:
        rootcounts=collections.Counter()
        for example in branches(row):
            sample,ledger=sampling(example);rootcounts.update(ledger)
            if sample:
                for key,value in sample.items():arrays[key].append(value)
        counts.update(rootcounts);roots.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],counts=dict(rootcounts)))
    path=DEST/(tag+'.npz');tmp=DEST/(tag+'.tmp.npz')
    np.savez(tmp,**{k:np.concatenate(v,axis=0) for k,v in arrays.items()});tmp.replace(path)
    value=dict(identity=IDENTITY,path=str(path),sha256=sha(path),counts=dict(counts),roots=roots,seconds=time.perf_counter()-start)
    dump(report,value);return value


def source_rows(split):
    if split=='train':
        manifest=json.loads(FINAL.read_text());rows=manifest['rows']
        if any(r['split']!='train' or r['source_split']!='train' for r in rows):raise ValueError('protected/nontraining data in fitting snapshot')
        return rows,FINAL
    if split=='dev':
        path=OLD/'snapshot/manifest.json';m=json.loads(path.read_text())
        return [dict(r,folder=str(OLD/'snapshot/roots'/r['root_id'])) for r in m['rows'] if r['split']=='dev'],path
    raise ValueError('protected payloads are unavailable to Stage2')


def prepare(split='train',workers=4):
    global JOBS,DEST,IDENTITY
    JOBS,manifest=source_rows(split);DEST=OUT/'prepared'/split;DEST.mkdir(parents=True,exist_ok=True)
    IDENTITY=dict(schema=SCHEMA,split=split,snapshot=str(manifest),snapshot_sha256=sha(manifest),sources=source_identity(),
        samples_per_stratum=SAMPLES_PER_STRATUM,objective='inverse inclusion weighted full-domain multiclass log score; equal Root and within-Root Query weight',
        input_files=['legal_root.json.gz'],target_files=['queries.json.gz','outcomes.json.gz','future_observations.json.gz'],
        privileged_files_opened=False)
    identity_path=DEST/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=IDENTITY:raise ValueError('Stage2 preparation source changed during run')
    dump(identity_path,IDENTITY);reports=[];counts=collections.Counter();start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(chunk,[(i,min(i+100,len(JOBS))) for i in range(0,len(JOBS),100)]):
            reports.append(report);counts.update(report['counts']);roots=sum(len(r['roots']) for r in reports)
            progress=dict(roots=roots,total_roots=len(JOBS),counts=dict(counts),seconds=time.perf_counter()-start)
            dump(DEST/'progress.json',progress);print(json.dumps(progress),flush=True)
    rootrows=[r for report in reports for r in report['roots']]
    if len(rootrows)!=len(JOBS) or len({r['root_id'] for r in rootrows})!=len(JOBS):raise ValueError('incomplete Stage2 Root accounting')
    dump(DEST/'manifest.json',dict(schema=SCHEMA,identity=IDENTITY,roots=len(JOBS),scenes=len({r['scene_id'] for r in rootrows}),counts=dict(counts),
        shards=[{k:r[k] for k in ('path','sha256','counts','roots')} for r in reports],failures=[],seconds=time.perf_counter()-start))
    for p in DEST.iterdir():
        if p.is_file():p.chmod(0o444)
    DEST.chmod(0o555)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','dev'],default='train');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();prepare(**vars(args))
