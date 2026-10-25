"""Extract frozen-model probabilities once; exact hard-rule sufficient statistics."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
import os
import time
import numpy as np
from .common import (ROOT,OUT,EVALUATION,CHECKPOINT_SHA,THRESHOLDS,EDGES,
                     make_model,labels,confusion,save,digest)

MODEL=None
IDENTITY=None

def extraction_identity():
    files=[__file__,str(ROOT/'scripts/stage2_diagnostic/common.py')]
    for folder in ('stage2','stage2_quality','stage2_segment'):
        files.extend(str(p) for p in (ROOT/'eventwm_graph'/folder).glob('*.py'))
    return dict(protocol_sha256=digest(OUT/'protocol.json'),
        precision_contract_sha256=digest(OUT/'precision_contract.json'),
        source_sha256={str(__import__('pathlib').Path(p).relative_to(ROOT)):digest(p) for p in sorted(files)})

def initialize(device):
    global MODEL,IDENTITY
    MODEL=make_model(device)
    IDENTITY=extraction_identity()

def threshold_confusions(y,p):
    """One histogram gives all 13 exact >= threshold decisions, including ties."""
    reveal=(p[:,1]+p[:,2]).astype(np.float64); cls=(p[:,2]>p[:,1]).astype(np.int64)
    bucket=np.searchsorted(THRESHOLDS,reveal,side='right')
    h=np.bincount((y.astype(np.int64)*2+cls)*(len(THRESHOLDS)+1)+bucket,
                  minlength=6*(len(THRESHOLDS)+1)).reshape(3,2,-1)
    above=np.cumsum(h[:,:,::-1],axis=2)[:,:,::-1]
    total=np.bincount(y,minlength=3)
    result=np.zeros((len(THRESHOLDS),3,3),np.int64)
    for j in range(len(THRESHOLDS)):
        result[j,:,1:]=above[:,:,j+1]
        result[j,:,0]=total-result[j,:,1:].sum(1)
    return result

def probability_histogram(y,p):
    nb=len(EDGES)-1; hist=np.zeros((4,3,nb),np.int64)
    for variable in range(4):
        value=p[:,variable] if variable<3 else p[:,1]+p[:,2]
        b=np.clip(np.searchsorted(EDGES,value,side='right')-1,0,nb-1)
        hist[variable]=np.bincount(y.astype(np.int64)*nb+b,minlength=3*nb).reshape(3,nb)
    return hist

def window_statistics(ex,delta,original):
    p=delta['probabilities']; keys=delta['candidate_cells']; y=labels(keys,ex['target']);n=len(y)
    cm=confusion(y,p.argmax(1))
    np.testing.assert_array_equal(cm,np.asarray(original['candidate']['confusion']))
    if n!=original['candidate']['candidates']:raise ValueError('candidate domain cardinality changed')
    if (int((y==1).sum()),int((y==2).sum()))!=(original['new_free'],original['new_blocked']):raise ValueError('groundtruth changed')
    if original['candidate']['out_of_domain_positive']:raise ValueError('unexpected source-domain miss')
    truep=p[np.arange(n),y]; multi_ce=-np.log(np.clip(truep,1e-9,1))
    multi_br=np.square(p).sum(1)-2*truep+1
    np.testing.assert_allclose([multi_ce.mean(),multi_br.mean()],original['candidate']['scores'],atol=2e-7,rtol=1e-5)
    reveal=p[:,1]+p[:,2]; obs=y>0
    binary_br=(reveal.astype(np.float64)-obs)**2
    binary_ce=-np.log(np.clip(np.where(obs,reveal,1-reveal),1e-9,1))
    observed_count=int(obs.sum())
    bins=np.minimum((reveal*20).astype(np.int64),19)
    calibration=np.array([np.bincount(bins,minlength=20),np.bincount(bins,weights=reveal,minlength=20),np.bincount(bins,weights=obs,minlength=20)],np.float64)/max(n,1)
    # A fixed coarse world-relative grid measures expected spatial count error.
    pos=(keys.astype(np.float64)+.5)*float(ex['graph']['resolution'])
    center=ex['trajectory'][0,[1,3]]; displacement=pos-center
    g=np.clip(np.floor((displacement+12)/6).astype(int),0,3);grid=g[:,0]*4+g[:,1]
    expected_grid=np.bincount(grid,weights=reveal,minlength=16)
    true_grid=np.bincount(grid,weights=obs,minlength=16)
    identifier={k:ex[k] for k in ('query_id','query_aliases','start_node','end_node')}
    record=dict(identifier,groups=original['groups'],duration=original['duration'],poses=original['poses'],
        candidates=n,domain_sha256=__import__('hashlib').sha256(keys.astype('<i4').tobytes()).hexdigest(),
        confusion=np.concatenate([cm[None],threshold_confusions(y,p)]).tolist(),
        scores=[float(multi_ce.mean()),float(multi_br.mean()),float(binary_ce.mean()),float(binary_br.mean())],
        observed_scores=[float(multi_ce[obs].sum()),float(multi_br[obs].sum()),observed_count],
        true_counts=[observed_count,int((y==1).sum()),int((y==2).sum())],
        expected_counts=[float(reveal.sum(dtype=np.float64)),float(p[:,1].sum(dtype=np.float64)),float(p[:,2].sum(dtype=np.float64))],
        spatial_expected=expected_grid.tolist(),spatial_truth=true_grid.tolist(),
        spatial_count_l1=float(np.abs(expected_grid-true_grid).sum()),
        zero_new=not observed_count)
    return record,p.copy(),y,calibration

def root_worker(row):
    from eventwm_graph.stage2_segment.data import replay
    dest=OUT/'probability'/'roots'; dest.mkdir(parents=True,exist_ok=True)
    path=dest/(row['root_id']+'.json.gz')
    if path.exists():
        with gzip.open(path,'rt') as f:saved=json.load(f)
        if saved['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('incompatible resumed root')
        if saved['extraction_identity']!=IDENTITY:raise ValueError('diagnostic protocol/source changed during resume')
        if digest(dest/(row['root_id']+'.probabilities.npz'))!=saved['probability_cache_sha256']:raise ValueError('probability cache corrupt')
        if digest(dest/(row['root_id']+'.hist.npz'))!=saved['histogram_sha256']:raise ValueError('histogram cache corrupt')
        if digest(EVALUATION/(row['root_id']+'.json.gz'))!=saved['original_root_sha256']:raise ValueError('original evaluation changed')
        return {'root_id':row['root_id'],'seconds':0.,'segments':len(saved['records']),'resumed':True}
    with gzip.open(EVALUATION/(row['root_id']+'.json.gz'),'rt') as f:original=json.load(f)
    original_by_key={(r['start_node'],r['end_node']):r for r in original['records']}
    start=time.perf_counter(); records=[]; probabilities=[]; truths=[];offsets=[0]
    hist=np.zeros((4,3,len(EDGES)-1),np.float64);pooled=np.zeros_like(hist,dtype=np.int64);cal=np.zeros((3,20))
    for ex in replay(row,'mixed'):
        if 'audit' in ex:continue
        source=original_by_key[(ex['start_node'],ex['end_node'])]
        delta=MODEL.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        rec,p,y,c=window_statistics(ex,delta,source)
        records.append(rec);probabilities.append(p);truths.append(y);offsets.append(offsets[-1]+len(y))
        h=probability_histogram(y,p);pooled+=h;hist+=h/max(len(y),1);cal+=c
    if len(records)!=len(original['records']):raise ValueError('window identities changed')
    count=len(records);hist/=count;cal/=count
    cache=dest/(row['root_id']+'.probabilities.npz')
    np.savez(cache,p=np.concatenate(probabilities),y=np.concatenate(truths),offsets=np.asarray(offsets,np.int64))
    histfile=dest/(row['root_id']+'.hist.npz')
    np.savez_compressed(histfile,histogram=hist,pooled=pooled,calibration=cal)
    result=dict(root_id=row['root_id'],scene_id=row['scene_id'],checkpoint_sha256=CHECKPOINT_SHA,records=records,
        extraction_identity=IDENTITY,original_root_sha256=digest(EVALUATION/(row['root_id']+'.json.gz')),
        probability_cache_sha256=digest(cache),histogram_sha256=digest(histfile),seconds=time.perf_counter()-start,
        original_argmax_confusions_exact=True,original_domain_target_counts_exact=True)
    with gzip.open(path.with_suffix(path.suffix+'.tmp'),'wt') as f:json.dump(result,f,allow_nan=False)
    path.with_suffix(path.suffix+'.tmp').replace(path)
    return {'root_id':row['root_id'],'seconds':result['seconds'],'segments':count,'resumed':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    if not (OUT/'protocol.json').exists():raise ValueError('freeze protocol first')
    from eventwm_graph.stage2.data import source_rows
    rows,_=source_rows('dev')
    if args.limit:rows=rows[:args.limit]
    start=time.perf_counter();finished=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.device,)) as pool:
        jobs=[pool.submit(root_worker,row) for row in rows]
        for future in as_completed(jobs):
            result=future.result();finished.append(result)
            if len(finished)%20==0 or len(finished)==len(rows):
                print(json.dumps({'roots':len(finished),'total':len(rows),'segments':sum(r['segments'] for r in finished),'seconds':time.perf_counter()-start}),flush=True)
    save(OUT/'probability'/('complete.json' if not args.limit else 'pilot_complete.json'),
         dict(roots=len(finished),segments=sum(r['segments'] for r in finished),seconds=time.perf_counter()-start,
              workers=args.workers,device=args.device,checkpoint_sha256=CHECKPOINT_SHA,
              code_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),extraction_identity=extraction_identity()))

if __name__=='__main__':main()
