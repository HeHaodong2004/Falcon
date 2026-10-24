"""Unchanged scene/Root boundary; full candidate domain and every dev Query."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import time
import numpy as np
from .api import Stage2Predictor,apply_graph_delta
from .common import OUT,SCHEMA,dump,sha,source_identity
from .data import branches,source_rows


def target_arrays(example,keys):
    n=len(keys);y=np.zeros(n,np.uint8);h=np.zeros(n,np.float64)
    target=example['target'];tk=np.asarray(list(target),np.int32).reshape(-1,2)
    if not n:return y,h,list(target)
    low=keys.min(0);span=int(keys[:,1].max()-low[1]+1)
    codes=(keys[:,0].astype(np.int64)-low[0])*span+keys[:,1]-low[1]
    tc=(tk[:,0].astype(np.int64)-low[0])*span+tk[:,1]-low[1];at=np.searchsorted(codes,tc);safe=np.minimum(at,n-1)
    valid=(at<n)&np.all(keys[safe]==tk,axis=1)
    values=list(target.values())
    y[at[valid]]=[values[i][0] for i in np.flatnonzero(valid)];h[at[valid]]=[values[i][1] for i in np.flatnonzero(valid)]
    return y,h,[tuple(key) for key in tk[~valid]]


def rates(matrix):
    result={}
    for state,name in [(1,'new_free'),(2,'new_blocked')]:
        tp=float(matrix[state,state]);actual=float(matrix[state].sum());predicted=float(matrix[:,state].sum())
        p=tp/predicted if predicted else 0.;r=tp/actual if actual else 0.
        result[name]=dict(precision=p,recall=r,f1=2*tp/(actual+predicted) if actual+predicted else 0.,true_positive=tp,actual=actual,predicted=predicted)
    actual=float(matrix[1:,:].sum());predicted=float(matrix[:,1:].sum())
    intersection=float(np.trace(matrix[1:,1:]));union=actual+predicted-intersection
    result['delta_cell_iou']=intersection/union if union else 1.
    reveal_intersection=float(matrix[1:,1:].sum());reveal_union=actual+predicted-reveal_intersection
    result['reveal_set_iou']=reveal_intersection/reveal_union if reveal_union else 1.
    return result


def evaluate(checkpoint,name='final_dev'):
    dest=OUT/'evaluation'/name;dest.mkdir(parents=True,exist_ok=False)
    rows,source=source_rows('dev');predictor=Stage2Predictor(checkpoint)
    dump(dest/'protocol.json',dict(schema=SCHEMA,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source=str(source),source_sha256=sha(source),
        sources=source_identity(),roots=len(rows),scenes=len({r['scene_id'] for r in rows}),selection='all original development Roots, every stored Query, no outcome cap',
        substantial_information='at least100 truly newly observed cells, predeclared',decision='argmax of raw3-class probabilities; no development threshold tuning',
        bootstrap=dict(unit='scene',draws=2000,seed=2411),unknown='initially absent and unobserved in legal terminal branch; outside candidate domain predicts UNKNOWN'))
    scene_matrices=collections.defaultdict(lambda:np.zeros((3,3),np.float64));total=np.zeros((3,3),np.int64);substantial=np.zeros((3,3),np.int64)
    totals=collections.Counter();reliability=np.zeros((3,15,3),np.float64);support=collections.defaultdict(set)
    logscore=0.;brier=0.;baseline_brier=0.;weight_sum=0.;height_sum=0.;height_count=0
    start=time.perf_counter()
    with gzip.open(dest/'queries.jsonl.gz','wt') as stream:
        for root_index,row in enumerate(rows):
            rootmatrix=np.zeros((3,3),np.float64)
            for example in branches(row):
                before=time.perf_counter();delta=predictor.predict(partial_graph=example['graph'],trajectory=example['trajectory']);predict_seconds=time.perf_counter()-before
                y,truth_height,miss=target_arrays(example,delta['candidate_cells']);states=delta['states'];p=delta['probabilities'];n=len(y)
                matrix=np.bincount(3*y.astype(np.int64)+states,minlength=9).reshape(3,3)
                for key in miss:matrix[example['target'][key][0],0]+=1
                total+=matrix;rootmatrix+=matrix/example['queries_in_root']
                is_substantial=len(example['target'])>=100
                if is_substantial:substantial+=matrix;support['substantial_roots'].add(row['root_id']);support['substantial_scenes'].add(row['scene_id'])
                denom=n+len(miss);w=1/(example['queries_in_root']*max(denom,1))
                if denom:
                    ce=-np.log(np.clip(p[np.arange(n),y],1e-9,1.));squared=np.square(p).sum(-1)-2*p[np.arange(n),y]+1
                    logscore+=w*(float(ce.sum())+len(miss)*-np.log(1e-9));brier+=w*(float(squared.sum())+2*len(miss))
                    baseline_brier+=w*(2*np.count_nonzero(y)+2*len(miss));weight_sum+=w*denom
                    for state in range(3):
                        bins=np.minimum((p[:,state]*15).astype(int),14)
                        reliability[state,:,0]+=w*np.bincount(bins,minlength=15)
                        reliability[state,:,1]+=w*np.bincount(bins,weights=p[:,state],minlength=15)
                        reliability[state,:,2]+=w*np.bincount(bins,weights=y==state,minlength=15)
                        if miss:
                            bin_index=14 if state==0 else 0
                            reliability[state,bin_index,0]+=w*len(miss)
                            reliability[state,bin_index,1]+=w*len(miss)*(state==0)
                            reliability[state,bin_index,2]+=w*sum(example['target'][k][0]==state for k in miss)
                    known=y>0;height_sum+=float(np.abs(delta['heights'][known]-truth_height[known]).sum());height_count+=int(known.sum())
                before=time.perf_counter();future=apply_graph_delta(example['graph'],delta);update_seconds=time.perf_counter()-before
                expected={tuple(c['cell']):c for c in example['graph']['cells']};observed={tuple(c['cell']):c for c in future['cells']}
                if len(observed)!=len(expected)+len(delta['static_updates']) or any(observed[k]!=v for k,v in expected.items()):raise ValueError('deterministic update changed existing map')
                for cell in delta['static_updates']:
                    actual=observed[tuple(cell['cell'])]
                    if actual['state']!=cell['state'] or actual['height']!=cell['height']:raise ValueError('graph update state/height mismatch')
                totals.update(queries=1,candidate_cells=n,new_cells=len(example['target']),out_of_domain_positive=len(miss),changed_known_cells=example['known_changes'],
                    graph_update_correct=1,substantial_queries=int(is_substantial),zero_execution=int(example['executed_nodes']==0))
                record=dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=example['query_id'],termination=example['termination'],
                    duration=float(example['trajectory'][-1,0]-example['trajectory'][0,0]),executed_nodes=example['executed_nodes'],new_cells=len(example['target']),
                    candidate_cells=n,confusion=matrix.tolist(),predict_seconds=predict_seconds,apply_seconds=update_seconds,**rates(matrix))
                stream.write(json.dumps(record)+'\n')
            scene_matrices[row['scene_id']]+=rootmatrix/sum(r['scene_id']==row['scene_id'] for r in rows)
            if root_index%25==0:
                progress=dict(roots=root_index+1,total_roots=len(rows),queries=totals['queries'],seconds=time.perf_counter()-start)
                print(json.dumps(progress),flush=True);dump(dest/'progress.json',progress)
    matrices=np.array(list(scene_matrices.values()));balanced=rates(matrices.mean(0));rng=np.random.default_rng(2411)
    samples=[rates(matrices[rng.integers(0,len(matrices),len(matrices))].mean(0)) for _ in range(2000)]
    for metric in ('new_free','new_blocked'):
        for stat in ('precision','recall','f1'):
            balanced[metric][stat+'_ci95']=np.quantile([s[metric][stat] for s in samples],[.025,.975]).tolist()
    calibration={}
    for state,name in enumerate(('unknown','new_free','new_blocked')):
        mass,prob,truth=reliability[state].T;valid=mass>0
        calibration[name]=dict(ece=float(np.abs(prob-truth).sum()/max(mass.sum(),1e-12)),
            bins=[dict(lower=i/15,upper=(i+1)/15,root_weight=float(mass[i]),probability=float(prob[i]/mass[i]),frequency=float(truth[i]/mass[i])) for i in np.flatnonzero(valid)])
    result=dict(schema=SCHEMA,status='RESEARCH_EVALUATION_NO_RELEASE_ACCEPTANCE_CLAIM',checkpoint_sha256=sha(checkpoint),roots=len(rows),scenes=len(matrices),
        counts=dict(totals),pooled=rates(total),pooled_confusion=total.tolist(),scene_balanced=balanced,
        substantial_information=dict(**rates(substantial),confusion=substantial.tolist(),roots=len(support['substantial_roots']),scenes=len(support['substantial_scenes'])),
        multiclass_logloss=logscore/weight_sum,multiclass_brier=brier/weight_sum,persist_unknown_brier=baseline_brier/weight_sum,
        calibration=calibration,height_mae_on_observed_cells=height_sum/max(height_count,1),height_support=height_count,
        graph_update_correctness=totals['graph_update_correct']/totals['queries'],seconds=time.perf_counter()-start,
        limitations=['Initial UNKNOWN-cell reveal only; changed known-cell observations counted, not modeled.',
            'Jaw-camera pose and dynamic-human occluders absent from requested inputs; base-pose conditioning retains irreducible uncertainty.',
            'Historical development cohort is previously used research data, not fresh protected confirmation.',
            'Observation-limited robot configuration-space map; no hidden geometry reconstruction or physical safety certification.'])
    dump(dest/'metrics.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--name',default='final_dev');args=p.parse_args();evaluate(**vars(args))
