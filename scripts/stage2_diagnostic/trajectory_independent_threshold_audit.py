"""Independent direct-mask verification of frozen cached Stage2 threshold decisions.

No production/helper/scoring module is imported. No model is loaded. Inputs are
read only. Every saved window is compared with its authenticated audit record
and original frozen released-model confusion. Audit-only output is immutable.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/stage2_diagnostic_20261025'
SOURCE = BASE / 'probability/roots'
ORIGINAL = ROOT / 'output/stage2_segment_20261024/evaluation/final_segment_dev'
DEST = BASE / 'trajectory/independent_probability_audit_v1'
THETA = (.01,.02,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90)
RULES = ['argmax'] + [f'threshold_{v:.2f}' for v in THETA]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:return json.load(f)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')

def cm(y, predicted):
    return np.bincount(y.astype(np.int64)*3+predicted,minlength=9).reshape(3,3)

def verify_one(pathstr):
    path=Path(pathstr);r=read(path);rid=r['root_id']
    assert path.name == rid+'.json.gz'
    original=ORIGINAL/(rid+'.json.gz');old=read(original)
    assert sha(original)==r['original_root_sha256']
    assert old['root_id']==rid and old['scene_id']==r['scene_id']
    cache=SOURCE/(rid+'.probabilities.npz')
    assert sha(cache)==r['probability_cache_sha256']
    assert len(old['records'])==len(r['records'])
    matrices=[];zero=np.zeros((14,2),np.int64);cells=0
    with np.load(cache,allow_pickle=False) as z:
        p=z['p'];y=z['y'];offsets=z['offsets']
        assert p.dtype==np.float32 and y.dtype==np.uint8 and offsets.dtype==np.int64
        assert p.shape==(len(y),3) and len(offsets)==len(r['records'])+1
        assert offsets[0]==0 and offsets[-1]==len(y) and np.all(np.diff(offsets)>0)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.max(abs(p.sum(1)-1))<=2e-6
        assert np.all(y<3)
        for i,(record,reference) in enumerate(zip(r['records'],old['records'])):
            for key in ('query_id','query_aliases','start_node','end_node','duration','poses'):
                assert record[key]==reference[key],(rid,i,key)
            start,end=map(int,offsets[i:i+2]);pi=p[start:end];yi=y[start:end]
            assert len(yi)==record['candidates']
            own=np.empty((14,3,3),np.int64)
            own[0]=cm(yi,np.argmax(pi,axis=1))
            np.testing.assert_array_equal(own[0],reference['candidate']['confusion'])
            # Crucial precision contract: float32 add first, then float64 compare.
            revelation=(pi[:,1]+pi[:,2]).astype(np.float64)
            conditional=np.where(pi[:,1]>=pi[:,2],1,2)
            for j,threshold in enumerate(THETA,1):
                prediction=np.where(revelation>=np.float64(threshold),conditional,0)
                own[j]=cm(yi,prediction)
            np.testing.assert_array_equal(own,np.asarray(record['confusion'],np.int64))
            np.testing.assert_array_equal(own.sum(2),np.broadcast_to(np.bincount(yi,minlength=3),(14,3)))
            assert int(np.sum(yi>0))==int(record['true_counts'][0])
            if not np.any(yi):
                n=own[:,:,1:].sum((1,2));zero[:,0]+=(n==0);zero[:,1]+=n
            matrices.append(own);cells+=len(yi)
    matrices=np.asarray(matrices)
    return dict(root_id=rid,scene_id=r['scene_id'],windows=len(matrices),cells=cells,
                root_mean=matrices.mean(0).tolist(),pooled=matrices.sum(0).tolist(),zero=zero.tolist(),
                record_sha256=sha(path),cache_sha256=r['probability_cache_sha256'],original_sha256=r['original_root_sha256'])

def divide(a,b):return float(a/b) if b else 0.

def rates(c):
    c=np.asarray(c,np.float64);result={}
    for i,label in enumerate(('unknown','free','blocked')):
        tp=c[i,i];denp=c[:,i].sum();denr=c[i].sum()
        result[label]={'precision':divide(tp,denp),'recall':divide(tp,denr),'f1':divide(2*tp,denp+denr)}
    exact=c[1,1]+c[2,2];actual=c[1:,:].sum();predicted=c[:,1:].sum();both=c[1:,1:].sum()
    result.update(label_aware_delta_iou=divide(exact,actual+predicted-exact),
                  revelation_iou=divide(both,actual+predicted-both),
                  revelation_precision=divide(both,predicted),revelation_recall=divide(both,actual),
                  revelation_f1=divide(2*both,actual+predicted),
                  terminal_label_agreement_on_true_revelations=divide(exact,actual),
                  missed_revelations=float(c[1:,0].sum()),incorrect_free_blocked=float(c[1,2]+c[2,1]),
                  false_revelations=float(c[0,1:].sum()),false_revelation_rate=divide(c[0,1:].sum(),c[0].sum()),
                  wrong_free_blocked_rate=divide(c[1,2]+c[2,1],actual),
                  false_free_unknown_count=float(c[0,1]),false_free_blocked_count=float(c[2,1]),
                  false_free_unknown_fraction=divide(c[0,1],c[:,1].sum()),
                  false_free_blocked_fraction=divide(c[2,1],c[:,1].sum()),
                  false_free_on_actual_blocked_rate=divide(c[2,1],c[2].sum()))
    return result

def compare_tree(actual,expected):
    differences=[]
    for k,v in actual.items():
        if isinstance(v,dict):differences.extend(compare_tree(v,expected[k]))
        else:
            delta=abs(v-expected[k]);assert delta<=1e-12,(k,v,expected[k],delta)
            differences.append(delta)
    return differences

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    assert 1<=args.workers<=4
    DEST.mkdir(parents=True,exist_ok=False)
    protocol=read(BASE/'protocol.json');precision=read(BASE/'precision_contract.json');summary=read(BASE/'probability/summary.json')
    assert tuple(protocol['thresholds'])==THETA
    paths=sorted(SOURCE.glob('*.json.gz'));assert len(paths)==2110
    identity=dict(source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),precision_contract_sha256=sha(BASE/'precision_contract.json'),
                  probability_summary_sha256=sha(BASE/'probability/summary.json'),original_metrics_sha256=sha(ORIGINAL/'metrics.json'),
                  expected_roots=2110,expected_scenes=108,expected_windows=47128,workers=args.workers,
                  method='Independent numpy direct masks on every F32 cached probability and label; no production threshold or histogram code imported',
                  neural_inference=False,training=False,production_changed=False)
    write(DEST/'identity.json',identity);start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(verify_one,str(path)) for path in paths]
        for f in as_completed(jobs):
            r=f.result();records.append(r)
            write(DEST/'roots'/(r['root_id']+'.json'),r)
            if len(records)%100==0 or len(records)==len(paths):print(json.dumps(dict(roots=len(records),seconds=time.perf_counter()-start)),flush=True)
    records.sort(key=lambda x:x['root_id']);scene=defaultdict(list)
    for r in records:scene[r['scene_id']].append(np.asarray(r['root_mean']))
    names=sorted(scene);arrays=np.asarray([np.mean(scene[s],axis=0) for s in names]);allmean=arrays.mean(0)
    actual={rule:rates(matrix) for rule,matrix in zip(RULES,allmean)}
    delta=compare_tree(actual,summary['rules'])
    pooled=np.sum([r['pooled'] for r in records],axis=0);zero=np.sum([r['zero'] for r in records],axis=0)
    for j,rule in enumerate(RULES):
        np.testing.assert_array_equal(pooled[j],summary['pooled_confusions'][rule])
        assert int(zero[j,0])==summary['zero_new_observation'][rule]['exact_empty']
        assert int(zero[j,1])==summary['zero_new_observation'][rule]['false_cells']
    selection=[i for i,s in enumerate(names) if s in protocol['threshold_selection_scenes']]
    validation=[i for i,s in enumerate(names) if s in protocol['diagnostic_validation_scenes']]
    assert len(selection)==70 and len(validation)==38 and set(selection).isdisjoint(validation)
    val={rule:rates(c) for rule,c in zip(RULES,arrays[validation].mean(0))}
    delta+=compare_tree(val,summary['diagnostic_scene_validation']['metrics']['rules'])
    selected={}
    for cohort,m in [('all',allmean),('selection70',arrays[selection].mean(0))]:
        candidates=[rates(c) for c in m[1:]]
        selected[cohort]={metric:RULES[1+int(np.argmax([r[field] for r in candidates]))] for metric,field in [('graphdelta_iou','label_aware_delta_iou'),('revelation_f1','revelation_f1')]}
    for metric,rule in selected['all'].items():assert rule==summary['development_selected'][metric]
    assert selected['selection70']==summary['diagnostic_scene_validation']['selected_rules']
    assert len(names)==108 and sum(r['windows'] for r in records)==47128
    assert sha(__file__)==identity['source_sha256'] and sha(BASE/'probability/summary.json')==identity['probability_summary_sha256']
    np.savez_compressed(DEST/'independent_scene_confusions.npz',scene_ids=np.asarray(names),matrices=arrays)
    result=dict(status='PASS',identity=identity,roots=len(records),scenes=len(names),windows=sum(r['windows'] for r in records),
                evaluated_cells=sum(r['cells'] for r in records),all_14_per_window_confusions_exact=True,
                original_argmax_per_window_confusions_exact=True,all_cache_hashes_verified=True,
                all_pooled_confusions_exact=True,all_zero_new_counts_exact=True,
                maximum_summary_absolute_difference=max(delta),rules=actual,validation_rules=val,selected_rules=selected,
                all_iou_gain_ratio=actual['threshold_0.20']['label_aware_delta_iou']/actual['argmax']['label_aware_delta_iou'],
                elapsed_seconds=time.perf_counter()-start,
                scope='Verifies decisions and score aggregation from authenticated saved p/y; does not independently rerun inference or reconstruct legal targets. Earlier independent replay reviews cover target/source causality. 38 scenes are diagnostic validation, not historically untouched confirmation.')
    write(DEST/'proof.json',result);print(json.dumps({k:result[k] for k in ('status','roots','scenes','windows','evaluated_cells','maximum_summary_absolute_difference','all_iou_gain_ratio','elapsed_seconds')}),flush=True)

if __name__=='__main__':main()
