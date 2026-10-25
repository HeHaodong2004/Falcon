"""All original development windows, exact candidate identities and full graphs.

The original baseline confusions are reused only after matching checkpoint,
window identity, exact domain digest and target counts to fresh legal replay.
No confirmation payload is opened. Final operating point is fixed at .20.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .predictor import make_predictor
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.evaluate import groups
from scripts.stage2_diagnostic.common import extended_rates

BASELINE=Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-segment-20261024/output/stage2_diagnostic_20261025/probability/roots')
MODEL=None;DEST=None;CHECKPOINT_SHA=None


def codes(keys):
    k=(np.asarray(keys,np.int64)+np.int64(2147483648)).astype(np.uint64)
    return (k[:,0]<<np.uint64(32))|k[:,1]


def labels(keys,target):
    y=np.zeros(len(keys),np.uint8);heights=np.zeros(len(keys),np.float64)
    if not target:return y,heights,[]
    tk=np.asarray(list(target),np.int32);tc=codes(tk);kc=codes(keys);at=np.searchsorted(kc,tc);valid=at<len(kc)
    if len(kc):valid&=kc[np.minimum(at,len(kc)-1)]==tc
    values=list(target.values());y[at[valid]]=[values[i][0]for i in np.flatnonzero(valid)];heights[at[valid]]=[values[i][1]for i in np.flatnonzero(valid)]
    return y,heights,[values[i]for i in np.flatnonzero(~valid)]


def initialize(checkpoint,dest,device,global_features=False):
    global MODEL,DEST,CHECKPOINT_SHA
    torch.set_num_threads(1);MODEL=make_predictor(checkpoint,device=device,threshold=.2,global_features=global_features);DEST=Path(dest);CHECKPOINT_SHA=sha(checkpoint)


def worker(row):
    path=DEST/(row['root_id']+'.json.gz')
    if path.exists():
        value=__import__('json').loads(gzip.decompress(path.read_bytes()))
        if value['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('stale evaluation receipt')
        return value
    frozen=__import__('json').loads(gzip.decompress((BASELINE/(row['root_id']+'.json.gz')).read_bytes()))
    assert frozen['checkpoint_sha256']=='1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd'
    old={(e['start_node'],e['end_node']):e for e in frozen['records']};records=[];audit=None;tick=time.perf_counter()
    with np.load(BASELINE/(row['root_id']+'.probabilities.npz'),allow_pickle=False)as z:old_y=z['y'];old_offsets=z['offsets']
    old_slices={(e['start_node'],e['end_node']):slice(int(old_offsets[i]),int(old_offsets[i+1]))for i,e in enumerate(frozen['records'])}
    initialization=member(row,'privileged_initialization.json.gz');human_count=initialization['episode']['info']['human_num']
    # Offline cohort metadata is intentionally read after model construction;
    # neither this count nor future camera truth is passed into inference.
    for ex in replay(row,'mixed'):
        if 'audit'in ex:audit=ex['audit'];continue
        delta=MODEL.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);keys=delta['candidate_cells'];p=delta['probabilities'];hard=delta['states']
        y,th,missing=labels(keys,ex['target']);prev=old[(ex['start_node'],ex['end_node'])]
        assert hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest()==prev['domain_sha256']
        assert len(keys)==prev['candidates'] and np.count_nonzero(y)+len(missing)==prev['true_counts'][0]
        np.testing.assert_array_equal(y,old_y[old_slices[(ex['start_node'],ex['end_node'])]])
        cm=np.bincount(3*y.astype(np.int64)+hard,minlength=9).reshape(3,3)
        for s,h in missing:cm[s,0]+=1
        truthp=p[np.arange(len(y)),y];scores=[float(-np.log(np.clip(truthp,1e-9,1)).mean()),float(np.mean(np.square(p).sum(1)-2*truthp+1))]
        # Verify every materialized graph's actual numerical cell columns.
        future=delta['future_partial_graph'];columns=future.cell_columns;prepared=MODEL.prepare_graph(ex['graph']);base=prepared.future_base.table
        fc=codes(columns['cell']);ki=np.searchsorted(fc,codes(base.keys));np.testing.assert_array_equal(columns['cell'][ki],base.keys)
        for name,expected in [('state',base.states),('height',base.heights),('first_seen',base.first),('last_seen',base.last)]:np.testing.assert_array_equal(columns[name][ki],expected)
        chosen=np.flatnonzero(hard);ni=np.searchsorted(fc,codes(keys[chosen]));np.testing.assert_array_equal(columns['cell'][ni],keys[chosen]);np.testing.assert_array_equal(columns['state'][ni],hard[chosen])
        assert len(columns['cell'])==len(base.keys)+len(chosen) and future['hypothetical'] is True
        correct=(y>0)&(hard==y);height=[float(abs(delta['heights'][correct]-th[correct]).sum()),int(correct.sum())]
        records.append(dict(start_node=ex['start_node'],end_node=ex['end_node'],cm=cm.tolist(),baseline_cm=prev['confusion'][6],scores=scores,
            groups=groups(ex),new_cells=len(ex['target']),poses=len(ex['trajectory']),empty_prediction=not bool(len(chosen)),
            candidates=len(keys),out_of_domain=len(missing),known_terminal_differences=ex['known_terminal_differences'],height_error=height,future_verified=True))
    assert audit is not None and len(records)==len(old)
    value=dict(root_id=row['root_id'],scene_id=row['scene_id'],human_count=human_count,checkpoint_sha256=CHECKPOINT_SHA,records=records,audit=audit,seconds=time.perf_counter()-tick)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gzip.compress(__import__('json').dumps(value,separators=(',',':')).encode()));return value


def summarize(roots):
    scenes=defaultdict(list);pooled=np.zeros((2,3,3),np.int64);zero=defaultdict(int);strata=defaultdict(list);known=0;outside=0;height=np.zeros(2);count=0
    for row in roots:
        rcm=[]
        for e in row['records']:
            cm=np.array([e['baseline_cm'],e['cm']],np.int64);rcm.append(cm);pooled+=cm;count+=1;known+=e['known_terminal_differences'];outside+=e['out_of_domain'];height+=e['height_error']
            for label in e['groups']:strata[label].append(cm)
            strata['human:zero'if row['human_count']==0 else'human:present'].append(cm)
            if not e['new_cells']:
                zero['windows']+=1;zero['candidate_exact_empty']+=e['empty_prediction'];zero['baseline_exact_empty']+=int(np.asarray(e['baseline_cm'])[:,1:].sum()==0)
        scenes[row['scene_id']].append(np.mean(rcm,axis=0))
    scene_cm=np.array([np.mean(v,axis=0)for k,v in sorted(scenes.items())]);mean=scene_cm.mean(0)
    metric=lambda cm:{name:extended_rates(c)for name,c in zip(['baseline','candidate'],cm)}
    rng=np.random.default_rng(2526);draws=[]
    for _ in range(2000):
        sample=scene_cm[rng.integers(0,len(scene_cm),len(scene_cm))].mean(0);rr=metric(sample);draws.append([rr['baseline']['label_aware_delta_iou'],rr['candidate']['label_aware_delta_iou'],rr['candidate']['label_aware_delta_iou']-rr['baseline']['label_aware_delta_iou']])
    return dict(roots=len(roots),scenes=len(scenes),windows=count,rules='Original equalScene/Root/window mean confusion -> rates; pooled is raw cell counts, no aggregation change.',
        hierarchical=metric(mean),pooled=metric(pooled),pooled_confusion=pooled.tolist(),zero_new=dict(zero),out_of_domain_positives=outside,
        known_terminal_differences=known,height_mae_on_correct_predicted_new_cells=float(height[0]/max(height[1],1)),
        bootstrap_95=dict(zip(['baseline','candidate','paired_delta'],np.quantile(draws,[.025,.975],axis=0).T.tolist())),
        pooled_strata={k:dict(windows=len(v),**metric(np.sum(v,axis=0)))for k,v in sorted(strata.items())})


def evaluate(checkpoint,workers=4,device='cuda:1',name='selected',smoke=False,global_features=False):
    dest=OUT/'full_evaluation'/name;dest.mkdir(parents=True,exist_ok=True);rows,manifest=source_rows('dev');tick=time.perf_counter();completed=[]
    if smoke:
        ids=set(read(HF/'protocol.json')['pilot_selection_roots'][:2]);rows=[r for r in rows if r['root_id']in ids]
    identity=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_manifest_sha256=sha(manifest),source_sha256=sha(__file__),threshold=.2,workers=workers,device=device,original_roots=len(rows),smoke_only=smoke,global_features=global_features)
    if (dest/'identity.json').exists():
        previous=read(dest/'identity.json')
        if previous['checkpoint_sha256']!=identity['checkpoint_sha256']or previous.get('global_features',False)!=global_features:raise ValueError('evaluation identity changed; use a fresh destination')
    save(dest/'identity.json',identity)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(checkpoint),str(dest),device,global_features))as pool:
        jobs={pool.submit(worker,row):row['root_id']for row in rows}
        for f in as_completed(jobs):
            try:completed.append(f.result())
            except BaseException as err:save(dest/'FAILURE.json',dict(root_id=jobs[f],error=repr(err)));raise
            if len(completed)%32==0 or len(completed)==len(rows):
                progress=dict(roots=len(completed),total=len(rows),seconds=time.perf_counter()-tick);save(dest/'progress.json',progress);print(progress,flush=True)
    result=summarize(completed)
    if not smoke:assert result['roots']==2110 and result['windows']==47128
    result.update(checkpoint_sha256=sha(checkpoint),threshold=.2,seconds=time.perf_counter()-tick);save(dest/'summary.json',result);print(result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='cuda:1');p.add_argument('--name',default='selected');p.add_argument('--smoke',action='store_true');p.add_argument('--global-features',action='store_true');evaluate(**vars(p.parse_args()))
