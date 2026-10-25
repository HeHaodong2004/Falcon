"""Full original development protocol plus predeclared diagnostic summaries.

Core rates/targets/scene bootstrap/proper score definitions are imported from
or exactly retained from the frozen evaluator. Extra diagnostics do not choose
weights/thresholds or change the raw-argmax deployment rule.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr,kendalltau
from eventwm_graph.stage2.api import Stage2Predictor as FrozenPredictor,apply_graph_delta as frozen_apply
from eventwm_graph.stage2.data import source_rows,branches
from eventwm_graph.stage2.evaluate import target_arrays,rates
from .api import Stage2Predictor,apply_graph_delta
from .common import OUT,ORIGINAL,SCHEMA,dump,sha,identity

THRESHOLDS=np.array([0.,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.])
MODELS=None

def initialize(checkpoint):
    global MODELS
    frozen=ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt'
    if sha(frozen)!='2c77ef1c6821f4c9dc3fac8d6128d1052f5bbdc2a449789b84e73c716a646ed4':raise ValueError('frozen baseline checkpoint changed')
    MODELS=dict(candidate=Stage2Predictor(checkpoint),baseline=FrozenPredictor(frozen))


def stratification(ex):
    p=ex['trajectory'];duration=float(p[-1,0]-p[0,0]);steps=np.linalg.norm(np.diff(p[:,[1,3]],axis=0),axis=1);distance=float(steps.sum())
    angle=np.diff(p[:,4]);rotation=float(np.abs(np.arctan2(np.sin(angle),np.cos(angle))).sum());stationary=float(np.mean(steps<1e-4)) if len(steps) else 1.
    cells=ex['graph']['cells'];keys=np.asarray([c['cell'] for c in cells],np.int32).reshape(-1,2)
    coverage=len(keys)/max(1,int(np.prod(keys.max(0)-keys.min(0)+1))) if len(keys) else 0.
    label=lambda value,limits:next((str(v) for v in limits if value<=v),'above')
    groups=['duration:'+label(duration,[.5,1.,2.]),'distance:'+label(distance,[.1,.5,1.]),'rotation:'+label(rotation,[.1,.5,float(np.pi)]),
        'new_cells:'+label(len(ex['target']),[0,99,999]),'initial_coverage:'+label(coverage,[.25,.5,.75]),
        'repeated_or_inplace:'+str(stationary>=.5 and rotation>.5)]
    return dict(duration=duration,distance=distance,rotation=rotation,stationary_fraction=stationary,initial_bbox_coverage=coverage,groups=groups)


def root_worker(row):
    start=time.perf_counter();records=[];statistics={key:dict(matrix=np.zeros((3,3),np.float64),pooled=np.zeros((3,3),np.int64),
        scores=np.zeros(4,np.float64),reliability=np.zeros((3,15,3),np.float64),height=np.zeros(2,np.float64),
        substantial=np.zeros((3,3),np.int64),pr=np.zeros((len(THRESHOLDS),4),np.float64)) for key in MODELS}
    counts=collections.Counter()
    for ex in branches(row):
        info=stratification(ex);record=dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=ex['query_id'],
            termination=ex['termination'],queries_in_root=ex['queries_in_root'],new_free=sum(v[0]==1 for v in ex['target'].values()),
            new_blocked=sum(v[0]==2 for v in ex['target'].values()),**info)
        candidate_keys=None
        for name,predictor in MODELS.items():
            before=time.perf_counter();delta=predictor.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);prediction_seconds=time.perf_counter()-before
            y,heights,miss=target_arrays(ex,delta['candidate_cells']);n=len(y);state=delta['states'];p=delta['probabilities'];s=statistics[name]
            if name=='candidate':candidate_keys=delta['candidate_cells']
            else:np.testing.assert_array_equal(candidate_keys,delta['candidate_cells'])
            cm=np.bincount(3*y.astype(np.int64)+state,minlength=9).reshape(3,3)
            for key in miss:cm[ex['target'][key][0],0]+=1
            s['matrix']+=cm/ex['queries_in_root'];s['pooled']+=cm
            if len(ex['target'])>=100:s['substantial']+=cm
            denom=n+len(miss);w=1/(ex['queries_in_root']*max(denom,1))
            if denom:
                ce=-np.log(np.clip(p[np.arange(n),y],1e-9,1.));brier=np.square(p).sum(-1)-2*p[np.arange(n),y]+1
                s['scores']+=np.array([w*(float(ce.sum())+len(miss)*-np.log(1e-9)),w*(float(brier.sum())+2*len(miss)),
                    w*(2*np.count_nonzero(y)+2*len(miss)),w*denom])
                for cls in range(3):
                    bins=np.minimum((p[:,cls]*15).astype(int),14)
                    s['reliability'][cls,:,0]+=w*np.bincount(bins,minlength=15)
                    s['reliability'][cls,:,1]+=w*np.bincount(bins,weights=p[:,cls],minlength=15)
                    s['reliability'][cls,:,2]+=w*np.bincount(bins,weights=y==cls,minlength=15)
                    if miss:
                        index=14 if cls==0 else 0;s['reliability'][cls,index,0]+=w*len(miss);s['reliability'][cls,index,1]+=w*len(miss)*(cls==0)
                        s['reliability'][cls,index,2]+=w*sum(ex['target'][k][0]==cls for k in miss)
                observed=y>0;s['height']+=np.array([float(np.abs(delta['heights'][observed]-heights[observed]).sum()),int(observed.sum())])
            actual=y>0;reveal=p[:,1]+p[:,2];pr=[]
            for threshold in THRESHOLDS:
                chosen=reveal>=threshold;tp=int(np.count_nonzero(chosen&actual));fp=int(np.count_nonzero(chosen&~actual));fn=int(np.count_nonzero(~chosen&actual))+len(miss);tn=int(np.count_nonzero(~chosen&~actual))
                pr.append([tp,fp,fn,tn])
            s['pr']+=np.asarray(pr)/ex['queries_in_root']
            update_seconds=0.
            if name=='candidate':
                before=time.perf_counter();future=apply_graph_delta(ex['graph'],delta);update_seconds=time.perf_counter()-before
                expected={tuple(c['cell']):c for c in ex['graph']['cells']};got={tuple(c['cell']):c for c in future['cells']}
                if len(got)!=len(expected)+len(delta['static_updates']) or any(got[k]!=v for k,v in expected.items()):raise ValueError('existing map mutated')
                for cell in delta['static_updates']:
                    gotcell=got[tuple(cell['cell'])]
                    if gotcell['state']!=cell['state'] or gotcell['height']!=cell['height']:raise ValueError('predicted update mismatch')
                counts.update(queries=1,candidate_cells=n,new_cells=len(ex['target']),out_of_domain_positive=len(miss),changed_known_cells=ex['known_changes'],
                    graph_update_correct=1,substantial_queries=int(len(ex['target'])>=100),zero_execution=int(ex['executed_nodes']==0))
            if name=='baseline':
                before=time.perf_counter();future=frozen_apply(ex['graph'],delta);update_seconds=time.perf_counter()-before
                expected={tuple(c['cell']):c for c in ex['graph']['cells']};got={tuple(c['cell']):c for c in future['cells']}
                if len(got)!=len(expected)+len(delta['static_updates']) or any(got[k]!=v for k,v in expected.items()):raise ValueError('frozen updater existing-map mismatch')
                for cell in delta['static_updates']:
                    if got[tuple(cell['cell'])]['state']!=cell['state'] or got[tuple(cell['cell'])]['height']!=cell['height']:raise ValueError('frozen predicted update mismatch')
            record[name]=dict(confusion=cm.tolist(),expected_free=float(p[:,1].sum()),expected_blocked=float(p[:,2].sum()),
                argmax_free=int(np.count_nonzero(state==1)),argmax_blocked=int(np.count_nonzero(state==2)),candidate_cells=n,
                predict_seconds=prediction_seconds,apply_seconds=update_seconds)
        # Geometry-only information-quantity proxy: legal candidates within10m
        # and60deg of ANY future base heading. It is not an optical truth mask.
        from .features import opportunity
        _,view=opportunity(candidate_keys,ex['trajectory'],float(ex['graph']['resolution']))
        record['geometry_expected_new']=int(np.count_nonzero(view[:,4]<1.))
        records.append(record)
    for s in statistics.values():
        for key in s:s[key]=s[key].tolist()
    return dict(root_id=row['root_id'],scene_id=row['scene_id'],records=records,statistics=statistics,counts=dict(counts),seconds=time.perf_counter()-start)


def bootstrap_metrics(matrices):
    balanced=rates(matrices.mean(0));rng=np.random.default_rng(2411);samples=[]
    for _ in range(2000):samples.append(rates(matrices[rng.integers(0,len(matrices),len(matrices))].mean(0)))
    for cls in ('new_free','new_blocked'):
        for metric in ('precision','recall','f1'):balanced[cls][metric+'_ci95']=np.quantile([r[cls][metric] for r in samples],[.025,.975]).tolist()
        balanced[cls]['false_discovery_rate']=1-balanced[cls]['precision']
    for metric in ('delta_cell_iou','reveal_set_iou'):balanced[metric+'_ci95']=np.quantile([r[metric] for r in samples],[.025,.975]).tolist()
    return balanced


def summarize(results,name):
    scene=collections.defaultdict(list);totals=collections.Counter();pooled=np.zeros((3,3),np.int64);substantial=np.zeros((3,3),np.int64)
    scores=np.zeros(4,np.float64);reliability=np.zeros((3,15,3),np.float64);height=np.zeros(2,np.float64);prs=np.zeros((len(THRESHOLDS),4),np.float64)
    for row in results:
        s=row['statistics'][name];scene[row['scene_id']].append(np.asarray(s['matrix']));pooled+=np.asarray(s['pooled'],np.int64);substantial+=np.asarray(s['substantial'],np.int64)
        scores+=s['scores'];reliability+=s['reliability'];height+=s['height'];prs+=s['pr'];totals.update(row['counts'])
    matrices=np.array([np.mean(v,axis=0) for v in scene.values()]);calibration={}
    for cls,label in enumerate(('unknown','new_free','new_blocked')):
        mass,prob,truth=reliability[cls].T
        calibration[label]=dict(ece=float(np.abs(prob-truth).sum()/max(mass.sum(),1e-12)),bins=[dict(lower=i/15,upper=(i+1)/15,root_weight=float(mass[i]),probability=float(prob[i]/mass[i]),frequency=float(truth[i]/mass[i])) for i in np.flatnonzero(mass>0)])
    substantial_roots={r['root_id'] for r in results if any(q['new_free']+q['new_blocked']>=100 for q in r['records'])}
    substantial_scenes={r['scene_id'] for r in results if r['root_id'] in substantial_roots}
    return dict(schema=SCHEMA,roots=len(results),scenes=len(scene),counts=dict(totals),pooled=rates(pooled),pooled_confusion=pooled.tolist(),
        scene_balanced=bootstrap_metrics(matrices),substantial_information=dict(**rates(substantial),confusion=substantial.tolist(),roots=len(substantial_roots),scenes=len(substantial_scenes)),
        multiclass_logloss=float(scores[0]/scores[3]),multiclass_brier=float(scores[1]/scores[3]),persist_unknown_brier=float(scores[2]/scores[3]),
        calibration=calibration,height_mae_on_observed_cells=float(height[0]/max(1,height[1])),height_support=int(height[1]),graph_update_correctness=totals['graph_update_correct']/totals['queries'],
        reveal_precision_recall=[dict(threshold=float(t),precision=float(a/max(a+b,1)),recall=float(a/max(a+c,1)),false_discovery_rate=float(b/max(a+b,1)),root_weighted_counts=[float(a),float(b),float(c),float(d)]) for t,(a,b,c,d) in zip(THRESHOLDS,prs)])


def query_ranking(results):
    output={};scene_names=sorted({r['scene_id'] for r in results})
    for model in ('candidate','baseline','geometry'):
        perroot=[];errors=collections.defaultdict(list)
        for root in results:
            q=root['records'];actual=np.array([r['new_free']+r['new_blocked'] for r in q],float)
            pred=np.array([r['geometry_expected_new'] if model=='geometry' else r[model]['expected_free']+r[model]['expected_blocked'] for r in q])
            errors['absolute_error'].extend(np.abs(pred-actual));errors['bias'].extend(pred-actual)
            if model!='geometry':
                for cls in ('free','blocked'):errors[cls+'_absolute_error'].extend(abs(r[model]['expected_'+cls]-r['new_'+cls]) for r in q)
            if len(q)>=2 and np.ptp(actual)>0:
                rho=float(spearmanr(actual,pred).statistic) if np.ptp(pred)>0 else 0.
                tau=float(kendalltau(actual,pred).statistic) if np.ptp(pred)>0 else 0.
                regret=float((actual.max()-actual[int(np.argmax(pred))])/max(1.,actual.max()))
                perroot.append(dict(scene_id=root['scene_id'],root_id=root['root_id'],spearman=rho,kendall=tau,relative_top1_quantity_regret=regret))
        output[model]=dict(quantity_errors={k:float(np.mean(v)) for k,v in errors.items()},ranking_roots=len(perroot),ranking_scenes=len({r['scene_id'] for r in perroot}),
            mean_root_spearman=float(np.mean([r['spearman'] for r in perroot])),mean_root_kendall=float(np.mean([r['kendall'] for r in perroot])),
            mean_relative_top1_quantity_regret=float(np.mean([r['relative_top1_quantity_regret'] for r in perroot])),per_root=perroot)
    output['interpretation']='Expected observation quantity and within-Root ranking only; no navigation utility, connectivity improvement or hidden-map truth is claimed.'
    return output


def strata(results):
    groups=collections.defaultdict(list)
    for root in results:
        for q in root['records']:
            for label in q['groups']:groups[label].append(q)
    output={}
    for label,records in groups.items():
        row=dict(queries=len(records),roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}))
        for name in ('baseline','candidate'):
            byroot=collections.defaultdict(list);root_scene={}
            for q in records:byroot[q['root_id']].append(np.asarray(q[name]['confusion']));root_scene[q['root_id']]=q['scene_id']
            byscene=collections.defaultdict(list)
            for rid,m in byroot.items():byscene[root_scene[rid]].append(np.mean(m,axis=0))
            row[name]=dict(pooled=rates(np.sum([np.asarray(q[name]['confusion']) for q in records],axis=0)),scene_balanced=bootstrap_metrics(np.array([np.mean(v,axis=0) for v in byscene.values()])))
        output[label]=row
    return output


def evaluate(checkpoint,workers=8):
    dest=OUT/'evaluation/full_dev';dest.mkdir(parents=True,exist_ok=True);rows,source=source_rows('dev');start=time.perf_counter();results=[]
    dump(dest/'protocol.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),sources=identity(),source_manifest=str(source),source_manifest_sha256=sha(source),
        roots=len(rows),scenes=len({r['scene_id'] for r in rows}),queries=sum(r['queries'] for r in rows),decision='fixed rawargmax; thresholds are diagnostics only',
        bootstrap='unchanged scene resampling2000seed2411',original_core_evaluator_sha256=sha(Path(__file__).parents[1]/'stage2/evaluate.py'),
        strata='predeclared duration/distance/rotation/repeatedposition/newcells/initialknownbboxcoverage; withinstratum Root then scene balancing',
        baseline='Frozen original checkpoint, actual original implementation, paired same Queries/candidate domains'))
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(checkpoint,)) as pool:
        for index,result in enumerate(pool.map(root_worker,rows)):
            results.append(result)
            if index%25==0:print(json.dumps(dict(roots=index+1,total=len(rows),seconds=time.perf_counter()-start)),flush=True)
    # Save sufficient statistics for independent audits; no need to rerun fits.
    with gzip.open(dest/'root_statistics.json.gz','wt') as f:json.dump(results,f)
    candidate=summarize(results,'candidate');baseline=summarize(results,'baseline')
    candidate.update(checkpoint_sha256=sha(checkpoint),seconds=time.perf_counter()-start,status='FINAL_FIXED_CANDIDATE_NO_DEV_TUNING')
    old=json.loads((ORIGINAL/'output/stage2_graphdelta_20261024/evaluation/optimized_dev/metrics.json').read_text())
    comparisons={}
    for key in ('roots','scenes','counts','pooled_confusion','multiclass_logloss','multiclass_brier','persist_unknown_brier','height_mae_on_observed_cells','height_support','graph_update_correctness','calibration'):
        comparisons[key]=baseline[key]==old[key]
    # Rates add FDR/IoU CIs but the inherited frozen numeric rates must match.
    for cls in ('new_free','new_blocked'):
        for metric in ('precision','recall','f1'):
            comparisons[cls+'_'+metric]=baseline['scene_balanced'][cls][metric]==old['scene_balanced'][cls][metric]
            comparisons[cls+'_'+metric+'_ci95']=baseline['scene_balanced'][cls][metric+'_ci95']==old['scene_balanced'][cls][metric+'_ci95']
    def maxdiff(a,b):
        if isinstance(a,dict):return max([maxdiff(v,b[k]) for k,v in a.items() if k in b]+[0.])
        if isinstance(a,list):return max([maxdiff(x,y) for x,y in zip(a,b)]+[0.]) if len(a)==len(b) else float('inf')
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)
        return 0. if a==b else float('inf')
    differences={k:maxdiff(baseline[k],old[k]) for k in ('pooled_confusion','scene_balanced','multiclass_logloss','multiclass_brier','calibration','height_mae_on_observed_cells')}
    dump(dest/'baseline_equivalence.json',dict(exact_fields=comparisons,all_exact=all(comparisons.values()),maximum_numeric_differences=differences,all_numeric_within_1e_12=all(x<=1e-12 for x in differences.values()),note='Aggregation order is Rootthenprocesssummary; last-bit score summation differences must be separately quantified, not silently ignored.'))
    dump(dest/'baseline_metrics.json',baseline);dump(dest/'metrics.json',candidate)
    dump(dest/'query_ranking.json',query_ranking(results));dump(dest/'strata.json',strata(results))
    print(json.dumps(candidate),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--workers',type=int,default=8);evaluate(**vars(p.parse_args()))
