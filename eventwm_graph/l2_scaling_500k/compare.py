"""Paired frozen-case statistics. No online inference or checkpoint selection."""
import collections
import gzip
import json
import math
from pathlib import Path
import numpy as np
from .common import BASE,OUT,sha,dump

HISTORICAL=BASE/'docs/l2/intermittent_wm_20261022/artifacts/evaluation'
ORIGINAL=OUT/'evaluation_arm_original_full/evaluation'
NEW=OUT/'evaluation_arm_new/evaluation'
FIELDS=('tracked_es','path_es','future_path_es','coverage50','coverage90','stop_brier','stop_probability','observed_stop','velocity_error','velocity_coverage90','direction_brier','diversity','variance_trace')

def practical_age(r):
    age=r.get('age')
    return age is not None and .75-1e-8<=age<=1.25+1e-8

def rows(folder,kind,model='D_StopMove'):
    with gzip.open(folder/(kind+'.jsonl.gz'),'rt') as f:
        return [r for line in f if (r:=json.loads(line))['model']==model]

def ci(records,key):
    root=collections.defaultdict(list)
    for r in records:
        x=r.get(key)
        if x is not None and math.isfinite(x):root[(r['scene_id'],r['root_id'])].append(x)
    scenes=collections.defaultdict(list)
    for (s,r),v in root.items():scenes[s].append(np.mean(v))
    a=np.array([np.mean(v) for _,v in sorted(scenes.items())]);n=len(a)
    if not n:return dict(mean=None,lo=None,hi=None,roots=0,scenes=0)
    boot=a[np.random.default_rng(73).integers(0,n,(1000,n))].mean(1)
    return dict(mean=float(a.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),roots=len(root),scenes=n)

def case_key(r,kind):
    return tuple(r.get(k) for k in ('root_id','cohort','decision','stream','physical','H'))

def pair(left,right,kind,fields):
    a={case_key(r,kind):r for r in left};b={case_key(r,kind):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right):raise ValueError('duplicate physical case key')
    paired=[]
    for k in sorted(a.keys()&b.keys(),key=repr):
        old,new=a[k],b[k];r={k:old.get(k) for k in ('root_id','scene_id','cohort','decision','stream','physical','H','age','age_bin','one_second','gap','gap_bin')}
        for field in fields:
            x,y=old.get(field),new.get(field)
            if x is not None and y is not None:r[field]=y-x;r['old_'+field]=x;r['new_'+field]=y
        paired.append(r)
    return paired,dict(old_cases=len(a),new_cases=len(b),common_cases=len(a.keys()&b.keys()),old_only=len(a.keys()-b.keys()),new_only=len(b.keys()-a.keys()))

def stats(left,right,paired,fields):
    return dict(original={k:ci(left,k) for k in fields},scaled={k:ci(right,k) for k in fields},
        common_original={k:ci(paired,'old_'+k) for k in fields},common_scaled={k:ci(paired,'new_'+k) for k in fields},
        paired_new_minus_original={k:ci(paired,k) for k in fields})

def forecast():
    a=[r for r in rows(ORIGINAL,'forecasts') if r['status']!='joint_supported'];b=[r for r in rows(NEW,'forecasts') if r['status']!='joint_supported']
    paired,ledger=pair(a,b,'forecasts',FIELDS);cells=[]
    assert ledger['old_only']==ledger['new_only']==0,'Every original physical target/failure/censor case must be retained'
    groups=sorted({(r['cohort'],r['age_bin'],r['H']) for r in a+b},key=repr)
    def matches(r,g):return (r['cohort'],r['age_bin'],r['H'])==g
    for g in groups:
        left=[r for r in a if matches(r,g)];right=[r for r in b if matches(r,g)];part=[r for r in paired if matches(r,g)]
        cells.append(dict(cohort=g[0],age_bin=g[1],H=g[2],original_status=dict(collections.Counter(r['status'] for r in left)),
            new_status=dict(collections.Counter(r['status'] for r in right)),**stats(left,right,part,FIELDS)))
    scopes=[]
    for cohort in ('native','masked'):
        for scope in ('all','visible','one_second'):
            def selected(r):
                return r['cohort']==cohort and (scope=='all' or scope=='visible' and r.get('age') is not None and r['age']<=1e-7 or scope=='one_second' and practical_age(r) and r['H']<=2.)
            left=list(filter(selected,a));right=list(filter(selected,b));part=list(filter(selected,paired))
            scopes.append(dict(cohort=cohort,scope=scope,original_status=dict(collections.Counter(r['status'] for r in left)),
                new_status=dict(collections.Counter(r['status'] for r in right)),**stats(left,right,part,FIELDS)))
    by_horizon=[]
    practical=[]
    for cohort in ('native','masked'):
        for H in (.5,1.,2.,3.):
            selected=lambda r:r['cohort']==cohort and r['H']==H
            by_horizon.append(dict(cohort=cohort,H=H,**stats(list(filter(selected,a)),list(filter(selected,b)),list(filter(selected,paired)),FIELDS)))
            selected=lambda r:r['cohort']==cohort and r['H']==H and practical_age(r)
            left=list(filter(selected,a));right=list(filter(selected,b));part=list(filter(selected,paired))
            practical.append(dict(cohort=cohort,H=H,original_status=dict(collections.Counter(r['status'] for r in left)),
                new_status=dict(collections.Counter(r['status'] for r in right)),**stats(left,right,part,FIELDS)))
    support=dict(original=dict(collections.Counter(r['status'] for r in a)),scaled=dict(collections.Counter(r['status'] for r in b)))
    baseline=[r for r in rows(ORIGINAL,'forecasts','B_MemoryCV') if r['status']!='joint_supported']
    vs_cv,_=pair(baseline,b,'forecasts',FIELDS);cv_contrasts=[]
    for cohort in ('native','masked'):
        for scope in ('all','visible','one_second'):
            selected=lambda r:r['cohort']==cohort and (scope=='all' or scope=='visible' and r.get('age') is not None and r['age']<=1e-7 or scope=='one_second' and practical_age(r) and r['H']<=2.)
            cv_contrasts.append(dict(cohort=cohort,scope=scope,**{m:ci(list(filter(selected,vs_cv)),m) for m in ('tracked_es','path_es','future_path_es')}))
    stream_sensitivity=[]
    for stream in (41,107):
        part=[r for r in paired if r['cohort']=='native' and practical_age(r) and r['H']<=2. and r['stream']==stream]
        stream_sensitivity.append(dict(stream=stream,**{m:ci(part,m) for m in ('tracked_es','path_es','future_path_es')}))
    return dict(case_ledger=ledger,support=support,cells=cells,scopes=scopes,by_horizon=by_horizon,practical_age_horizons=practical,new_vs_memory_cv=cv_contrasts,primary_stream_sensitivity=stream_sensitivity)

def reobservation():
    a=rows(ORIGINAL,'reobservation');b=rows(NEW,'reobservation')
    fields=('tracked_es_gain','path_es_gain','future_path_es_gain','coverage50_change','coverage90_change','association_accuracy','false_split','false_merge','duplicate','missing_antecedent_mass')
    paired,ledger=pair(a,b,'reobservation',fields);result=[]
    assert ledger['old_only']==ledger['new_only']==0,'Reobservation eligibility must be model independent'
    for scope in ('all','one_second'):
        for H in (.5,1.,2.,3.):
            def selected(r):return r['H']==H and r.get('gap') is not None and (scope=='all' or .75-1e-8<=r['gap']<=1.25+1e-8)
            left=list(filter(selected,a));right=list(filter(selected,b));part=list(filter(selected,paired))
            result.append(dict(scope=scope,H=H,original_status=dict(collections.Counter(r['prior_status']+'/'+r['posterior_status'] for r in left)),
                new_status=dict(collections.Counter(r['prior_status']+'/'+r['posterior_status'] for r in right)),**stats(left,right,part,fields)))
    return dict(case_ledger=ledger,cells=result)

def belief():
    a=rows(ORIGINAL,'beliefs');b=rows(NEW,'beliefs')
    result={}
    for name,part in (('original',a),('scaled',b)):
        result[name]=dict(states=len(part),roots=len({r['root_id'] for r in part}),scenes=len({r['scene_id'] for r in part}),
            zero_true_count_support=sum(not r['true_count_support'] for r in part),
            duplicate_ancestry_states=sum(r['expected_duplicate_ancestries']>0 for r in part),
            missing_preferred_ancestry_states=sum(r['expected_missing_preferred_ancestries']>0 for r in part),
            max_arrived_coordinate_error=max((r['max_arrived_coordinate_error'] or 0.) for r in part))
    return result

def physical():
    original=OUT/'evaluation_arm_original_full/physical.json'
    new=OUT/'evaluation_arm_new/physical.json'
    old_data=json.loads(original.read_text())
    a=[r for r in old_data['records'] if r['model']=='D_StopMove']
    b=[r for r in json.loads(new.read_text())['records'] if r['model']=='D_StopMove']
    key=lambda r:(r['root_id'],r['stream']);left={key(r):r for r in a};right={key(r):r for r in b};metrics=[]
    fields=sorted({k[10:] for r in a for k in r if k.startswith('predicted_')})
    for field in fields:
        part=[]
        for k in left.keys()&right.keys():
            x,y=left[k].get('predicted_'+field),right[k].get('predicted_'+field)
            truth=left[k].get('actual_'+field);assert truth==right[k].get('actual_'+field)
            if x is not None and y is not None and truth is not None:
                part.append(dict(root_id=left[k]['root_id'],scene_id=left[k]['scene_id'],original=x,scaled=y,actual=truth,
                    difference=y-x,absolute_error_change=abs(y-truth)-abs(x-truth)))
        metrics.append(dict(metric=field,**{name:ci(part,name) for name in ('original','scaled','actual','difference','absolute_error_change')}))
    return dict(original_rows=len(a),new_rows=len(b),common_rows=len(left.keys()&right.keys()),original_only=len(left.keys()-right.keys()),new_only=len(right.keys()-left.keys()),metrics=metrics,
        original_sha256=sha(original),new_sha256=sha(new))

def run():
    sources={str(p):sha(p) for p in (Path(__file__),ORIGINAL/'forecasts.jsonl.gz',ORIGINAL/'reobservation.jsonl.gz',NEW/'forecasts.jsonl.gz',NEW/'reobservation.jsonl.gz')}
    result=dict(forecasting=forecast(),reobservation=reobservation(),belief=belief(),physical=physical(),inputs=sources,
        inference_unchanged=True,ci='1000 paired scene bootstrap draws seed73; observations/streams average within root, then roots within scene; equal scene weights; no sampled-world independence claim')
    assert all(sha(Path(p))==h for p,h in sources.items())
    dump(OUT/'comparison.json',result)

if __name__=='__main__':run()
