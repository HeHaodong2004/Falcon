"""Count/spatial diagnostics; no policy, Query-ranking model or optimization."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .common import OUT,save,digest

def load_roots():
    roots=[]
    for path in sorted((OUT/'probability/roots').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:roots.append(json.load(f))
    if len(roots)!=2110:raise ValueError('full frozen probability audit required')
    return roots

def band(rec):
    return next(g for g in rec['groups'] if g.startswith('duration:'))

def window_weights(roots,scenes):
    selected=[r for r in roots if r['scene_id'] in scenes]; counts=defaultdict(int)
    for root in selected:counts[root['scene_id']]+=1
    for root in selected:
        weight=1/(len(counts)*counts[root['scene_id']]*len(root['records']))
        for rec in root['records']:yield rec,weight

def fit_baselines(roots,scenes):
    values={g:dict(mass=0.,truth=np.zeros(3),area=0.,values=[],weights=[]) for g in ('all','duration:<=0.5','duration:0.5-1','duration:1-2','duration:>2')}
    for rec,w in window_weights(roots,scenes):
        for g in ('all',band(rec)):
            v=values[g];v['mass']+=w;v['truth']+=w*np.array(rec['true_counts']);v['area']+=w*rec['candidates'];v['values'].append(rec['true_counts']);v['weights'].append(w)
    result={}
    for g,v in values.items():
        data=np.array(v['values']);weights=np.array(v['weights']);medians=[]
        for j in range(3):
            order=np.argsort(data[:,j]);at=np.searchsorted(np.cumsum(weights[order]),.5*weights.sum());medians.append(float(data[order[min(at,len(order)-1)],j]))
        result[g]=dict(mean_counts=(v['truth']/v['mass']).tolist(),median_counts=medians,
            area_prevalence=(v['truth']/v['area']).tolist(),weight_mass=v['mass'])
    return result

def moments(x,y):
    return np.stack([abs(x-y),x-y,x,y,x*x,y*y,x*y])

def rates(m):
    mae,bias,x,y,xx,yy,xy=m
    covariance=xy-x*y;vx=np.maximum(xx-x*x,0);vy=np.maximum(yy-y*y,0)
    valid=(vx>1e-12*np.maximum(xx,1))&(vy>1e-12*np.maximum(yy,1))
    correlation=np.divide(covariance,np.sqrt(vx*vy),out=np.zeros(3),where=valid)
    return {name:dict(mae=float(mae[j]),bias=float(bias[j]),mean_prediction=float(x[j]),mean_truth=float(y[j]),pearson=float(correlation[j]) if valid[j] else None)
            for j,name in enumerate(('revelation','free','blocked'))}

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());roots=load_roots()
    selection=set(protocol['threshold_selection_scenes']);validation=set(protocol['diagnostic_validation_scenes'])
    baseline=fit_baselines(roots,selection);names=['model','zero','global_mean','duration_mean','duration_median','duration_candidate_area']
    per_scene=defaultdict(list);spatial=defaultdict(list)
    for root in roots:
        if root['scene_id'] not in validation:continue
        m=[];sp=[]
        for rec in root['records']:
            y=np.array(rec['true_counts']);area=rec['candidates'];b=baseline[band(rec)]
            predictions=[np.array(rec['expected_counts']),np.zeros(3),np.array(baseline['all']['mean_counts']),
                np.array(b['mean_counts']),np.array(b['median_counts']),np.array(b['area_prevalence'])*area]
            m.append(np.stack([moments(x,y) for x in predictions]))
            eg=np.array(rec['spatial_expected']);tg=np.array(rec['spatial_truth']);valid=int(tg.sum()>0 and eg.sum()>0)
            tv=.5*abs(eg/eg.sum()-tg/tg.sum()).sum() if valid else 0.
            sp.append([rec['spatial_count_l1'],tv,valid,int(tg.sum()>0 and eg.sum()<=0)])
        per_scene[root['scene_id']].append(np.mean(m,axis=0));spatial[root['scene_id']].append(np.mean(sp,axis=0))
    scenes=sorted(per_scene);array=np.stack([np.mean(per_scene[s],axis=0) for s in scenes]);mean=array.mean(0)
    result=dict(validation_scenes=len(scenes),validation_roots=sum(len(rs) for rs in per_scene.values()),
        fitted_baselines=baseline,metrics={name:rates(m) for name,m in zip(names,mean)},
        weighting='equalwindows withinRoot/equalRoots withinScene/equalScenes; counts persegment, NOT totaluniqueacquiredcells acrossoverlappingwindows',
        caveat=protocol['split_caveat'],protocol_sha256=digest(OUT/'protocol.json'))
    rng=np.random.default_rng(2525);draws=defaultdict(list)
    for _ in range(2000):
        value=array[rng.integers(0,len(array),len(array))].mean(0)
        for j,name in enumerate(names):
            r=rates(value[j]);model=rates(value[0])
            for outcome,metrics in r.items():
                for key in ('mae','bias','pearson'):
                    if metrics[key] is not None:draws[(name,outcome,key)].append(metrics[key])
                    if name!='model' and metrics[key] is not None and model[outcome][key] is not None:
                        draws[(name,outcome,key,'model_minus_baseline')].append(model[outcome][key]-metrics[key])
    result['scene_ci95']={}
    for key,values in draws.items():result['scene_ci95']['/'.join(key)]=np.quantile(values,[.025,.975]).tolist()
    sp=np.mean([np.mean(spatial[s],axis=0) for s in scenes],axis=0)
    result['spatial']=dict(grid='4x4worldrelative6m bins centered onstart, outercoordinates clipped toedge bins',
        mean_spatial_count_l1=float(sp[0]),conditional_mean_total_variation=float(sp[1]/max(sp[2],1e-30)),
        undefined_zero_prediction_mass_weight=float(sp[3]),
        caveat='normalized spatial shape only when trueANDpredictedmasses positive; undefinedzero-predictioncasesexcludedANDcounted; proper probabilistic scores remain primary')
    save(OUT/'decision_relevance/counts.json',result)
    print(json.dumps({'metrics':result['metrics'],'spatial':result['spatial']}),flush=True)

if __name__=='__main__':main()
