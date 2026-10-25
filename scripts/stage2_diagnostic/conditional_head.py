"""Post-hoc FREE/BLOCKED-head diagnostic on truly revealed cells, not a predictor.

Ground-truth masking is used ONLY to score the already saved probabilities.
No oracle gate, graph reconstruction, model input or deployed baseline results
from this diagnostic.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import numpy as np
from .common import OUT,save

def root(path):
    with gzip.open(path,'rt') as f:meta=json.load(f)
    cm=np.zeros((2,2));sums=np.zeros(6)
    with np.load(path.parent/(meta['root_id']+'.probabilities.npz')) as cache:
        p=cache['p'];y=cache['y'];offsets=cache['offsets']
        for j,rec in enumerate(meta['records']):
            a,b=offsets[j:j+2];truth=y[a:b];prob=p[a:b];observed=truth>0
            q=prob[observed,1:3].astype(np.float64);q/=np.maximum(q.sum(1,keepdims=True),1e-30)
            t=truth[observed]-1;pred=np.where(q[:,0]>=q[:,1],0,1)
            cm+=np.bincount(2*t+pred,minlength=4).reshape(2,2)
            ce=-np.log(np.clip(q[np.arange(len(t)),t],1e-9,1))
            br=np.square(q).sum(1)-2*q[np.arange(len(t)),t]+1
            sums+=np.r_[ce.sum(),br.sum(),len(t),int((t==0).sum()),int((t==1).sum()),len(truth)]/max(len(truth),1)
    return dict(scene_id=meta['scene_id'],root_id=meta['root_id'],cm=cm/len(meta['records']),sums=sums/len(meta['records']))

def main():
    paths=sorted((OUT/'probability/roots').glob('*.json.gz'))
    if len(paths)!=2110:raise ValueError('complete probabilitycache required')
    scenes=defaultdict(list)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(root,paths,chunksize=4):scenes[r['scene_id']].append(r)
    cm=np.stack([np.mean([r['cm'] for r in rs],axis=0) for _,rs in sorted(scenes.items())]);sums=np.stack([np.mean([r['sums'] for r in rs],axis=0) for _,rs in sorted(scenes.items())])
    C=cm.mean(0);S=sums.mean(0);result=dict(roots=len(paths),scenes=len(scenes),
        conditional_confusion=C.tolist(),conditional_class_accuracy=float(np.trace(C)/C.sum()),
        conditional_free_recall=float(C[0,0]/C[0].sum()),conditional_blocked_recall=float(C[1,1]/C[1].sum()),
        conditional_logloss=float(S[0]/S[2]),conditional_sum_brier=float(S[1]/S[2]),
        caveat='GTrevealedcellmask is posthoc scoring only; this is NOT an achievable oracle-revelation GraphDelta forecast; conditional class accuracy uses original rawcount hierarchy, proper scores use cellnormalized hierarchy')
    protocol=json.loads((OUT/'protocol.json').read_text());selection=set(protocol['threshold_selection_scenes']);names=sorted(scenes)
    a=np.array([name in selection for name in names]);train=sums[a].mean(0);test=sums[~a].mean(0);prior=train[3:5]/train[2];prevalence=test[3:5]/test[2]
    result['diagnostic_validation']=dict(model_logloss=float(test[0]/test[2]),model_sum_brier=float(test[1]/test[2]),
        constant_fitted_label_prior=prior.tolist(),prior_logloss=float(-np.dot(prevalence,np.log(prior))),
        prior_sum_brier=float(1+np.sum(prior**2)-2*np.dot(prior,prevalence)))
    rng=np.random.default_rng(2525);draws=[]
    for _ in range(2000):
        c=cm[rng.integers(0,len(cm),len(cm))].mean(0);draws.append(np.trace(c)/c.sum())
    result['conditional_class_accuracy_scene_ci95']=np.quantile(draws,[.025,.975]).tolist()
    save(OUT/'probability/conditional_head.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
