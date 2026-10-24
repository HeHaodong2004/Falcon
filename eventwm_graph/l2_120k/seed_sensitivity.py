"""Bounded second-seed forecasting diagnostic; no fitting or model selection."""
import json
import argparse
import hashlib
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.repair_evaluation import scene_ci
from .data import OUT,SNAP
from .evaluate import HORIZONS,model_set,restore,truth_at,visible_ids,forecast_score,path_scores,maps_for


def root_seed(seed,root_id):
    return int.from_bytes(hashlib.sha256(('l2-120k-mc-v1:'+str(seed)+':'+root_id).encode()).digest()[:4],'big')


def main(seed=107,limit=256,root_keyed=False):
    models=model_set();cache=np.load(OUT/'dev.npy',allow_pickle=True).tolist()
    if limit:cache=cache[:limit]
    records=[];paths=[];failures=[];started=time.perf_counter()
    for i,c in enumerate(cache):
        folder=SNAP/'roots'/c['root_id'];legal=read(folder/'legal_root.json.gz')
        try:
            clock=np.unique(np.r_[0.,np.arange(1,63)*.048,HORIZONS,3.])
            draw_seed=root_seed(seed,c['root_id']) if root_keyed else seed
            futures={name:model.predict_shared(restore(model,c,legal['partial_graph'],name.startswith('A_')),
                3.,16,draw_seed,times=clock) for name,model in models.items()}
            with np.load(folder/'privileged_future.npz') as f:
                times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
            ids=visible_ids(legal,truth[0]);targets=[]
            for horizon in HORIZONS:
                target=truth_at(times,truth,c['timestamp']+horizon)
                if target is None:continue
                index=int(np.argmin(abs(next(iter(futures.values())).times-c['timestamp']-horizon)))
                targets.append((index,target))
                for name,f in futures.items():
                    meta=dict(root_id=c['root_id'],scene_id=c['scene_id'],model=name,horizon=horizon,seed=seed,draw_seed=draw_seed)
                    records.append({**meta,**forecast_score(f,index,target,truth[0],ids)})
                    paths.append({**meta,**path_scores(f,targets,ids,maps_for(f,truth[0]))})
        except Exception as e:failures.append(dict(root_id=c['root_id'],error=repr(e)))
        if (i+1)%50==0:print('seed',seed,'root_keyed',root_keyed,i+1,'seconds',round(time.perf_counter()-started,1),flush=True)
    table={(r['root_id'],r['horizon'],r['model']):r for r in records}
    contrasts={}
    for baseline in models:
        if baseline=='D_trained_graph':continue
        rows=[{**r,'difference':r['population_es3']-table[(r['root_id'],r['horizon'],baseline)]['population_es3']}
            for r in records if r['model']=='D_trained_graph']
        contrasts['D_minus_'+baseline]=scene_ci(rows,'difference')
    result=dict(records=records,paths=paths,failures=failures,primary_contrasts=contrasts,
        requested_roots=len(cache),seed=seed,root_keyed=root_keyed,worlds=16,seconds=time.perf_counter()-started,
        scope='post-fit numerical validation; fixed weights and all6models; root-keyed independent MC if requested, original shared-seed results retained; no tuning')
    name='forecast_root_keyed_'+str(seed)+'.json' if root_keyed else 'seed_sensitivity.json'
    (OUT/name).write_text(json.dumps(result,indent=2))
    print(json.dumps(contrasts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=107);p.add_argument('--limit',type=int,default=256)
    p.add_argument('--root-keyed',action='store_true');main(**vars(p.parse_args()))
