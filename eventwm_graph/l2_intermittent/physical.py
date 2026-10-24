"""Offline native-interval motion validation, with fixed physical ancestry."""
import argparse
import json
import math
import time
import numpy as np
from .common import OUT, OLD, SNAP, read, sha, seed_for, dump
from .model import IntermittentWorldModel
from .baselines import MemoryCV
from .evaluate import exact_ids, extend_ancestry, lineage_slot, to_previous, paired_ci
from eventwm_graph.l2_120k.model import load_model
from eventwm_graph.l2_120k.evaluate import restore


def features(x,dt):
    """x[world,time,person,xy]; native finite differences on BOTH sides."""
    v=np.diff(x,axis=1)/dt[None,:,None,None]
    s=np.linalg.norm(v,axis=-1);finite=np.isfinite(s);values=s[finite]
    if not len(values):return {}
    result={f'speed_p{q}':float(np.quantile(values,q/100)) for q in (50,95,99)}
    result.update(stop_frequency=float(np.mean(values<.1)),speed_over3=float(np.mean(values>3)),speed_over5=float(np.mean(values>5)))
    for threshold in (.1,.5,1.,1.5,2.,3.,5.):result['speed_cdf_'+str(threshold)]=float(np.mean(values<=threshold))
    good=finite[:,1:]&finite[:,:-1]
    result['restart_frequency']=float(np.mean(((s[:,:-1]<.1)&(s[:,1:]>.2))[good])) if good.any() else None
    result['stop_entry_frequency']=float(np.mean(((s[:,:-1]>.2)&(s[:,1:]<.1))[good])) if good.any() else None
    moving=good&(s[:,1:]>.1)&(s[:,:-1]>.1)
    cross=v[:,:-1,:,0]*v[:,1:,:,1]-v[:,:-1,:,1]*v[:,1:,:,0]
    dot=(v[:,:-1]*v[:,1:]).sum(-1)
    turn=np.abs(np.arctan2(cross,dot));length=(s[:,:-1]*dt[:-1][None,:,None]+s[:,1:]*dt[1:][None,:,None])/2
    result['turn_p95_rad']=float(np.quantile(turn[moving],.95)) if moving.any() else None
    for threshold in (.1,.5,1.,2.):result['turn_cdf_'+str(threshold)]=float(np.mean(turn[moving]<=threshold)) if moving.any() else None
    result['curvature_p95']=float(np.quantile((turn/np.maximum(length,1e-12))[moving],.95)) if moving.any() else None
    displacement=np.linalg.norm(x[:,-1]-x[:,0],axis=-1);displacement=displacement[np.isfinite(displacement)]
    result['displacement_p50']=float(np.quantile(displacement,.5)) if len(displacement) else None
    result['displacement_p95']=float(np.quantile(displacement,.95)) if len(displacement) else None
    durations=[];censored=[]
    for w in range(len(x)):
        for person in range(x.shape[2]):
            stopped=(s[w,:,person]<.1)&finite[w,:,person]
            starts=np.flatnonzero(stopped&~np.r_[False,stopped[:-1]])
            ends=np.flatnonzero(stopped&~np.r_[stopped[1:],False])
            for begin,end in zip(starts,ends):
                duration=float(dt[begin:end+1].sum())
                if begin==0 or end==len(dt)-1:censored.append(duration)
                else:durations.append(duration)
    result.update(stop_duration_p50=float(np.quantile(durations,.5)) if durations else None,
        stop_duration_p95=float(np.quantile(durations,.95)) if durations else None,
        completed_stop_runs_per_world=len(durations)/len(x),censored_stop_runs_per_world=len(censored)/len(x))
    for threshold in (.1,.25,.5,1.,2.):result['completed_stop_duration_cdf_'+str(threshold)]=float(np.mean(np.array(durations)<=threshold)) if durations else None
    return result


def old_model():
    return load_model(OLD/'training/checkpoint.pt',json.loads((OLD/'population_prior.json').read_text()),
        'docs/l2/graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json',json.loads((OLD/'motion.json').read_text()))


def summarize(records):
    summary=[];differences=[]
    for model in sorted({r['model'] for r in records}):
        part=[r for r in records if r['model']==model]
        keys=sorted({k[10:] for r in part for k in r if k.startswith('predicted_')})
        for key in keys:
            matched=[r for r in part if r.get('predicted_'+key) is not None and r.get('actual_'+key) is not None]
            delta=[dict(r,difference=r['predicted_'+key]-r['actual_'+key]) for r in matched]
            summary.append(dict(model=model,metric=key,predicted=paired_ci(matched,'predicted_'+key),actual=paired_ci(matched,'actual_'+key),difference=paired_ci(delta,'difference')))
    return summary


def run(limit=0):
    import torch
    torch.set_num_threads(1)
    path=OUT/'physical.json'
    if path.exists():raise ValueError('preserve existing physical evidence')
    law=json.loads((OUT/'motion.json').read_text());models={
        'A_CV':MemoryCV(0.,law=law),'B_MemoryCV':MemoryCV(.16,law=law),'B_HistoricalCV':MemoryCV(.64,law=law),
        'D_StopMove':IntermittentWorldModel.load(OUT/'training/checkpoint.pt'),
        'E_Initial':IntermittentWorldModel.load(OUT/'training/epoch0.pt')}
    sources={name:sha(OUT.parent.parent/'eventwm_graph/l2_intermittent'/name) for name in ('physical.py','model.py','dynamics.py','evaluate.py','baselines.py')}
    previous=old_model();bootstrap=load_model();rows=list(np.load(OLD/'dev.npy',allow_pickle=True))[:limit or None]
    records=[];failures=[];support=[];start=time.perf_counter()
    for ri,row in enumerate(rows):
        folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz')
        with np.load(folder/'privileged_future.npz') as f:ts=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        ids=sorted(exact_ids(legal['history'][-1],ts,truth))
        if not ids:continue
        take=ts<=ts[0]+3.+1e-8;ts=ts[take];truth=truth[take]
        if len(ts)<3:continue
        dt=np.diff(ts);initial=restore(bootstrap,row,legal['partial_graph']);actual=features(truth[:,ids][None],dt)
        support.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],people=len(ids),intervals=len(dt),duration=float(ts[-1]-ts[0]),interval_quantiles=np.quantile(dt,[0,.5,.95,1]).tolist()))
        for stream in (41,107):
            states={k:m.from_legacy(initial,seed_for(row['root_id'],stream+10000)) for k,m in models.items()}
            for name in list(models)+['C_Previous']:
                b=states['B_MemoryCV' if name=='C_Previous' else name];ancestry={};extend_ancestry(b,ancestry,ts,truth)
                try:
                    if any(lineage_slot(c,ancestry,p) is None for c in b.components if c.weight>0 for p in ids):
                        failures.append(dict(root_id=row['root_id'],model=name,stream=stream,error='represented ancestry unsupported'));continue
                    if name=='C_Previous':
                        future=previous.predict_shared(to_previous(previous,b),float(ts[-1]-ts[0]),32,seed_for(row['root_id'],stream),times=ts-ts[0])
                        origins=[b.components[i].origin for i in future.world_source_indices]
                        slots=[future.source.hypotheses[i].slot_ids for i in future.world_source_indices]
                    else:
                        future=models[name].predict_shared(b,float(ts[-1]-ts[0]),32,seed_for(row['root_id'],stream),times=ts-ts[0])
                        origins=[c.origin for c in future.endpoints[0]];slots=future.slots
                    x=[]
                    for w,(origin,ss) in enumerate(zip(origins,slots)):
                        indices=[min((ancestry[(origin,int(s))][1],int(s),i) for i,s in enumerate(ss) if ancestry.get((origin,int(s)),(-1,))[0]==p)[2] for p in ids]
                        x.append(future.positions[w][:,indices])
                    pred=features(np.asarray(x),dt)
                    records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],model=name,stream=stream,people=len(ids),
                        **{'predicted_'+k:v for k,v in pred.items()},**{'actual_'+k:v for k,v in actual.items()}))
                except Exception as exc:failures.append(dict(root_id=row['root_id'],model=name,stream=stream,error=repr(exc)))
        if (ri+1)%100==0:print(json.dumps(dict(roots=ri+1,scored=len(support),seconds=time.perf_counter()-start)),flush=True)
    dump(path,dict(records=records,summary=summarize(records),support=support,failures=failures,seconds=time.perf_counter()-start,
        checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),snapshot_sha256=sha(SNAP/'manifest.json'),sources=sources,
        source_stable=all(sha(OUT.parent.parent/'eventwm_graph/l2_intermittent'/name)==value for name,value in sources.items()),
        semantics='root-positive fixed physical ancestry; actual native recorded intervals <=3s; 32 worlds/two independent root streams; completed stop durations exclude censored runs'))


def reproduce_old():
    """Same 256 roots, seed71,16 worlds/all-population as frozen old diagnostic."""
    import torch
    torch.set_num_threads(1)
    path=OUT/'old_physical_reproduction.json'
    if path.exists():raise ValueError('preserve old physical reproduction')
    wm=old_model();rows=list(np.load(OLD/'dev.npy',allow_pickle=True))[:256];records=[];failures=[];start=time.perf_counter()
    for row in rows:
        folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz')
        with np.load(folder/'privileged_future.npz') as f:ts=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        keep=ts<=ts[0]+3.+1e-8;ts=ts[keep];truth=truth[keep]
        if len(ts)<3:continue
        try:
            b=restore(wm,row,legal['partial_graph']);future=wm.predict_shared(b,float(ts[-1]-ts[0]),16,71,times=ts-ts[0]);dt=np.diff(ts)
            pred=features(future.positions,dt);actual=features(truth[None],dt)
            records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],model='43187f24_all_population',
                **{'predicted_'+k:v for k,v in pred.items()},**{'actual_'+k:v for k,v in actual.items()}))
        except Exception as exc:failures.append(dict(root_id=row['root_id'],error=repr(exc)))
    dump(path,dict(records=records,summary=summarize(records),failures=failures,seconds=time.perf_counter()-start,
        noise_floor=dict(position_variance_rate=.0025,speed_threshold=.1,dt=.048,max_stop_probability=float(-np.expm1(-.1**2*.048/(2*.0025)))),
        scope='reproduce frozen matched diagnostic only; all-population support; main new physical scores are tracked people separately'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reproduce-old',action='store_true');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    reproduce_old() if a.reproduce_old else run(a.limit)
