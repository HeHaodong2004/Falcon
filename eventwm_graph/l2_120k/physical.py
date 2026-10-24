"""Native matched-interval motion plausibility, offline development only."""
import argparse
import json
import math
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.repair_evaluation import scene_ci
from .data import OUT,SNAP
from .evaluate import model_set,restore


def summarize(records):
    keys=sorted(set(k for r in records for k in r if k.startswith(('predicted_','true_','latent_'))))
    summary={name:{k:scene_ci([r for r in records if r['model']==name],k) for k in keys}
        for name in sorted({r['model'] for r in records})}
    matched={}
    for name in summary:
        matched[name]={}
        for key in keys:
            if not key.startswith('predicted_'):continue
            target='true_'+key[len('predicted_'):]
            rows=[{**r,'difference':r[key]-r[target]} for r in records
                if r['model']==name and r.get(key) is not None and r.get(target) is not None]
            matched[name][key]=dict(predicted=scene_ci(rows,key),actual=scene_ci(rows,target),
                predicted_minus_actual=scene_ci(rows,'difference'))
    return summary,matched


def run(limit=256):
    models=model_set();rows=np.load(OUT/'dev.npy',allow_pickle=True).tolist()[:limit]
    records=[];failures=[];started=time.perf_counter()
    for ri,c in enumerate(rows):
        folder=SNAP/'roots'/c['root_id'];legal=read(folder/'legal_root.json.gz')
        with np.load(folder/'privileged_future.npz') as f:times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        take=times<=times[0]+3.+1e-8;times=times[take];truth=truth[take]
        if len(times)<3:continue
        dt=np.diff(times)
        v=np.diff(truth,axis=0)/dt[:,None,None]
        def features(positions,velocity,tracked=None):
            if positions.ndim==3:positions=positions[None];velocity=velocity[None]
            finite=np.isfinite(velocity).all(-1)
            speed=np.linalg.norm(velocity,axis=-1)[finite]
            if not len(speed):return {}
            out={f'speed_p{q}':float(np.quantile(speed,q/100)) for q in (50,90,95,99)}
            out.update(speed_over3=float(np.mean(speed>3)),speed_over5=float(np.mean(speed>5)),
                stop_frequency=float(np.mean(speed<.1)))
            acceleration=(velocity[:,1:]-velocity[:,:-1])/((dt[1:]+dt[:-1])/2)[None,:,None,None]
            acceleration=np.linalg.norm(acceleration,axis=-1);valid=np.isfinite(acceleration)
            out['acceleration_p95']=float(np.quantile(acceleration[valid],.95)) if valid.any() else None
            s=np.linalg.norm(velocity,axis=-1);moving=(s[:,1:]>.1)&(s[:,:-1]>.1)
            cosine=np.sum(velocity[:,1:]*velocity[:,:-1],axis=-1)/np.maximum(s[:,1:]*s[:,:-1],1e-12)
            turn=np.arccos(np.clip(cosine,-1,1));out['turn_p95_rad']=float(np.quantile(turn[moving],.95)) if moving.any() else None
            good=np.isfinite(s[:,1:])&np.isfinite(s[:,:-1])
            out['restart_frequency']=float(np.mean(((s[:,:-1]<.1)&(s[:,1:]>.2))[good])) if good.any() else None
            displacement=np.linalg.norm(positions[:,-1]-positions[:,0],axis=-1);displacement=displacement[np.isfinite(displacement)]
            out['displacement_p95']=float(np.quantile(displacement,.95)) if len(displacement) else None
            return out
        actual=features(truth,v)
        for name,wm in models.items():
            try:
                b=restore(wm,c,legal['partial_graph'],name.startswith('A_'))
                f=wm.predict_shared(b,float(times[-1]-times[0]),16,71,times=times-times[0])
                matched=(f.positions[:,1:]-f.positions[:,:-1])/dt[None,:,None,None]
                pred=features(f.positions,matched)
                latent=f.velocities[:,1:];ss=np.linalg.norm(latent,axis=-1);ss=ss[np.isfinite(ss)]
                records.append(dict(root_id=c['root_id'],scene_id=c['scene_id'],model=name,duration=float(times[-1]-times[0]),
                    **{'predicted_'+k:v for k,v in pred.items()},**{'true_'+k:v for k,v in actual.items()},
                    latent_speed_over3=float(np.mean(ss>3)) if len(ss) else None))
            except Exception as e:failures.append(dict(root_id=c['root_id'],model=name,error=repr(e)))
        if (ri+1)%50==0:print('physical',ri+1,round(time.perf_counter()-started,1),flush=True)
    summary,matched=summarize(records)
    (OUT/'physical.json').write_text(json.dumps(dict(records=records,summary=summary,matched_summary=matched,failures=failures,
        seconds=time.perf_counter()-started,scope='first256 metadata-ordered dev roots; native actual intervals, available prefix <=3s; no horizon-support substitutions'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=256);p.add_argument('--summarize-existing',action='store_true');args=p.parse_args()
    if args.summarize_existing:
        path=OUT/'physical.json';data=json.loads(path.read_text())
        data['summary'],data['matched_summary']=summarize(data['records'])
        data['summary_qualification']='Per-feature paired nonempty truth/prediction support; all-predicted and all-truth root summaries also retained. No new model fitting.'
        path.write_text(json.dumps(data,indent=2))
    else:run(args.limit)
