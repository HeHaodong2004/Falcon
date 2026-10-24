"""Offline calibration of supported fixed-query events; no query selection."""
import argparse
import collections
import json
import math
import time
import numpy as np
from .common import OUT, OLD, SNAP, BASE, read, canonical, centers, dump, sha, seed_for
from .model import IntermittentWorldModel
from .observations import predict_observations, condition_observation_event
from .evaluate import paired_ci
from eventwm_graph.l2_120k.model import load_model
from eventwm_graph.l2_120k.evaluate import restore


def loss(p,y):
    value=p if y else 1-p
    return -math.log(value) if value>0 else 'INFINITE_WITH_ZERO_SUPPORT'


def run(limit=0):
    import torch
    torch.set_num_threads(1)
    destination=OUT/'optical_validation.json'
    if destination.exists():raise ValueError('preserve existing optical results')
    model=IntermittentWorldModel.load(OUT/'training/checkpoint.pt');bootstrap=load_model()
    rows=list(np.load(OLD/'dev.npy',allow_pickle=True))[:limit or None]
    records=[];failures=[];normalization=[];clock=[];start=time.perf_counter()
    sources={n:sha(BASE/'eventwm_graph/l2_intermittent'/n) for n in ('model.py','dynamics.py','observations.py','optical_validation.py')}
    for ri,row in enumerate(rows):
        folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz');query,obs=canonical(folder)
        now=row['timestamp'];n=min(len(query['compiled']['actions']),62)
        actual=[ob for ob in obs[:n] if ob['timestamp']<=now+3.+1e-8]
        # One held human tape contains nominal planned and actual endpoint times.
        times=np.unique(np.r_[0.,np.arange(1,63)*.048,3.,[ob['timestamp']-now for ob in actual]])
        for stream in (41,107):
            meta=dict(root_id=row['root_id'],scene_id=row['scene_id'],stream=stream)
            try:
                b=model.from_legacy(restore(bootstrap,row,legal['partial_graph']),seed_for(row['root_id'],stream+10000))
                future=model.predict_shared(b,3.,32,seed_for(row['root_id'],stream),times=times)
                event=predict_observations(model,b,query,future)
                if not event['successor_supported']:
                    failures.append(dict(**meta,status=event['status']));continue
                residual=abs(event['none_probability']+sum(event['first_detection_mass'])-1)
                weights_error=0.
                if event['none_probability']>0:
                    child=condition_observation_event(model,future,event,'none')
                    expected=future.weights*event['per_world_none']/event['none_probability']
                    weights_error=max(abs(np.array([c.weight for c in child.components])-expected))
                normalization.append(dict(**meta,first_plus_none_error=float(residual),conditional_weights_error=float(weights_error)))
                labels=[];available=[]
                for i,ob in enumerate(actual):
                    ok=bool(ob.get('human_sensor_available',True) and ob.get('observation_valid',True));available.append(ok)
                    y=bool(len(centers(ob)));labels.append(y)
                    if stream==41:clock.append(float(ob['timestamp']-(now+(i+1)*.048)))
                    if not ok:continue
                    p=event['endpoint_any_probability'][i]
                    records.append(dict(**meta,kind='planned_endpoint',index=i,predicted=p,observed=float(y),brier=(p-y)**2,nll=loss(p,y)))
                    j=int(np.argmin(abs(future.times-ob['timestamp'])))
                    regimes=np.array([c.visibility_mode for c in future.endpoints.worlds])
                    pd=model.observation_law.detection(future.positions[:,j],ob['robot_pose'],regimes)
                    hd=np.array(model.observation_law.hidden_detection)[regimes,None]*np.isfinite(future.hidden_positions[:,j,:,0])
                    none=np.prod(1-pd,axis=1)*np.prod(1-hd,axis=1)
                    p=float(future.weights@(1-none))
                    records.append(dict(**meta,kind='recorded_pose_endpoint',index=i,predicted=p,observed=float(y),brier=(p-y)**2,nll=loss(p,y)))
                if len(actual)==len(event['endpoint_times']) and all(available):
                    y=any(labels);p=event['any_probability']
                    likelihood=event['none_probability'] if not y else p
                    first=next((i for i,v in enumerate(labels) if v),None)
                    first_mass=event['none_probability'] if first is None else event['first_detection_mass'][first]
                    records.append(dict(**meta,kind='complete_available_planned_prefix',predicted=p,observed=float(y),brier=(p-y)**2,
                        nll=-math.log(likelihood) if likelihood>0 else 'INFINITE_WITH_ZERO_SUPPORT',
                        first_detection_nll=-math.log(first_mass) if first_mass>0 else 'INFINITE_WITH_ZERO_SUPPORT',
                        endpoints=len(labels),complete_query=bool(len(labels)==len(query['compiled']['actions']))))
                else:
                    failures.append(dict(**meta,status='prefix_censored_or_unavailable',available_endpoints=len(actual),planned_endpoints=len(event['endpoint_times'])))
            except Exception as exc:failures.append(dict(**meta,status='exception',error=repr(exc)))
        if (ri+1)%100==0:print(json.dumps(dict(roots=ri+1,seconds=time.perf_counter()-start)),flush=True)
    summary=[];bins=[]
    for kind in sorted({r['kind'] for r in records}):
        part=[r for r in records if r['kind']==kind]
        numeric=[dict(r,nll=r['nll'] if isinstance(r['nll'],(float,int)) else None,
                      first_detection_nll=r.get('first_detection_nll') if isinstance(r.get('first_detection_nll'),(float,int)) else None) for r in part]
        summary.append(dict(kind=kind,metrics={m:paired_ci(numeric,m) for m in ('predicted','observed','brier','nll','first_detection_nll')},
            zero_support_rows=sum(isinstance(r['nll'],str) for r in part),zero_support_roots=len({r['root_id'] for r in part if isinstance(r['nll'],str)}),
            unconditional_nll='INFINITE_WITH_ZERO_SUPPORT' if any(isinstance(r['nll'],str) for r in part) else 'finite'))
        for j in range(10):
            selected=[r for r in part if j<=min(9,int(10*r['predicted']))<j+1]
            bins.append(dict(kind=kind,bin=j,predicted=paired_ci(selected,'predicted'),observed=paired_ci(selected,'observed')))
    dump(destination,dict(records=records,summary=summary,calibration_bins=bins,failures=failures,normalization=normalization,
        native_minus_nominal_clock_quantiles=np.quantile(clock,[0,.5,.95,.99,1]).tolist() if clock else [],
        max_normalization_error=max((r['first_plus_none_error'] for r in normalization),default=0.),
        max_conditional_weight_error=max((r['conditional_weights_error'] for r in normalization),default=0.),
        checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),sources=sources,source_stable=all(sha(BASE/'eventwm_graph/l2_intermittent'/n)==v for n,v in sources.items()),
        seconds=time.perf_counter()-start,semantics='32 worlds/two root-keyed streams; range/sector approximate optics; planned prefix <=2.976s; recorded future poses are OFFLINE optical diagnostic only; no privileged geometry'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);run(p.parse_args().limit)
