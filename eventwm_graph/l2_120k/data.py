"""Offline audit and supervision. No online module imports this truth reader."""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.snapshot import sha
from .model import load_model

OUT = Path(os.environ.get('L2_REPRO_OUTPUT','output/l2_graph_wm_120k_20261021')).resolve()
if not OUT.is_relative_to(Path(__file__).resolve().parents[2]/'output'):
    raise ValueError('research artifacts must stay in this worktree output directory')
SNAP = OUT/'snapshot'


def audit():
    manifest=json.loads((SNAP/'manifest.json').read_text())
    population=np.full(17,.001); summaries=collections.defaultdict(list)
    motion, motion_roots=[],[]; sampling=collections.Counter();versions=collections.Counter()
    started=time.perf_counter()
    for i,row in enumerate(manifest['rows']):
        folder=SNAP/'roots'/row['root_id']; m=json.loads((folder/'manifest.json').read_text())
        l=read(folder/'legal_root.json.gz')
        with np.load(folder/'privileged_future.npz') as f:
            times=f['timestamps'].copy(); pos=f['positions'][...,[0,2]]
        n=pos.shape[1];hist=l['history'];now=hist[-1]['timestamp'];key=row['split']
        summaries[key+':count'].append(n);summaries[key+':history_length'].append(len(hist))
        summaries[key+':duration'].append(float(times[-1]-times[0]))
        summaries[key+':history_duration'].append(now-hist[0]['timestamp'])
        summaries[key+':track_age'].extend([now-t['last_seen_time'] for t in hist[-1]['tracks']])
        sampling.update(m['sampling']['query_diagnostics'].get('accepted_by_category',{}))
        versions[l['sensor']['protocol'] if isinstance(l.get('sensor'),dict) and 'protocol' in l['sensor'] else str(l.get('sensor'))]+=1
        if key=='train':
            if n>=len(population): population=np.pad(population,(0,n+1-len(population)),constant_values=.001)
            population[n]+=1
            if len(times)>3 and n:
                dt=np.diff(times);velocity=np.diff(pos,axis=0)/dt[:,None,None]
                dv=np.diff(velocity,axis=0).reshape(-1,2)
                good=np.isfinite(dv).all(1);dv=dv[good]
                if len(dv):
                    select=np.linspace(0,len(dv)-1,min(32,len(dv))).astype(int)
                    motion.extend(dv[select]);motion_roots.extend([row['root_id']]*len(select))
                summaries['train:speed'].extend(np.linalg.norm(velocity,axis=-1).ravel().tolist())
        if (i+1)%2000==0:print('audit',i+1,round(time.perf_counter()-started,1),flush=True)
    population/=population.sum();(OUT/'population_prior.json').write_text(json.dumps(population.tolist()))
    dv=np.asarray(motion);_,indices,counts=np.unique(motion_roots,return_inverse=True,return_counts=True)
    weights=1./counts[indices];weights/=weights.sum();p=-np.expm1(-2*.048)
    def objective(logparams):
        variance, jump=np.exp(logparams)
        var=np.array([variance,variance+jump*jump]);squared=(dv*dv).sum(1)
        lp=-np.log(2*np.pi*var)[None,:]-.5*squared[:,None]/var[None,:]
        return float(-(weights*logsumexp(lp+np.log([1-p,p]),axis=1)).sum())
    fit=minimize(objective,np.log([.005,.2]),bounds=[(-14,1),(-4.6,np.log(2))],method='L-BFGS-B')
    motion_law=dict(maneuver_rate=2., reset_std=float(np.exp(fit.x[1])), maneuver_retention=1.)
    (OUT/'motion.json').write_text(json.dumps(motion_law,indent=2))
    quantiles={k:dict(n=len(v),mean=float(np.mean(v)) if v else None,
        quantiles=np.quantile(v,[0,.05,.5,.9,.95,.99,1]).tolist() if v else []) for k,v in summaries.items()}
    report=dict(manifest_sha256=sha(SNAP/'manifest.json'),splits=manifest['splits'],
        source_outcomes=manifest['source_outcomes'],source_roots=manifest['source_roots'],
        distributions=quantiles,sampling=dict(sampling),sensor_versions=dict(versions),
        scalar_motion_fit=dict(success=bool(fit.success),message=str(fit.message),
            roots=len(counts),increments=len(dv),objective=float(fit.fun),
            nuisance_velocity_innovation_variance=float(np.exp(fit.x[0])),**motion_law,
            interpretation='train-only root-balanced finite-difference innovation calibration; no hard speed clipping'),
        seconds=time.perf_counter()-started)
    (OUT/'data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['scalar_motion_fit']),flush=True)


def prepare(split, limit=0, start_index=0, stop_index=0, tag=''):
    manifest=json.loads((SNAP/'manifest.json').read_text())
    population=json.loads((OUT/'population_prior.json').read_text());motion=json.loads((OUT/'motion.json').read_text())
    wm=load_model(population=population, motion=motion,
        observation='docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/observation_fit.json')
    rows=[r for r in manifest['rows'] if r['split']==split]
    rows=rows[start_index:stop_index or None]
    if limit:rows=rows[:limit]
    basename=split+('_'+tag if tag else '_smoke' if limit else '')
    result=[];failures=[];started=time.perf_counter()
    for ri,row in enumerate(rows):
        folder=SNAP/'roots'/row['root_id'];l=read(folder/'legal_root.json.gz')
        try:b=wm._initialize_history(l['history'],l['partial_graph'])
        except Exception as e:
            failures.append(dict(root_id=row['root_id'],error=repr(e)));continue
        with np.load(folder/'privileged_future.npz') as f:
            times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        # Every root is retained for evaluation. All TRAIN roots supervise the
        # separate population prior in audit(); only matched trajectories train
        # network motion parameters. Never claim an empty motion loss trains count.
        components=[]
        for h in b.hypotheses:
            ids=np.full(h.n,-1,int)
            # Only exact root-visible centers establish training correspondence.
            for i in np.flatnonzero(np.isclose(h.last_seen,b.timestamp,atol=1e-7,rtol=0)):
                if truth.shape[1]:
                    d=np.linalg.norm(truth[0]-h.mean[i,:2],axis=-1);j=int(d.argmin())
                    if d[j]<1e-5:ids[i]=j
            selected=np.flatnonzero(ids>=0)
            if len(selected) and len(times)>1:
                # Initial mixture components already carry fixed intentions. Reset
                # only for a newly declared forecasting episode, as initialize does.
                components.append(dict(weight=h.weight,mean=h.mean.astype(np.float32),
                    covariance=h.covariance.astype(np.float32),age=(b.timestamp-h.last_seen).astype(np.float32),
                    graph=b.graph.local_features(h.mean[:,:2]).astype(np.float32),
                    selected=selected,truth=truth[:,ids[selected]].astype(np.float32),
                    maneuver_mean=h.maneuver_mean,maneuver_covariance=h.maneuver_covariance,
                    maneuver_probability=h.maneuver_probability,maneuver_until=h.maneuver_until))
        result.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],components=components,
            times=times[:np.searchsorted(times,b.timestamp+3.+1e-8)].copy(),
            count=truth.shape[1],history_length=len(l['history']),
            approximation=b.evidence.get('normalized_approximation'),
            root_retained_mass=sum(h.weight for h in b.hypotheses),
            filtered_hypotheses=b.hypotheses,filtered_evidence=b.evidence,
            timestamp=b.timestamp,robot_pose=b.robot_pose))
        # Trim supervision to the same actual timestamps; no resampled native grid.
        for c in components:c['truth']=c['truth'][:len(result[-1]['times'])]
        if (ri+1)%250==0:print(split,ri+1,'supervised',sum(bool(r['components']) for r in result),
            'failures',len(failures),'seconds',round(time.perf_counter()-started,1),flush=True)
        if (ri+1)%1000==0:np.save(OUT/(basename+'_partial.npy'),np.array(result,dtype=object),allow_pickle=True)
    np.save(OUT/(basename+'.npy'),np.array(result,dtype=object),allow_pickle=True)
    (OUT/(basename+'_prepare.json')).write_text(json.dumps(dict(roots=len(rows),
        built=len(result),supervised=sum(bool(r['components']) for r in result),failures=failures,
        seconds=time.perf_counter()-started,manifest_sha256=sha(SNAP/'manifest.json'),motion=motion,
        start_index=start_index,stop_index=stop_index,tag=tag),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','train','dev']);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--stop-index',type=int,default=0);p.add_argument('--tag',default='');a=p.parse_args()
    if a.command=='audit':audit()
    else:prepare(a.command,a.limit,a.start_index,a.stop_index,a.tag)
