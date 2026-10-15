"""Supplementary motion, correlation and eroded-navmesh diagnostics.

Targets only define evaluation strata/scores after legal forecasts are drawn.
The map is an eroded navigation proxy, not a human contact or wall oracle.
No predicted path is clipped, projected, rejected or removed from tail counts.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.window_evaluate import aggregate
from .occupancy_prior import atomic_save
from .evaluation_population import population as resolve_population

STRATA=('all','high_acceleration','turning','slow','fast','near_supported_human',
        'near_navmesh_boundary','navmesh_interior','no_current_observation',
        'dense_scene','long_history','short_history')


def map_values(points,features,support):
    shape=points.shape[:-1];size=features.shape[-1]
    index=((points+8.)*(size/16.)).floor().long()
    inside=((index>=0)&(index<size)).all(-1)
    cell=(index[...,1].clamp(0,size-1)*size+index[...,0].clamp(0,size-1)).reshape(shape[0],-1)
    known=support.flatten(1).gather(1,cell).reshape(shape)&inside
    walk=features[:,1].flatten(1).gather(1,cell).reshape(shape)>.5
    clearance=2*features[:,0].flatten(1).gather(1,cell).reshape(shape)
    return inside,known,walk,clearance


def variogram(a,b,weights,truth,complete,kind):
    """Unbiased two-bank squared moment score, power .5, not strictly proper."""
    length=truth.shape[1];n=truth.shape[2]
    if kind.startswith('temporal'):
        lag=int(kind.rsplit('_',1)[1])
        if length<=lag:return truth.new_zeros(len(truth)),torch.zeros(len(truth),dtype=torch.bool,device=truth.device)
        feature=lambda paths:(paths[:,:,lag:]-paths[:,:,:-lag]).norm(dim=-1).sqrt()
        target=(truth[:,lag:]-truth[:,:-lag]).norm(dim=-1).sqrt()
        eligible=complete[:,None].expand(-1,length-lag,-1)
    else:
        i,j=torch.triu_indices(n,n,1,device=truth.device)
        feature=lambda paths:(paths[:,:,:,i]-paths[:,:,:,j]).norm(dim=-1).sqrt()
        target=(truth[:,:,i]-truth[:,:,j]).norm(dim=-1).sqrt()
        eligible=(complete[:,i]&complete[:,j])[:,None].expand(-1,length,-1)
    first=(feature(a)*weights[:,:,None,None]).sum(1);second=(feature(b)*weights[:,:,None,None]).sum(1)
    score=(target-first)*(target-second);count=eligible.sum((1,2))
    return (score*eligible).sum((1,2))/count.clamp_min(1),count>0


def weighted_quantile(values,weights,probability=.95):
    index=values.argsort();v=values[index];w=weights[index]
    threshold=probability*w.sum();at=torch.searchsorted(w.cumsum(0),threshold).clamp_max(len(v)-1)
    return float(v[at])


@torch.no_grad()
def run(cfg):
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed diagnostic is immutable')
    guard=CooperativeGuard(out);torch.set_num_threads(1);start=time.time()
    ids,population_identity=resolve_population(cfg)
    state=torch.load(cfg['checkpoint'],map_location='cpu')
    if set(state.get('fit_root_ids',[]))&set(ids):raise ValueError('Evaluation leakage')
    if cfg.get('development_limit') is not None:ids=ids[:int(cfg['development_limit'])]
    model=load_model(cfg['checkpoint']).eval();roots=load_roots(cfg['manifest'],ids,guard)
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    attach_geometry(roots,cfg['geometry_manifest'],guard)
    windows=[(r,j,w) for r in roots for j,w in enumerate(r['windows'])];rows=[];tails=[];g=torch.Generator().manual_seed(62111)
    identity=dict(config=cfg,source_sha256=sha256(__file__),**{key+'_sha256':sha256(cfg[key]) for key in
        ('checkpoint','manifest','protocol','geometry_manifest')})
    checkpoint=out/'diagnostic.resume.pt';cursor=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['identity']!=identity:raise ValueError('Diagnostic resume identity changed')
        rows,tails,cursor=saved['rows'],saved['tails'],saved['cursor'];g.set_state(saved['rng'])
    for offset in range(cursor,len(windows),8):
        guard.check();chosen=windows[offset:offset+8];batch=prepared_batch([w for _,_,w in chosen],'cpu')
        physical=batch if cfg.get('physical_uses_geometry',False) or model.recipe.get('variant')=='geometry_occupancy_v2' else dict(
            batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
        belief=encode(model,physical);f=model.imagine(belief,horizon_s=3.)
        if 48%f.weights.shape[1]:raise ValueError('Exactly 48 stratified samples required')
        draws=48//f.weights.shape[1];a,weights,_=f.sample(draws,g);b,_,_=f.sample(draws,g)
        visible=torch.zeros_like(f.valid)
        for i,(_,_,window) in enumerate(chosen):
            humans=window['base'].humans;visible[i,:len(humans)]=humans[:,-1,4]>.5
        # Only the following lines access dynamic future supervision.
        for horizon in (1.,2.,3.):
            length=round(horizon*4);truth=batch['targets']['track_positions'][:,:length]
            complete=batch['targets']['track_valid'][:,:length].all(1);aa=a[:,:,:length];bb=b[:,:,:length]
            mean=(f.positions[:,:,:length]*f.weights[:,:,None,None,None]).sum(1)
            error=(mean-truth).norm(dim=-1);velocity=(truth[:,1:]-truth[:,:-1])/.25
            acceleration=(velocity[:,1:]-velocity[:,:-1]).norm(dim=-1)/.25
            speed=velocity.norm(dim=-1).mean(1);high_acc=acceleration.amax(1)>1.
            first=velocity[:,0];last=velocity[:,-1]
            cosine=(first*last).sum(-1)/(first.norm(dim=-1)*last.norm(dim=-1)).clamp_min(1e-6)
            turn=(cosine<2**-.5)&(first.norm(dim=-1)>.2)&(last.norm(dim=-1)>.2)
            pair=torch.cdist(truth[:,0],truth[:,0]);n=pair.shape[-1]
            eligible=complete[:,None]&~torch.eye(n,dtype=torch.bool)[None]
            nearest=pair.masked_fill(~eligible,float('inf')).amin(-1)
            _,known,walk,clearance=map_values(truth[:,0],batch['geometry'],batch['geometry_support'])
            strata=dict(all=complete,high_acceleration=complete&high_acc,turning=complete&turn,
                slow=complete&(speed<.25),fast=complete&(speed>=1.25),near_supported_human=complete&(nearest<1.),
                near_navmesh_boundary=complete&known&(clearance<.5),navmesh_interior=complete&known&walk&(clearance>=.5),
                no_current_observation=complete&~(visible&belief.core.valid).any(1)[:,None])
            for i,(_,_,window) in enumerate(chosen):
                for name,condition in [('dense_scene',window['metadata']['physical_humans_target_only']>=6),
                    ('long_history',window['metadata']['history_span_s']>=3.),('short_history',window['metadata']['history_span_s']<1.)]:
                    if name not in strata:strata[name]=torch.zeros_like(complete)
                    strata[name][i]=complete[i]&condition
            for name,support in strata.items():
                moments={kind:variogram(aa,bb,weights,truth,support,kind) for kind in ('temporal_1','temporal_4','spatial')}
                for i,(root,j,window) in enumerate(chosen):
                    if not support[i].any():continue
                    selected=(mean[i]-truth[i])[:,support[i]]
                    row=dict(root_id=root['root_id'],scene=root['scene'],window=j,horizon=horizon,stratum=name,
                        category=window['metadata']['category'],scene_type=root['scene'].split(':')[0],agents=int(support[i].sum()),
                        ade=float(error[i,:,support[i]].mean()),fde=float(error[i,-1,support[i]].mean()),
                        bias_x=float(selected[...,0].mean()),bias_y=float(selected[...,1].mean()))
                    row.update({kind:float(value[i]) for kind,(value,valid) in moments.items() if valid[i]});rows.append(row)
            inside,known,walk,_=map_values(aa,batch['geometry'],batch['geometry_support'])
            speeds=(aa[:,:,1:]-aa[:,:,:-1]).norm(dim=-1)/.25
            acc=(aa[:,:,2:]-2*aa[:,:,1:-1]+aa[:,:,:-2]).norm(dim=-1)/(.25**2)
            for i,(root,j,_) in enumerate(chosen):
                valid=f.valid[i]
                if not valid.any():continue
                values=speeds[i,:,:,valid];accelerations=acc[i,:,:,valid]
                w=weights[i,:,None,None].expand_as(values);wa=weights[i,:,None,None].expand_as(accelerations)
                grid_weight=weights[i,:,None,None].expand_as(known[i,:,:,valid])
                coverage=known[i,:,:,valid];den=(grid_weight*coverage).sum()
                row=dict(root_id=root['root_id'],scene=root['scene'],window=j,horizon=horizon,
                    speed_gt3=float(((values>3)*w).sum()/w.sum()),speed_p95=weighted_quantile(values.flatten(),w.flatten()),
                    acceleration_gt6=float(((accelerations>6)*wa).sum()/wa.sum()),acceleration_p95=weighted_quantile(accelerations.flatten(),wa.flatten()),
                    outside_grid=float(((~inside[i,:,:,valid])*grid_weight).sum()/grid_weight.sum()),
                    known_map_fraction=float(den/grid_weight.sum()))
                if den>0:row['nonwalkable_navmesh_proxy']=float(((~walk[i,:,:,valid])*coverage*grid_weight).sum()/den)
                tails.append(row)
        save_json(out/'progress.json',dict(status='diagnosing_physics',windows=offset+len(chosen),total_windows=len(windows)))
        if (offset+len(chosen))%64==0 or offset+len(chosen)==len(windows):
            atomic_save(dict(identity=identity,rows=rows,tails=tails,cursor=offset+len(chosen),rng=g.get_state()),checkpoint)
    scores=('ade','fde','bias_x','bias_y','temporal_1','temporal_4','spatial');tail_keys=('speed_gt3','speed_p95','acceleration_gt6','acceleration_p95','outside_grid','known_map_fraction','nonwalkable_navmesh_proxy')
    summaries={};tail_summaries={}
    for population in ('origin','all_windows'):
        summaries[population]={};tail_summaries[population]={}
        for horizon in (1.,2.,3.):
            selected=[r for r in rows if r['horizon']==horizon and (population=='all_windows' or r['window']==0)]
            groups={name:[r for r in selected if r['stratum']==name] for name in STRATA}
            groups.update({'category:'+str(value):[r for r in selected if r['stratum']=='all' and r['category']==value]
                           for value in sorted({r['category'] for r in selected},key=str)})
            groups.update({'scene_type:'+value:[r for r in selected if r['stratum']=='all' and r['scene_type']==value]
                           for value in sorted({r['scene_type'] for r in selected})})
            summaries[population][str(horizon)]={name:dict(roots=len({r['root_id'] for r in group}),
                metrics={key:float(np.mean([r[key] for r in aggregate(group,scores) if key in r])) if any(key in r for r in group) else None for key in scores}) for name,group in groups.items()}
            tail_rows=aggregate([r for r in tails if r['horizon']==horizon and (population=='all_windows' or r['window']==0)],tail_keys)
            tail_summaries[population][str(horizon)]={key:scene_interval(tail_rows,key) for key in tail_keys}
            guard.check()
    result=dict(status='physics_diagnosis_complete',config=cfg,source_sha256=sha256(__file__),runtime_s=time.time()-start,
        smoke_only=cfg.get('development_limit') is not None,development_roots=len(ids),
        checkpoint_sha256=sha256(cfg['checkpoint']),manifest_sha256=sha256(cfg['manifest']),geometry_manifest_sha256=sha256(cfg['geometry_manifest']),
        protocol_sha256=sha256(cfg['protocol']),records=rows,tail_records=tails,summary=summaries,tails=tail_summaries,
        semantics=dict(variogram='Power .5 of Euclidean displacements; two independent 48-path stratified banks give an unbiased squared moment score, which may be negative at finite K. Supplementary, non-strictly proper.',
            tails='All legal forecast tracks; weighted scenario/time/agent mean within window, windows averaged within root. Speed/acceleration include readout noise.',
            geometry='Eroded navmesh proxy only. Outside-grid and missing-map mass reported separately; no samples rejected.',
            strata='Target-only acceleration/turn/proximity and scene metadata after inference. Near-human means another supported target within 1m.',
            units='Distances metres, speed m/s, acceleration m/s^2; variogram m. Sampling unchanged across compared models.'),
        confirmation=population_identity['confirmation'],population_identity=population_identity)
    save_json(out/'result.json',result);print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(read(parser.parse_args().config))


if __name__=='__main__':main()
