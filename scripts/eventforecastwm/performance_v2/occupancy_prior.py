"""Root-balanced, shrunk anonymous occupancy conditioned on legal context.

This is a marginal occupancy readout, not invented identity trajectories.
Geometry is an eroded-navmesh covariate; it never hard-masks human occupancy.
The trajectory distribution and its parameters remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import occupancy_targets, save_json
from eventbeliefwm.metrics import binary_summary
from eventforecastwm.benchmark import load_model, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.readouts import tracked_occupancy
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from eventforecastwm.window_evaluate import aggregate


def legal_bins(batch):
    prepared=batch['prepared_history'];valid=prepared.valid
    count=valid.sum(1).clamp_max(3).long()
    last=(prepared.lengths-1).clamp_min(0).long()
    index=last[:,None,None,None].expand(-1,valid.shape[1],1,1)
    visible=(batch['base'].humans[...,4:5].gather(2,index).squeeze(-1).squeeze(-1)>.5)&valid
    older=(valid & (-prepared.last_observed_times>=1.)).any(1)
    context=count*4+visible.any(1).long()*2+older.long()
    geometry=batch['geometry'];support=batch['geometry_support']
    # 0 unavailable; 1 nonwalkable; 2 walkable with <=0.5m clearance;
    # 3 walkable with >0.5m clearance. This is a learned proxy, not a contact label.
    spatial=torch.where(support,1+(geometry[:,1]>.5).long()*(1+(geometry[:,0]>.25).long()),0)
    return context,spatial


def predict(tables,batch,*,geometry=True,strength=128.,length=12,history_context=True):
    context,spatial=legal_bins(batch);device=context.device
    num=tables['positive'].to(device);den=tables['exposure'].to(device)
    prior=tables['prior'][:length].to(device)
    if not history_context:
        num=num.sum(0,keepdim=True);den=den.sum(0,keepdim=True);context=torch.zeros_like(context)
    if not geometry:
        probability=(num.sum(1)[:,:length]+strength*prior)/(den.sum(1)[:,:length]+strength)
        return probability[context]
    probability=(num[:,:,:length]+strength*prior)/(den[:,:,:length]+strength)
    selected=probability[context]
    gather=spatial[:,None,None].expand(-1,1,length,-1,-1)
    return selected.gather(1,gather).squeeze(1)


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);os.replace(temporary,path)


@torch.no_grad()
def fit(cfg,model,guard,identity):
    out=Path(cfg['run']);ids=read(cfg['protocol'])['subsets'][str(cfg.get('scale','full'))]
    resume=out/'fit.resume.pt';positive=torch.zeros(16,4,20,32,32);exposure=torch.zeros_like(positive)
    global_positive=torch.zeros(20,32,32);global_exposure=torch.zeros(20);cursor=0
    if resume.exists():
        state=torch.load(resume,map_location='cpu')
        if state['identity']!=identity:raise ValueError('Occupancy fitting identity changed')
        positive,exposure=state['positive'],state['exposure'];cursor=state['cursor']
        global_positive,global_exposure=state['global_positive'],state['global_exposure']
    for start in range(cursor,len(ids),64):
        guard.check();roots=load_roots(cfg['manifest'],ids[start:start+64],guard)
        prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
        attach_geometry(roots,cfg['geometry_manifest'],guard)
        for root in roots:
            batch=prepared_batch(root['windows'],'cpu');_,anonymous=occupancy_targets(batch,20)
            valid=batch['targets']['physical_exposure'];weights=valid/valid.sum(0).clamp_min(1)
            global_positive+=(anonymous*weights[...,None,None]).sum(0)
            global_exposure+=valid.any(0)
            context,spatial=legal_bins(batch)
            index=(context[:,None,None]*4+spatial)[:,None].expand(-1,20,-1,-1).flatten(1)
            weighted=weights[...,None,None].expand_as(anonymous)
            positive.view(64,-1).scatter_add_(0,index,(anonymous*weighted).flatten(1))
            exposure.view(64,-1).scatter_add_(0,index,weighted.flatten(1))
        cursor=min(start+64,len(ids))
        atomic_save(dict(identity=identity,cursor=cursor,positive=positive,exposure=exposure,
            global_positive=global_positive,global_exposure=global_exposure),resume)
        save_json(out/'progress.json',dict(status='fitting_anonymous_occupancy',roots=cursor,total_roots=len(ids)))
    prior=(global_positive+.5)/(global_exposure[:,None,None]+1.)
    tables=dict(positive=positive,exposure=exposure,prior=prior,fit_root_ids=ids,
        identity=identity,semantics='Anonymous physical human-center cell occupancy; root mean over supported windows')
    atomic_save(tables,out/'prior.pt');return tables


@torch.no_grad()
def assess(cfg,model,tables,guard):
    roots=load_roots(cfg['manifest'],read(cfg['protocol'])['development'],guard)
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    attach_geometry(roots,cfg['geometry_manifest'],guard)
    variants=dict(unconditional=None,context_128=(False,128.,True),context_512=(False,512.,True),
                  context_map_128=(True,128.,True),context_map_512=(True,512.,True),
                  map_only_128=(True,128.,False),map_only_512=(True,512.,False))
    names=cfg.get('variants',['unconditional','context_128','context_512','context_map_128','context_map_512'])
    variants={name:variants[name] for name in names}
    if 'unconditional' not in variants:raise ValueError('The matched unconditional readout is required')
    windows=[(root,j,w) for root in roots for j,w in enumerate(root['windows'])];records=[]
    for offset in range(0,len(windows),16):
        guard.check();chosen=windows[offset:offset+16]
        batch=prepared_batch([w for _,_,w in chosen],'cpu')
        # The primary physical model was trained with unavailable geometry.
        # Supplying new maps to its unused geometry weights would change means.
        physical_batch=batch if cfg.get('trajectory_geometry',False) else dict(batch,
            geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
        belief=encode(model,physical_batch)
        forecast=model.imagine(belief,horizon_s=3.);tracked=tracked_occupancy(forecast)
        target,anonymous=occupancy_targets(batch,12)
        for name,variant in variants.items():
            prior=tables['prior'][:12][None].expand(len(chosen),-1,-1,-1) if variant is None else predict(
                tables,batch,geometry=variant[0],strength=variant[1],history_context=variant[2])
            combined=1-(1-tracked)*(1-prior)
            for i,(root,j,_) in enumerate(chosen):
                for horizon in (1.,3.):
                    length=round(horizon*4)
                    if not bool(batch['targets']['physical_exposure'][i,:length].all()):continue
                    for population,truth,probability in (('total',target,combined),('anonymous',anonymous,prior)):
                        records.append(dict(root_id=root['root_id'],scene=root['scene'],window=j,
                            horizon=horizon,variant=name,population=population,
                            **binary_summary(truth[i,:length].numpy(),probability[i,:length].numpy())))
        if offset%128==0:save_json(Path(cfg['run'])/'progress.json',dict(status='evaluating_anonymous_occupancy',windows=offset+len(chosen),total_windows=len(windows)))
    keys=('average_precision','brier','nll','positive_nll');summary={}
    for distribution in ('origin','all_windows'):
        summary[distribution]={}
        for horizon in (1.,3.):
            summary[distribution][str(horizon)]={}
            for population in ('total','anonymous'):
                grouped={}
                for name in variants:
                    selected=[r for r in records if r['variant']==name and r['population']==population and
                              r['horizon']==horizon and (distribution=='all_windows' or r['window']==0)]
                    grouped[name]=aggregate(selected,keys)
                baseline={r['root_id']:r for r in grouped['unconditional']}
                summary[distribution][str(horizon)][population]={name:dict(
                    metrics={key:scene_interval(rows,key) for key in keys},
                    paired_delta={key:scene_interval(rows,key,baseline) for key in keys}) for name,rows in grouped.items()}
    return dict(records=records,summary=summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);cfg=read(parser.parse_args().config)
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed occupancy experiment is immutable')
    torch.set_num_threads(1);start=time.time();guard=CooperativeGuard(out)
    identity=dict(config=cfg,source_sha256=sha256(__file__),checkpoint_sha256=sha256(cfg['checkpoint']),
        manifest_sha256=sha256(cfg['manifest']),geometry_manifest_sha256=sha256(cfg['geometry_manifest']),
        protocol_sha256=sha256(cfg['protocol']))
    save_json(out/'identity.json',identity);model=load_model(cfg['checkpoint']).eval()
    if set(read(cfg['protocol'])['development']) & set(read(cfg['protocol'])['subsets'][str(cfg.get('scale','full'))]):
        raise ValueError('Fit/development leakage')
    if cfg.get('reuse_prior'):
        if sha256(cfg['reuse_prior'])!=cfg['reuse_prior_sha256']:raise ValueError('Frozen anonymous prior changed')
        tables=torch.load(cfg['reuse_prior'],map_location='cpu');parent=tables['identity']['config']
        if any(parent[key]!=cfg[key] for key in ('manifest','protocol','geometry_manifest')):
            raise ValueError('Anonymous prior fitting population changed')
        if tables['fit_root_ids']!=read(cfg['protocol'])['subsets'][str(cfg.get('scale','full'))]:
            raise ValueError('Anonymous prior fitting roots changed')
        prior_path=Path(cfg['reuse_prior'])
    else:tables=fit(cfg,model,guard,identity);prior_path=out/'prior.pt'
    result=assess(cfg,model,tables,guard)
    result.update(status='occupancy_experiment_complete',identity=identity,runtime_s=time.time()-start,
        prior_sha256=sha256(prior_path),fit_roots=len(tables['fit_root_ids']),
        trajectory_parameters_changed=False,confirmation=False,
        limitation='Marginal anonymous readout only; no exact anonymous identity paths or joint temporal law claimed')
    save_json(out/'result.json',result);print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'])),flush=True)


if __name__=='__main__':main()
