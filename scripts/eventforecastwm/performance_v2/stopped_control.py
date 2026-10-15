"""Bounded training-only persistent stopped-motion mixture experiment.

This offline control is deliberately not a released recurrent model. Its branch
is persistent for the entire sampled future; a positive result must earn a full
belief/update implementation. Both branches use the frozen parent's OU law.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import occupancy_targets, save_json
from eventbeliefwm.metrics import binary_summary
from eventforecastwm.agent_scale_calibration import pack_supported
from eventforecastwm.benchmark import load_model, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.model_agent_scale import component_log_densities
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from eventforecastwm.window_evaluate import aggregate
from .occupancy_prior import predict, atomic_save


def stopped_forecast(f,belief):
    position=belief.core.last_observed_positions[:,None,None].expand_as(f.positions)
    return replace(f,positions=position,velocities=torch.zeros_like(f.velocities))


def branch_density(f,stopped,truth,mask):
    a,count=component_log_densities(f,truth,mask)
    b,other=component_log_densities(stopped,truth,mask)
    if not torch.equal(count,other):raise AssertionError('Branch support changed')
    log_scale=f.agent_scale_probabilities.clamp_min(1e-12).log()
    return torch.stack([(a+log_scale).logsumexp(-1),(b+log_scale).logsumexp(-1)],-1),count


def joint_log_probability(density,counts,weights,probability):
    # Unsupported agents integrate to one and cannot become negative examples.
    agent=(density+probability.log()[:,None]).logsumexp(-1)
    return ((agent*(counts>0)[:,None]).sum(-1)+weights.clamp_min(1e-12).log()).logsumexp(-1)


def mixed_paths(paths,f,stopped,probability,generator):
    b,s,t,n,_=f.positions.shape;draws=paths.shape[1]//s
    moving=f.positions[:,:,None].expand(b,s,draws,t,n,2).reshape_as(paths)
    stationary=stopped.positions[:,:,None].expand(b,s,draws,t,n,2).reshape_as(paths)
    branch=torch.rand(b,s*draws,1,n,1,generator=generator,device=paths.device)<probability[:,None,None,:,1:2]
    return paths+branch*(stationary-moving)


def weighted_energy(a,b,weights,truth,mask):
    m=mask[...,None].to(a);scale=(2*m.sum((1,2,3))).clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (weights*first).sum(1)-.5*(weights[:,:,None]*weights[:,None]*pair).sum((1,2))


def agent_cell_mass(f,size=32,extent=8.):
    edges=torch.linspace(-extent,extent,size+1,device=f.positions.device,dtype=f.positions.dtype)
    base=f.covariance[...,0,0]+f.readout_scale.square();mass=0.
    for index,variance in enumerate(f.scale_values):
        sigma=(base*variance).sqrt()
        cdf=.5*(1+torch.erf((edges-f.positions[...,None])/(sigma[...,None,None]*math.sqrt(2))))
        axis=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
        mass=mass+axis[...,1,:,None]*axis[...,0,None,:]*f.agent_scale_probabilities[...,index][:,:,None,:,None,None]
    return mass*f.valid[:,None,None,:,None,None]


def mixed_occupancy(f,stopped,probability):
    p=probability[:,None,None,:,:,None,None]
    branches=torch.stack([agent_cell_mass(f),agent_cell_mass(stopped)],4)
    mass=(branches*p).sum(4)
    return ((1-(1-mass).prod(3))*f.weights[:,:,None,None,None]).sum(1)


def gate():
    head=nn.Linear(8,2);nn.init.zeros_(head.weight)
    with torch.no_grad():head.bias.copy_(torch.tensor([math.log(.95),math.log(.05)]))
    return head


def fit(cfg,model,guard,identity):
    out=Path(cfg['run']);ids=read(cfg['protocol'])['subsets'][str(cfg.get('fit_scale','1024'))]
    if set(ids)&set(read(cfg['protocol'])['development']):raise ValueError('Fitting leakage')
    device=cfg.get('device','cpu');model.to(device);cache=out/'densities.pt'
    if cache.exists():
        saved=torch.load(cache,map_location=device)
        if saved['identity']!=identity:raise ValueError('Density identity changed')
        tensors=saved['tensors'];root_ids=saved['root_ids']
    else:
        cached=[];root_ids=[];parts=out/'density_parts';parts.mkdir(exist_ok=True)
        with torch.no_grad():
            for start in range(0,len(ids),64):
                guard.check();part=parts/f'{start:06d}.pt';part_ids=ids[start:start+64]
                if part.exists():
                    if sha256(part)!=read(part.with_suffix('.json'))['sha256']:raise ValueError('Density chunk bytes changed')
                    saved=torch.load(part,map_location='cpu')
                    if saved['identity']!=identity or saved['fit_root_ids']!=part_ids:raise ValueError('Density chunk identity changed')
                    if saved['tensors'] is not None:cached.append(saved['tensors'])
                    root_ids.extend(saved['root_ids']);continue
                roots=load_roots(cfg['manifest'],part_ids,guard);part_cached=[];part_roots=[]
                prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
                windows=[(r['root_id'],w) for r in roots for w in r['windows']]
                for offset in range(0,len(windows),16):
                    selected=windows[offset:offset+16];batch=prepared_batch([w for _,w in selected],device)
                    belief=encode(model,batch);f=model.imagine(belief,horizon_s=3.)
                    context=f.legal_context;stationary=stopped_forecast(f,belief)
                    raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
                    d,c=branch_density(f,stationary,batch['targets']['track_positions'][:,:12],mask)
                    keep=c.sum(-1)>0
                    part_cached.append(tuple(x[keep].cpu() for x in (d,c,context,f.weights)))
                    part_roots.extend(rid for (rid,_),ok in zip(selected,keep.tolist()) if ok)
                tensors=pack_supported(part_cached) if part_roots else None
                atomic_save(dict(identity=identity,fit_root_ids=part_ids,tensors=tensors,root_ids=part_roots),part)
                save_json(part.with_suffix('.json'),dict(sha256=sha256(part),roots=len(part_ids),supported_windows=len(part_roots)))
                if tensors is not None:cached.append(tensors)
                root_ids.extend(part_roots)
                save_json(out/'progress.json',dict(status='density_cache',roots=min(start+64,len(ids)),total_roots=len(ids)))
        tensors=pack_supported(cached)
        atomic_save(dict(identity=identity,tensors=tensors,root_ids=root_ids),cache)
    d,c,x,w=(t.to(device) for t in tensors);head=gate().to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=.03);step=0;curve=[]
    occurrences=Counter(root_ids);root_weight=d.new_tensor([1/occurrences[rid] for rid in root_ids])/len(occurrences)
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location=device)
        if state['identity']!=identity:raise ValueError('Gate resume identity changed')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer']);step=state['step'];curve=state['curve']
    def save():
        atomic_save(dict(identity=identity,head=head.state_dict(),optimizer=optimizer.state_dict(),step=step,curve=curve),out/'last.pt')
    try:
        for current in range(step,cfg.get('steps',512)):
            guard.check();lp=joint_log_probability(d,c,w,head(x).softmax(-1))
            nll=(-lp/c.sum(-1)*root_weight).sum();penalty=.001*head.weight.square().sum();loss=nll+penalty
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite regime likelihood')
            optimizer.zero_grad();loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();step=current+1
            if step%32==0 or step==1:curve.append(dict(step=step,nll=float(nll),penalty=float(penalty)));save()
        save()
    except BaseException:
        save();raise
    receipt=dict(steps=step,fit_roots=len(ids),supported_roots=len(occurrences),supported_windows=len(root_ids),curve=curve,
        parameters=sum(p.numel() for p in head.parameters()),physical_parameters_frozen=True,
        note='Distribution mean changes by mixing a stationary alternative; physical branch parameters and noise stay frozen')
    atomic_save(dict(identity=identity,head=head.cpu().state_dict(),fit_root_ids=ids,receipt=receipt),out/'gate.pt')
    save_json(out/'fit.json',receipt);return head.cpu(),receipt


@torch.no_grad()
def assess(cfg,model,head,guard):
    model.cpu();protocol=read(cfg['protocol']);roots=load_roots(cfg['manifest'],protocol['development'],guard)
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    attach_geometry(roots,cfg['geometry_manifest'],guard)
    tables=torch.load(cfg['occupancy_prior'],map_location='cpu')
    if set(tables['fit_root_ids'])&set(protocol['development']):raise ValueError('Occupancy leakage')
    windows=[(r,j,w) for r in roots for j,w in enumerate(r['windows'])];records=[]
    for horizon in (1.,3.):
        length=round(horizon*4);rng=torch.Generator().manual_seed(73199+int(horizon*10))
        branch_rng=torch.Generator().manual_seed(93211+int(horizon*10))
        for offset in range(0,len(windows),8):
            guard.check();selected=windows[offset:offset+8];batch=prepared_batch([w for _,_,w in selected],'cpu')
            physical=dict(batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
            belief=encode(model,physical);f=model.imagine(belief,horizon_s=horizon);stationary=stopped_forecast(f,belief)
            probability=head(f.legal_context).softmax(-1)
            stop_weight=cfg.get('stop_weight',1.)
            if not 0<=stop_weight<=1:raise ValueError('Invalid mixture dilution')
            stop=probability[...,1]*stop_weight
            probability=torch.stack([1-stop,stop],-1)
            raw=batch['targets']['track_valid'][:,:length];mask=raw.all(1)[:,None].expand_as(raw)
            truth=batch['targets']['track_positions'][:,:length];d,c=branch_density(f,stationary,truth,mask)
            a,weights,_=f.sample(8,rng);b,_,_=f.sample(8,rng)
            mixed_a=mixed_paths(a,f,stationary,probability,branch_rng);mixed_b=mixed_paths(b,f,stationary,probability,branch_rng)
            prior=predict(tables,batch,length=length,geometry=True,strength=128.,history_context=False)
            occupancy_truth,_=occupancy_targets(batch,length)
            no_stop=torch.stack([torch.ones_like(probability[...,0]),torch.zeros_like(probability[...,1])],-1)
            for name,pr,aa,bb in (('parent',no_stop,a,b),('stopped_mixture',probability,mixed_a,mixed_b)):
                es=weighted_energy(aa,bb,weights,truth,mask)
                lp=joint_log_probability(d,c,f.weights,pr)
                mean=((f.positions*pr[:,None,None,:,0:1]+stationary.positions*pr[:,None,None,:,1:2])*f.weights[:,:,None,None,None]).sum(1)
                distance=(mean-truth).norm(dim=-1);ade=(distance*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
                tracked=mixed_occupancy(f,stationary,pr);occupancy=1-(1-tracked)*(1-prior)
                for i,(root,j,_) in enumerate(selected):
                    row=dict(root_id=root['root_id'],scene=root['scene'],window=j,horizon=horizon,variant=name)
                    if c[i].sum()>0:row.update(energy=float(es[i]),nll=float(-lp[i]/c[i].sum()),mean_ade=float(ade[i]),
                        stop_probability=float(pr[i,f.valid[i],1].mean()))
                    if bool(batch['targets']['physical_exposure'][i,:length].all()):
                        row.update({'occupancy_'+k:v for k,v in binary_summary(occupancy_truth[i].numpy(),occupancy[i].numpy()).items()})
                    records.append(row)
            save_json(Path(cfg['run'])/'progress.json',dict(status='evaluation',horizon=horizon,windows=offset+len(selected),total_windows=len(windows)))
    keys=('energy','nll','mean_ade','stop_probability','occupancy_average_precision','occupancy_brier','occupancy_nll')
    summary={}
    for population in ('origin','all_windows'):
        summary[population]={}
        for horizon in (1.,3.):
            grouped={name:aggregate([r for r in records if r['variant']==name and r['horizon']==horizon and
                (population=='all_windows' or r['window']==0)],keys) for name in ('parent','stopped_mixture')}
            reference={r['root_id']:r for r in grouped['parent']}
            summary[population][str(horizon)]={name:dict(metrics={key:scene_interval(rows,key) for key in keys},
                paired_delta={key:scene_interval(rows,key,reference) for key in keys}) for name,rows in grouped.items()}
    return dict(records=records,summary=summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);cfg=read(parser.parse_args().config)
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed control is immutable')
    torch.set_num_threads(1);torch.manual_seed(20261015);start=time.time()
    identity=dict(config=cfg,source_sha256=sha256(__file__),**{k+'_sha256':sha256(cfg[k]) for k in
        ('checkpoint','protocol','manifest','geometry_manifest','occupancy_prior')})
    if cfg.get('reuse_gate'):identity['gate_sha256']=sha256(cfg['reuse_gate'])
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Control identity changed')
    save_json(out/'identity.json',identity)
    try:
        guard=CooperativeGuard(out);model=load_model(cfg['checkpoint']).eval()
        for p in model.parameters():p.requires_grad_(False)
        if cfg.get('reuse_gate'):
            saved=torch.load(cfg['reuse_gate'],map_location='cpu');parent=saved['identity']['config']
            if sha256(cfg['reuse_gate'])!=cfg['reuse_gate_sha256']:raise ValueError('Reused gate changed')
            if any(parent[k]!=cfg[k] for k in ('checkpoint','protocol','manifest','fit_scale')):
                raise ValueError('Incompatible gate population or physical law')
            head=gate();head.load_state_dict(saved['head']);fitting=saved['receipt']
            fitting=dict(fitting,reused_gate=str(cfg['reuse_gate']),new_optimizer_steps=0)
        elif (out/'fit.json').exists():
            saved=torch.load(out/'gate.pt',map_location='cpu')
            if saved['identity']!=identity:raise ValueError('Fitted gate identity changed')
            head=gate();head.load_state_dict(saved['head']);fitting=read(out/'fit.json')
        else:head,fitting=fit(cfg,model,guard,identity)
        result=assess(cfg,model,head,guard)
        result.update(status='probability_control_complete',identity=identity,fitting=fitting,runtime_s=time.time()-start,
            sampling='48 stratified paths per bank for both laws; common Gaussian draws; persistent independent agent branch',
            limitation='Offline fixed-history experiment. Not a released recurrent update implementation.',confirmation=False)
        save_json(out/'result.json',result);print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'])),flush=True)
    except BaseException as error:save_json(out/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
