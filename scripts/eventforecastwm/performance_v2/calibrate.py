"""Controlled fixed-mean conditional probability fit, balanced by root.

All-window likelihood averages supported windows inside each fitting root.
Targets enter density evaluation only; legal contexts are computed beforehand
with every legal track. This does not alter the physical mean or tracker.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.agent_scale_calibration import pack_supported
from eventforecastwm.benchmark import load_model
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.model_agent_scale import AgentScaleWorldModel, component_log_densities
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from . import density_cache


def tensor_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def root_horizon_weights(root_ids, row_horizons, like):
    """Equal roots, then available horizons, then supported windows.

    Horizons with no supported window for a root contribute no invented
    target. Repeated query branches never become independent observations.
    """
    if not root_ids or len(root_ids)!=len(row_horizons):
        raise ValueError('Missing or misaligned root/horizon identities')
    counts=Counter(zip(root_ids,row_horizons));available=defaultdict(set)
    for rid,horizon in counts:available[rid].add(horizon)
    weights=like.new_tensor([1./(len(available[rid])*counts[(rid,h)])
                            for rid,h in zip(root_ids,row_horizons)])/len(available)
    if not torch.allclose(weights.sum(),weights.new_tensor(1.)):
        raise AssertionError('Unnormalized root/horizon weights')
    return weights


def fit(cfg):
    run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():raise FileExistsError('Completed calibration is immutable')
    guard=CooperativeGuard(run)
    identity=dict(config=cfg,checkpoint_sha256=sha256(cfg['checkpoint']),manifest_sha256=sha256(cfg['manifest']),
                  protocol_sha256=sha256(cfg['protocol']),source_sha256=sha256(__file__))
    if cfg.get('incremental_density_cache'):
        identity['density_cache_sha256']=sha256(density_cache.__file__)
    if (run/'identity.json').exists() and read(run/'identity.json')!=identity:raise ValueError('Calibration resume identity changed')
    save_json(run/'identity.json',identity)
    device=cfg.get('device','cuda:1');torch.set_num_threads(1)
    torch.manual_seed(cfg.get('seed',20261015));np.random.seed(cfg.get('seed',20261015))
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
        if torch.cuda.mem_get_info(device)[0]<8*1024**3:raise RuntimeError('Research stopped: GPU reserve below 8 GiB')
        torch.cuda.reset_peak_memory_stats(device)
    original=load_model(cfg['checkpoint']).eval()
    if not isinstance(original,AgentScaleWorldModel):raise ValueError('Controlled experiment requires per-agent variance parent')
    recipe=dict(original.recipe);recipe['conditional']=cfg['conditional']
    recipe['age_aware']=cfg.get('age_aware',recipe['age_aware'])
    model=AgentScaleWorldModel(**recipe).eval();model.load_state_dict(original.state_dict())
    physical_hash=tensor_hash(model.core.state_dict())
    parent=torch.load(cfg['checkpoint'],map_location='cpu')
    protocol=read(cfg['protocol']);ids=protocol['subsets'][str(cfg.get('fit_scale','full'))]
    if set(ids)&set(protocol['development']):raise ValueError('Development entered calibration')
    horizons=tuple(float(h) for h in cfg.get('horizons',[3.]))
    if not horizons or len(set(horizons))!=len(horizons) or any(h not in (1.,3.) for h in horizons):
        raise ValueError('Probability control supports distinct one/three-second horizons')
    if horizons!=(3.,) and not cfg.get('incremental_density_cache'):
        raise ValueError('Multi-horizon fitting requires recoverable density chunks')
    started=time.time();cache=run/'densities.pt'
    if cfg.get('density_source'):
        source=Path(cfg['density_source']);source_identity=read(source/'identity.json')
        fields=('checkpoint','manifest','protocol','windows','fit_scale','age_aware','horizons')
        if any(source_identity['config'].get(k)!=cfg.get(k) for k in fields):raise ValueError('Incompatible density cache reuse')
        if source_identity['source_sha256']!=identity['source_sha256']:raise ValueError('Density implementation changed')
        if sha256(source/'densities.pt')!=read(source/'densities.json')['sha256']:raise ValueError('Reused density bytes changed')
        saved=torch.load(source/'densities.pt',map_location='cpu')
        density,counts,context,weights=saved['tensors'];root_ids=saved['root_ids']
        row_horizons=saved.get('row_horizons',[3.]*len(root_ids))
    elif cache.exists():
        if sha256(cache)!=read(run/'densities.json')['sha256']:raise ValueError('Calibration cache changed')
        saved=torch.load(cache,map_location='cpu')
        if saved['identity']!=identity:raise ValueError('Calibration cache identity changed')
        density,counts,context,weights=saved['tensors'];root_ids=saved['root_ids']
        row_horizons=saved.get('row_horizons',[3.]*len(root_ids))
    elif cfg.get('incremental_density_cache'):
        model.to(device)
        @torch.no_grad()
        def build(selected_ids):
            roots=load_roots(cfg['manifest'],selected_ids,guard)
            prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
            windows=[(r['root_id'],w) for r in roots for w in (r['windows'] if cfg['windows']=='all' else r['windows'][:1])]
            cached=[];root_ids=[]
            for start in range(0,len(windows),32):
                selected=windows[start:start+32];batch=prepared_batch([w for _,w in selected],device)
                ticks=round(horizon/.25)
                f=model.imagine(encode(model,batch),horizon_s=horizon)
                raw=batch['targets']['track_valid'][:,:ticks];mask=raw.all(1)[:,None].expand_as(raw)
                d,c=component_log_densities(f,batch['targets']['track_positions'][:,:ticks],mask)
                keep=c.sum(-1)>0
                cached.append(tuple(x[keep].cpu() for x in (d,c,f.legal_context,f.weights)))
                root_ids.extend(rid for (rid,_),ok in zip(selected,keep.tolist()) if ok)
            return cached,root_ids
        cached=[];root_ids=[];row_horizons=[]
        for horizon in horizons:
            directory=run if horizons==(3.,) else run/f'horizon_{horizon:g}'
            directory.mkdir(exist_ok=True)
            part_identity=dict(identity,horizon=horizon) if horizons!=(3.,) else identity
            tensors,rows=density_cache.materialize(directory,part_identity,ids,build,guard)
            cached.append(tensors);root_ids.extend(rows);row_horizons.extend([horizon]*len(rows))
        density,counts,context,weights=pack_supported(cached)
        torch.save(dict(identity=identity,tensors=(density,counts,context,weights),root_ids=root_ids,row_horizons=row_horizons),run/'densities.tmp')
        os.replace(run/'densities.tmp',cache);save_json(run/'densities.json',dict(sha256=sha256(cache)))
    else:
        model.to(device);cached=[];root_ids=[]
        with torch.no_grad():
            for offset in range(0,len(ids),64):
                guard.check();roots=load_roots(cfg['manifest'],ids[offset:offset+64],guard)
                prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
                windows=[(r['root_id'],w) for r in roots for w in (r['windows'] if cfg['windows']=='all' else r['windows'][:1])]
                for start in range(0,len(windows),32):
                    selected=windows[start:start+32]
                    batch=prepared_batch([w for rid,w in selected],device)
                    f=model.imagine(encode(model,batch),horizon_s=3.)
                    raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
                    d,c=component_log_densities(f,batch['targets']['track_positions'][:,:12],mask)
                    keep=c.sum(-1)>0
                    cached.append(tuple(x.cpu() for x in (d[keep],c[keep],f.legal_context[keep],f.weights[keep])))
                    root_ids.extend(rid for (rid,w),ok in zip(selected,keep.tolist()) if ok)
                save_json(run/'progress.json',dict(status='building_fixed_density_cache',roots=min(offset+64,len(ids)),total_roots=len(ids),supported_windows=len(root_ids)))
        density,counts,context,weights=pack_supported(cached);row_horizons=[3.]*len(root_ids)
        torch.save(dict(identity=identity,tensors=(density,counts,context,weights),root_ids=root_ids),run/'densities.tmp')
        os.replace(run/'densities.tmp',cache);save_json(run/'densities.json',dict(sha256=sha256(cache)))
    density,counts,context,weights=(x.to(device) for x in (density,counts,context,weights))
    model.to(device)
    if set(row_horizons)!=set(horizons):raise ValueError('Density cache horizon membership changed')
    occurrences=Counter(root_ids)
    root_weight=root_horizon_weights(root_ids,row_horizons,density)
    # Prefix-complete support is nested: every complete three-second window
    # also has a complete first second. Count windows once at shortest horizon.
    supported_windows=sum(h==min(horizons) for h in row_horizons)
    for p in model.parameters():p.requires_grad_(False)
    model.scale_logits.requires_grad_(True);parameters=[model.scale_logits]
    if cfg['conditional']:
        for p in model.scale_context.parameters():p.requires_grad_(True);parameters.append(p)
    optimizer=torch.optim.Adam(parameters,lr=cfg.get('lr',.03))
    step=0;curve=[];steps=cfg.get('steps',512)
    if (run/'last.pt').exists():
        last=torch.load(run/'last.pt',map_location=device)
        if last['identity']!=identity:raise ValueError('Optimizer resume identity changed')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);step=last['step'];curve=last['curve']
    def checkpoint(status):
        torch.save(dict(identity=identity,recipe=model.recipe,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,curve=curve),run/'last.tmp')
        os.replace(run/'last.tmp',run/'last.pt')
        save_json(run/'progress.json',dict(status=status,step=step,target_steps=steps,fit_roots=len(ids),supported_roots=len(occurrences),supported_windows=supported_windows,likelihood_window_horizon_pairs=len(root_ids)))
    try:
        for step0 in range(step,steps):
            guard.check()
            logp=model.probabilities(context,weights.shape[1]).clamp_min(1e-12).log()
            component=((density+logp).logsumexp(-1)*(counts>0)[:,None]).sum(-1)
            per_window=-(component+weights.clamp_min(1e-12).log()).logsumexp(-1)/counts.sum(-1)
            nll=(per_window*root_weight).sum()
            penalty=cfg.get('regularization',.001)*sum(p.square().sum() for p in model.scale_context.parameters()) if cfg['conditional'] else nll*0
            loss=nll+penalty
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite conditional likelihood')
            optimizer.zero_grad();loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('Invalid calibration gradient')
            optimizer.step();step=step0+1
            if step%32==0 or step==1:
                curve.append(dict(step=step,nll=float(nll.detach()),penalty=float(penalty.detach())))
                checkpoint('running')
        checkpoint('calibration_complete')
    except BaseException as error:
        checkpoint('paused_or_failed');save_json(run/'failure.json',dict(error=repr(error),step=step));raise
    if tensor_hash(model.core.state_dict())!=physical_hash:raise AssertionError('Physical mean changed during probability calibration')
    result=dict(status='calibration_complete',steps=step,fit_roots=len(ids),supported_roots=len(occurrences),
                supported_windows=supported_windows,likelihood_window_horizon_pairs=len(root_ids),
                supported_windows_by_horizon=dict(Counter(row_horizons)),curve=curve,physical_hash=physical_hash,physical_means_unchanged=True,
                runtime_s=time.time()-started,recipe=model.recipe,identity=identity,
                objective='mean roots, then available horizons, then supported-window exact NLL per coordinate, plus context L2 penalty',
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)
    model.cpu()
    torch.save(dict(recipe=model.recipe,model=model.state_dict(),calibration=result,parent_checkpoint_sha256=identity['checkpoint_sha256'],
                    fit_root_ids=ids,anonymous_occupancy_prior=parent['anonymous_occupancy_prior'],
                    anonymous_exposure_counts=parent.get('anonymous_exposure_counts'),physical_source_parent=cfg['checkpoint']),run/'model.pt')
    result['checkpoint_sha256']=sha256(run/'model.pt');save_json(run/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','curve')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);fit(read(p.parse_args().config))


if __name__=='__main__':main()
