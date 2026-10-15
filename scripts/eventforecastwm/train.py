"""Forecasting-only root-balanced training with exact optimizer/RNG recovery."""
import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from eventbeliefwm.model import PreparedHistory
from eventbeliefwm.metrics import trajectory_energy,mean_valid
from eventbeliefwm.data import sha256,read
from eventbeliefwm.experiment import save_json
from eventforecastwm.data import batch_windows
from eventforecastwm.model import ForecastWorldModel,trajectory_log_probability,energy_score
from eventforecastwm.guard import Guard

torch.set_num_threads(1)


def fingerprints():
    here=Path(__file__).parent
    return {name:sha256(here/name) for name in ('model.py','data.py','train.py','guard.py')}


def load_roots(manifest,ids,guard=None):
    manifest=Path(manifest);rows={r['root_id']:r for r in read(manifest)['roots']};out=[]
    for rid in ids:
        if guard:guard.check()
        row=rows[rid];path=manifest.parent/row['file']
        if sha256(path)!=row['sha256']:raise ValueError('Frozen payload changed: '+rid)
        value=torch.load(path,map_location='cpu')
        if value['root_id']!=rid:raise ValueError('Root identity mismatch')
        out.append(value)
    return out


def prepare_roots(model,roots,guard=None):
    windows=[w for r in roots for w in r['windows']]
    with torch.no_grad():
        for offset in range(0,len(windows),64):
            if guard:guard.check()
            block=windows[offset:offset+64];batch=batch_windows(block)
            prepared=model.core.prepare_history(batch['base'],history_timestamps=batch['history_timestamps'],
                track_acquisition_times=batch['track_acquisition_times'])
            for i,w in enumerate(block):
                n=max(1,w['base'].humans.shape[0]);t=w['base'].robot.shape[0]
                value=prepared.select(torch.tensor([i]),n=n,t=t)
                w['_prepared']=PreparedHistory(**{f.name:(getattr(value,f.name).clone() if isinstance(getattr(value,f.name),torch.Tensor)
                                                         else getattr(value,f.name)) for f in fields(value)})


def prepared_batch(windows,device):
    batch=batch_windows(windows,device);b,n,t=batch['base'].humans.shape[:3]
    values={}
    for f in fields(PreparedHistory):
        name=f.name;first=getattr(windows[0]['_prepared'],name)
        if not isinstance(first,torch.Tensor):values[name]=first;continue
        shape=(b,n,t,13) if name=='clean' else (b,n,*first.shape[2:]) if name!='lengths' else (b,)
        target=first.new_zeros(shape)
        for i,w in enumerate(windows):
            v=getattr(w['_prepared'],name)
            slices=(slice(i,i+1),)+tuple(slice(0,d) for d in v.shape[1:])
            target[slices]=v
        values[name]=target.to(device)
    batch['prepared_history']=PreparedHistory(**values)
    return batch


def encode(model,batch):
    return model.encode_history(batch['base'],{'features':batch['geometry'],'support':batch['geometry_support']},
        batch['sensor_config'],batch['robot_geometry'],prepared_history=batch.get('prepared_history'),
        history_timestamps=batch['history_timestamps'],track_acquisition_times=batch['track_acquisition_times'])


def loss(model,batch,objective='distribution',nll_weight=.05,horizon=3.):
    forecast=model.imagine(encode(model,batch),horizon_s=horizon);l=forecast.positions.shape[2]
    gt=batch['targets']['track_positions'][:,:l];mask=batch['targets']['track_valid'][:,:l]
    if objective=='legacy':es,supported=trajectory_energy(forecast.positions,gt,mask,forecast.weights)
    else:es,supported=energy_score(forecast,gt,mask,draws=max(1,6//forecast.weights.shape[1]))
    lp,count=trajectory_log_probability(forecast,gt,mask)
    energy=mean_valid(es,supported);nll=mean_valid(-lp/count.clamp_min(1),count>0)
    value=energy+nll_weight*nll
    return value,dict(energy=float(energy.detach()),nll=float(nll.detach()),loss=float(value.detach()),
        supported_roots=int(supported.sum()),effective_modes=float((-(forecast.weights*forecast.weights.clamp_min(1e-12).log()).sum(1)).exp().mean().detach()))


@torch.no_grad()
def evaluate(model,roots,device,batch_size=16,horizons=(1.,3.),draws=24):
    was_training=model.training;model.eval();result={}
    for h in horizons:
        records=[];l=round(h/.25);generator=torch.Generator(device=device);generator.manual_seed(73199+int(h*10))
        for offset in range(0,len(roots),batch_size):
            selected=roots[offset:offset+batch_size];batch=prepared_batch([r['windows'][0] for r in selected],device)
            f=model.imagine(encode(model,batch),horizon_s=h)
            gt=batch['targets']['track_positions'][:,:l];raw=batch['targets']['track_valid'][:,:l]
            complete=raw.all(1);mask=complete[:,None].expand_as(raw)
            es,supported=energy_score(f,gt,mask,max(1,math.ceil(draws/f.weights.shape[1])),generator)
            discrete,_=trajectory_energy(f.positions,gt,mask,f.weights)
            lp,count=trajectory_log_probability(f,gt,mask)
            distances=(f.positions-gt[:,None]).norm(dim=-1)
            ade=(distances*mask[:,None]).sum((2,3))/mask.sum((1,2))[:,None].clamp_min(1)
            fde=(distances[:,:,-1]*complete[:,None]).sum(-1)/complete.sum(-1)[:,None].clamp_min(1)
            mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)
            mean_error=(mean-gt).norm(dim=-1)
            mean_ade=(mean_error*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
            mean_fde=(mean_error[:,-1]*complete).sum(-1)/complete.sum(-1).clamp_min(1)
            mode=f.weights.argmax(1)
            variance=f.position_variance
            for i,r in enumerate(selected):
                row=dict(root_id=r['root_id'],scene=r['scene'],agents=int(complete[i].sum()),
                    physical_supported=bool(batch['targets']['physical_exposure'][i,:l].all()))
                if supported[i]:
                    row.update(energy=float(es[i]),discrete_energy=float(discrete[i]),nll=float(-lp[i]/count[i]),
                        mean_ade=float(mean_ade[i]),mean_fde=float(mean_fde[i]),map_ade=float(ade[i,mode[i]]),map_fde=float(fde[i,mode[i]]),
                        minade_centers=float(ade[i].min()),minfde_centers=float(fde[i].min()),
                        expected_ade=float((ade[i]*f.weights[i]).sum()),
                        effective_modes=float((-(f.weights[i]*f.weights[i].clamp_min(1e-12).log()).sum()).exp()),
                        marginal_sigma=float(variance[i,:,:,complete[i]].sqrt().mean()))
                records.append(row)
        keys=('energy','discrete_energy','nll','mean_ade','mean_fde','map_ade','map_fde','minade_centers','minfde_centers','expected_ade','effective_modes','marginal_sigma')
        result[str(h)]=dict(roots=len(records),trajectory_roots=sum('energy' in r for r in records),
            agents=sum(r['agents'] for r in records),physical_roots=sum(r['physical_supported'] for r in records),
            metrics={k:float(np.mean([r[k] for r in records if k in r])) if any(k in r for r in records) else None for k in keys},records=records)
    model.train(was_training);return result


def train(config_path):
    cfg=read(config_path);run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():raise FileExistsError('Completed result is immutable')
    protocol=read(cfg['protocol']);ids=protocol['subsets'][str(cfg['scale'])]
    dev_ids=protocol['development'];device=cfg.get('device','cuda:1');seed=cfg['seed']
    identity=dict(config=cfg,manifest_sha256=sha256(cfg['manifest']),protocol_sha256=sha256(cfg['protocol']),sources=fingerprints())
    if (run/'identity.json').exists() and read(run/'identity.json')!=identity:raise ValueError('Exact resume identity changed')
    save_json(run/'identity.json',identity)
    guard=Guard(run);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if not device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Substantial training requires available CUDA')
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0]<4*1024**3:raise RuntimeError('GPU free memory below4GiB')
    torch.cuda.reset_peak_memory_stats(device)
    model=ForecastWorldModel(**cfg['model'])
    started=time.time();roots=load_roots(cfg['manifest'],ids,guard);dev=load_roots(cfg['manifest'],dev_ids,guard)
    prepare_roots(model,roots+dev,guard)
    save_json(run/'data_usage.json',dict(fit_roots=len(roots),development_roots=len(dev),
        fit_windows=sum(len(r['windows']) for r in roots),prepare_seconds=time.time()-started,
        root_ids=ids,development_ids=dev_ids))
    model.to(device);save_json(run/'architecture.json',model.parameter_counts())
    physical=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('core.')]
    uncertainty=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('core.')]
    groups=[dict(params=uncertainty,lr_factor=cfg.get('noise_lr_factor',10.),weight_decay=0.)]
    if physical:groups.append(dict(params=physical,lr_factor=1.,weight_decay=cfg.get('weight_decay',.01)))
    optimizer=torch.optim.AdamW(groups,lr=cfg['lr'])
    rng=np.random.default_rng(seed);order=rng.permutation(len(roots)).tolist();cursor=0;step=0;presentations=0;seen=set()
    best=float('inf');best_step=None;elapsed=0.;last=run/'last.pt';history=[]
    if last.exists():
        state=torch.load(last,map_location=device)
        if state['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['sampler_rng'];torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        order,cursor,step,presentations=state['order'],state['cursor'],state['step'],state['presentations']
        best,best_step,elapsed,seen=state['best'],state['best_step'],state['elapsed'],set(state['seen'])
    starting=time.time();batch_size=cfg.get('batch_size',32)
    def checkpoint(status):
        state=dict(identity=identity,recipe=model.recipe,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
            order=order,cursor=cursor,step=step,presentations=presentations,seen=sorted(seen),best=best,best_step=best_step,
            elapsed=elapsed+time.time()-starting,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        torch.save(state,run/'last.pt.tmp');os.replace(run/'last.pt.tmp',last)
        save_json(run/'progress.json',dict(status=status,step=step,target_steps=cfg['steps'],unique_roots=len(seen),
            available_roots=len(roots),root_presentations=presentations,effective_epochs=presentations/len(roots),
            elapsed_s=state['elapsed'],best_step=best_step,best_dev_energy=best if np.isfinite(best) else None))
        return state
    try:
        model.train()
        while step<cfg['steps']:
            guard.check()
            if (run/'STOP_RESEARCH').exists():checkpoint('stopped');return
            if cursor>=len(order):order=rng.permutation(len(roots)).tolist();cursor=0
            ix=order[cursor:cursor+batch_size];cursor+=len(ix);presentations+=len(ix);seen.update(ids[i] for i in ix)
            windows=[roots[i]['windows'][int(rng.integers(len(roots[i]['windows'])))] for i in ix]
            batch=prepared_batch(windows,device)
            # Explicit schedule, fixed by optimizer step; no early LR change on inspected eval roots.
            warm=min(1.,(step+1)/cfg.get('warmup',256))
            fraction=max(0.,step-cfg.get('warmup',256))/max(1,cfg['steps']-cfg.get('warmup',256))
            lr=cfg['lr']*warm*(.1+.9*.5*(1+math.cos(math.pi*fraction))) if cfg.get('schedule','cosine')=='cosine' else cfg['lr']
            for group in optimizer.param_groups:group['lr']=lr*group['lr_factor']
            optimizer.zero_grad(set_to_none=True)
            value,metrics=loss(model,batch,cfg.get('objective','distribution'),cfg.get('nll_weight',.05),cfg.get('horizon',3.))
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite forecast objective')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite forecast gradient')
            optimizer.step();step+=1
            if step%32==0 or step<=4:
                record=dict(step=step,seconds=elapsed+time.time()-starting,lr=lr,gradient_norm=float(norm),
                    root_presentations=presentations,effective_epochs=presentations/len(roots),unique_roots=len(seen),**metrics)
                with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if step%cfg.get('eval_every',512)==0 or step==cfg['steps']:
                scores=evaluate(model,dev,device,batch_size,horizons=(1.,3.),draws=24)
                score=scores['3.0']['metrics']['energy'];improved=score is not None and score<best
                if improved:best=score;best_step=step
                state=checkpoint('running')
                if improved:torch.save(state,run/'best.pt.tmp');os.replace(run/'best.pt.tmp',run/'best.pt')
                save_json(run/f'dev_step{step:06d}.json',scores)
                with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,best_step=best_step,
                    scores={h:v['metrics'] for h,v in scores.items()}))+'\n')
            elif step%128==0:checkpoint('running')
        checkpoint('complete')
        save_json(run/'result.json',dict(status='training_complete',steps=step,training_roots=len(seen),
            presentations=presentations,epochs=presentations/len(roots),duration_s=elapsed+time.time()-starting,
            best_step=best_step,best_dev_energy=best,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
            best_sha256=sha256(run/'best.pt'),last_sha256=sha256(last)))
    except BaseException as error:
        checkpoint('failed');save_json(run/'failure.json',dict(error=repr(error),step=step));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();train(a.config)


if __name__=='__main__':main()
