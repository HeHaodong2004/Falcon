"""Proper path Energy Score using samples from the deployed compound OU law.

Linear OU factors and native velocity jumps allow exact path sampling at four
arbitrary native timestamps without executing every intermediate state in torch.
Intention is persistent across the entire path. No Gaussian long-path NLL is used.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch

from eventwm_graph.l2_partial.neural import GraphDynamics
from eventwm_graph.l2_partial.probability import transition_matrices
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT, SNAP


def psd_root(c):
    e,u=np.linalg.eigh(c)
    if np.min(e)<-1e-7: raise ValueError('indefinite training covariance')
    return (u*np.sqrt(np.maximum(e,0))[...,None,:])@u.swapaxes(-1,-2)


def path_factors(times, until):
    t=np.asarray(times)-times[0];k=.7;H=len(t)
    a=-np.expm1(-k*t)/k
    covariance=np.zeros((H,H))
    for i,s in enumerate(t):
        for j,u in enumerate(t):
            v=min(s,u);_,Q,_,_=transition_matrices(v,k)
            ad=-np.expm1(-k*abs(s-u))/k
            covariance[i,j]=Q[0,0]+ad*Q[0,2]
    nextgrid=(math.floor((times[0]+1e-9)/.048)+1)*.048
    if abs(times[0]/.048-round(times[0]/.048))<1e-7 and until<=times[0]+1e-9:
        nextgrid=times[0]
    grids=np.arange(nextgrid,times[-1]+1e-9,.048)
    delta=np.maximum(times[:,None]-grids[None,:],0.)
    jump=-np.expm1(-k*delta)/k
    return a,psd_root(covariance),psd_root(.0025*np.minimum(t[:,None],t[None,:])),jump


def component_draw(row,rng):
    p=np.array([c['weight'] for c in row['components']]);p/=p.sum()
    return row['components'][rng.choice(len(p),p=p)]


def make_batch(rows, rng, motion, samples=4, device='cpu'):
    B=len(rows);N=max(len(c['mean']) for r in rows for c in r['components']);K=3;S=samples;H=4
    arrays={k:np.zeros(shape,np.float32) for k,shape in dict(mean=(B,N,4),age=(B,N),graph=(B,N,18),mask=(B,N),
        target=(B,N,H,2),valid=(B,N,H),base=(B,N,K,S,H,2),ou=(B,N,K,S,H,2),
        target_factor=(B,H)).items()}
    jump_p=-np.expm1(-motion['maneuver_rate']*.048)
    for b,row in enumerate(rows):
        c=component_draw(row,rng);n=len(c['mean']);alltimes=row['times']
        take=np.linspace(1,len(alltimes)-1,min(H,len(alltimes)-1)).astype(int)
        validH=len(take);take=np.pad(take,(0,H-validH),mode='edge');times=alltimes[take]
        a,L,Brown,J=path_factors(np.r_[alltimes[0],times],c['maneuver_until'])
        a,L,Brown,J=a[1:],L[1:,1:],Brown[1:,1:],J[1:]
        t=times-alltimes[0]
        arrays['mean'][b,:n]=c['mean'];arrays['age'][b,:n]=c['age'];arrays['graph'][b,:n]=c['graph'];arrays['mask'][b,:n]=1
        selected=c['selected'];arrays['target'][b,selected]=c['truth'][take].transpose(1,0,2)
        arrays['valid'][b,selected,:validH]=1
        arrays['target_factor'][b]=t-a
        # Independent paths for every intention mode ensure unbiased cross-mode
        # ES pair terms. Initial state is sampled from the actual retained mixture.
        if c['maneuver_mean'] is not None:
            gm,gc,gp=c['maneuver_mean'],c['maneuver_covariance'],c['maneuver_probability']
            choice=(rng.random((n,K,S,1))>np.cumsum(gp,axis=-1)[:,None,None,:]).sum(-1)
            init=gm[np.arange(n)[:,None,None],choice]
            root=psd_root(gc)[np.arange(n)[:,None,None],choice]
        else:
            init=np.broadcast_to(c['mean'][:,None,None,:],(n,K,S,4))
            root=np.broadcast_to(psd_root(c['covariance'])[:,None,None],(n,K,S,4,4))
        init=init+np.einsum('nksij,nksj->nksi',root,rng.normal(size=(n,K,S,4)))
        base=init[...,None,:2]+a[None,None,None,:,None]*init[...,None,2:]
        base+=np.einsum('ij,nksjd->nksid',Brown,rng.normal(size=(n,K,S,H,2)))
        if J.shape[1]:
            impulse=(rng.random((n,K,S,J.shape[1],1))<jump_p)*rng.normal(size=(n,K,S,J.shape[1],2))*motion['reset_std']
            base+=np.einsum('hg,nksgd->nkshd',J,impulse)
        arrays['base'][b,:n]=base
        arrays['ou'][b,:n]=np.einsum('ij,nksjd->nksid',L,rng.normal(size=(n,K,S,H,2)))
    return {k:torch.as_tensor(v,device=device) for k,v in arrays.items()}


def mixture_path_energy(draws, logits, target, valid):
    """Unbiased ES estimator for a 3-mode path mixture; independent MC paths."""
    B,N,K,S,H,D=draws.shape
    present=valid.sum(-1)
    scale=(valid/present.clamp(min=1)[...,None]).sqrt()
    paths=(draws*scale[:,:,None,None,:,None]).reshape(B,N,K,S,H*D)
    truth=(target*scale[...,None]).reshape(B,N,1,1,H*D)
    # sqrt(eps) is subtracted to keep coincident paths exactly zero and gradients
    # finite; the 1e-8 m smoothing is explicitly numerical, below scoring precision.
    def norm(x):return (x.square().sum(-1)+1e-16).sqrt()-1e-8
    first=norm(paths-truth).mean(-1)
    flat=paths.reshape(B,N,K*S,H*D)
    distance=norm(flat[:,:,:,None,:]-flat[:,:,None,:,:]).reshape(B,N,K,S,K,S)
    pair=distance.sum((3,5))/(S*S)
    diag=torch.arange(K,device=draws.device)
    pair[:,:,diag,diag]*=S/(S-1)
    p=torch.softmax(logits,-1)
    score=(first*p).sum(-1)-.5*torch.einsum('bnk,bnkl,bnl->bn',p,pair,p)
    people=(present>0).to(score.dtype)
    return (score*people).sum(-1)/people.sum(-1).clamp(min=1)


def objective(network,b):
    target,q,logits=network(b['mean'],b['age'],b['graph'],b['mask'])
    draws=b['base']+b['target_factor'][:,None,None,None,:,None]*target[:,:,:,None,None,:]
    draws=draws+q.sqrt()[:,:,:,None,None,None]*b['ou']
    return mixture_path_energy(draws,logits,b['target'],b['valid']).mean()


def evaluate(network,rows,motion,device):
    rng=np.random.default_rng(212);values=[]
    with torch.no_grad():
        for i in range(0,len(rows),48):
            part=rows[i:i+48];b=make_batch(part,rng,motion,8,device)
            values.append((len(part),float(objective(network,b))))
    return sum(n*v for n,v in values)/sum(n for n,_ in values)


def train(epochs=50,smoke=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.manual_seed(211);np.random.seed(211);random.seed(211)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.20)
    train_rows=[r for r in np.load(OUT/('train_smoke.npy' if smoke else 'train.npy'),allow_pickle=True) if r['components']]
    dev_rows=[r for r in np.load(OUT/('dev_smoke.npy' if smoke else 'dev.npy'),allow_pickle=True) if r['components']]
    motion=json.loads((OUT/'motion.json').read_text());dest=OUT/('training_smoke' if smoke else 'training');dest.mkdir(exist_ok=False)
    if motion.get('maneuver_retention',1.)!=1.:
        raise ValueError('this path-factor objective requires mean-preserving impulses')
    net=GraphDynamics(64).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.00005)
    rng=np.random.default_rng(211);history=[];best=math.inf;stale=0;started=time.perf_counter();cpu0=time.process_time()
    source={str(p):sha(p) for folder in ('eventwm_graph/l2_120k','eventwm_graph/l2_partial') for p in Path(folder).glob('*.py')}
    configuration=dict(seed=211,epochs=epochs,minimum_epochs=15,patience=8,improvement=.001,batch_size=48,
        device=device,parameters=sum(p.numel() for p in net.parameters()),motion=motion,
        training_roots=len(train_rows),dev_roots=len(dev_rows),width=64,
        objective='root-balanced persistent-mixture path ES; 4 samples/mode train,8 dev',
        objective_version='compound-native-persistent-path-ES-2',native_dt=.048,relaxation=.7,
        independent_position_variance_rate=.0025,
        manifest_sha256=sha(SNAP/'manifest.json'),sources=source,
        train_cache_sha256=sha(OUT/('train_smoke.npy' if smoke else 'train.npy')),
        dev_cache_sha256=sha(OUT/('dev_smoke.npy' if smoke else 'dev.npy')),
        population_prior_sha256=sha(OUT/'population_prior.json'),motion_sha256=sha(OUT/'motion.json'),
        observation_sha256=sha(Path('docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/observation_fit.json')),
        frozen_secondary_event_head_sha256=sha(Path('docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json')),
        gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        gpu_memory_free_total=torch.cuda.mem_get_info() if device=='cuda' else None,
        gpu_utilization_status='NVML unavailable: driver/library mismatch; CUDA works; no invented hardware-utilization percentage')
    (dest/'configuration.json').write_text(json.dumps(configuration,indent=2))
    for epoch in range(epochs+1):
        losses=[];net.train();cuda_step_ms=0.
        if epoch:
            for i in range(0,len(train_rows),48):
                if i==0:order=rng.permutation(len(train_rows))
                part=[train_rows[j] for j in order[i:i+48]]
                b=make_batch(part,rng,motion,4,device);opt.zero_grad(set_to_none=True)
                if device=='cuda':
                    event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True);event_start.record()
                loss=objective(net,b)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();losses.append(float(loss.detach()))
                if device=='cuda':
                    event_end.record();event_end.synchronize();cuda_step_ms+=event_start.elapsed_time(event_end)
        net.eval();val=evaluate(net,dev_rows,motion,device)
        record=dict(epoch=epoch,train_es=float(np.mean(losses)) if losses else None,dev_es=val,
            learning_rate=opt.param_groups[0]['lr'],elapsed_s=time.perf_counter()-started,
            cpu_seconds=time.process_time()-cpu0,rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_allocated_bytes=torch.cuda.memory_allocated() if device=='cuda' else 0,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            cuda_training_step_elapsed_ms=cuda_step_ms,
            cuda_timing_semantics='stream elapsed for forward/backward/optimizer; not NVML hardware utilization')
        history.append(record);print(json.dumps(record),flush=True)
        checkpoint=dict(state_dict=net.cpu().state_dict(),width=64,diffusion_scale=1.,optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch,dev_path_es=val,configuration=configuration,
            numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state() if device=='cuda' else None)
        torch.save(checkpoint,dest/'last.pt')
        if epoch==0:torch.save(checkpoint,dest/'epoch0.pt')
        if epoch and val<best:
            improvement=best-val;best=val;torch.save(checkpoint,dest/'checkpoint.pt')
            stale=0 if improvement>=.001 else stale+1
        elif epoch:stale+=1
        net.to(device)
        (dest/'learning.json').write_text(json.dumps(history,indent=2))
        if not smoke and epoch>=15 and stale>=8:break
        if epoch:scheduler.step()
    configuration.update(completed_epochs=epoch,selected_dev_path_es=best,seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu0,checkpoint_sha256=sha(dest/'checkpoint.pt'),
        stop_reason='predeclared_patience' if epoch<epochs else 'maximum_epochs')
    (dest/'training.json').write_text(json.dumps(configuration,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=50);p.add_argument('--smoke',action='store_true');train(**vars(p.parse_args()))
