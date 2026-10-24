"""Offline train-only scalar fit and root-balanced native-path neural fitting."""
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
from scipy.optimize import minimize
from .common import OUT, OLD, SNAP, BASE, dump, sha, seed_for
from .dynamics import DEFAULT, LAW_VERSION
from .neural import IntermittentDynamics
from eventwm_graph.l2_120k.learning import mixture_path_energy, component_draw, psd_root


def prepare():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((SNAP/'manifest.json').read_text())
    rate_rows=[];speed_samples=[];stopped_speed=[];start=time.perf_counter()
    for ri,row in enumerate(manifest['rows']):
        if row['split']!='train':continue
        with np.load(SNAP/'roots'/row['root_id']/'privileged_future.npz') as f:
            ts=f['timestamps'];x=f['positions'][...,[0,2]]
        if len(ts)<3 or not x.shape[1]:continue
        v=np.diff(x,axis=0)/np.diff(ts)[:,None,None];speed=np.linalg.norm(v,axis=-1);s=speed<.1
        rate_rows.append((s[:-1].ravel(),s[1:].ravel(),np.broadcast_to(np.diff(ts)[1:,None],s[1:].shape).ravel()))
        speed_samples.extend(np.quantile(speed,[.5,.95,.99]).tolist())
        stopped_speed.extend(speed[s][::max(1,int(s.sum())//32)].tolist())
    # Root-balanced binary transition composite likelihood under exact CTMC
    # transition probabilities. Finite-difference labels are a proxy state.
    def loss(z):
        a,b=np.exp(z);pi=a/(a+b);vals=[]
        for s0,s1,dt in rate_rows:
            p=pi+(s0.astype(float)-pi)*np.exp(-(a+b)*dt)
            vals.append(-np.mean(np.log(np.where(s1,p,1-p))))
        return float(np.mean(vals))
    fit=minimize(loss,np.log([.8,1.3]),method='L-BFGS-B',bounds=[(-4,3),(-4,3)])
    law={**DEFAULT,'stop_rate':float(np.exp(fit.x[0])),'restart_rate':float(np.exp(fit.x[1]))}
    dump(OUT/'motion.json',law)
    dump(OUT/'motion_fit.json',dict(roots=len(rate_rows),success=bool(fit.success),objective=float(fit.fun),law=law,
         method='root-balanced finite-difference Stop/Move CTMC composite likelihood; labels speed<.1m/s',
         stop_reset_std_is_fixed=.015,stop_stationary_velocity_std=math.sqrt(law['stop_diffusion']/(2*law['stop_relaxation'])),
         old_position_noise_stop_upper_bound=float(-np.expm1(-.1**2*.048/(2*.0025))),seconds=time.perf_counter()-start))
    for split in ('train','dev'):
        cached=np.load(OLD/(split+'.npy'),allow_pickle=True);result=[]
        for row in cached:
            if not row['components']:continue
            with np.load(SNAP/'roots'/row['root_id']/'privileged_future.npz') as f:
                times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
            keep=times<=times[0]+5.+1e-8;times=times[keep];truth=truth[keep]
            cs=[]
            for c in row['components']:
                ids=[]
                for p in c['truth'][0]:
                    d=np.linalg.norm(truth[0]-p,axis=-1);j=int(d.argmin())
                    if d[j]>2e-4:raise ValueError('lost frozen training correspondence')
                    ids.append(j)
                cs.append({**c,'truth':truth[:,ids].astype(np.float32)})
            result.append({**row,'components':cs,'times':times})
        np.save(OUT/(split+'.npy'),np.array(result,dtype=object),allow_pickle=True)
        print('prepared',split,len(result),flush=True)
    dump(OUT/'preparation.json',dict(snapshot_sha256=sha(SNAP/'manifest.json'),old_train_sha256=sha(OLD/'train.npy'),
         old_dev_sha256=sha(OLD/'dev.npy'),train_sha256=sha(OUT/'train.npy'),dev_sha256=sha(OUT/'dev.npy'),seconds=time.perf_counter()-start))


def ou_coeff(dt,k):
    e=np.exp(-k*dt);a=-np.expm1(-k*dt)/k
    vv=-np.expm1(-2*k*dt)/(2*k);pv=(1-e)**2/(2*k*k)
    pp=(dt-2*a+vv)/(k*k)
    small=k*dt<1e-3
    pp=np.where(small,dt**3*(1/3-k*dt/4+7*(k*dt)**2/60),pp)
    lv=np.sqrt(np.maximum(vv,0));lp=np.divide(pv,lv,out=np.zeros_like(pv),where=lv>0)
    return e,a,lv,lp,np.sqrt(np.maximum(pp-lp**2,0))


def sample_factors(mean,cov,times,law,rng,samples,mixture=None):
    """Vectorized exact conditional-CTMC path sampler, factored for gradients."""
    n=len(mean);K=3;S=samples;shape=(n,K,2,S);H=len(times)
    init=mean[:,None,None,None,:]+np.einsum('nij,nkasj->nkasi',psd_root(cov),rng.normal(size=shape+(4,)))
    if mixture is not None:
        gm,gc,gp=mixture
        choice=(rng.random(shape+(1,))>np.cumsum(gp,axis=-1)[:,None,None,None,:]).sum(-1)
        idx=np.arange(n)[:,None,None,None]
        init=gm[idx,choice]+np.einsum('nkasij,nkasj->nkasi',psd_root(gc)[idx,choice],rng.normal(size=shape+(4,)))
    base=init.copy();noise=np.zeros_like(base);coef=np.zeros(shape+(2,))
    stopped=np.broadcast_to(np.array([False,True])[None,None,:,None],shape).copy()
    base[:,:,1,:,2:]=rng.normal(size=(n,K,S,2))*law['stop_reset_std']
    nxt=rng.exponential(1/np.where(stopped,law['restart_rate'],law['stop_rate']))
    clock=np.zeros(shape);outs=[[],[],[]]
    for end in times:
        while np.any(clock<end-1e-12):
            active=clock<end-1e-12
            until=np.minimum(nxt,end);dt=np.where(active,np.maximum(0,until-clock),0.)
            k=np.where(stopped,law['stop_relaxation'],law['move_relaxation']);e,a,lv,lp,lpp=ou_coeff(dt,k)
            for arr in (base,noise):
                arr[...,:2]+=a[...,None]*arr[...,2:];arr[...,2:]*=e[...,None]
            coef[...,0]+=a*coef[...,1];coef[...,1]*=e
            moving=~stopped;coef[...,0]+=(dt-a)*moving;coef[...,1]+=(1-e)*moving
            z=rng.normal(size=shape+(2,2));incp=lp[...,None]*z[...,0,:]+lpp[...,None]*z[...,1,:];incv=lv[...,None]*z[...,0,:]
            base[...,:2]+=incp*stopped[...,None]*math.sqrt(law['stop_diffusion'])
            base[...,2:]+=incv*stopped[...,None]*math.sqrt(law['stop_diffusion'])
            noise[...,:2]+=incp*moving[...,None];noise[...,2:]+=incv*moving[...,None]
            clock=np.where(active,until,clock)
            change=active&(nxt<=end+1e-12)
            stopped^=change
            bs=rng.normal(size=shape+(2,))*law['stop_reset_std']
            ns=rng.normal(size=shape+(2,))*math.sqrt(law['restart_variance_factor'])
            base[...,2:]=np.where(change[...,None],bs*stopped[...,None],base[...,2:])
            noise[...,2:]=np.where(change[...,None],ns*(~stopped)[...,None],noise[...,2:])
            coef[...,1]=np.where(change,(~stopped).astype(float),coef[...,1])
            nxt=np.where(change,clock+rng.exponential(1/np.where(stopped,law['restart_rate'],law['stop_rate'])),nxt)
        outs[0].append(base[...,:2].copy());outs[1].append(noise[...,:2].copy());outs[2].append(coef[...,0].copy())
    return np.stack(outs[0],-2),np.stack(outs[1],-2),np.stack(outs[2],-1)


def make_batch(rows,rng,law,samples=4,device='cpu',gap_weight=True,selection_log=None):
    B=len(rows);N=max(len(c['mean']) for r in rows for c in r['components']);H=4
    shapes=dict(mean=(B,N,4),age=(B,N),graph=(B,N,18),mask=(B,N),target=(B,N,H,2),valid=(B,N,H),
                base=(B,N,6,samples,H,2),noise=(B,N,6,samples,H,2),coef=(B,N,6,samples,H))
    arr={k:np.zeros(s,np.float32) for k,s in shapes.items()}
    for b,row in enumerate(rows):
        c=component_draw(row,rng);n=len(c['mean']);ts=row['times'];rel=ts-ts[0]
        age=0.;fallback=False
        if gap_weight and rng.random()<.5:
            age=float(rng.choice([.5,1.,1.5,2.]));desired=age+np.array([.0,.5,1.,2.])
            take=np.searchsorted(rel,desired);take=take[take<len(ts)]
            if not len(take):take=np.array([len(ts)-1]);fallback=True
        else:
            end=np.searchsorted(rel,3.+1e-8);take=np.linspace(1,max(1,end-1),4).astype(int)
        take=np.maximum(1,take);take=np.minimum(len(ts)-1,take);validH=len(take);take=np.pad(take,(0,H-validH),mode='edge')
        times=rel[take];arr['mean'][b,:n]=c['mean'];arr['age'][b,:n]=c['age'];arr['graph'][b,:n]=c['graph'];arr['mask'][b,:n]=1
        if selection_log is not None:selection_log.append(dict(root_id=row['root_id'],requested_age=age,fallback=fallback,target_timestamps=ts[take].tolist(),unique_targets=len(np.unique(take))))
        arr['target'][b,c['selected']]=c['truth'][take].transpose(1,0,2);arr['valid'][b,c['selected'],:validH]=1
        mixture=(c['maneuver_mean'],c['maneuver_covariance'],c['maneuver_probability']) if c['maneuver_mean'] is not None else None
        base,noise,coef=sample_factors(c['mean'],c['covariance'],times,law,rng,samples,mixture)
        arr['base'][b,:n]=base.reshape(n,6,samples,H,2);arr['noise'][b,:n]=noise.reshape(n,6,samples,H,2);arr['coef'][b,:n]=coef.reshape(n,6,samples,H)
    return {k:torch.as_tensor(v,device=device) for k,v in arr.items()}


def objective(net,b):
    target,q,logits,stop=net(b['mean'],b['age'],b['graph'],b['mask'])
    target=target[:,:, :,None,:].expand(-1,-1,-1,2,-1).reshape(*target.shape[:2],6,2)
    q=q[:,:,:,None].expand(-1,-1,-1,2).reshape(*q.shape[:2],6)
    lp=torch.log_softmax(logits,-1)[...,:,None]+torch.stack([torch.nn.functional.logsigmoid(-stop),torch.nn.functional.logsigmoid(stop)],-1)[...,None,:]
    lp=lp.reshape(*logits.shape[:2],6)
    draws=b['base']+b['coef'][...,None]*target[:,:,:,None,None,:]+q.sqrt()[:,:,:,None,None,None]*b['noise']
    return mixture_path_energy(draws,lp,b['target'],b['valid']).mean()


def evaluate(net,rows,law,device,gap_weight=True):
    rng=np.random.default_rng(222);scores=[]
    with torch.no_grad():
        for i in range(0,len(rows),48):
            part=rows[i:i+48];scores.append((len(part),float(objective(net,make_batch(part,rng,law,8,device,gap_weight)))))
    return sum(n*s for n,s in scores)/sum(n for n,s in scores)


def train(epochs=80):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.manual_seed(221);np.random.seed(221);random.seed(221)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.2)
    rows={s:list(np.load(OUT/(s+'.npy'),allow_pickle=True)) for s in ('train','dev')};law=json.loads((OUT/'motion.json').read_text())
    dest=OUT/'training';dest.mkdir(exist_ok=False)
    net=IntermittentDynamics(64).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.00005)
    cfg=dict(seed=221,width=64,parameters=sum(p.numel() for p in net.parameters()),law=law,law_version=LAW_VERSION,
             device=device,epochs=epochs,minimum_epochs=30,patience=12,min_improvement=.001,batch_size=48,
             objective='root-balanced proper path Energy Score; persistent 3 velocity modes x initial Stop/Move; 4 paths/mode train,8 dev; half gap-shifted targets',
             train_roots=len(rows['train']),dev_roots=len(rows['dev']),snapshot_sha256=sha(SNAP/'manifest.json'),
             sources={str(p.relative_to(BASE)):sha(p) for p in Path(__file__).parent.glob('*.py')},
             protocol_sha256=sha(BASE/'docs/l2/intermittent_wm_20261022/PROTOCOL.md'),
             caches={s:sha(OUT/(s+'.npy')) for s in ('train','dev')},gpu=torch.cuda.get_device_name() if device=='cuda' else None)
    dump(dest/'configuration.json',cfg);rng=np.random.default_rng(221);history=[];best=math.inf;significant=math.inf;last_significant=0
    start=time.perf_counter();cpu0=time.process_time()
    for epoch in range(epochs+1):
        losses=[];net.train();selection_log=[]
        if epoch:
            order=rng.permutation(len(rows['train']))
            for i in range(0,len(order),48):
                part=[rows['train'][j] for j in order[i:i+48]];b=make_batch(part,rng,law,4,device,selection_log=selection_log)
                opt.zero_grad(set_to_none=True);loss=objective(net,b)
                if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();losses.append(float(loss))
        net.eval();val=evaluate(net,rows['dev'],law,device)
        rec=dict(epoch=epoch,train_es=float(np.mean(losses)) if losses else None,dev_es=val,elapsed_s=time.perf_counter()-start,
                 cpu_s=time.process_time()-cpu0,learning_rate=opt.param_groups[0]['lr'],rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
        history.append(rec);print(json.dumps(rec),flush=True)
        if epoch:
            import gzip
            with gzip.open(dest/('selections_%03d.json.gz'%epoch),'wt') as f:json.dump(selection_log,f)
        if epoch:scheduler.step()
        ck=dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},width=64,epoch=epoch,dev_path_es=val,configuration=cfg,
                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),numpy_rng_state=rng.bit_generator.state,
                python_rng_state=random.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state() if device=='cuda' else None)
        torch.save(ck,dest/'last.pt')
        if epoch==0:torch.save(ck,dest/'epoch0.pt')
        if epoch and val<best:best=val;torch.save(ck,dest/'checkpoint.pt')
        if epoch and val<=significant-.001:significant=val;last_significant=epoch
        dump(dest/'learning.json',history)
        if epoch>=30 and epoch-last_significant>=12:break
    dump(dest/'training.json',dict(**cfg,completed_epochs=epoch,selected_dev_path_es=best,significant_best=significant,
         last_significant_epoch=last_significant,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0,
         stop_reason='significant_best_patience' if epoch<epochs else 'maximum_epochs',checkpoint_sha256=sha(dest/'checkpoint.pt')))
    ck=torch.load(dest/'checkpoint.pt',map_location=device);net.load_state_dict(ck['state_dict'])
    dump(dest/'focused_ablation.json',dict(selected_with_gap_objective=evaluate(net,rows['dev'],law,device),
         selected_without_gap_objective=evaluate(net,rows['dev'],law,device,False),
         interpretation='same selected checkpoint evaluated on two declared objectives; not a separately trained causal ablation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:train(a.epochs)
