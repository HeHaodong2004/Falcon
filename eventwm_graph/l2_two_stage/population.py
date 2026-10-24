"""Exact injective multi-Bernoulli + Poisson current-population likelihood.

Physical IDs label offline trajectories only. This loss matches the entire
unordered nonvisible position set and has no privileged association input.
Conditional on one persistent global mode, stale episode positions must be
independent Gaussians. Shared future innovations therefore exclude coefficient0.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import jobs_for,native_index,read,selected_branch
from .features import positives


def attach_targets(records,split):
    if all('set_positions' in r and 'minimum_count' in r for r in records):
        return dict(anchors=len(records),ambiguous_current_multiset=sum(not r['set_valid'] for r in records),method='Manifest-bound native population targets from full preparation')
    folders={r['root_id']:Path(folder) for folder,r in jobs_for('dev' if split.startswith('dev') else 'train')}
    truth_cache={};failures=0;minimum_cache={}
    for r in records:
        root=r['root_id']
        if root not in truth_cache:
            with np.load(folders[root]/'privileged_future.npz',allow_pickle=False) as z:
                truth_cache[root]=(z['timestamps'].copy(),z['positions'][...,[0,2]].copy())
            if 'minimum_count' not in r:
                legal=read(folders[root]/'legal_root.json.gz');_,_,branch=selected_branch(folders[root])
                minimum=max(len(positives(ob)) for ob in legal['history']);table=[minimum]
                for node in branch:minimum=max(minimum,len(positives(node['observation'])));table.append(minimum)
                minimum_cache[root]=table
        if 'minimum_count' not in r:r['minimum_count']=minimum_cache[root][r['anchor_index']]
        ts,truth=truth_cache[root];y=truth[native_index(ts,r['timestamp'])];remaining=list(range(len(y)));valid=True
        # Condition on exactly the arrived positive center multiset. No earlier
        # source-ID -> physical correspondence is used to train population.
        for i in np.flatnonzero(r['age']==0):
            close=[p for p in remaining if np.linalg.norm(y[p]-r['origin'][i])<2e-4]
            if not close:valid=False;break
            if len(close)>1 and np.max(np.linalg.norm(y[close]-y[close[0]],axis=-1))>1e-7:
                valid=False;break
            remaining.remove(close[0])
        r['set_positions']=y[remaining].astype(np.float32);r['set_valid']=float(valid)
        failures+=not valid
    return dict(anchors=len(records),ambiguous_current_multiset=failures,
        method='Current exact detections removed injectively; every other physical position retained without source association.')


def add_batch(records,batch,device):
    u=max(1,max(len(r['set_positions']) for r in records));positions=np.zeros((len(records),u,2),np.float32);mask=np.zeros((len(records),u),np.float32)
    for i,r in enumerate(records):positions[i,:len(r['set_positions'])]=r['set_positions'];mask[i,:len(r['set_positions'])]=1
    batch['set_positions']=torch.as_tensor(positions,device=device);batch['set_mask']=torch.as_tensor(mask,device=device)
    batch['set_valid']=torch.as_tensor(np.array([r['set_valid'] for r in records],np.float32),device=device)
    batch['minimum_count']=torch.as_tensor(np.array([r['minimum_count'] for r in records],int),device=device)
    return batch


def safe_logsumexp(x,dim):
    all_absent=torch.isneginf(x).all(dim=dim,keepdim=True)
    safe=torch.where(all_absent,torch.zeros_like(x),x)
    answer=torch.logsumexp(safe,dim=dim)
    return torch.where(all_absent.squeeze(dim),torch.full_like(answer,-torch.inf),answer)


def assignment_log_likelihood(log_exist,log_absent,log_density,log_intensity,rate):
    """Batched exact sum over injective assignments, with Poisson leftovers.

    log_density=[batch,mode,source,target]; all source and target permutations
    are summed, without assignment of one person twice to the same target.
    The reference finite-set integration measure is sum_n (1/n!) dY^n.
    """
    b,m,s,n=log_density.shape;device=log_density.device;states=1<<n
    masks=torch.arange(states,device=device);bits=torch.arange(n,device=device)
    contained=(masks[:,None]&(1<<bits[None]))!=0
    previous=masks[:,None]^(1<<bits[None])
    previous=previous.clamp_min(0)
    dp=torch.full((b,m,states),-torch.inf,device=device,dtype=log_density.dtype)
    dp[:,:,0]=0.
    for source in range(s):
        absent=dp+log_absent[:,source,None,None]
        if n:
            assigned=dp[:,:,previous]+log_exist[:,source,None,None,None]+log_density[:,:,source,None,:]
            assigned=assigned.masked_fill(~contained[None,None],-torch.inf)
            dp=safe_logsumexp(torch.cat((absent[...,None],assigned),dim=-1),-1)
        else:dp=absent
    leftover=torch.where((~contained)[None,None],log_intensity[:,:,None,:],0.).sum(-1)
    return safe_logsumexp(dp+leftover,-1)-rate[:,None]


def set_nll(p,batch,condition_minimum=False,compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    from .tilted_count import is_tilted,tilted_set_nll
    if is_tilted(p):
        return tilted_set_nll(_single_rate_set_nll,p,batch,condition_minimum,compulsory_unique_episode,max_speed,compulsory_observed_clique)
    from .mixed_count import is_mixed,mixed_set_nll
    if is_mixed(p):
        return mixed_set_nll(_single_rate_set_nll,p,batch,condition_minimum,
            compulsory_unique_episode,max_speed,compulsory_observed_clique)
    return _single_rate_set_nll(p,batch,condition_minimum,compulsory_unique_episode,max_speed,compulsory_observed_clique)


def _single_rate_set_nll(p,batch,condition_minimum=False,compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    result=torch.zeros(len(batch['origin']),device=batch['origin'].device,dtype=batch['origin'].dtype)
    sizes=batch['set_mask'].sum(-1).long()
    for n in torch.unique(sizes).tolist():
        selected=torch.nonzero(sizes==n,as_tuple=True)[0]
        y=batch['set_positions'][selected,:n];age=batch['age'][selected]
        means=p['means'][selected,:,:,0].permute(0,2,1,3)
        scales=p['scales'][selected,:,:,0].permute(0,2,1,3)
        mu=batch['origin'][selected,None]+means*age[:,None,:,None]
        safe_age=torch.where(age==0,torch.ones_like(age),age)
        sigma=scales*safe_age[:,None,:,None]
        delta=(y[:,None,None]-mu[:,:,:,None])/sigma[:,:,:,None]
        log_density=-.5*delta.square().sum(-1)-sigma.log().sum(-1)[:,:,:,None]-1.8378770664093453
        if max_speed is not None:
            duration=safe_age[:,None,:,None,None]
            measured=batch['velocity'][selected,None,:,None]
            last=(batch['source_last_positions'][selected,None,:,None] if 'source_last_positions' in batch
                  else batch['origin'][selected,None,:,None]-measured*duration)
            bounded_velocity=(y[:,None,None]-last)/duration
            inside=1-bounded_velocity.square().sum(-1)/max_speed**2
            safe_inside=inside.clamp_min(1e-12)
            raw_velocity=bounded_velocity/torch.sqrt(safe_inside[...,None])
            raw_mean=measured+means[:,:,:,None]
            raw_scale=scales[:,:,:,None]
            log_density=-.5*((raw_velocity-raw_mean)/raw_scale).square().sum(-1)-raw_scale.log().sum(-1)-1.8378770664093453
            log_density=log_density-2*duration.squeeze(-1).log()-2*safe_inside.log()
            log_density=log_density.masked_fill(inside<=0,-torch.inf)
        optional=(batch['mask'][selected]>0)&(age>0)
        compulsory=torch.zeros_like(optional)
        if compulsory_unique_episode:compulsory|=optional&(batch['mask'][selected].sum(-1,keepdim=True)==1)
        if compulsory_observed_clique:compulsory|=optional&(batch['mask'][selected].sum(-1,keepdim=True)==batch['minimum_count'][selected,None])
        log_exist=F.logsigmoid(p['presence_logits'][selected]).masked_fill(~optional,-torch.inf)
        log_absent=F.logsigmoid(-p['presence_logits'][selected]).masked_fill(~optional,0.)
        log_exist=log_exist.masked_fill(compulsory,0.)
        log_absent=log_absent.masked_fill(compulsory,-torch.inf)
        relative=y-batch['robotxy'][selected,None]
        spatial_delta=(relative[:,:,None]-p['spatial_mean'][selected,None])/p['spatial_scale'][selected,None]
        spatial_log=-.5*spatial_delta.square().sum(-1)-p['spatial_scale'][selected].log().sum(-1)[:,None]-1.8378770664093453
        log_intensity=torch.logsumexp(spatial_log+F.log_softmax(p['spatial_logits'][selected],-1)[:,None],dim=-1)+p['rate'][selected].log()[:,None]
        loglaw=assignment_log_likelihood(log_exist,log_absent,log_density,log_intensity[:,None].expand(-1,4,-1),p['rate'][selected])
        log_mixture=torch.logsumexp(loglaw+F.log_softmax(p['mode_logits'][selected],-1),-1)
        result[selected]=-log_mixture*batch['set_valid'][selected]
    if condition_minimum:
        from .count import torch_log_normalizer
        certain=((batch['age']==0)&(batch['mask']>0)).sum(-1)
        remaining_minimum=(batch['minimum_count']-certain).clamp_min(0)
        if ((batch['set_mask'].sum(-1)<remaining_minimum)&(batch['set_valid']>0)).any():raise ValueError('native population target below legal historical count floor')
        optional=(batch['age']>0)&(batch['mask']>0)
        compulsory=torch.zeros_like(optional)
        if compulsory_unique_episode:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==1)
        if compulsory_observed_clique:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==batch['minimum_count'][:,None])
        if ((batch['set_mask'].sum(-1)<compulsory.sum(-1))&(batch['set_valid']>0)).any():raise ValueError('native population target cannot lose a compulsory observed person')
        minimum=(remaining_minimum-compulsory.sum(-1)).clamp_min(0)
        optional=optional&(~compulsory)
        result=result+torch_log_normalizer(p['presence_logits'],optional,p['rate'],minimum).to(result.dtype)*batch['set_valid']
    return result
