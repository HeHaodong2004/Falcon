"""Proper-score training for the actual history-conditioned population law.

Current observations are normalized by an exact-in-expectation negative phase.
Accepted current coordinates/latents are DETACHED; future motion uses pathwise
gradients. Independent accepted samples supply the missing normalization score
term for proper future Energy/kernel scores. No gradient is taken through a
discrete acceptance or association decision.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import digamma
from .conditioned import draw_conditioned_current,pad_current,source_coordinates
from .count import torch_log_normalizer
from .neural import basis
from .population import set_nll
from .speed import bound_torch


def make_batch(records,device='cuda:0'):
    from .data import collate
    from .future_population_targets import current_measurement_overlay
    batch=collate(records,device);b=len(records)
    remaining=[]
    for r in records:
        history=r['positive_history'];start=int(history.positive_offsets[-2])
        overlay,unobserved=current_measurement_overlay(r['population_current_positions'],history.positive_xyz[start:,[0,2]])
        remaining.append(unobserved)
    u=max(1,max(len(p) for p in remaining));count=max(1,max(r['population_future_positions'].shape[1] for r in records))
    current=np.zeros((b,u,2));current_mask=np.zeros((b,u))
    future=np.zeros((b,8,count,2));people=np.zeros((b,count))
    for i,(r,y) in enumerate(zip(records,remaining)):
        current[i,:len(y)]=y;current_mask[i,:len(y)]=1
        n=r['population_future_positions'].shape[1];future[i,:,:n]=r['population_future_positions'];people[i,:n]=1
    for key,value in dict(set_positions=current,set_mask=current_mask,set_valid=np.ones(b),
            minimum_count=np.array([r['minimum_count'] for r in records],int),
            population_future_positions=future,population_person_mask=people,
            population_future_valid=np.array([r['population_future_valid'] for r in records])).items():
        batch[key]=torch.as_tensor(value,device=device)
    return batch


def sample_latents(records,parameters,graphs,worlds,rng,unique=True,guidance=False):
    if worlds<3:raise ValueError('independent normalization correction needs at least3 worlds')
    b=len(records);n=parameters['means'].shape[1];rows=[];indices=[];lineages=[];diagnostics=[];raw_last=[]
    numpy_parameters={k:v.detach().cpu().numpy() for k,v in parameters.items()}
    for i,record in enumerate(records):
        p={k:v[i] for k,v in numpy_parameters.items()};history=record['positive_history']
        sampled,source_events,stats=draw_conditioned_current(record,p,history,graphs[i],worlds,rng,unique)
        current,present=pad_current(sampled,len(record['keys']))
        if guidance:
            from .guided_association import lineage as guided_lineage
            witness=guided_lineage(graphs[i],current,present,source_events,
                np.array([row.source_positions for row in sampled]).reshape(worlds,len(record['keys']),2),rng,source_first=True)
        else:witness=graphs[i].lineage(current,present,rng)
        rows.append(sampled);indices.append(source_events);lineages.append(witness);diagnostics.append(stats)
        raw_last.append(source_coordinates(record,history)[0])
    hidden=max(len(row.hidden_positions) for sample in rows for row in sample)
    output=dict(mode=np.zeros((b,worlds),int),source_present=np.zeros((b,worlds,n),bool),
        initial=np.zeros((b,worlds,n,2)),hidden_components=np.zeros((b,worlds,hidden),int),
        hidden_positions=np.zeros((b,worlds,hidden,2)),hidden_mask=np.zeros((b,worlds,hidden),bool),
        current=np.zeros((b,worlds,n+hidden,2)),present=np.zeros((b,worlds,n+hidden),bool),
        source_person=np.zeros((b,worlds,n),int),raw_last=np.zeros((b,n,2)),minimum=np.array([d['minimum_current_count'] for d in diagnostics]))
    for i,sample in enumerate(rows):
        count=len(records[i]['keys']);output['raw_last'][i,:count]=raw_last[i]
        for w,row in enumerate(sample):
            u=len(row.hidden_positions);output['mode'][i,w]=row.mode;output['source_present'][i,w,:count]=row.source_present
            output['initial'][i,w,:count]=row.initial_coefficients;output['hidden_components'][i,w,:u]=row.hidden_components
            output['hidden_positions'][i,w,:u]=row.hidden_positions;output['hidden_mask'][i,w,:u]=True
            output['current'][i,w,:count]=row.source_positions;output['current'][i,w,n:n+u]=row.hidden_positions
            output['present'][i,w,:count]=row.source_present;output['present'][i,w,n:n+u]=True
            person=lineages[i].event_person[w,indices[i]]
            output['source_person'][i,w,:count]=np.where(person<count,person,person+n-count)
    return output,diagnostics


def tensors(latents,device):
    return {k:torch.as_tensor(v,device=device) for k,v in latents.items()}


def latent_current_log_probability(p,batch,z,unique=True):
    """Generated joint latent density; not an oracle ownership likelihood.

    Gaussian coefficient0 coordinates are a fixed, parameter-independent map
    to current physical positions. Its Jacobian is parameter-independent, so
    it cancels from every score-function gradient used here.
    """
    b,m=z['mode'].shape;n=p['presence_logits'].shape[1];device=z['mode'].device
    ib=torch.arange(b,device=device)[:,None];person=torch.arange(n,device=device)[None,None,:]
    mode=z['mode'];logq=F.log_softmax(p['mode_logits'],dim=-1)[ib,mode].double()
    known=batch['mask']>0;certain=(batch['age']<1e-7)&known
    if unique:certain=certain|(known&(known.sum(-1,keepdim=True)==1))
    optional=known&~certain
    source=z['source_present']
    log_presence=torch.where(source,F.logsigmoid(p['presence_logits'])[:,None],F.logsigmoid(-p['presence_logits'])[:,None])
    logq=logq+(log_presence*optional[:,None]).sum(-1)
    count=z['hidden_mask'].sum(-1);rate=p['rate'].double()
    minimum=(z['minimum']-certain.sum(-1)).clamp_min(0)
    normalizer=torch_log_normalizer(p['presence_logits'],optional,p['rate'],minimum)
    logq=logq+count*rate[:,None].log()-rate[:,None]-torch.lgamma(count+1.)-normalizer[:,None]
    means=p['means'][ib[:,:,None],person,mode[:,:,None],0].double()
    scales=p['scales'][ib[:,:,None],person,mode[:,:,None],0].double()
    gaussian=-.5*((z['initial']-means)/scales).square().sum(-1)-scales.log().sum(-1)-math.log(2*math.pi)
    logq=logq+(gaussian*known[:,None]).sum(-1)
    component=z['hidden_components'];index=ib[:,:,None]
    mu=p['spatial_mean'][index,component].double();scale=p['spatial_scale'][index,component].double()
    hidden_gaussian=-.5*((z['hidden_positions']-batch['robotxy'][:,None,None]-mu)/scale).square().sum(-1)-scale.log().sum(-1)-math.log(2*math.pi)
    hidden_mixture=F.log_softmax(p['spatial_logits'],-1)[index,component]
    logq=logq+((hidden_gaussian+hidden_mixture)*z['hidden_mask']).sum(-1)
    return logq


def future_paths(p,batch,z):
    """Reparameterize future marks conditional on accepted fixed current states."""
    b,m=z['mode'].shape;n=p['presence_logits'].shape[1];device=z['mode'].device
    ib=torch.arange(b,device=device)[:,None,None];person=torch.arange(n,device=device)[None,None,:]
    means=p['means'][ib,person,z['mode'][:,:,None],1:].double()
    scales=p['scales'][ib,person,z['mode'][:,:,None],1:].double()
    noise=torch.randn_like(means);common=torch.randn((b,m,1,4,2),device=device,dtype=means.dtype)
    coefficient=means+scales*(noise*.979795897+common*.2)
    velocity=bound_torch(batch['velocity'][:,None,:,None]+z['initial'][:,:,:,None]+coefficient,3.)
    phi=basis(batch['times'].double())
    source=z['current'][:, :, None, :n]+torch.einsum('bmnrc,btr->bmtnc',velocity,phi)
    component=z['hidden_components'];index=torch.arange(b,device=device)[:,None,None]
    mu=p['spatial_velocity'][index,component].double();scale=p['spatial_velocity_scale'][index,component].double()
    hidden_velocity=bound_torch(mu+scale[...,None]*torch.randn_like(mu),3.)
    hidden=z['hidden_positions'][:,:,None]+batch['times'][:,None,:,None,None]*hidden_velocity[:,:,None]
    return torch.cat((source,hidden),dim=3)


def set_kernel(paths,present,truth,truth_present,valid_times,path_bandwidth=3.,set_bandwidth=2.):
    b,m,t,n,_=paths.shape;people=truth.shape[2]
    norm=valid_times.sum(-1).clamp_min(1).sqrt()
    predicted=(paths*valid_times[:,None,:,None,None]).permute(0,1,3,2,4).reshape(b,m*n,t*2)/norm[:,None,None]
    actual=(truth*valid_times[:,:,None,None]).permute(0,2,1,3).reshape(b,people,t*2)/norm[:,None,None]
    combined=torch.cat((predicted,actual),1)
    distance=torch.cdist(combined,combined).square()
    inner=torch.exp(-distance/(2*path_bandwidth**2))
    membership=torch.zeros((b,m*n+people,m+1),device=paths.device,dtype=paths.dtype)
    for w in range(m):membership[:,w*n:(w+1)*n,w]=present[:,w]
    membership[:,m*n:,m]=truth_present
    embedded=membership.transpose(1,2)@inner@membership
    diagonal=torch.diagonal(embedded,dim1=1,dim2=2)
    squared=(diagonal[:,:,None]+diagonal[:,None,:]-2*embedded).clamp_min(0)
    return torch.exp(-squared/(2*set_bandwidth**2))


def proper_pair_losses(paths,batch,z,logq,kernel):
    """Unbiased normalization correction, independent of each scored pair."""
    b,m,t,n,_=paths.shape;source_count=batch['target'].shape[1]
    index=z['source_person'][:,:,:,None,None].expand(-1,-1,-1,t,2)
    tracked=torch.gather(paths.permute(0,1,3,2,4),2,index)
    mask=batch['target_mask'][:,None,:,:,None]
    target=batch['target'][:,None]
    difference=((tracked-target)*mask).flatten(2)
    norm=batch['target_mask'].sum((1,2)).clamp_min(1).sqrt()
    first=difference.norm(dim=-1)/norm[:,None]
    scores=[];values=[];set_scores=[];set_values=[]
    for i in range(m):
        for j in range(i+1,m):
            cross=((tracked[:,i]-tracked[:,j])*mask[:,0]).flatten(1).norm(dim=-1)/norm
            es=.5*(first[:,i]+first[:,j]-cross)
            other=[k for k in range(m) if k not in (i,j)]
            normalization_score=logq[:,i]+logq[:,j]-2*logq[:,other].mean(-1)
            scores.append(es+es.detach()*(normalization_score-normalization_score.detach()))
            values.append(es.detach())
            ks=kernel[:,i,j]-kernel[:,i,m]-kernel[:,j,m]+1.
            set_scores.append(ks+ks.detach()*(normalization_score-normalization_score.detach()))
            set_values.append(ks.detach())
    return torch.stack(scores).mean(0),torch.stack(set_scores).mean(0),dict(
        tracked_es=torch.stack(values).mean(0),population_kernel=torch.stack(set_values).mean(0))


def loss(p,batch,records,graphs,rng,worlds=4,current_weight=.05,population_weight=.2,unique=True,guidance=False):
    # Native clocks and measurement ages stay float64; float32 feature tensors
    # remain the neural encoder's adequate numerical representation only.
    batch=dict(batch);p={key:value.double() for key,value in p.items()}
    age=np.zeros(tuple(batch['age'].shape),np.float64)
    for i,record in enumerate(records):age[i,:len(record['keys'])]=record['age']
    batch['age']=torch.as_tensor(age,device=batch['x'].device)
    batch['times']=torch.as_tensor(np.array([r['times'] for r in records],np.float64),device=batch['x'].device)
    raw,diagnostics=sample_latents(records,p,graphs,worlds,rng,unique,guidance);z=tensors(raw,batch['x'].device)
    logq=latent_current_log_probability(p,batch,z,unique)
    paths=future_paths(p,batch,z)
    truth=batch['population_future_positions'].double();truth_present=batch['population_person_mask']
    valid=batch['population_future_valid'].double()
    kernel=set_kernel(paths,z['present'],truth,truth_present,valid)
    tracked,population,terms=proper_pair_losses(paths,batch,z,logq,kernel)
    current_batch=dict(batch,source_last_positions=z['raw_last'],minimum_count=z['minimum'],
        origin=batch['origin'].double(),age=batch['age'].double())
    nll=set_nll(p,current_batch,True,unique,3.)
    logz=torch.as_tensor([digamma(worlds)-digamma(d['proposals_examined_through_last_accept'])+d.get('log_normalizer_multiplier',0.) for d in diagnostics],device=nll.device,dtype=nll.dtype)
    # The scalar is the independently justified log-score estimator; gradients
    # use the exact accepted-latent normalizer identity, not log plug-in values.
    surrogate=nll+logq.mean(-1)
    value=nll.detach()+logz
    normalized_nll=surrogate+(value-surrogate.detach())
    has_tracked=(batch['target_mask'].sum((1,2))>0);has_future=(valid.sum(-1)>0)
    per_anchor=tracked*has_tracked+population_weight*population*has_future+current_weight*normalized_nll
    terms.update(current_nll=value.detach(),normalizer_log_estimate=logz.detach(),proposals=np.array([d['proposals_generated'] for d in diagnostics]))
    return (per_anchor*batch['weight']).mean(),terms
