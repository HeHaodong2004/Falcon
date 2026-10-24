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


def sample_latents(records,parameters,graphs,worlds,rng,unique=True,guidance=False,include_lineage=True):
    if worlds<3:raise ValueError('independent normalization correction needs at least3 worlds')
    b=len(records);n=parameters['means'].shape[1];rows=[];indices=[];lineages=[];diagnostics=[];raw_last=[]
    numpy_parameters={k:v.detach().cpu().numpy() for k,v in parameters.items()}
    for i,record in enumerate(records):
        p={k:v[i] for k,v in numpy_parameters.items()};history=record['positive_history']
        sampled,source_events,stats=draw_conditioned_current(record,p,history,graphs[i],worlds,rng,unique)
        current,present=pad_current(sampled,len(record['keys']))
        if not include_lineage:witness=None
        elif guidance:
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
            if include_lineage:
                person=lineages[i].event_person[w,indices[i]]
                output['source_person'][i,w,:count]=np.where(person<count,person,person+n-count)
    if not include_lineage:del output['source_person']
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
    known=batch['mask']>0;certain=(batch['age']==0)&known
    if unique:certain=certain|(known&(known.sum(-1,keepdim=True)==1))
    optional=known&~certain
    source=z['source_present']
    log_presence=torch.where(source,F.logsigmoid(p['presence_logits'])[:,None],F.logsigmoid(-p['presence_logits'])[:,None])
    logq=logq+(log_presence*optional[:,None]).sum(-1)
    count=z['hidden_mask'].sum(-1);rate=p['rate'].double()
    minimum=(z['minimum']-certain.sum(-1)).clamp_min(0)
    from .mixed_count import is_mixed,torch_mixture_log_normalizer,torch_residual_log_probability
    from .tilted_count import is_tilted,torch_log_potential,torch_tilt_log_normalizer
    if is_tilted(p):
        total=count+source.sum(-1)
        normalizer=torch_tilt_log_normalizer(p,p['presence_logits'],optional,z['minimum'],certain.sum(-1))
        logq=logq+count*rate[:,None].log()-rate[:,None]-torch.lgamma(count+1.)
        logq=logq+torch_log_potential(p,total)-normalizer[:,None]
    elif is_mixed(p):
        normalizer=torch_mixture_log_normalizer(p,p['presence_logits'],optional,minimum)
        logq=logq+torch_residual_log_probability(p,count)-normalizer[:,None]
    else:
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


def visible_initial_gaussian(p,batch,z):
    """Current-coordinate Gaussian score that integrates out at EXACT age zero.

    Other initial coefficients, including every stale/absent episode, must
    remain in the score because current feasibility or association can use them.
    """
    b,m=z['mode'].shape;n=p['presence_logits'].shape[1];device=z['mode'].device
    ib=torch.arange(b,device=device)[:,None,None];person=torch.arange(n,device=device)[None,None,:]
    mean=p['means'][ib,person,z['mode'][:,:,None],0].double()
    scale=p['scales'][ib,person,z['mode'][:,:,None],0].double()
    gaussian=-.5*((z['initial']-mean)/scale).square().sum(-1)-scale.log().sum(-1)-math.log(2*math.pi)
    visible=(batch['age']==0)&(batch['mask']>0)
    return (gaussian*visible[:,None]).sum(-1)


def future_paths(p,batch,z,visible_initial_reparameterization=False):
    """Reparameterize future marks conditional on accepted fixed current states."""
    b,m=z['mode'].shape;n=p['presence_logits'].shape[1];device=z['mode'].device
    ib=torch.arange(b,device=device)[:,None,None];person=torch.arange(n,device=device)[None,None,:]
    means=p['means'][ib,person,z['mode'][:,:,None],1:].double()
    scales=p['scales'][ib,person,z['mode'][:,:,None],1:].double()
    noise=torch.randn_like(means);common=torch.randn((b,m,1,4,2),device=device,dtype=means.dtype)
    coefficient=means+scales*(noise*.979795897+common*.2)
    initial=z['initial']
    if visible_initial_reparameterization:
        mean0=p['means'][ib,person,z['mode'][:,:,None],0].double()
        scale0=p['scales'][ib,person,z['mode'][:,:,None],0].double()
        standard=(initial-mean0.detach())/scale0.detach()
        reparameterized=mean0+scale0*standard
        visible=((batch['age']==0)&(batch['mask']>0))[:,None,:,None]
        # Forward values and RNG stay exactly unchanged. Only the valid
        # age-zero coordinate's gradient becomes pathwise.
        initial=initial+torch.where(visible,reparameterized-reparameterized.detach(),torch.zeros_like(initial))
    velocity=bound_torch(batch['velocity'][:,None,:,None]+initial[:,:,:,None]+coefficient,3.)
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


def proper_pair_losses(paths,batch,z,logq,kernel,marginal_weight=0.):
    """Unbiased normalization correction, independent of each scored pair."""
    if not math.isfinite(marginal_weight) or marginal_weight<0:raise ValueError('nonnegative finite marginal score weight required')
    b,m,t,n,_=paths.shape;source_count=batch['target'].shape[1]
    index=z['source_person'][:,:,:,None,None].expand(-1,-1,-1,t,2)
    tracked=torch.gather(paths.permute(0,1,3,2,4),2,index)
    mask=batch['target_mask'][:,None,:,:,None]
    target=batch['target'][:,None]
    difference=((tracked-target)*mask).flatten(2)
    norm=batch['target_mask'].sum((1,2)).clamp_min(1).sqrt()
    first=difference.norm(dim=-1)/norm[:,None]
    scores=[];values=[];set_scores=[];set_values=[];point_scores=[];point_values=[]
    if marginal_weight:
        point_mask=mask[:,0,:,:,0]
        point_norm=batch['target_mask'].sum((1,2)).clamp_min(1)
        point_first=((tracked-target).norm(dim=-1)*point_mask[:,None]).sum((2,3))/point_norm[:,None]
    for i in range(m):
        for j in range(i+1,m):
            cross=((tracked[:,i]-tracked[:,j])*mask[:,0]).flatten(1).norm(dim=-1)/norm
            es=.5*(first[:,i]+first[:,j]-cross)
            other=[k for k in range(m) if k not in (i,j)]
            normalization_score=logq[:,i]+logq[:,j]-2*logq[:,other].mean(-1)
            scores.append(es+es.detach()*(normalization_score-normalization_score.detach()))
            values.append(es.detach())
            if marginal_weight:
                point_cross=((tracked[:,i]-tracked[:,j]).norm(dim=-1)*point_mask).sum((1,2))/point_norm
                point_es=.5*(point_first[:,i]+point_first[:,j]-point_cross)
                point_scores.append(point_es+point_es.detach()*(normalization_score-normalization_score.detach()))
                point_values.append(point_es.detach())
            ks=kernel[:,i,j]-kernel[:,i,m]-kernel[:,j,m]+1.
            set_scores.append(ks+ks.detach()*(normalization_score-normalization_score.detach()))
            set_values.append(ks.detach())
    tracked_score=torch.stack(scores).mean(0)
    point_value=torch.zeros_like(tracked_score)
    if marginal_weight:
        tracked_score=tracked_score+marginal_weight*torch.stack(point_scores).mean(0)
        point_value=torch.stack(point_values).mean(0)
    return tracked_score,torch.stack(set_scores).mean(0),dict(
        tracked_es=torch.stack(values).mean(0),marginal_es=point_value,population_kernel=torch.stack(set_values).mean(0))


def current_cover_certificate(record,graph):
    """Certify A==1 only when all historical paths end at actual current sinks.

    The full causal DAG needs at least c paths for its c latest-time vertices.
    If its minimum d equals c, its path-cover tails are exactly those vertices.
    The proposal contains all of their canonical, forced age-zero atoms. No
    population target, weak historical count bound or stale-source extrapolation
    participates in this certificate. Co-located events retain multiplicity.
    """
    from .mixed_count import legal_context
    history=record['positive_history']
    legal_context(record,history,graph)
    raw_last,events=source_coordinates(record,history)
    current=np.asarray(record['age'])==0
    expected=np.arange(history.positive_offsets[-2],history.positive_offsets[-1])
    if not np.array_equal(np.sort(events[current]),expected):
        raise ValueError('current proposal atoms do not cover every actual current event once')
    c=len(expected);d=graph.minimum_current_count
    if d<c:raise ValueError('full history DAG minimum is below its actual current sinks')
    return d==c,raw_last,d


def _leading_subset(values,index,rows):
    # Keep ALL trailing source/spatial padding. Re-collating a smaller batch
    # would change component indices and is not the original proposal law.
    return {key:value.index_select(0,index) if torch.is_tensor(value) and value.ndim and value.shape[0]==rows else value
            for key,value in values.items()}


def _covered_current_loss(p,batch,records,graphs,rng,worlds,unique,analytic_empty_normalizer):
    if isinstance(worlds,bool) or not isinstance(worlds,(int,np.integer)) or worlds<3:
        raise ValueError('current normalization requires at least3 integer worlds')
    count=len(records);device=batch['x'].device
    if len(graphs)!=count or batch['origin'].shape[:2]!=batch['mask'].shape:
        raise ValueError('current calibration batch differs from its legal sources')
    width=batch['origin'].shape[1]
    raw_last=np.zeros((count,width,2));minimum=np.empty(count,np.int64);covered=np.zeros(count,bool)
    for i,(record,graph) in enumerate(zip(records,graphs)):
        n=len(record['keys'])
        if n>width or not np.array_equal(batch['mask'][i].detach().cpu().numpy()>0,np.arange(width)<n):
            raise ValueError('current calibration source mask differs from its legal record')
        covered[i],positions,dag_minimum=current_cover_certificate(record,graph)
        minimum[i]=max(int(record['minimum_count']),int(dag_minimum))
        raw_last[i,:n]=positions
    current_batch=dict(batch,source_last_positions=torch.as_tensor(raw_last,device=device),
        minimum_count=torch.as_tensor(minimum,device=device),origin=batch['origin'].double(),age=batch['age'].double())
    nll=set_nll(p,current_batch,True,unique,3.)
    negative=torch.zeros_like(nll);logz=torch.zeros_like(nll);proposals=np.zeros(count,np.int64)
    remaining=np.flatnonzero(~covered)
    if len(remaining):
        index=torch.as_tensor(remaining,device=device)
        subp=_leading_subset(p,index,count);subbatch=_leading_subset(batch,index,count)
        subrecords=[records[i] for i in remaining];subgraphs=[graphs[i] for i in remaining]
        raw,diagnostics=sample_latents(subrecords,subp,subgraphs,worlds,rng,unique,False,include_lineage=False)
        z=tensors(raw,device);sublogq=latent_current_log_probability(subp,subbatch,z,unique)
        negative=negative.index_copy(0,index,sublogq.mean(-1))
        estimates=torch.as_tensor([digamma(worlds)-digamma(d['proposals_examined_through_last_accept'])+
            d.get('log_normalizer_multiplier',0.) for d in diagnostics],device=device,dtype=nll.dtype)
        logz=logz.index_copy(0,index,estimates)
        proposals[remaining]=[d['proposals_generated'] for d in diagnostics]
    surrogate=nll+negative;value=nll.detach()+logz
    normalized=surrogate+(value-surrogate.detach());zero=torch.zeros_like(value)
    empty=np.array([g.detections==0 for g in graphs]) if analytic_empty_normalizer else np.zeros(count,bool)
    terms=dict(tracked_es=zero,marginal_es=zero,population_kernel=zero,current_nll=value.detach(),
        normalizer_log_estimate=logz.detach(),proposals=proposals,
        analytic_empty_normalizer=empty.astype(np.float64),analytic_current_cover=covered.astype(np.float64))
    return (normalized*batch['weight']).mean(),terms


def loss(p,batch,records,graphs,rng,worlds=4,current_weight=.05,population_weight=.2,unique=True,guidance=False,marginal_weight=0.,visible_initial_reparameterization=False,current_likelihood_only=False,analytic_empty_normalizer=False,analytic_current_cover=False):
    # Native clocks and measurement ages stay float64; float32 feature tensors
    # remain the neural encoder's adequate numerical representation only.
    batch=dict(batch);p={key:value.double() for key,value in p.items()}
    if analytic_empty_normalizer and not current_likelihood_only:
        raise ValueError('the analytic-empty option is implemented for count-likelihood calibration only')
    if type(analytic_current_cover) is not bool or analytic_current_cover and not current_likelihood_only:
        raise ValueError('analytic current cover requires current-likelihood-only calibration')
    if current_likelihood_only:
        from .tilted_count import is_tilted,LOGITS
        if not is_tilted(p) or any(value.requires_grad for key,value in p.items() if key!=LOGITS):
            raise ValueError('count likelihood calibration requires the total-count head and frozen base')
        validity=batch.get('set_valid')
        if (not torch.is_tensor(validity) or validity.shape!=(len(records),) or
                not torch.isfinite(validity).all() or not (validity==1).all()):
            raise ValueError('count calibration requires a supported native current target for every record')
        if marginal_weight or visible_initial_reparameterization:
            raise ValueError('future-only gradient controls do not apply to current count calibration')
    age=np.zeros(tuple(batch['age'].shape),np.float64)
    for i,record in enumerate(records):age[i,:len(record['keys'])]=record['age']
    batch['age']=torch.as_tensor(age,device=batch['x'].device)
    if not current_likelihood_only:
        batch['times']=torch.as_tensor(np.array([r['times'] for r in records],np.float64),device=batch['x'].device)
    if analytic_current_cover:
        return _covered_current_loss(p,batch,records,graphs,rng,worlds,unique,analytic_empty_normalizer)
    raw,diagnostics=sample_latents(records,p,graphs,worlds,rng,unique,guidance,include_lineage=not current_likelihood_only);z=tensors(raw,batch['x'].device)
    logq=latent_current_log_probability(p,batch,z,unique)
    if current_likelihood_only:
        # Kappa is normalized given the accepted current world. Count likelihood
        # integrates it out; no future trajectory or lineage is needed here.
        current_batch=dict(batch,source_last_positions=z['raw_last'],minimum_count=z['minimum'],
            origin=batch['origin'].double(),age=batch['age'].double())
        nll=set_nll(p,current_batch,True,unique,3.)
        logz=torch.as_tensor([digamma(worlds)-digamma(d['proposals_examined_through_last_accept'])+d.get('log_normalizer_multiplier',0.) for d in diagnostics],device=nll.device,dtype=nll.dtype)
        negative=logq.mean(-1)
        exact_empty=np.zeros(len(graphs),bool)
        if analytic_empty_normalizer:
            for i,(record,graph) in enumerate(zip(records,graphs)):
                actual_count=int(record['positive_history'].positive_offsets[-1])
                if actual_count!=graph.detections:
                    raise ValueError('analytic normalization graph differs from actual positive history')
                exact_empty[i]=(actual_count==0 and graph.detections==0 and graph.minimum_current_count==0)
        for is_empty,diag in zip(exact_empty,diagnostics):
            if is_empty and (diag['proposals_examined_through_last_accept']!=worlds or diag.get('log_normalizer_multiplier',0.)!=0.):
                raise RuntimeError('empty positive history did not certify exact acceptance probability one')
        empty=torch.as_tensor(exact_empty,device=nll.device)
        negative=torch.where(empty,torch.zeros_like(negative),negative)
        logz=torch.where(empty,torch.zeros_like(logz),logz)
        surrogate=nll+negative;value=nll.detach()+logz
        normalized_nll=surrogate+(value-surrogate.detach())
        zero=torch.zeros_like(value)
        terms=dict(tracked_es=zero,marginal_es=zero,population_kernel=zero,current_nll=value.detach(),
            normalizer_log_estimate=logz.detach(),proposals=np.asarray([d['proposals_generated'] for d in diagnostics]),
            analytic_empty_normalizer=exact_empty.astype(np.float64))
        return (normalized_nll*batch['weight']).mean(),terms
    if visible_initial_reparameterization:
        gaussian=visible_initial_gaussian(p,batch,z)
        # Mixed coordinates: q(other current latents)*phi(visible standard
        # noise). A and kappa do not use that noise at EXACT age zero. Keep
        # original numerical logq values while removing only its zero-mean
        # visible Gaussian score, for both the future and negative phase.
        logq=logq-(gaussian-gaussian.detach())
    paths=future_paths(p,batch,z,visible_initial_reparameterization)
    truth=batch['population_future_positions'].double();truth_present=batch['population_person_mask']
    valid=batch['population_future_valid'].double()
    kernel=set_kernel(paths,z['present'],truth,truth_present,valid)
    tracked,population,terms=proper_pair_losses(paths,batch,z,logq,kernel,marginal_weight)
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
