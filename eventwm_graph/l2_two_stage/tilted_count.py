"""Positive total-cardinality potentials on the unchanged Bernoulli/PPP base.

The last potential applies to ALL counts >= cutoff. It is not a population cap.
All normalizers below use the unconditioned base, before its legal count floor.
"""
import numpy as np
from scipy.special import logsumexp
import torch
from torch.nn import functional as F
from .count import count_log_probability,suffix_tails,torch_log_normalizer

LOGITS='total_count_tilt_logits'


def is_tilted(parameters):
    result=LOGITS in parameters
    if result and ('residual_count_rates' in parameters or 'residual_count_logits' in parameters):
        raise ValueError('mixed residual and total-count tilt families cannot be combined')
    return result


def integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(value)


def numpy_weights(parameters):
    if not is_tilted(parameters):raise ValueError('total-count potential field required')
    raw=np.asarray(parameters[LOGITS])
    if raw.dtype.kind not in 'fiu' or raw.ndim!=1 or len(raw)<2 or not np.isfinite(raw).all():
        raise ValueError('finite positive potentials required for every total-count class')
    value=np.asarray(raw,np.float64)
    with np.errstate(over='ignore',invalid='ignore'):value=value-value.max()
    if not np.isfinite(value).all():raise ValueError('total-count log-potential difference exceeds numeric support')
    return value-logsumexp(value)


def numpy_log_normalizer(parameters,source_logits,age,minimum,certain=None,shift=0):
    minimum=integer(minimum,'minimum');shift=integer(shift,'shift')
    weights=numpy_weights(parameters);cutoff=len(weights)-1;rate=float(parameters['rate'])
    if not np.isfinite(rate) or rate<=0:raise ValueError('positive finite PPP base rate required')
    finite=[]
    for n in range(minimum,max(minimum,cutoff-shift)):
        mass=count_log_probability(source_logits,age,rate,n,0,certain)
        if np.isfinite(mass):finite.append(weights[n+shift]+mass)
    lower=max(minimum,cutoff-shift,0)
    tail=suffix_tails(source_logits,age,rate,lower,certain)[0][0,lower]
    if not np.isfinite(tail):raise RuntimeError('positive unbounded count-tail normalizer underflow')
    answer=float(logsumexp(finite+[weights[-1]+tail]))
    if not np.isfinite(answer):raise RuntimeError('total-count tilt normalizer underflow')
    return answer


def numpy_count_log_probability(parameters,source_logits,age,count,minimum=0,certain=None,shift=0):
    count=integer(count,'count');minimum=integer(minimum,'minimum');shift=integer(shift,'shift')
    if count<minimum:return -np.inf
    weights=numpy_weights(parameters)
    mass=count_log_probability(source_logits,age,float(parameters['rate']),count,0,certain)
    return float(weights[min(count+shift,len(weights)-1)]+mass-
        numpy_log_normalizer(parameters,source_logits,age,minimum,certain,shift))


def torch_weights(parameters):
    if not is_tilted(parameters):raise ValueError('total-count potential field required')
    raw=parameters[LOGITS]
    if (not torch.is_tensor(raw) or raw.dtype==torch.bool or raw.is_complex() or raw.ndim!=2
            or len(raw)<1 or raw.shape[1]<2 or not torch.isfinite(raw).all()):
        raise ValueError('finite batched total-count potential tensor required')
    value=F.log_softmax(raw.double(),dim=-1)
    if not torch.isfinite(value).all():raise ValueError('total-count log-potential difference exceeds numeric support')
    return value


def torch_log_potential(parameters,counts):
    weights=torch_weights(parameters)
    if (not torch.is_tensor(counts) or counts.ndim not in (1,2) or counts.shape[0]!=len(weights)
            or counts.dtype==torch.bool or counts.is_complex() or not torch.isfinite(counts).all()
            or (counts<0).any() or (counts.is_floating_point() and not torch.equal(counts,counts.floor()))):
        raise ValueError('integer total-count batch required')
    flat=counts.reshape(len(weights),-1).long().clamp_max(weights.shape[1]-1)
    return weights.gather(1,flat).reshape(counts.shape)


def torch_tilt_log_normalizer(parameters,source_logits,optional,minimum,certain_count):
    weights=torch_weights(parameters);rate=parameters['rate'].double();b=len(weights);cutoff=weights.shape[1]-1
    if (rate.shape!=(b,) or not torch.isfinite(rate).all() or (rate<=0).any()
            or source_logits.ndim!=2 or source_logits.shape[0]!=b or not torch.isfinite(source_logits).all()
            or optional.shape!=source_logits.shape or optional.dtype!=torch.bool):
        raise ValueError('invalid Bernoulli/PPP base for total-count potentials')
    for value in (minimum,certain_count):
        if (value.shape!=(b,) or value.dtype==torch.bool or value.is_complex() or not torch.isfinite(value).all()
                or (value<0).any() or (value.is_floating_point() and not torch.equal(value,value.floor()))):
            raise ValueError('integer full count floor and certain-source counts required')
    n=torch.arange(cutoff,device=rate.device,dtype=torch.float64)
    # Direct PMF recursion, never subtraction of almost-equal survival values.
    pmf=n[None]*rate[:,None].log()-rate[:,None]-torch.lgamma(n[None]+1.)
    yes=F.logsigmoid(source_logits.double()).masked_fill(~optional,-torch.inf)
    no=F.logsigmoid(-source_logits.double()).masked_fill(~optional,0.)
    for i in range(source_logits.shape[1]-1,-1,-1):
        shifted=torch.cat((torch.full_like(pmf[:,:1],-torch.inf),pmf[:,:-1]),dim=-1)
        pmf=torch.logaddexp(no[:,i,None]+pmf,yes[:,i,None]+shifted)
    index=n.long()[None]-certain_count.long()[:,None]
    finite=pmf.gather(1,index.clamp_min(0))+weights[:,:cutoff]
    finite=finite.masked_fill((index<0)|(n[None]<minimum[:,None]),-torch.inf)
    tail_floor=(torch.maximum(minimum,torch.full_like(minimum,cutoff))-certain_count).clamp_min(0)
    tail=torch_log_normalizer(source_logits,optional,rate,tail_floor)+weights[:,-1]
    result=torch.logsumexp(torch.cat((finite,tail[:,None]),dim=1),dim=1)
    if not torch.isfinite(result).all():raise RuntimeError('total-count tilt normalizer underflow')
    return result


def tilted_set_nll(single_rate_function,parameters,batch,condition_minimum=False,
                   compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    weights=torch_weights(parameters);validity=batch['set_valid']
    if (validity.shape!=(len(weights),) or validity.is_complex() or not torch.isfinite(validity).all()
            or not ((validity==0)|(validity==1)).all()):
        raise ValueError('finite binary current-target validity required')
    valid=validity>0
    if not valid.all():
        result=weights.sum(-1)*0;selected=torch.nonzero(valid,as_tuple=True)[0]
        if len(selected):
            def subset(values):return {k:v[selected] if torch.is_tensor(v) and v.ndim and len(v)==len(valid) else v for k,v in values.items()}
            scored=tilted_set_nll(single_rate_function,subset(parameters),subset(batch),condition_minimum,
                compulsory_unique_episode,max_speed,compulsory_observed_clique)
            result=result.index_copy(0,selected,scored)
        return result
    optional=(batch['age']>0)&(batch['mask']>0)
    current=(batch['age']==0)&(batch['mask']>0)
    compulsory=torch.zeros_like(optional)
    if compulsory_unique_episode:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==1)
    if compulsory_observed_clique:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==batch['minimum_count'][:,None])
    optional=optional&~compulsory
    # Stale compulsory sources are ALREADY included in the unobserved target set.
    total=batch['set_mask'].sum(-1).long()+current.sum(-1)
    minimum=batch['minimum_count'] if condition_minimum else torch.zeros_like(total)
    if condition_minimum and (total<minimum).any():
        raise ValueError('native population target below legal historical count floor')
    normalizer=torch_tilt_log_normalizer(parameters,parameters['presence_logits'],optional,minimum,
        current.sum(-1)+compulsory.sum(-1))
    base=single_rate_function(parameters,batch,False,compulsory_unique_episode,max_speed,compulsory_observed_clique)
    return (base-torch_log_potential(parameters,total)+normalizer)*validity
