"""Finite mixed-Poisson residual cardinality with exact legal count conditioning.

The count component affects cardinality only. Spatial/velocity marks and the
motion-world mode are independent of it conditional on the residual count.
"""
import numpy as np
from scipy.special import logsumexp
import torch
from torch.nn import functional as F
from .count import suffix_tails,count_log_probability,torch_log_normalizer

LOGITS='residual_count_logits'
RATES='residual_count_rates'


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(value)


def is_mixed(parameters):
    if (LOGITS in parameters)!=(RATES in parameters):
        raise ValueError('both residual mixture parameters are required')
    if RATES in parameters and 'total_count_tilt_logits' in parameters:
        raise ValueError('mixed residual and total-count tilt families cannot be combined')
    return RATES in parameters


def numpy_components(parameters,palm=False):
    if is_mixed(parameters):
        raw_rates=np.asarray(parameters[RATES]);raw_logits=np.asarray(parameters[LOGITS])
    else:
        raw_rates=np.asarray([parameters['rate']]);raw_logits=np.zeros(1,np.float64)
    if raw_rates.dtype.kind not in 'fiu' or raw_logits.dtype.kind not in 'fiu':
        raise ValueError('residual mixture parameters must be real numeric arrays')
    rates=np.asarray(raw_rates,np.float64);logits=np.asarray(raw_logits,np.float64)
    if (rates.ndim!=1 or not len(rates) or logits.shape!=rates.shape or
            not np.isfinite(rates).all() or np.any(rates<=0) or
            np.any(np.isnan(logits)|np.isposinf(logits)) or not np.isfinite(logits).any()):
        raise ValueError('positive finite residual rates and supported mixture logits required')
    active=np.isfinite(logits);rates=rates[active];logits=logits[active]
    with np.errstate(over='ignore',invalid='ignore'):logits=logits-logits.max()
    if not np.isfinite(logits).all():raise ValueError('residual logit difference exceeds numeric support')
    log_weights=logits-logsumexp(logits)
    if palm:log_weights=log_weights+np.log(rates);log_weights-=logsumexp(log_weights)
    return rates,log_weights


def numpy_log_normalizer(source_logits,age,rates,count_logits,minimum,certain=None):
    minimum=_integer(minimum,'minimum')
    p={RATES:rates,LOGITS:count_logits};rates,weights=numpy_components(p)
    values=np.asarray([suffix_tails(source_logits,age,float(rate),int(minimum),certain)[0][0,int(minimum)] for rate in rates])
    if not np.isfinite(values).all():raise RuntimeError('active count component normalizer underflow')
    answer=float(logsumexp(weights+values))
    if not np.isfinite(answer):raise RuntimeError('mixed count normalizer underflow; no fabricated worlds')
    return answer


def numpy_count_log_probability(parameters,source_logits,age,count,minimum=0,certain=None,palm=False):
    count=_integer(count,'count');minimum=_integer(minimum,'minimum')
    rates,weights=numpy_components(parameters,palm)
    if count<minimum:return -np.inf
    norms=np.asarray([suffix_tails(source_logits,age,float(rate),int(minimum),certain)[0][0,int(minimum)] for rate in rates])
    if not np.isfinite(norms).all():raise RuntimeError('active count component normalizer underflow')
    values=np.asarray([count_log_probability(source_logits,age,float(rate),int(count),int(minimum),certain) for rate in rates])
    return float(logsumexp(weights+norms+values)-logsumexp(weights+norms))


def torch_components(parameters):
    if not is_mixed(parameters):raise ValueError('mixed residual count fields required')
    original_rates=parameters[RATES];original_logits=parameters[LOGITS]
    if (not torch.is_tensor(original_rates) or not torch.is_tensor(original_logits) or
            original_rates.dtype==torch.bool or original_logits.dtype==torch.bool or
            original_rates.is_complex() or original_logits.is_complex()):
        raise ValueError('neural count parameters must be real numeric tensors')
    rates=original_rates.double();logits=original_logits.double()
    if rates.ndim!=2 or logits.shape!=rates.shape or rates.shape[0]<1 or rates.shape[1]<1:
        raise ValueError('nonempty batch and residual component dimensions required')
    if not torch.isfinite(rates).all() or (rates<=0).any() or not torch.isfinite(logits).all():
        raise ValueError('neural residual mixture must be finite and positive')
    weights=F.log_softmax(logits,dim=-1)
    if not torch.isfinite(weights).all():raise ValueError('residual logit difference exceeds numeric support')
    return rates,weights


def torch_residual_log_probability(parameters,counts):
    rates,weights=torch_components(parameters)
    if (not isinstance(counts,torch.Tensor) or counts.ndim!=2 or counts.shape[0]!=len(rates)
            or counts.dtype==torch.bool or counts.is_complex() or not torch.isfinite(counts).all()
            or (counts<0).any() or (counts.is_floating_point() and not torch.equal(counts,counts.floor()))):
        raise ValueError('residual counts require batched nonnegative integers')
    u=counts.double()[...,None]
    return torch.logsumexp(weights[:,None]+u*rates[:,None].log()-rates[:,None]-torch.lgamma(u+1),dim=-1)


def torch_mixture_log_normalizer(parameters,source_logits,optional,minimum,return_components=False):
    rates,weights=torch_components(parameters)
    if (source_logits.ndim!=2 or source_logits.shape[0]!=len(rates) or optional.shape!=source_logits.shape
            or optional.dtype!=torch.bool or minimum.shape!=(len(rates),) or minimum.dtype==torch.bool
            or minimum.is_complex() or not torch.isfinite(minimum).all() or (minimum<0).any()
            or (minimum.is_floating_point() and not torch.equal(minimum,minimum.floor()))):
        raise ValueError('invalid source masks or count-floor shape/value')
    norms=torch.stack([torch_log_normalizer(source_logits,optional,rates[:,j],minimum) for j in range(rates.shape[1])],dim=-1)
    mixed=torch.logsumexp(weights+norms,dim=-1)
    return (mixed,norms,weights) if return_components else mixed


def mixed_set_nll(single_rate_function,parameters,batch,condition_minimum=False,
                  compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    """Mix unnormalized Janossy densities, then condition ONCE on the count floor."""
    rates,weights=torch_components(parameters)
    validity=batch['set_valid']
    if (not torch.is_tensor(validity) or validity.shape!=(len(rates),) or validity.is_complex()
            or not torch.isfinite(validity).all() or not ((validity==0)|(validity==1)).all()):
        raise ValueError('set validity must be a finite binary batch mask')
    valid=batch['set_valid']>0
    if not valid.all():
        result=(rates.sum(-1)+weights.sum(-1))*0
        selected=torch.nonzero(valid,as_tuple=True)[0]
        if len(selected):
            def subset(values):
                return {key:value[selected] if torch.is_tensor(value) and value.ndim and len(value)==len(valid) else value
                        for key,value in values.items()}
            scored=mixed_set_nll(single_rate_function,subset(parameters),subset(batch),condition_minimum,
                compulsory_unique_episode,max_speed,compulsory_observed_clique)
            result=result.index_copy(0,selected,scored)
        return result
    # Use the same forced/optional source semantics as the single-rate law.
    optional=(batch['age']>0)&(batch['mask']>0)
    forced=(batch['age']==0)&(batch['mask']>0)
    compulsory=torch.zeros_like(optional)
    if compulsory_unique_episode:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==1)
    if compulsory_observed_clique:compulsory|=optional&(batch['mask'].sum(-1,keepdim=True)==batch['minimum_count'][:,None])
    optional=optional&~compulsory
    if condition_minimum:
        minimum=(batch['minimum_count']-forced.sum(-1)-compulsory.sum(-1)).clamp_min(0)
        logz,norms,_=torch_mixture_log_normalizer(parameters,parameters['presence_logits'],optional,minimum,True)
    else:
        logz=torch.zeros_like(rates[:,0]);norms=torch.zeros_like(rates)
    components=[]
    for j in range(rates.shape[1]):
        branch=dict(parameters,rate=rates[:,j])
        nll=single_rate_function(branch,batch,condition_minimum,compulsory_unique_episode,max_speed,compulsory_observed_clique)
        components.append(-nll+norms[:,j]*batch['set_valid'])
    value=-torch.logsumexp(weights+torch.stack(components,dim=-1),dim=-1)+logz*batch['set_valid']
    return value*batch['set_valid']


def legal_context(features,history,graph):
    """Three causal cardinality facts, never a simulator population label."""
    graph.validate_certificate()
    from .association import _columns
    names=('frame_timestamps','positive_offsets','positive_xyz','observation_valid','human_sensor_available')
    if any(not np.array_equal(getattr(graph,name),column) for name,column in zip(names,_columns(history))):
        raise ValueError('cardinality graph differs from the complete actual history')
    if float(history.timestamps[-1])!=float(features['timestamp']) or graph.timestamp!=float(features['timestamp']):
        raise ValueError('cardinality context clock differs from actual history')
    visible=int(history.positive_offsets[-1]-history.positive_offsets[-2])
    ages=np.asarray(features['age'])
    if (ages.shape!=(len(features['keys']),) or not np.isfinite(ages).all() or np.any(ages<0)
            or int(np.count_nonzero(ages==0))!=visible):
        raise ValueError('exact-current proposal atoms differ from actual current positive observations')
    if len(features['keys'])>len(history.positive_xyz) or visible>len(features['keys']):
        raise ValueError('proposal episodes lack complete actual positive evidence')
    return np.asarray([graph.minimum_current_count,visible,len(features['keys'])],np.float32)/10.
