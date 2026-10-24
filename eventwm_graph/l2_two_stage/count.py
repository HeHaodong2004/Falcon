"""Exact minimum-population conditioning of Bernoulli + Poisson laws."""
import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
import torch
from torch.nn import functional as F


def certain_sources(age,minimum_count,unique=False,observed_clique=False):
    """Legal co-observation witness, never a privileged identity assertion.

    IfN remembered keys equalsL simultaneous actual detections, every key
    co-occurred in that frame. They cannot all be aliases of fewer people.
    No such assertion is made for ambiguousN>L episode collections.
    """
    age=np.asarray(age)
    witness=(unique and len(age)==1) or (observed_clique and len(age)>0 and len(age)==minimum_count)
    return (age==0)|bool(witness)


def suffix_tails(logits,age,rate,minimum,certain=None):
    logits=np.asarray(logits,float);age=np.asarray(age);n=len(logits)
    yes=-np.logaddexp(0.,-logits);no=-np.logaddexp(0.,logits)
    certain=age==0 if certain is None else np.asarray(certain,bool)|(age==0)
    if certain.shape!=age.shape:raise ValueError('certain source mask mismatch')
    yes=np.where(certain,0.,yes);no=np.where(certain,-math.inf,no)
    table=np.empty((n+1,minimum+1));table[-1,0]=0
    if minimum:table[-1,1:]=poisson.logsf(np.arange(1,minimum+1)-1,rate)
    for i in range(n-1,-1,-1):
        table[i,0]=0
        for needed in range(1,minimum+1):table[i,needed]=np.logaddexp(no[i]+table[i+1,needed],yes[i]+table[i+1,needed-1])
    return table,yes,no


def sample_conditioned(logits,age,rate,minimum,worlds,rng,certain=None):
    table,yes,no=suffix_tails(logits,age,rate,minimum,certain)
    if not np.isfinite(table[0,minimum]):raise RuntimeError('minimum-population normalizer underflow; no fabricated worlds')
    present=np.zeros((worlds,len(logits)),bool);needed=np.full(worlds,minimum,int)
    for i in range(len(logits)):
        logp=yes[i]+table[i+1,np.maximum(needed-1,0)]-table[i,needed]
        probability=np.exp(logp).clip(0,1)
        present[:,i]=rng.random(worlds)<probability;needed=np.maximum(needed-present[:,i],0)
    counts=np.zeros(worlds,int)
    for lower in np.unique(needed):
        selected=np.flatnonzero(needed==lower)
        if lower==0:counts[selected]=rng.poisson(rate,len(selected));continue
        survival=poisson.sf(int(lower)-1,rate)
        if not survival>0:raise RuntimeError('truncated Poisson tail underflow')
        uniform=(rng.random(len(selected))+.5*np.finfo(float).eps).clip(np.finfo(float).tiny,1-np.finfo(float).eps)
        sampled=poisson.isf(uniform*survival,rate)
        rare=(rate<float(lower) or survival<1e-8)
        fallback=np.ones(len(selected),bool) if rare else (~np.isfinite(sampled))|(sampled<lower)
        if fallback.any():
            u=uniform[fallback];mass=float(np.exp(poisson.logpmf(lower,rate)-poisson.logsf(int(lower)-1,rate)))
            cumulative=mass;value=int(lower);answer=np.full(len(u),value,int);pending=u>cumulative
            while pending.any():
                value+=1;mass*=rate/value;cumulative+=mass
                answer[pending]=value;pending=u>cumulative
                if value-int(lower)>100000 or (mass==0 and pending.any()):raise RuntimeError('conditional Poisson recurrence exhausted numeric support')
            sampled[fallback]=answer
        if not np.isfinite(sampled).all():raise RuntimeError('truncated Poisson inverse unsupported')
        counts[selected]=sampled.astype(int)
    if np.any(present.sum(-1)+counts<minimum):raise RuntimeError('minimum-population sampler invariant failed')
    return present,counts,float(table[0,minimum])


def sample_total_count(logits,age,rate,total,worlds,rng,certain=None):
    """Exact Bernoulli/Poisson allocation conditional on a fixed TOTAL count.

    This is a PMF suffix recursion, not truncated-tail conditioning. It enables
    rigorous stratified integration of the history-conditioned count law.
    """
    logits=np.asarray(logits,float);age=np.asarray(age);total=int(total)
    if total<0:raise ValueError('nonnegative total count required')
    forced=(age==0) if certain is None else np.asarray(certain,bool)|(age==0)
    yes=np.where(forced,0.,-np.logaddexp(0.,-logits));no=np.where(forced,-math.inf,-np.logaddexp(0.,logits))
    table=np.empty((len(logits)+1,total+1));table[-1]=poisson.logpmf(np.arange(total+1),rate)
    for i in range(len(logits)-1,-1,-1):
        table[i,0]=no[i]+table[i+1,0]
        table[i,1:]=np.logaddexp(no[i]+table[i+1,1:],yes[i]+table[i+1,:-1])
    if not np.isfinite(table[0,total]):raise ValueError('fixed total has zero proposal probability')
    remaining=np.full(worlds,total,int);present=np.zeros((worlds,len(logits)),bool)
    for i in range(len(logits)):
        logp=yes[i]+table[i+1,np.maximum(remaining-1,0)]-table[i,remaining]
        probability=np.where(remaining>0,np.exp(logp).clip(0,1),0.)
        present[:,i]=rng.random(worlds)<probability;remaining-=present[:,i]
    if np.any(remaining<0) or np.any(present.sum(-1)+remaining!=total):raise RuntimeError('fixed-count sampler invariant failed')
    return present,remaining,float(table[0,total])


def torch_log_normalizer(logits,optional,rate,minimum):
    """Differentiable exact suffix recursion; Poisson gamma tails in float64."""
    maximum=int(minimum.max());b=len(rate);dtype=torch.float64;device=rate.device
    if maximum==0:return torch.zeros_like(rate,dtype=dtype)
    k=torch.arange(1,maximum+1,device=device,dtype=dtype)
    tail=torch.special.gammainc(k[None],rate.double()[:,None]).log()
    tail=torch.cat((torch.zeros((b,1),device=device,dtype=dtype),tail),dim=-1)
    yes=F.logsigmoid(logits.double()).masked_fill(~optional,-torch.inf)
    no=F.logsigmoid(-logits.double()).masked_fill(~optional,0.)
    for i in range(logits.shape[-1]-1,-1,-1):
        tail=torch.cat((torch.zeros((b,1),device=device,dtype=dtype),
            torch.logaddexp(no[:,i,None]+tail[:,1:],yes[:,i,None]+tail[:,:-1])),dim=-1)
    normalizer=tail[torch.arange(b,device=device),minimum.long()]
    if not torch.isfinite(normalizer).all():raise RuntimeError('population normalizer underflow')
    return normalizer


def count_log_probability(logits,age,rate,count,minimum_count=0,certain=None):
    if count<minimum_count:return -math.inf
    certain=(np.asarray(age)==0) if certain is None else np.asarray(certain,bool)|(np.asarray(age)==0)
    normalizer=suffix_tails(logits,age,rate,minimum_count,certain)[0][0,minimum_count]
    count-=int(np.count_nonzero(certain))
    if count<0:return -math.inf
    polynomial=np.array([0.])
    for logit in np.asarray(logits)[~certain]:
        yes=-np.logaddexp(0.,-float(logit));no=-np.logaddexp(0.,float(logit))
        previous=polynomial;polynomial=np.full(len(previous)+1,-math.inf)
        polynomial[:-1]=previous+no;polynomial[1:]=np.logaddexp(polynomial[1:],previous+yes)
    return float(logsumexp([mass+poisson.logpmf(count-i,rate) for i,mass in enumerate(polynomial) if i<=count])-normalizer)
