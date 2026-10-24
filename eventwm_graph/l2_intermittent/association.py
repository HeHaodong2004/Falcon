"""Exact finite-prior assignment conditioning, without a permutation beam.

The suffix table sums complete injective assignments in O(n D 2**D).
Tracked misses and residual hidden-count likelihoods enter the evidence before
sampling. For singular emissions we use common isotropic small-ball limits:
minimum total dimension, with unit-ball volumes in the leading coefficient.
This is an explicit convention, not a Lebesgue density for point masses.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _add(a,b):
    if a == -np.inf:return b
    if b == -np.inf:return a
    m=max(a,b)
    return m+math.log(math.exp(a-m)+math.exp(b-m))


@njit(cache=True)
def posterior_table(log_positive,ranks,log_miss,log_birth,hidden_den,hidden_miss,birth_rank):
    n,d=log_positive.shape;size=1<<d
    dimension=np.full((n+1,size),100000,np.int32)
    mass=np.full((n+1,size),-np.inf)
    entropy=np.zeros((n+1,size))
    count=np.zeros((n+1,size))
    for mask in range(size):
        nb=0;lp=0.
        for z in range(d):
            if not mask & (1<<z):nb+=1;lp+=log_birth[z]
        if nb>len(hidden_den):continue
        for k in range(nb):lp+=math.log(hidden_den[k])
        lp+=hidden_miss[nb]
        if not math.isfinite(lp):continue
        dimension[n,mask]=birth_rank*nb;mass[n,mask]=lp;count[n,mask]=1.
    for i in range(n-1,-1,-1):
        for mask in range(size):
            best=dimension[i+1,mask];total=log_miss[i]+mass[i+1,mask]
            for z in range(d):
                if mask & (1<<z) or not math.isfinite(log_positive[i,z]):continue
                child=mask | (1<<z);rank=ranks[i]+dimension[i+1,child]
                lp=log_positive[i,z]+mass[i+1,child]
                if rank<best:best=rank;total=lp
                elif rank==best:total=_add(total,lp)
            dimension[i,mask]=best;mass[i,mask]=total
            if not math.isfinite(total):continue
            if dimension[i+1,mask]==best:
                lp=log_miss[i]+mass[i+1,mask];p=math.exp(lp-total)
                if p>0:entropy[i,mask]+=p*(entropy[i+1,mask]-math.log(p))
                count[i,mask]+=count[i+1,mask]
            for z in range(d):
                if mask & (1<<z) or not math.isfinite(log_positive[i,z]):continue
                child=mask | (1<<z)
                if ranks[i]+dimension[i+1,child]!=best:continue
                lp=log_positive[i,z]+mass[i+1,child];p=math.exp(lp-total)
                if p>0:entropy[i,mask]+=p*(entropy[i+1,child]-math.log(p))
                count[i,mask]+=count[i+1,child]
    return dimension,mass,entropy[0,0],count[0,0]


@njit(cache=True)
def sample_assignment(log_positive,ranks,log_miss,dimension,mass,uniforms):
    n,d=log_positive.shape;assign=np.full(d,-1,np.int64);mask=0;logp=0.
    for i in range(n):
        u=uniforms[i];cumulative=0.;chosen=-1
        if dimension[i+1,mask]==dimension[i,mask]:
            p=math.exp(log_miss[i]+mass[i+1,mask]-mass[i,mask])
            cumulative+=p
            if u<cumulative:chosen=-2;logp+=math.log(p)
        for z in range(d):
            if chosen!=-1:break
            if mask & (1<<z) or not math.isfinite(log_positive[i,z]):continue
            child=mask | (1<<z)
            if ranks[i]+dimension[i+1,child]!=dimension[i,mask]:continue
            p=math.exp(log_positive[i,z]+mass[i+1,child]-mass[i,mask])
            cumulative+=p
            if u<cumulative+1e-13:chosen=z;logp+=math.log(p)
        if chosen==-1:raise ValueError('invalid exact assignment conditional')
        if chosen>=0:assign[chosen]=i;mask|=1<<chosen
    return assign,logp


@njit(cache=True)
def association_marginals(log_positive,ranks,log_miss,dimension,mass):
    """Exact P(track i emits center z), and P(center z is a hidden birth)."""
    n,d=log_positive.shape;reach=np.zeros(1<<d);reach[0]=1.
    matched=np.zeros((n,d))
    for i in range(n):
        following=np.zeros_like(reach)
        for mask in range(len(reach)):
            if reach[mask]==0:continue
            if dimension[i+1,mask]==dimension[i,mask]:
                following[mask]+=reach[mask]*math.exp(log_miss[i]+mass[i+1,mask]-mass[i,mask])
            for z in range(d):
                if mask & (1<<z) or not math.isfinite(log_positive[i,z]):continue
                child=mask | (1<<z)
                if ranks[i]+dimension[i+1,child]!=dimension[i,mask]:continue
                p=reach[mask]*math.exp(log_positive[i,z]+mass[i+1,child]-mass[i,mask])
                matched[i,z]+=p;following[child]+=p
        reach=following
    birth=np.ones(d)-matched.sum(axis=0)
    return matched,birth


def prepare(logdensity,ranks,log_pd,miss,birth,hidden_den,hidden_miss,birth_rank=2):
    d=logdensity.shape[1]
    if d>16:raise ValueError('UNSUPPORTED_ASSOCIATION_DETECTION_BUDGET')
    volumes=np.array([0.,math.log(2.),math.log(math.pi)])
    positive=np.asarray(logdensity,float)+log_pd[None]+volumes[np.asarray(ranks),None]
    birth=np.asarray(birth,float)+volumes[birth_rank]
    log_miss=np.log(np.asarray(miss,float))
    table=posterior_table(positive,np.asarray(ranks,np.int64),log_miss,birth,
                          np.asarray(hidden_den,float),np.asarray(hidden_miss,float),int(birth_rank))
    return positive,log_miss,table
