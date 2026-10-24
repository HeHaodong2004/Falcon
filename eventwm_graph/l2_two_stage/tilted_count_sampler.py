"""Exact global total-count tilt of one Bernoulli-plus-Poisson allocation law.

For base remainder count N=B+U, weights a=softmax(tilt_logits) multiply its
mass by a[min(N+shift, cutoff)]. The last weight covers the ENTIRE unbounded
tail, not a point count at cutoff. A finite-count PMF suffix table and one SF
suffix table prepare all classes; individual classes only take immutable views.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp,pdtrc
from scipy.stats import poisson

from .fast_count import PreparedCountSampler,_prepare_prefix
from .types import immutable_array

VERSION='global_total_count_tilt_finite_pmf_plus_unbounded_tail_1'


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(value)


def _vector(value,name):
    value=np.asarray(value)
    if value.ndim!=1 or value.dtype.kind not in 'fiu':
        raise ValueError(name+' must be a real numeric vector')
    value=np.asarray(value,np.float64)
    if not np.isfinite(value).all():raise ValueError(name+' must be finite')
    return value


def _view(component,minimum,total=None):
    """Lower columns are the same suffix event, never SF/PMF interchange."""
    if total is None:
        if component.total is not None or minimum>component.minimum:
            raise ValueError('unprepared SF floor')
        maximum=minimum;prefixes=component.prefixes[:minimum]
    else:
        if component.total is None or total>component.total or total<minimum:
            raise ValueError('unprepared fixed-total PMF')
        maximum=total;prefixes=()
    normalizer=float(component.suffix[0,maximum])
    if not np.isfinite(normalizer):
        raise RuntimeError('count allocation has zero or unsupported numeric mass')
    result=object.__new__(PreparedCountSampler)
    for name,value in [('rate',component.rate),('minimum',minimum),('total',total),
        ('log_normalizer',normalizer),('suffix',immutable_array(component.suffix[:,:maximum+1])),
        ('probabilities',immutable_array(component.probabilities[:,:maximum+1])),('prefixes',prefixes)]:
        object.__setattr__(result,name,value)
    return result


def _prepare_base_tables(logits,age,rate,minimum,maximum,certain):
    """One Poisson grid and batched SF/PMF recursions; old allocation draws.

    PMFs are computed directly, never by subtracting survival probabilities.
    Each logaddexp has the old operands/order. SF column0 is exactly zero;
    PMF column0 propagates the no-source mass. Higher unused PMF columns do
    not affect any lower column. Large grids retain the reference fallback.
    """
    upper=max(minimum,maximum,int(math.ceil(rate+8*math.sqrt(rate+1))))
    if upper>65536:
        return (PreparedCountSampler(logits,age,rate,minimum,certain=certain),
                PreparedCountSampler(logits,age,rate,0,certain=certain,total=maximum))
    width=max(minimum,maximum)
    # For these validated integer thresholds and positive rate, SciPy's
    # Poisson _sf is pdtrc(floor(k),rate), and its _logsf applies np.log.
    # Preserve the k=-1 support value explicitly; no tail is approximated.
    sf_grid=np.empty(upper+2,np.float64);sf_grid[0]=0.
    with np.errstate(divide='ignore'):
        sf_grid[1:]=np.log(pdtrc(np.arange(upper+1),rate))
    pmf_grid=poisson.logpmf(np.arange(upper+1),rate)
    yes=np.where(certain,0.,-np.logaddexp(0.,-logits))
    no=np.where(certain,-np.inf,-np.logaddexp(0.,logits))
    tables=np.empty((2,len(logits)+1,width+1),np.float64)
    tables[0,-1]=sf_grid[:width+1]
    tables[1,-1]=pmf_grid[:width+1]
    for i in range(len(logits)-1,-1,-1):
        tables[0,i,0]=0.
        tables[1,i,0]=no[i]+tables[1,i+1,0]
        tables[:,i,1:]=np.logaddexp(no[i]+tables[:,i+1,1:],yes[i]+tables[:,i+1,:-1])
    if not np.isfinite(tables[0,0,minimum]) or not np.isfinite(tables[1,0,maximum]):
        raise RuntimeError('base SF/PMF normalizer unsupported; no class dropped')
    probabilities=np.zeros((2,len(logits),width+1),np.float64)
    needed=np.arange(width+1)
    for i in range(len(logits)):
        supported=np.isfinite(tables[:,i])
        with np.errstate(invalid='ignore'):
            logp=yes[i]+tables[:,i+1][:,np.maximum(needed-1,0)]-tables[:,i]
        probabilities[:,i][supported]=np.exp(logp[supported]).clip(0,1)
        probabilities[1,i,0]=0.
    prefixes=tuple(_prepare_prefix(rate,lower,(sf_grid,pmf_grid)) for lower in range(1,minimum+1))
    prepared=[]
    for index,limit,total in [(0,minimum,None),(1,maximum,maximum)]:
        result=object.__new__(PreparedCountSampler)
        for name,value in [('rate',rate),('minimum',minimum if total is None else 0),('total',total),
            ('log_normalizer',float(tables[index,0,limit])),
            ('suffix',immutable_array(tables[index,:,:limit+1])),
            ('probabilities',immutable_array(probabilities[index,:,:limit+1])),
            ('prefixes',prefixes if total is None else ())]:object.__setattr__(result,name,value)
        prepared.append(result)
    return tuple(prepared)


@dataclass(frozen=True,init=False)
class PreparedTiltedCountSampler:
    """Immutable allocation law proportional to base(B,U) a[N+shift] 1[N>=d].

    ``total`` refers to the REMAINDER N; Palm adds its distinguished person
    separately. A fixed total cancels the tilt in the allocation conditional.
    ``log_normalizer`` always includes that tilt factor and the unconditioned
    base event mass. All tilt logits must be finite, so every possible count
    retains positive mathematical weight. Numerical failures fail closed.
    """
    logits:np.ndarray
    age:np.ndarray
    certain:np.ndarray
    rate:float
    minimum:int
    total:object
    shift:int
    tilt_logits:np.ndarray
    log_tilt_weights:np.ndarray
    cutoff:int
    source_count:int
    forced_count:int
    log_normalizer:float
    class_totals:np.ndarray
    class_log_probabilities:np.ndarray
    samplers:tuple
    base_sf:object
    base_pmf:object
    overflow_minimum:object
    version:str

    def __init__(self,logits,age,rate,minimum,tilt_logits,certain=None,total=None,shift=0):
        logits=_vector(logits,'logits');age=_vector(age,'age')
        weights=_vector(tilt_logits,'tilt_logits')
        if age.shape!=logits.shape or np.any(age<0):raise ValueError('matching nonnegative source ages required')
        if not len(weights):raise ValueError('nonempty positive count-tilt support required')
        raw_rate=np.asarray(rate)
        if raw_rate.ndim!=0 or raw_rate.dtype.kind not in 'fiu':raise ValueError('real scalar Poisson rate required')
        rate=float(raw_rate)
        if not math.isfinite(rate) or rate<=0:raise ValueError('finite positive Poisson rate required')
        minimum=_integer(minimum,'minimum');shift=_integer(shift,'shift')
        if total is not None:
            total=_integer(total,'total')
            if total<minimum:raise ValueError('fixed remainder total is below its floor')
        # Only an actual current positive observation certifies this proposal
        # source. Every strictly positive age keeps its continuous uncertainty.
        forced=age==0
        if certain is not None:
            certain=np.asarray(certain,bool)
            if certain.shape!=age.shape:raise ValueError('certain mask mismatch')
            forced|=certain
        with np.errstate(over='ignore',invalid='ignore'):
            centered=weights-weights.max()
            log_weights=centered-logsumexp(centered)
        if not np.isfinite(log_weights).all():raise RuntimeError('finite count-tilt log weight overflow')
        self._configure(logits,age,rate,minimum,weights,log_weights,forced,total,shift)

    def _configure(self,logits,age,rate,minimum,weights,log_weights,certain,total,shift,sf=None,pmf=None):
        cutoff=len(weights)-1;forced_count=int(certain.sum())
        sf_pool=sf;pmf_pool=pmf
        if total is not None:
            if pmf_pool is not None and pmf_pool.total>=total:
                component=_view(pmf_pool,minimum,total)
            else:component=PreparedCountSampler(logits,age,rate,minimum,certain=certain,total=total)
            classes=[total];samplers=[component];overflow=None
            log_masses=np.array([component.log_normalizer+log_weights[min(total+shift,cutoff)]])
        elif np.all(weights==weights[0]) or max(minimum,forced_count)+shift>=cutoff:
            # The tilt is constant on all possible base counts. Delegation
            # preserves the complete old sampler output AND RNG stream.
            component=(_view(sf_pool,minimum) if sf_pool is not None and sf_pool.minimum>=minimum
                       else PreparedCountSampler(logits,age,rate,minimum,certain=certain))
            sf_pool=component if sf_pool is None else sf_pool
            classes=[-1];samplers=[component];overflow=minimum
            log_masses=np.array([component.log_normalizer+log_weights[-1]])
        else:
            overflow=max(minimum,cutoff-shift)
            maximum=cutoff-shift-1
            finite=list(range(max(minimum,forced_count),maximum+1))
            if finite and sf_pool is None and pmf_pool is None:
                sf_pool,pmf_pool=_prepare_base_tables(logits,age,rate,overflow,maximum,certain)
            component=(_view(sf_pool,overflow) if sf_pool is not None and sf_pool.minimum>=overflow
                       else PreparedCountSampler(logits,age,rate,overflow,certain=certain))
            sf_pool=component if sf_pool is None else sf_pool
            if finite:
                if pmf_pool is None or pmf_pool.total<maximum:
                    # ONE PMF recursion prepares every finite class. Its
                    # highest total has positive support because maximum>=F.
                    pmf_pool=PreparedCountSampler(logits,age,rate,0,certain=certain,total=maximum)
                finite_samplers=[_view(pmf_pool,minimum,count) for count in finite]
            else:finite_samplers=[]
            classes=finite+[-1];samplers=finite_samplers+[component]
            log_masses=np.array([item.log_normalizer+log_weights[count+shift]
                                 for count,item in zip(finite,finite_samplers)]
                                +[component.log_normalizer+log_weights[-1]],np.float64)
        if not np.isfinite(log_masses).all():
            raise RuntimeError('positive count-tilt class has unsupported numeric mass; no class dropped')
        centered=log_masses-log_masses.max()
        log_sum=float(logsumexp(centered))
        normalizer=float(log_masses.max()+log_sum)
        log_probabilities=centered-log_sum
        if not np.isfinite(normalizer) or not np.isfinite(log_probabilities).all():
            raise RuntimeError('count-tilt normalizer unsupported; no fabricated allocation')
        for name,value in [('logits',immutable_array(logits)),('age',immutable_array(age)),
            ('certain',immutable_array(certain)),('rate',rate),('minimum',minimum),('total',total),('shift',shift),
            ('tilt_logits',immutable_array(weights)),('log_tilt_weights',immutable_array(log_weights)),
            ('cutoff',cutoff),('source_count',len(logits)),('forced_count',forced_count),
            ('log_normalizer',normalizer),('class_totals',immutable_array(np.array(classes,np.int64))),
            ('class_log_probabilities',immutable_array(log_probabilities)),('samplers',tuple(samplers)),
            ('base_sf',sf_pool),('base_pmf',pmf_pool),('overflow_minimum',overflow),('version',VERSION)]:
            object.__setattr__(self,name,value)

    def recondition(self,minimum,shift,total=None):
        """Share fixed base tables for a lower floor / larger Palm count shift.

        New fixed totals use existing PMF columns where available and otherwise
        one fresh fixed-total recursion. SF and PMF tables never substitute for
        each other. Source probabilities, rate and tilt weights cannot change.
        """
        minimum=_integer(minimum,'minimum');shift=_integer(shift,'shift')
        if self.total is not None:raise ValueError('recondition the unfixed floor sampler')
        if minimum>self.minimum or shift<self.shift:
            raise ValueError('only lower floors and nondecreasing count shifts are reusable')
        if total is not None:
            total=_integer(total,'total')
            if total<minimum:raise ValueError('fixed remainder total is below its floor')
        result=object.__new__(type(self))
        result._configure(self.logits,self.age,self.rate,minimum,self.tilt_logits,self.log_tilt_weights,
                          self.certain,total,shift,self.base_sf,self.base_pmf)
        return result

    def sample(self,worlds,rng):
        worlds=_integer(worlds,'worlds')
        if not isinstance(rng,np.random.Generator):raise ValueError('numpy Generator required')
        if len(self.samplers)==1:return self.samplers[0].sample(worlds,rng)
        present=np.zeros((worlds,self.source_count),bool);hidden=np.empty(worlds,np.int64)
        if not worlds:return present,hidden
        # Every finite log weight participates; exponentiation cannot silently
        # delete either a rare finite class or the unbounded overflow branch.
        selected=np.argmax(self.class_log_probabilities[None,:]+
                           rng.gumbel(size=(worlds,len(self.samplers))),axis=1)
        for index,sampler in enumerate(self.samplers):
            rows=np.flatnonzero(selected==index)
            if not len(rows):continue
            source,count=sampler.sample(len(rows),rng)
            present[rows]=source;hidden[rows]=count
        return present,hidden
