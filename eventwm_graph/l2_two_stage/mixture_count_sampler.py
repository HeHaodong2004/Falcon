"""Prepared finite-mixture count conditioning without truncating hidden counts.

The component is a latent *rate*, not a person identity. For the floor event C,
its posterior is proportional to pi[j] P_j(C). For a fixed total it is instead
proportional to pi[j] P_j(N=total). Each selected component uses the existing
unbounded Bernoulli/Poisson allocation sampler. Preparation is parameter
dependent and must remain inside the complete-call inference timer.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson

from .fast_count import PreparedCountSampler,_prepare_prefix
from .types import immutable_array

VERSION = 'prepared_positive_rate_mixture_posterior_log_gumbel_1'


def _real_vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in 'fiu':
        raise ValueError(name+' must be a real one-dimensional numeric array')
    return np.asarray(array, np.float64)


def _prepared_component(rate,minimum,total,table,probabilities,prefixes):
    result=object.__new__(PreparedCountSampler)
    maximum=minimum if total is None else total
    for name,value in (('rate',float(rate)),('minimum',minimum),('total',total),
        ('log_normalizer',float(table[0,maximum])),('suffix',immutable_array(table)),
        ('probabilities',immutable_array(probabilities)),('prefixes',prefixes)):
        object.__setattr__(result,name,value)
    return result


def _prepare_pair(logits,age,rates,minimum,certain,total):
    """Same two old component tables; batch only independent scalar operations.

    The zero floor column remains exactly zero. Every other logaddexp has the
    same two operands and operation order as count.py. The old prefix factory
    and old component sampler remain authoritative, including tail fallbacks.
    """
    logits=np.asarray(logits,np.float64);age=np.asarray(age,np.float64)
    if logits.ndim!=1 or age.shape!=logits.shape or not np.isfinite(logits).all():
        raise ValueError('finite one-dimensional logits and matching ages required')
    if not np.isfinite(age).all() or np.any(age<0):
        raise ValueError('finite nonnegative age required')
    if int(minimum)!=minimum or minimum<0:raise ValueError('nonnegative integer minimum required')
    minimum=int(minimum)
    forced=age==0
    if certain is not None:
        certain=np.asarray(certain,bool)
        if certain.shape!=age.shape:raise ValueError('certain source mask mismatch')
        forced|=certain
    yes=np.where(forced,0.,-np.logaddexp(0.,-logits))
    no=np.where(forced,-math.inf,-np.logaddexp(0.,logits))
    if total is not None:
        if int(total)!=total or total<minimum:
            raise ValueError('fixed integer total must be at least the minimum')
        total=int(total)
    maximum=minimum if total is None else total
    table=np.empty((len(rates),len(logits)+1,maximum+1),np.float64)
    shared=None
    if total is None:
        if minimum:
            upper=max(max(minimum,int(math.ceil(rate+8*math.sqrt(rate+1)))) for rate in rates)
            if upper>65536:
                return tuple(PreparedCountSampler(logits,age,rate,minimum,certain=forced)
                             for rate in rates)
            # The extra SF value at -1 is exactly the prefix factory's index0.
            sf=poisson.logsf(np.arange(-1,upper+1)[None,:],rates[:,None])
            pmf=poisson.logpmf(np.arange(upper+1)[None,:],rates[:,None])
            shared=(sf,pmf)
            table[:,-1,1:]=sf[:,1:minimum+1]
        table[:,-1,0]=0.
    else:table[:,-1]=poisson.logpmf(np.arange(total+1)[None,:],rates[:,None])
    for i in range(len(logits)-1,-1,-1):
        table[:,i,0]=0. if total is None else no[i]+table[:,i+1,0]
        table[:,i,1:]=np.logaddexp(no[i]+table[:,i+1,1:],yes[i]+table[:,i+1,:-1])
    if not np.isfinite(table[:,0,maximum]).all():
        raise RuntimeError('population normalizer underflow; no fabricated worlds')
    probabilities=np.zeros((len(rates),len(logits),maximum+1),np.float64)
    needed=np.arange(maximum+1)
    for i in range(len(logits)):
        supported=np.isfinite(table[:,i])
        with np.errstate(invalid='ignore'):
            logp=yes[i]+table[:,i+1][:,np.maximum(needed-1,0)]-table[:,i]
        probabilities[:,i][supported]=np.exp(logp[supported]).clip(0,1)
        if total is not None:probabilities[:,i,0]=0.
    components=[]
    for j,rate in enumerate(rates):
        prefixes=() if total is not None else tuple(
            _prepare_prefix(float(rate),lower,(shared[0][j],shared[1][j]))
            for lower in range(1,minimum+1))
        components.append(_prepared_component(rate,minimum,total,table[j],probabilities[j],prefixes))
    return tuple(components)


@dataclass(frozen=True, init=False)
class PreparedMixtureCountSampler:
    """Immutable count allocation conditional on a floor or one fixed total.

    ``rates`` are finite and strictly positive. Exact ``-inf`` count logits
    denote zero-mass padding and preserve their component indices. Every finite
    component stays in log-domain categorical sampling; no probability cutoff
    or exponentiation-to-zero discards an active rate. A one-component model
    delegates directly to PreparedCountSampler, including its RNG coupling.

    Multi-component sampling groups rows by their independently sampled latent
    rate, then scatters allocations into the original world order. Its random
    number coupling differs from the one-component sampler; its conditional
    probability law is the specified mixture.
    """
    rates: np.ndarray
    log_count_probabilities: np.ndarray
    log_component_normalizers: np.ndarray
    component_log_probabilities: np.ndarray
    components: tuple
    active_components: np.ndarray
    log_normalizer: float
    source_count: int
    minimum: int
    total: object
    version: str

    def __init__(self, logits, age, rates, count_logits, minimum,
                 certain=None, total=None):
        rates = _real_vector(rates, 'rates')
        count_logits = _real_vector(count_logits, 'count_logits')
        if (not len(rates) or count_logits.shape != rates.shape
                or not np.isfinite(rates).all() or np.any(rates <= 0)):
            raise ValueError('nonempty matching finite positive rates required')
        if (np.isnan(count_logits).any() or np.isposinf(count_logits).any()
                or not np.isfinite(count_logits).any()):
            raise ValueError('count logits require finite or -inf entries and finite support')
        active = np.flatnonzero(np.isfinite(count_logits))
        with np.errstate(over='ignore', invalid='ignore'):
            log_prior = count_logits-logsumexp(count_logits)
        if not np.isfinite(log_prior[active]).all():
            raise RuntimeError('finite count component log weight overflow')

        components = [None]*len(rates)
        prepared=(_prepare_pair(logits,age,rates[active],minimum,certain,total)
                  if len(active)==2 else tuple(PreparedCountSampler(logits,age,rates[j],
                      minimum,certain=certain,total=total) for j in active))
        for component,sampler in zip(active,prepared):
            if not np.isfinite(sampler.log_normalizer):
                raise RuntimeError('active count component normalizer unsupported')
            components[component] = sampler
        self._install(rates,log_prior,components,active)

    def _install(self,rates,log_prior,components,active):
        log_constraints=np.full(len(rates),-np.inf,np.float64)
        for component in active:log_constraints[component]=components[component].log_normalizer
        with np.errstate(over='ignore', invalid='ignore'):
            log_joint = log_prior+log_constraints
            log_normalizer = float(logsumexp(log_joint))
            log_posterior = log_joint-log_normalizer
        if (not np.isfinite(log_normalizer)
                or not np.isfinite(log_posterior[active]).all()):
            raise RuntimeError('count mixture normalizer unsupported; no component discarded')
        first = components[int(active[0])]
        for name, value in (
                ('rates', immutable_array(rates)),
                ('log_count_probabilities', immutable_array(log_prior)),
                ('log_component_normalizers', immutable_array(log_constraints)),
                ('component_log_probabilities', immutable_array(log_posterior)),
                ('components', tuple(components)),
                ('active_components', immutable_array(active)),
                ('log_normalizer', log_normalizer),
                ('source_count', len(first.probabilities)),
                ('minimum', first.minimum), ('total', first.total),
                ('version', VERSION)):
            object.__setattr__(self, name, value)

    def recondition(self,minimum,count_logits):
        """Reuse a floor table for a lower floor and new component log weights.

        Suffix column k is the same tail probability regardless of the maximum
        prepared k. Rates/source probabilities are immutable. Fixed-total PMF
        tables and unprepared zero-weight components cannot use this operation.
        No random draws occur during preparation or reconditioning.
        """
        if self.total is not None:
            raise ValueError('fixed-total PMF tables cannot be reused as floor tables')
        if int(minimum)!=minimum or minimum<0 or minimum>self.minimum:
            raise ValueError('reconditioning requires an integer lower or equal floor')
        minimum=int(minimum)
        weights=_real_vector(count_logits,'count_logits')
        if (weights.shape!=self.rates.shape or np.isnan(weights).any() or np.isposinf(weights).any()
                or not np.isfinite(weights).any()):
            raise ValueError('matching finite or -inf component logits required')
        active=np.flatnonzero(np.isfinite(weights))
        if any(self.components[j] is None for j in active):
            raise ValueError('cannot activate an unprepared zero-weight component')
        with np.errstate(over='ignore',invalid='ignore'):log_prior=weights-logsumexp(weights)
        if not np.isfinite(log_prior[active]).all():raise RuntimeError('finite count component log weight overflow')
        components=[None]*len(self.rates)
        for j in active:
            previous=self.components[j]
            components[j]=_prepared_component(previous.rate,minimum,None,
                previous.suffix[:,:minimum+1],previous.probabilities[:,:minimum+1],previous.prefixes[:minimum])
        result=object.__new__(type(self));result._install(self.rates,log_prior,components,active)
        return result

    def sample(self, worlds, rng):
        if (not isinstance(worlds, (int, np.integer))
                or isinstance(worlds, (bool, np.bool_)) or worlds < 0
                or not isinstance(rng, np.random.Generator)):
            raise ValueError('nonnegative integer worlds and numpy Generator required')
        worlds = int(worlds)
        # This path consumes precisely the old sampler's random draws. It also
        # avoids irrelevant component RNG when only one nonzero rate remains.
        if len(self.active_components) == 1:
            return self.components[int(self.active_components[0])].sample(worlds, rng)
        present = np.zeros((worlds, self.source_count), bool)
        hidden = np.empty(worlds, np.int64)
        if not worlds:
            return present, hidden
        selected = self.active_components[np.argmax(
            self.component_log_probabilities[self.active_components][None, :]
            +rng.gumbel(size=(worlds, len(self.active_components))), axis=1)]
        for component in self.active_components:
            rows = np.flatnonzero(selected == component)
            if not len(rows):
                continue
            component_present, component_hidden = self.components[int(component)].sample(
                len(rows), rng)
            present[rows] = component_present
            hidden[rows] = component_hidden
        return present, hidden
