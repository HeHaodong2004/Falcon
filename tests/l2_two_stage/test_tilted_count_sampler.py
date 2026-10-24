"""Independent enumeration, full-tail, shifted-Palm and allocation checks."""
import itertools
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import expit,logsumexp
from scipy.stats import poisson

from eventwm_graph.l2_two_stage.fast_count import PreparedCountSampler
from eventwm_graph.l2_two_stage.tilted_count_sampler import PreparedTiltedCountSampler


def mass(logits,age,rate,n,certain=None):
    p=expit(logits);forced=np.asarray(age)==0
    if certain is not None:forced|=certain
    p=np.where(forced,1.,p)
    return sum(np.prod(np.where(bits,p,1-p))*poisson.pmf(n-sum(bits),rate)
               for bits in itertools.product((0,1),repeat=len(p)))


def tail(logits,age,rate,lower,certain=None):
    p=expit(logits);forced=np.asarray(age)==0
    if certain is not None:forced|=certain
    p=np.where(forced,1.,p)
    return sum(np.prod(np.where(bits,p,1-p))*poisson.sf(lower-sum(bits)-1,rate)
               for bits in itertools.product((0,1),repeat=len(p)))


def normalizer(logits,age,rate,minimum,weights,shift,certain=None):
    alpha=np.exp(weights-logsumexp(weights));cutoff=len(weights)-1
    return sum(alpha[n+shift]*mass(logits,age,rate,n,certain)
               for n in range(minimum,max(minimum,cutoff-shift)))+alpha[-1]*tail(
                   logits,age,rate,max(minimum,cutoff-shift),certain)


@pytest.mark.parametrize('minimum',[0,1,3,6])
@pytest.mark.parametrize('shift',[0,1,3,7])
def test_exact_class_and_global_normalizers_by_independent_enumeration(minimum,shift):
    logits=np.array([-.8,.3]);age=np.array([1.,0.]);rate=.7
    weights=np.array([-.5,.3,-1.1,.8,-.2])
    sampler=PreparedTiltedCountSampler(logits,age,rate,minimum,weights,shift=shift)
    expected=normalizer(logits,age,rate,minimum,weights,shift)
    np.testing.assert_allclose(np.exp(sampler.log_normalizer),expected,rtol=3e-14,atol=1e-16)
    alpha=np.exp(weights-logsumexp(weights));classes=[]
    for count in sampler.class_totals:
        if count<0:classes.append(alpha[-1]*tail(logits,age,rate,sampler.overflow_minimum))
        else:classes.append(alpha[min(count+shift,len(alpha)-1)]*mass(logits,age,rate,int(count)))
    np.testing.assert_allclose(np.exp(sampler.class_log_probabilities),np.array(classes)/expected,
                               rtol=3e-14,atol=1e-16)


@pytest.mark.parametrize('minimum,shift',[(0,0),(2,1)])
def test_joint_allocation_probabilities_match_analytic_tilt(minimum,shift):
    logits=np.array([-.8,.3]);age=np.ones(2);rate=1.3
    weights=np.array([-.7,.1,.6,-.5,.3,1.2,-.1,.2])
    sampler=PreparedTiltedCountSampler(logits,age,rate,minimum,weights,shift=shift)
    present,hidden=sampler.sample(160000,np.random.default_rng(731+shift))
    probability=expit(logits);alpha=np.exp(weights-logsumexp(weights))
    assert np.all(present.sum(1)+hidden>=minimum)
    for bits in itertools.product((0,1),repeat=2):
        source_mass=np.prod(np.where(bits,probability,1-probability))
        match=np.all(present==bits,axis=1)
        for u in range(11):
            count=sum(bits)+u
            expected=(source_mass*poisson.pmf(u,rate)*alpha[min(count+shift,len(alpha)-1)]
                      /np.exp(sampler.log_normalizer) if count>=minimum else 0.)
            observed=np.mean(match&(hidden==u))
            assert abs(expected-observed)<=6*np.sqrt(expected*(1-expected)/len(hidden))+2/len(hidden)


@pytest.mark.parametrize('minimum,total',[(0,None),(1,None),(6,None),(1,3),(2,12)])
def test_constant_tilt_delegates_old_output_and_rng_exactly(minimum,total):
    logits=np.array([-.9,.2,1.5,-2.]);age=np.array([.2,0.,2.,.8]);certain=[False,False,False,True]
    old=PreparedCountSampler(logits,age,.7,minimum,certain=certain,total=total)
    new=PreparedTiltedCountSampler(logits,age,.7,minimum,np.zeros(8),certain=certain,total=total)
    r0=np.random.default_rng(291);r1=np.random.default_rng(291)
    a,b=old.sample(137,r0);c,d=new.sample(137,r1)
    assert a.tobytes()==c.tobytes() and b.tobytes()==d.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state
    assert new.log_normalizer==old.log_normalizer-np.log(8.)


@pytest.mark.parametrize('total',[0,1,4,12,256])
def test_fixed_remainder_total_cancels_arbitrary_shifted_tilt(total):
    weights=np.array([-.7,.1,.6,-.5,.3,1.2,-.1,.2])
    new=PreparedTiltedCountSampler([],[],.7,0,weights,total=total,shift=1)
    old=PreparedCountSampler([],[],.7,0,total=total)
    r0=np.random.default_rng(4);r1=np.random.default_rng(4)
    a,b=old.sample(71,r0);c,d=new.sample(71,r1)
    assert a.tobytes()==c.tobytes() and b.tobytes()==d.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state
    expected=poisson.logpmf(total,.7)+(weights-logsumexp(weights))[min(total+1,7)]
    np.testing.assert_allclose(new.log_normalizer,expected,rtol=0,atol=2e-14)


def test_lower_floor_and_palm_shift_reuse_is_fresh_constructor_bitwise():
    values=dict(logits=[-.8,.3,1.1],age=[1.,0.,.9],rate=.7,minimum=5,
                tilt_logits=[-.7,.1,.6,-.5,.3,1.2,-.1,.2])
    original=PreparedTiltedCountSampler(**values)
    for minimum,shift,total in [(4,1,None),(0,1,None),(4,1,5),(4,1,12),(0,8,None)]:
        reused=original.recondition(minimum,shift,total)
        fresh=PreparedTiltedCountSampler(**dict(values,minimum=minimum,shift=shift,total=total))
        assert reused.log_normalizer==fresh.log_normalizer
        assert reused.class_totals.tobytes()==fresh.class_totals.tobytes()
        assert reused.class_log_probabilities.tobytes()==fresh.class_log_probabilities.tobytes()
        r0=np.random.default_rng(83);r1=np.random.default_rng(83)
        a,b=reused.sample(211,r0);c,d=fresh.sample(211,r1)
        assert a.tobytes()==c.tobytes() and b.tobytes()==d.tobytes()
        assert r0.bit_generator.state==r1.bit_generator.state


def test_one_sf_and_one_pmf_recursion_not_seven_full_preparations():
    import eventwm_graph.l2_two_stage.tilted_count_sampler as module
    original=module.PreparedCountSampler.__init__;calls=[]
    def tracked(self,*args,**kwargs):
        calls.append((args,kwargs));return original(self,*args,**kwargs)
    with patch.object(module.PreparedCountSampler,'__init__',tracked):
        prepared=PreparedTiltedCountSampler([-.8,.3],[1.,1.],.7,0,[-.7,.1,.6,-.5,.3,1.2,-.1,.2])
        assert len(calls)<=2
        assert prepared.base_sf is not None and prepared.base_pmf is not None
        before=len(calls)
        prepared.recondition(0,1)
        assert len(calls)==before


def test_overflow_remains_unbounded_and_rare_finite_log_weights_are_kept():
    class TailOnly(np.random.Generator):
        def gumbel(self,loc=0.,scale=1.,size=None):
            result=np.zeros(size);result[:,-1]=2000.;return result
    sampler=PreparedTiltedCountSampler([],[],100.,0,[0.,-900.,-900.,-900.])
    assert np.isfinite(sampler.class_log_probabilities).all()
    assert np.exp(sampler.log_tilt_weights[-1])==0.
    _,counts=sampler.sample(64,TailOnly(np.random.PCG64(71)))
    assert counts.max()>100 and np.all(counts>=3)


def test_analytic_pmf_not_tail_subtraction_preserves_tiny_low_count_mass():
    sampler=PreparedTiltedCountSampler([],[],1000.,0,[0.,1.,-.5,.3])
    assert sampler.class_totals.tolist()==[0,1,2,-1]
    assert np.isfinite(sampler.class_log_probabilities).all()
    assert sampler.base_pmf.suffix[0,0]==-1000.


def test_logit_common_offset_is_stable_and_all_arrays_are_immutable():
    sampler=PreparedTiltedCountSampler([],[],.7,0,np.full(8,1e308))
    np.testing.assert_allclose(sampler.log_tilt_weights,-np.log(8.),rtol=0,atol=0)
    for name in ['logits','age','certain','tilt_logits','log_tilt_weights','class_totals','class_log_probabilities']:
        with pytest.raises(ValueError):getattr(sampler,name).setflags(write=True)


@pytest.mark.parametrize('field,value',[('rate',0.),('rate',-1.),('rate',np.inf),('rate',1+0j),
    ('minimum',True),('minimum',1.5),('minimum',-1),('shift',True),('shift',-.1),
    ('tilt_logits',[]),('tilt_logits',[0.,np.nan]),('tilt_logits',[0.,-np.inf]),
    ('tilt_logits',[0.,np.inf]),('tilt_logits',[0+0j]),('age',[-1.]),('age',[np.nan])])
def test_invalid_parameters_fail_closed(field,value):
    values=dict(logits=[.1],age=[1.],rate=.7,minimum=0,tilt_logits=[0.,1.])
    values[field]=value
    with pytest.raises((ValueError,RuntimeError)):PreparedTiltedCountSampler(**values)


def test_positive_unrepresentable_tail_is_failure_not_removed_class():
    with pytest.raises(RuntimeError):PreparedTiltedCountSampler([],[],1e-200,0,[0.,.1,.2,.3])


def test_strictly_positive_age_aliases_keep_single_person_support():
    sampler=PreparedTiltedCountSampler([.3,.2],[1e-9,2e-9],.7,1,[0.,2.,.1])
    assert sampler.forced_count==0 and 1 in sampler.class_totals
    present,hidden=sampler.sample(2000,np.random.default_rng(113))
    assert np.any(present.sum(1)+hidden==1)
