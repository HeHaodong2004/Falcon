"""Analytic, independent integral, mark and conditional-moment checks."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp,softmax
from scipy.stats import norm,truncnorm
from eventwm_graph.l2_two_stage.gaussian_rectangle import prepare_rectangle


def parameters(mean,scale,logits=None):
    mean=np.asarray(mean,float).reshape(-1,2);scale=np.broadcast_to(np.asarray(scale,float),mean.shape).copy()
    return dict(spatial_mean=mean,spatial_scale=scale,
                spatial_logits=np.zeros(len(mean)) if logits is None else np.asarray(logits,float))


def test_mixture_log_mass_matches_analytic_rectangle_probability():
    p=parameters([[-2.,-.3],[.5,.8],[4.,5.]],[[1.,2.],[.5,.8],[2.,3.]],[2.,0.,-1.])
    robot=np.array([1.,-2.]);lo=np.array([-.5,-3.]);hi=np.array([2.,.5])
    prepared=prepare_rectangle(p,robot,lo,hi)
    a=(lo-robot-p['spatial_mean'])/p['spatial_scale'];b=(hi-robot-p['spatial_mean'])/p['spatial_scale']
    expected=np.dot(softmax(p['spatial_logits']),np.prod(norm.cdf(b)-norm.cdf(a),axis=1))
    assert abs(prepared.log_mass-math.log(expected))<2e-14


def test_extreme_tail_log_mass_remains_finite_against_independent_scaled_integral():
    for lo,hi in (([35.,-40.],[36.,-39.]),([100.,100.],[100.01,100.01]),
                  ([1.,-1.],[1.+1e-12,-1.+1e-12])):
        prepared=prepare_rectangle(parameters([[0.,0.]],1.),[0.,0.],lo,hi)
        expected=0.
        for a,b in zip(lo,hi):
            mode=min(max(0.,a),b);width=b-a
            integral=quad(lambda u:math.exp(-.5*((a+width*u)-mode)*((a+width*u)+mode)),
                          0.,1.,epsabs=1e-13,epsrel=1e-13)[0]
            expected+=math.log(width)-.5*mode*mode-.5*math.log(2*math.pi)+math.log(integral)
        assert math.isfinite(prepared.log_mass)
        assert abs(prepared.log_mass-expected)<3e-11
        _,xy=prepared.sample(1000,np.random.default_rng(19))
        assert np.all(xy>=lo) and np.all(xy<=hi)


def test_narrow_and_standardized_underflow_intervals_keep_positive_log_mass():
    for scale,width in ((1.,1e-300),(1e200,1e-200)):
        prepared=prepare_rectangle(parameters([[0.,0.]],scale),[0.,0.],[0.,0.],[width,width])
        expected=2*(math.log(width)-math.log(scale)-.5*math.log(2*math.pi))
        assert abs(prepared.log_mass-expected)<1e-12
        assert np.isfinite(prepared.axis_log_mass).all()
        _,xy=prepared.sample(20000,np.random.default_rng(3));unit=xy/width
        assert np.all((unit>=0)&(unit<=1))
        np.testing.assert_allclose(unit.mean(0),.5,atol=.015,rtol=0)
        np.testing.assert_allclose(unit.var(0),1/12,atol=.004,rtol=0)


def test_marks_and_conditional_coordinate_moments_match_reference_law():
    p=parameters([[-1.,-.3],[.5,.8],[3.,2.]],[[1.,2.],[.5,.8],[2.,3.]],[1.,0.,-.5])
    lo=np.array([-.5,-1.]);hi=np.array([2.,1.5]);prepared=prepare_rectangle(p,[0.,0.],lo,hi)
    labels,xy=prepared.sample(60000,np.random.default_rng(729))
    expected=np.exp(prepared.conditional_log_weights)
    for i in range(3):
        selected=labels==i;n=int(selected.sum());observed=n/len(labels)
        assert abs(observed-expected[i])<6*math.sqrt(expected[i]*(1-expected[i])/len(labels))+2/len(labels)
        assert n>1000
        for axis in range(2):
            mean,var,_,kurtosis=truncnorm.stats(prepared.standardized_lower[i,axis],prepared.standardized_upper[i,axis],
                loc=prepared.means[i,axis],scale=prepared.scales[i,axis],moments='mvsk')
            assert abs(xy[selected,axis].mean()-mean)<6*math.sqrt(var/n)+1e-10
            variance_se=math.sqrt(((kurtosis+3)*var*var-(n-3)/(n-1)*var*var)/n)
            assert abs(xy[selected,axis].var(ddof=1)-var)<6*variance_se+1e-10


def test_no_log_component_is_removed_when_ordinary_weight_underflows():
    class ForcedGumbel(np.random.Generator):
        def gumbel(self,loc=0.,scale=1.,size=None):
            answer=np.zeros(size);answer[:,0]=20000.;return answer
    prepared=prepare_rectangle(parameters([[0.,0.],[0.,0.]],1.,[-10000.,0.]),[0.,0.],[-1.,-1.],[1.,1.])
    assert np.isfinite(prepared.conditional_log_weights).all()
    assert math.exp(prepared.conditional_log_weights[0])==0.
    component,xy=prepared.sample(32,ForcedGumbel(np.random.PCG64(3)))
    np.testing.assert_array_equal(component,0)
    assert np.isfinite(xy).all()


def test_exact_zero_weight_batch_padding_preserves_active_component_indices():
    p=parameters([[8.,9.],[0.,0.],[-7.,4.],[1.,.5]],1.,[-np.inf,2.,-np.inf,-3.])
    prepared=prepare_rectangle(p,[0.,0.],[-1.,-1.],[1.,1.])
    unpadded={key:value[[1,3]].copy() for key,value in p.items()}
    direct=prepare_rectangle(unpadded,[0.,0.],[-1.,-1.],[1.,1.])
    assert prepared.log_mass==direct.log_mass
    assert np.isneginf(prepared.conditional_log_weights[[0,2]]).all()
    labels,xy=prepared.sample(5000,np.random.default_rng(13))
    expected,truth=direct.sample(5000,np.random.default_rng(13))
    np.testing.assert_array_equal(labels,np.array([1,3])[expected])
    np.testing.assert_array_equal(xy,truth)
    for logits in ((-np.inf,-np.inf),(np.nan,0.),(np.inf,0.)):
        with pytest.raises(ValueError):
            prepare_rectangle(parameters([[0.,0.],[0.,0.]],1.,logits),[0.,0.],[-1.,-1.],[1.,1.])


def test_far_tail_sampler_has_correct_excess_scale():
    prepared=prepare_rectangle(parameters([[0.,0.]],1.),[0.,0.],[1000.,1000.],[1000.01,1000.01])
    _,xy=prepared.sample(12000,np.random.default_rng(35))
    excess=(xy-1000.)*1000.
    expected=1-10/math.expm1(10)
    np.testing.assert_allclose(excess.mean(0),expected,atol=.06,rtol=0)
    assert prepared.log_mass<-1e5


def test_immutability_validation_and_absolute_location():
    p=parameters([[0.,0.]],1.);robot=np.array([2.,-3.]);lo=robot-.1;hi=robot+.2
    prepared=prepare_rectangle(p,robot,lo,hi);p['spatial_mean'][0]=999.;robot[:]=999.;lo[:]=999.
    for field in ('lower','upper','means','scales','axis_log_mass','conditional_log_weights'):
        with pytest.raises(ValueError):getattr(prepared,field).setflags(write=True)
    _,xy=prepared.sample(50,np.random.default_rng(4))
    assert np.all(xy>=[1.9,-3.1]) and np.all(xy<=[2.2,-2.8])
    for invalid in (np.nan,np.inf,-np.inf):
        bad=parameters([[invalid,0.]],1.)
        with pytest.raises(ValueError):prepare_rectangle(bad,[0.,0.],[-1.,-1.],[1.,1.])
    for lower,upper in (([0.,0.],[0.,1.]),([1.,0.],[0.,1.])):
        with pytest.raises(ValueError):prepare_rectangle(parameters([[0.,0.]],1.),[0.,0.],lower,upper)
    with pytest.raises(ValueError):prepare_rectangle(parameters([[0.,0.]],0.),[0.,0.],[-1.,-1.],[1.,1.])
    assert prepared.sample(0,np.random.default_rng(1))[1].shape==(0,2)
    with pytest.raises(ValueError):prepared.sample(1.5,np.random.default_rng(1))
