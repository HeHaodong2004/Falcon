"""Independent probability identities for residual mixed-Poisson worlds.

These are mathematical regressions, not synthetic forecasting acceptance.
"""
import itertools
import math

import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal, poisson
import torch

from eventwm_graph.l2_two_stage import mixed_count as mixed
from eventwm_graph.l2_two_stage.population import set_nll


def test_count_floor_and_palm_remainder_match_direct_probability_sum():
    logits=np.array([-.2,.4]);age=np.array([0.,1.])
    rates=np.array([.4,2.1]);weights=np.array([.3,.7])
    parameters={mixed.RATES:rates,mixed.LOGITS:np.log(weights)}
    probabilities=np.array([1.,expit(logits[1])])

    def reference(n,minimum,prior):
        mass=normalizer=0.
        for bits in itertools.product((0,1),repeat=2):
            p=np.prod(np.where(bits,probabilities,1-probabilities))
            normalizer+=p*float(prior@poisson.sf(minimum-sum(bits)-1,rates))
            if n>=minimum:mass+=p*float(prior@poisson.pmf(n-sum(bits),rates))
        return mass/normalizer,normalizer

    for minimum in (0,2,4):
        actual=mixed.numpy_log_normalizer(logits,age,rates,np.log(weights),minimum)
        assert abs(actual-math.log(reference(max(1,minimum),minimum,weights)[1]))<1e-13
        pmf=[]
        for n in range(81):
            value=mixed.numpy_count_log_probability(parameters,logits,age,n,minimum)
            probability=math.exp(value)
            assert abs(probability-reference(n,minimum,weights)[0])<1e-13
            pmf.append(probability)
            if n>=max(1,minimum):
                # The Palm helper scores its shifted remainder count; the
                # added residual atom is represented by the external -1.
                biased=weights*rates;biased/=biased.sum()
                expected=reference(n-1,max(0,minimum-1),biased)[0]
                value=mixed.numpy_count_log_probability(parameters,logits,age,n-1,max(0,minimum-1),palm=True)
                assert abs(math.exp(value)-expected)<1e-13
        # This finite reference sum has an independently negligible tail;
        # the production count family itself is unbounded.
        assert poisson.sf(77,rates.max())<1e-80
        assert abs(sum(pmf)-1)<1e-13


def test_colocated_multiset_likelihood_and_gradients_include_multiplicity_once():
    dtype=torch.float64
    rates=torch.tensor([[.4,2.1]],dtype=dtype,requires_grad=True)
    logits=torch.tensor([[-.7,.2]],dtype=dtype,requires_grad=True)
    parameters={
        mixed.RATES:rates,mixed.LOGITS:logits,
        'rate':torch.ones(1,dtype=dtype),
        'means':torch.zeros((1,1,4,5,2),dtype=dtype),
        'scales':torch.full((1,1,4,5,2),.5,dtype=dtype),
        'presence_logits':torch.zeros((1,1),dtype=dtype),
        'mode_logits':torch.zeros((1,4),dtype=dtype),
        'spatial_logits':torch.zeros((1,4),dtype=dtype),
        'spatial_mean':torch.zeros((1,4,2),dtype=dtype),
        'spatial_scale':torch.ones((1,4,2),dtype=dtype)}
    batch=dict(origin=torch.zeros((1,1,2),dtype=dtype),velocity=torch.zeros((1,1,2),dtype=dtype),
        robotxy=torch.zeros((1,2),dtype=dtype),age=torch.ones((1,1),dtype=dtype),
        mask=torch.ones((1,1),dtype=dtype),minimum_count=torch.tensor([2]),
        set_positions=torch.tensor([[[.25,0.],[.25,0.]]],dtype=dtype),
        set_mask=torch.ones((1,2),dtype=dtype),set_valid=torch.ones(1,dtype=dtype))
    actual=set_nll(parameters,batch,condition_minimum=True,compulsory_unique_episode=True).sum()
    source_density=multivariate_normal.pdf([.25,0.],cov=np.eye(2)*.25)
    residual_density=multivariate_normal.pdf([.25,0.],cov=np.eye(2))
    prior=torch.softmax(logits,dim=-1)
    # One compulsory source can match either occurrence of the co-located
    # two-person multiset. One residual remains in each injective assignment.
    numerator=2*source_density*residual_density*(prior*rates*torch.exp(-rates)).sum()
    normalizer=(prior*(1-torch.exp(-rates))).sum()
    expected=-numerator.log()+normalizer.log()
    assert abs(actual.item()-expected.item())<1e-13
    actual_grad=torch.autograd.grad(actual,(rates,logits),retain_graph=True)
    expected_grad=torch.autograd.grad(expected,(rates,logits))
    assert all(torch.max(abs(a-b))<2e-13 for a,b in zip(actual_grad,expected_grad))


def test_floor_gradient_respects_mixture_posterior_and_optional_source():
    rates=torch.tensor([[.4,2.1]],dtype=torch.float64,requires_grad=True)
    logits=torch.tensor([[-.7,.2]],dtype=torch.float64,requires_grad=True)
    source=torch.tensor([[.4]],dtype=torch.float64,requires_grad=True)
    parameters={mixed.RATES:rates,mixed.LOGITS:logits}
    actual=mixed.torch_mixture_log_normalizer(parameters,source,torch.ones_like(source,dtype=torch.bool),torch.tensor([1])).sum()
    # P(B+U>=1)=1-P(B=0)*sum_j pi_j exp(-lambda_j).
    expected=torch.log(1-(1-torch.sigmoid(source[0,0]))*(torch.softmax(logits,-1)*torch.exp(-rates)).sum())
    assert abs(actual.item()-expected.item())<1e-14
    a=torch.autograd.grad(actual,(rates,logits,source),retain_graph=True)
    b=torch.autograd.grad(expected,(rates,logits,source))
    assert all(torch.max(abs(x-y))<2e-13 for x,y in zip(a,b))


def test_censored_current_target_is_not_evaluated_or_counted_as_density_truth():
    rates=torch.tensor([[.4,2.1]],dtype=torch.float64,requires_grad=True)
    logits=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
    parameters={mixed.RATES:rates,mixed.LOGITS:logits}
    def forbidden(*args,**kwargs):
        raise AssertionError('censored target was evaluated')
    result=mixed.mixed_set_nll(forbidden,parameters,{'set_valid':torch.zeros(1)},True)
    assert torch.equal(result,torch.zeros_like(result))
    result.sum().backward()
    assert torch.equal(rates.grad,torch.zeros_like(rates))
    assert torch.equal(logits.grad,torch.zeros_like(logits))
