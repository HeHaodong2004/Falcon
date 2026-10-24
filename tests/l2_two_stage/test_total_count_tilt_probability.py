"""Independent count-potential identities and exact observation-clock regressions.

These establish implementation mathematics, not forecasting acceptance.
"""
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from eventwm_graph.l2_two_stage import tilted_count as tilt
from eventwm_graph.l2_two_stage.conditioned import source_coordinates
from eventwm_graph.l2_two_stage.count import count_log_probability
from eventwm_graph.l2_two_stage.population import set_nll
from eventwm_graph.l2_two_stage.positive_history import PositiveSetHistory


def test_total_floor_normalizer_gradients_with_one_current_and_one_optional_source():
    rate=torch.tensor([.7],dtype=torch.float64,requires_grad=True)
    source=torch.tensor([[.2,.4]],dtype=torch.float64,requires_grad=True)
    logits=torch.tensor([[-.3,.7,.2,-.1]],dtype=torch.float64,requires_grad=True)
    p={'rate':rate,tilt.LOGITS:logits}
    actual=tilt.torch_tilt_log_normalizer(p,source,torch.tensor([[False,True]]),
        torch.tensor([2]),torch.tensor([1])).sum()
    # TotalN=1+B+U. OnlyN2 andN>=3 survive the floor. The overflow
    # term integrates the WHOLE tail, not only the pointN3.
    b=torch.sigmoid(source[0,1]);lam=rate[0];e=torch.exp(-lam)
    p2=(1-b)*lam*e+b*e
    tail=(1-b)*(1-e*(1+lam))+b*(1-e)
    a=torch.softmax(logits[0],0)
    expected=(a[2]*p2+a[3]*tail).log()
    assert abs(actual.item()-expected.item())<2e-14
    left=torch.autograd.grad(actual,(rate,source,logits),retain_graph=True)
    right=torch.autograd.grad(expected,(rate,source,logits))
    assert all(torch.max(abs(x-y))<2e-13 for x,y in zip(left,right))


def _parameters_and_batch(age,points):
    d=torch.float64
    p={'rate':torch.tensor([.7],dtype=d,requires_grad=True),
        tilt.LOGITS:torch.tensor([[-.3,.7,.2,-.1]],dtype=d,requires_grad=True),
        'means':torch.zeros((1,1,4,5,2),dtype=d),
        'scales':torch.full((1,1,4,5,2),.5,dtype=d),
        'presence_logits':torch.zeros((1,1),dtype=d),'mode_logits':torch.zeros((1,4),dtype=d),
        'spatial_logits':torch.zeros((1,4),dtype=d),'spatial_mean':torch.zeros((1,4,2),dtype=d),
        'spatial_scale':torch.ones((1,4,2),dtype=d)}
    points=torch.tensor(points,dtype=d).reshape(1,-1,2)
    batch=dict(origin=torch.zeros((1,1,2),dtype=d),velocity=torch.zeros((1,1,2),dtype=d),
        robotxy=torch.zeros((1,2),dtype=d),age=torch.tensor([[age]],dtype=d),mask=torch.ones((1,1),dtype=d),
        minimum_count=torch.tensor([2]),set_positions=points,set_mask=torch.ones(points.shape[:2],dtype=d),
        set_valid=torch.ones(1,dtype=d))
    return p,batch


@pytest.mark.parametrize('current',[False,True])
def test_set_likelihood_total_count_distinguishes_current_atoms_and_stale_compulsory_source(current):
    positions=[[.25,0.]] if current else [[.25,0.],[.25,0.]]
    p,batch=_parameters_and_batch(0. if current else 1.,positions)
    actual=set_nll(p,batch,True,True).sum()
    lam=p['rate'][0];e=torch.exp(-lam);a=torch.softmax(p[tilt.LOGITS][0],0)
    residual=multivariate_normal.pdf([.25,0.],cov=np.eye(2))
    if current:
        # One fixed actual detection is not inside the residual target set.
        # The floor2 removes U0; current1+residual1 has potentiala2.
        numerator=lam*e*residual*a[2]
        normalizer=a[2]*lam*e+a[3]*(1-e*(1+lam))
    else:
        # The stale compulsory source is ALREADY one of the two target
        # occurrences, so totalN2, notN3. Either duplicate is its assignment.
        source=multivariate_normal.pdf([.25,0.],cov=np.eye(2)*.25)
        numerator=2*source*residual*lam*e*a[2]
        normalizer=a[2]*lam*e+a[3]*(1-e*(1+lam))
    expected=-numerator.log()+normalizer.log()
    assert abs(actual.item()-expected.item())<2e-13
    left=torch.autograd.grad(actual,(p['rate'],p[tilt.LOGITS]),retain_graph=True)
    right=torch.autograd.grad(expected,(p['rate'],p[tilt.LOGITS]))
    assert all(torch.max(abs(x-y))<3e-13 for x,y in zip(left,right))


def test_positive_tiny_age_aliases_are_not_two_certain_current_people():
    for age in (1e-9,1e-8,1e-7):
        assert math.isfinite(count_log_probability([.3,.2],[age,2*age],.7,1,1))
    assert count_log_probability([.3,.2],[0.,0.],.7,1,1)==-np.inf


def test_fallback_source_recovery_uses_exact_current_status_then_nearest_clock():
    history=PositiveSetHistory('','','',2,np.array([0.,1e-9,2e-9]),np.array([0,1,2,4]),
        np.zeros((4,3)),np.ones(3,bool),np.ones(3,bool))
    features=dict(keys=('a',),timestamp=2e-9,age=np.array([0.]),origin=np.zeros((1,2)),velocity=np.zeros((1,2)))
    assert source_coordinates(features,history)[1].tolist()==[2]
    assert source_coordinates(dict(features,age=np.array([1e-9])),history)[1].tolist()==[1]
    both=dict(features,keys=('a','b'),age=np.zeros(2),origin=np.zeros((2,2)),velocity=np.zeros((2,2)))
    assert source_coordinates(both,history)[1].tolist()==[2,3]
    with pytest.raises(ValueError):source_coordinates(dict(features,last_times=np.array([1e-9])),history)
    with pytest.raises(ValueError):source_coordinates(dict(features,last_times=np.array([3e-9])),history)
