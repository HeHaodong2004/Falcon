import unittest
from dataclasses import replace
import torch
from eventforecastwm.model_conditional_scale import ConditionalScaleWorldModel,component_log_densities
from eventforecastwm.model_scale_mixture import trajectory_log_probability
from eventforecastwm.composable_sampling import sample_transition
from test_forecast_model import legal


class TestConditionalScale(unittest.TestCase):
    def test_joint_density_gradients_and_persistence(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        m=ConditionalScaleWorldModel(kind='stochastic');b=m.encode_history(legal())
        f=m.imagine(b,horizon_s=1.);truth=f.positions[:,0].detach()+.2;mask=torch.ones(1,4,1,dtype=torch.bool)
        terms,count=component_log_densities(f,truth,mask)
        expected=(terms+f.weights.log()[...,None]+f.scale_probabilities.log()).flatten(1).logsumexp(1)
        actual,_=trajectory_log_probability(f,truth,mask)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-5))
        (-actual/count).mean().backward();self.assertGreater(float(m.scale_context.weight.grad.norm()),0.)
        first=m.imagine(b,horizon_s=.5);second=m.imagine(first.next_belief,horizon_s=.5)
        g=torch.Generator().manual_seed(3);h=torch.Generator().manual_seed(3)
        whole,_=sample_transition(f,6,g);left,state=sample_transition(first,6,h);right,_=sample_transition(second,6,h,state)
        self.assertTrue(torch.equal(whole,torch.cat([left,right],2)))
        corrected=m.observe(first.next_belief,dict(positions=first.positions[:,0,-1].detach()+.3,
            observed=torch.ones(1,1,dtype=torch.bool),acquisition_times=torch.full((1,1),.5),timestamp_s=.5))
        self.assertTrue(torch.equal(corrected.legal_context,b.legal_context))
        self.assertFalse(torch.equal(corrected.scale_probabilities,b.scale_probabilities))
        m.imagine(corrected,horizon_s=.5)


if __name__=='__main__':unittest.main()
