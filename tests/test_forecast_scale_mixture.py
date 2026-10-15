import unittest
from dataclasses import replace
import torch
from eventforecastwm.model_scale_mixture import ScaleMixtureWorldModel,trajectory_log_probability,occupancy,marginal_cdf
from eventforecastwm.model_damped import trajectory_log_probability as gaussian
from eventforecastwm.composable_sampling import sample_transition
from test_forecast_model import legal


class TestFiniteSceneScale(unittest.TestCase):
    def test_exact_mixture_likelihood_composition_and_update(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        m=ScaleMixtureWorldModel(kind='deterministic',scale_std=(.5,1.,2.))
        b=m.encode_history(legal());f=m.imagine(b,horizon_s=1.)
        target=f.positions[:,0].detach()+.15;mask=torch.ones(1,4,1,dtype=torch.bool)
        lp,count=trajectory_log_probability(f,target,mask);terms=[]
        for v in f.scale_values:
            component=replace(f,initial_covariance=f.initial_covariance*v,acceleration_scale=f.acceleration_scale*v.sqrt(),readout_scale=f.readout_scale*v.sqrt())
            terms.append(gaussian(component,target,mask)[0])
        expected=torch.stack(terms,1).logsumexp(1)-torch.tensor(3.).log()
        self.assertTrue(torch.allclose(lp,expected,atol=2e-5));(-lp/count).mean().backward()
        self.assertGreater(float(m.scale_logits.grad.norm()),0.)
        first=m.imagine(b,horizon_s=.5);rest=m.imagine(first.next_belief,horizon_s=.5)
        g=torch.Generator().manual_seed(8);h=torch.Generator().manual_seed(8)
        all_paths,_=sample_transition(f,6,g);left,state=sample_transition(first,6,h);right,_=sample_transition(rest,6,h,state)
        self.assertTrue(torch.equal(all_paths,torch.cat([left,right],2)))
        corrected=m.observe(first.next_belief,dict(positions=first.positions[:,0,-1].detach()+.2,
            observed=torch.ones(1,1,dtype=torch.bool),acquisition_times=torch.full((1,1),.5),timestamp_s=.5))
        self.assertTrue(torch.allclose(corrected.scale_probabilities.sum(-1),torch.ones(1,1)))
        self.assertFalse(torch.equal(corrected.scale_probabilities,first.next_belief.scale_probabilities))
        self.assertTrue(torch.isfinite(occupancy(f)).all());self.assertTrue(torch.isfinite(marginal_cdf(f,target[:,None])).all())


if __name__=='__main__':unittest.main()
