import unittest
import torch
from eventforecastwm.model_damped import DampedForecastWorldModel,process,trajectory_log_probability,energy_score
from test_forecast_model import legal


class TestDampedResidual(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(4)

    def test_exact_semigroup_and_stationary_velocity_variance(self):
        rate=torch.tensor(.8);scale=torch.tensor(1.3)
        f,q=process(rate,scale,.25);full,Q=process(rate,scale,1.)
        cov=torch.zeros(2,2)
        for _ in range(4):cov=f@cov@f.T+q
        self.assertTrue(torch.allclose(full,torch.linalg.matrix_power(f,4),atol=1e-7))
        self.assertTrue(torch.allclose(Q,cov,atol=1e-7))
        _,limit=process(rate,scale,100.)
        self.assertAlmostEqual(float(limit[1,1]),float(scale.square()*.25/(2*rate)),places=6)

    def test_nll_matches_full_temporal_gaussian(self):
        model=DampedForecastWorldModel(kind='cv');forecast=model.imagine(model.encode_history(legal()),horizon_s=.75)
        truth=forecast.positions[:,0]+.1;mask=torch.ones(1,3,1,dtype=torch.bool)
        actual,count=trajectory_log_probability(forecast,truth,mask)
        f,q=process(forecast.decay_rate,forecast.acceleration_scale,.25);cov=forecast.initial_covariance[0,0,0];states=[]
        for _ in range(3):cov=f@cov@f.T+q;states.append(cov)
        full=torch.zeros(3,3)
        for i in range(3):
            for j in range(i,3):
                full[i,j]=(states[i]@torch.linalg.matrix_power(f,j-i).T)[0,0];full[j,i]=full[i,j]
        full+=torch.eye(3)*forecast.readout_scale.square()
        expected=2*torch.distributions.MultivariateNormal(torch.zeros(3),full).log_prob(torch.ones(3)*.1)
        self.assertTrue(torch.allclose(actual[0],expected,atol=1e-4));self.assertEqual(int(count),6)

    def test_composition_and_useful_decay_gradient(self):
        model=DampedForecastWorldModel(age_aware=True);belief=model.encode_history(legal())
        full=model.imagine(belief,horizon_s=3.);left=model.imagine(belief,horizon_s=1.);right=model.imagine(left.next_belief,horizon_s=2.)
        self.assertTrue(torch.equal(full.positions[:,:,4:],right.positions))
        self.assertTrue(torch.equal(full.next_belief.covariance,right.next_belief.covariance))
        a=torch.Generator().manual_seed(7);b=torch.Generator().manual_seed(7)
        paths,_,_=full.sample(2,a);p,_,state=left.sample(2,b);q,_,_=right.sample(2,b,state)
        self.assertTrue(torch.equal(paths,torch.cat([p,q],2)))
        mask=torch.ones(1,12,1,dtype=torch.bool);truth=full.positions[:,0].detach()+.3
        lp,count=trajectory_log_probability(full,truth,mask);es,_=energy_score(full,truth,mask)
        (es.mean()-.05*(lp/count).mean()).backward()
        self.assertGreater(float(model.raw_decay_rate.grad.abs()),0.)


if __name__=='__main__':unittest.main()
