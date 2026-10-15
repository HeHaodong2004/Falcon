from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from eventforecastwm.benchmark import load_model
from eventforecastwm.distribution import trajectory_log_probability
from eventforecastwm.model_agent_scale import AgentScaleWorldModel
from eventforecastwm.observations import protect_observe
from eventforecastwm.readouts import tracked_occupancy,marginal_cdf
from eventforecastwm.performance_v2.regime import RegimeWorldModel,branches
from eventforecastwm.performance_v2.stopped_control import (
    stopped_forecast,branch_density,joint_log_probability,mixed_occupancy)
from test_forecast_model import legal


class TestRegime(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(241)
        parent=AgentScaleWorldModel(mean_variant='posterior_v1',age_aware=True)
        self.model=protect_observe(RegimeWorldModel(parent.recipe)).eval()
        self.model.physical.load_state_dict(parent.state_dict())
        self.model.stop_head.weight.data.zero_();self.model.stop_head.bias.data.zero_()

    def test_offline_density_occupancy_and_distribution_mean_parity(self):
        model=self.model;belief=model.encode_history(legal());f=model.imagine(belief,horizon_s=3.)
        parent=model.physical.imagine(belief.moving,horizon_s=3.);stopped=stopped_forecast(parent,belief.moving)
        probability=torch.tensor([[[.75,.25]]]);truth=parent.positions[:,0].detach()+.2
        mask=torch.ones_like(truth[...,0],dtype=torch.bool)
        d,c=branch_density(parent,stopped,truth,mask)
        lp,count=trajectory_log_probability(f,truth,mask)
        self.assertTrue(torch.equal(count,c.sum(-1)))
        self.assertTrue(torch.allclose(lp,joint_log_probability(d,c,parent.weights,probability),atol=1e-6))
        self.assertTrue(torch.allclose(tracked_occupancy(f),mixed_occupancy(parent,stopped,probability),atol=2e-7))
        self.assertTrue(torch.equal(f.positions,.75*parent.positions+.25*stopped.positions))
        locations=torch.linspace(-8,8,33)[None,None,None,None,None]
        expected=.75*marginal_cdf(parent,locations)+.25*marginal_cdf(stopped,locations)
        self.assertTrue(torch.allclose(marginal_cdf(f,locations),expected,atol=1e-7))
        poisoned=replace(legal(),query=torch.full_like(legal().query,float('nan')))
        other=model.imagine(model.encode_history(poisoned),candidate_query=torch.full((1,7),float('nan')),horizon_s=3.)
        self.assertTrue(torch.equal(f.positions,other.positions));self.assertTrue(torch.equal(f.branch_probabilities,other.branch_probabilities))

    def check_composition(self,belief):
        model=self.model;whole=model.imagine(belief,horizon_s=3.);left=model.imagine(belief,horizon_s=1.)
        right=model.imagine(left.next_belief,horizon_s=2.)
        for field in ('positions','velocities','moving_positions','stopped_positions','covariance'):
            self.assertTrue(torch.equal(getattr(whole,field)[:,:,4:],getattr(right,field)),field)
        g1=torch.Generator().manual_seed(713);g2=torch.Generator().manual_seed(713)
        expected,_,_=whole.sample(8,g1);a,_,state=left.sample(8,g2);b,_,_=right.sample(8,g2,state)
        self.assertTrue(torch.equal(expected,torch.cat([a,b],2)))
        self.assertTrue(torch.isfinite(whole.position_variance).all());self.assertTrue((whole.position_variance>0).all())

    def test_composition_and_persistent_regimes_before_and_after_correction(self):
        belief=self.model.encode_history(legal());self.check_composition(belief)
        future=self.model.imagine(belief,horizon_s=1.)
        observation=dict(positions=future.moving_positions[:,0,-1]+.2,observed=torch.ones(1,1,dtype=torch.bool),
            acquisition_times=torch.ones(1,1),timestamp_s=1.)
        updated=self.model.observe(future.next_belief,observation);self.check_composition(updated)
        self.assertFalse(torch.equal(updated.branch_probabilities,belief.branch_probabilities))

    def test_bayesian_branch_variance_and_scene_posterior(self):
        data=legal();data=replace(data,humans=data.humans.repeat(1,2,1,1),human_padding=data.human_padding.repeat(1,2))
        belief=self.model.imagine(self.model.encode_history(data),horizon_s=1.).next_belief
        target=belief.anchor[:,0].clone();target[:,1]+=torch.tensor([.3,-.2])
        observation=dict(positions=target,observed=torch.ones(1,2,dtype=torch.bool),
            acquisition_times=torch.ones(1,2),timestamp_s=1.)
        variance=belief.moving.physical.covariance[...,0,0]+self.model.physical.log_readout.exp().square()
        means=torch.stack([belief.moving.core.positions,belief.anchor+belief.stopped_residual_mean[...,0,:]],-2)
        variances=variance[...,None,None]*belief.moving.scale_values
        error=(target[:,None,:,None]-means).square().sum(-1)[...,None]
        likelihood=torch.exp(-.5*error/variances)/(2*math.pi*variances)
        rho=torch.stack([belief.moving.agent_scale_probabilities,belief.stopped_scale_probabilities],-2)
        joint=likelihood*rho*belief.branch_probabilities[...,None]
        evidence=joint.sum((-1,-2));expected_branch=joint.sum(-1)/evidence[...,None]
        expected_scene=belief.core.scenario_weights*evidence.prod(-1);expected_scene/=expected_scene.sum(-1,keepdim=True)
        updated=self.model.observe(belief,observation)
        self.assertTrue(torch.allclose(updated.branch_probabilities,expected_branch,atol=1e-6))
        self.assertTrue(torch.allclose(updated.core.scenario_weights,expected_scene,atol=1e-6))
        self.assertTrue(torch.allclose(updated.stopped_scale_probabilities,joint[...,1,:]/joint[...,1,:].sum(-1,keepdim=True),atol=1e-6))

    def test_missing_arrivals_and_future_rejection(self):
        belief=self.model.imagine(self.model.encode_history(legal()),horizon_s=1.).next_belief
        empty=dict(positions=torch.full((1,1,2),float('nan')),observed=torch.zeros(1,1,dtype=torch.bool),
                   acquisition_times=torch.full((1,1),float('nan')),timestamp_s=1.)
        updated=self.model.observe(belief,empty)
        self.assertTrue(torch.allclose(updated.branch_probabilities,belief.branch_probabilities,atol=1e-7))
        self.assertTrue(torch.isfinite(self.model.imagine(updated,horizon_s=1.).positions).all())
        with self.assertRaises(ValueError):self.model.observe(belief,dict(empty,positions=torch.zeros(1,1,2),
            observed=torch.ones(1,1,dtype=torch.bool),acquisition_times=torch.full((1,1),1.1)))

    def test_vectorized_conditional_ranks_preserve_regime_density(self):
        from eventforecastwm.performance_v2.conditional_coverage import agent_density,sample_agent_density
        from eventforecastwm.sampling import joint_samples
        forecast=self.model.imagine(self.model.encode_history(legal()),horizon_s=1.)
        paths,_=joint_samples(forecast,5,torch.Generator().manual_seed(412))
        mask=torch.ones(1,4,1,dtype=torch.bool)
        expected=torch.stack([agent_density(forecast,paths[:,i],mask)[0] for i in range(5)],1)
        actual=sample_agent_density(forecast,paths,mask,chunk=3)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5,rtol=1e-6))

    def test_serialization_and_hybrid_occupancy_composition(self):
        from eventforecastwm.performance_v2.hybrid import OccupancyWorldModel
        model=OccupancyWorldModel(self.model.recipe);model.physical.load_state_dict(self.model.state_dict())
        model.anonymous_probabilities.copy_(torch.rand_like(model.anonymous_probabilities)*.01)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(dict(recipe=model.recipe,model=model.state_dict()),path)
            loaded=load_model(path);belief=loaded.encode_history(legal())
            whole=loaded.imagine(belief,horizon_s=3.);left=loaded.imagine(belief,horizon_s=1.)
            right=loaded.imagine(left.next_belief,horizon_s=2.)
            self.assertTrue(torch.equal(whole.anonymous_occupancy[:,4:],right.anonymous_occupancy))
            self.assertTrue(torch.equal(whole.positions[:,:,4:],right.positions))
            expected=model.imagine(model.encode_history(legal()),horizon_s=3.)
            self.assertTrue(torch.equal(whole.positions,expected.positions))


if __name__=='__main__':unittest.main()
