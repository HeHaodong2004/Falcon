import unittest
from dataclasses import replace
import torch
from eventforecastwm.model_agent_scale import AgentScaleWorldModel,trajectory_log_probability,occupancy,marginal_cdf
from eventforecastwm.model_damped import trajectory_log_probability as gaussian
from eventforecastwm.composable_sampling import sample_transition
from eventforecastwm.observations import protect_observe
from test_forecast_model import legal


def two_agents():
    value=legal();value.humans=value.humans.repeat(1,2,1,1);value.human_padding=value.human_padding.repeat(1,2)
    value.humans[:,1,:,0]+=.8;value.humans[:,1,-1,4]=0.
    return value


class TestAgentScale(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(9)

    def test_joint_density_matches_product_of_agent_mixtures(self):
        model=AgentScaleWorldModel(kind='deterministic',scale_std=(.5,1.,2.))
        f=model.imagine(model.encode_history(two_agents()),horizon_s=1.)
        truth=f.positions[:,0].detach()+torch.tensor([.05,.8])[None,None,:,None]
        mask=torch.ones(1,4,2,dtype=torch.bool);lp,count=trajectory_log_probability(f,truth,mask)
        terms=[]
        for n in range(2):
            components=[]
            for v in f.scale_values:
                one=replace(f,positions=f.positions[:,:,:,n:n+1],velocities=f.velocities[:,:,:,n:n+1],
                    initial_covariance=f.initial_covariance[:,:,n:n+1]*v,
                    acceleration_scale=f.acceleration_scale*v.sqrt(),readout_scale=f.readout_scale*v.sqrt())
                components.append(gaussian(one,truth[:,:,n:n+1],mask[:,:,n:n+1])[0])
            terms.append(torch.stack(components,-1).logsumexp(-1)-torch.tensor(3.).log())
        self.assertTrue(torch.allclose(lp,sum(terms),atol=3e-5));self.assertEqual(int(count),16)
        (-lp/count).mean().backward();self.assertGreater(float(model.scale_context.weight.grad.norm()),0.)
        self.assertTrue(torch.isfinite(occupancy(f)).all());self.assertTrue(torch.isfinite(marginal_cdf(f,truth[:,None])).all())
        mask[:,:,1]=False;first,_=trajectory_log_probability(f,truth,mask);truth[:,:,1]=1000.
        second,_=trajectory_log_probability(f,truth,mask);self.assertTrue(torch.equal(first,second))

    def test_composition_and_selective_real_observation(self):
        model=protect_observe(AgentScaleWorldModel());b=model.encode_history(two_agents())
        full=model.imagine(b,horizon_s=1.);first=model.imagine(b,horizon_s=.5);rest=model.imagine(first.next_belief,horizon_s=.5)
        g=torch.Generator().manual_seed(7);h=torch.Generator().manual_seed(7)
        all_paths,_=sample_transition(full,6,g);left,state=sample_transition(first,6,h);right,_=sample_transition(rest,6,h,state)
        self.assertTrue(torch.equal(all_paths,torch.cat([left,right],2)))
        pos=first.positions[:,0,-1].detach()+.2;pos[:,1]=float('nan')
        corrected=model.observe(first.next_belief,dict(positions=pos,observed=torch.tensor([[True,False]]),
            acquisition_times=torch.tensor([[.5,float('nan')]]),timestamp_s=.5))
        self.assertTrue(torch.allclose(corrected.agent_scale_probabilities[:,:,1],b.agent_scale_probabilities[:,:,1]))
        self.assertFalse(torch.allclose(corrected.agent_scale_probabilities[:,:,0],b.agent_scale_probabilities[:,:,0]))
        self.assertTrue(torch.isfinite(model.imagine(corrected,horizon_s=.5).positions).all())


if __name__=='__main__':unittest.main()
