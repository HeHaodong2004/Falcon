import unittest
import torch
from eventforecastwm.model import ForecastWorldModel,trajectory_log_probability,energy_score,sample_components
from eventquerywm.inputs import tensorize,collate


def legal():
    history=[]
    for t in [0.,.25,.5]:
        history.append(dict(timestamp=t,robot_position=[0.,0.,0.],robot_yaw=0.,previous_command=[0.,0.],tracking_available=True,
            human_tracks=[dict(track_id='legal',position=[t,0.,0.],estimated_velocity=[999.,0.,0.],
                visible_now=True,valid=True,velocity_valid=False,observation_age=0.,first_seen_time=0.,num_observations=3)]))
    return collate([tensorize(dict(history=history,history_images=None,
        query=dict(nominal_timed_trajectory=[[0.,0.,0.,0.,0.],[3.,0.,0.,0.,0.]])))])


class TestForecastModel(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(3)

    def test_analytic_nll_matches_full_gaussian(self):
        model=ForecastWorldModel(kind='cv',noise='integrated')
        f=model.imagine(model.encode_history(legal()),horizon_s=.75)
        target=f.positions[:,0]+.1;mask=torch.ones(1,3,1,dtype=torch.bool)
        lp,count=trajectory_log_probability(f,target,mask)
        F=torch.tensor([[1.,.25],[0.,1.]]);g=torch.tensor([.5*.25**2,.25])
        P=f.initial_covariance[0,0,0];Q=f.acceleration_scale.square()*g[:,None]*g[None]
        state=[]
        for i in range(3):P=F@P@F.T+Q;state.append(P)
        cov=torch.zeros(3,3)
        for i in range(3):
            for j in range(i,3):
                cov[i,j]=(state[i]@torch.linalg.matrix_power(F,j-i).T)[0,0]
                cov[j,i]=cov[i,j]
        cov+=torch.eye(3)*f.readout_scale.square()
        dist=torch.distributions.MultivariateNormal(torch.zeros(3),covariance_matrix=cov)
        expected=2*dist.log_prob(torch.ones(3)*.1)
        self.assertTrue(torch.allclose(lp[0],expected,atol=1e-4));self.assertEqual(int(count),6)

    def test_composition_and_query_independence(self):
        model=ForecastWorldModel(noise='integrated');b=model.encode_history(legal())
        full=model.imagine(b,horizon_s=3.)
        first=model.imagine(b,torch.ones(1,8,16,10),horizon_s=1.)
        rest=model.imagine(first.next_belief,torch.randn(1,33,16,10),horizon_s=2.)
        self.assertTrue(torch.equal(full.positions[:,:,4:],rest.positions))
        self.assertTrue(torch.equal(full.next_belief.covariance,rest.next_belief.covariance))
        self.assertTrue(torch.equal(full.weights,rest.weights))
        g1=torch.Generator().manual_seed(78);g2=torch.Generator().manual_seed(78)
        whole,_,_=full.sample(draws=4,generator=g1)
        left,_,state=first.sample(draws=4,generator=g2)
        right,_,_=rest.sample(draws=4,generator=g2,initial_state=state)
        self.assertTrue(torch.equal(whole,torch.cat([left,right],2)))

    def test_missing_targets_do_not_contribute(self):
        model=ForecastWorldModel(noise='integrated');f=model.imagine(model.encode_history(legal()),horizon_s=1.)
        truth=f.positions[:,0].detach().clone();mask=torch.ones(1,4,1,dtype=torch.bool);mask[:,1:3]=False
        first,_=trajectory_log_probability(f,truth,mask);truth[:,1:3]=1000
        second,_=trajectory_log_probability(f,truth,mask)
        self.assertTrue(torch.equal(first,second))

    def test_gradient_and_actual_observation_correction(self):
        model=ForecastWorldModel(noise='integrated');b=model.encode_history(legal());f=model.imagine(b,horizon_s=1.)
        truth=f.positions[:,0].detach()+.1;mask=torch.ones(1,4,1,dtype=torch.bool)
        lp,count=trajectory_log_probability(f,truth,mask);es,_=energy_score(f,truth,mask)
        loss=es.mean()-.05*(lp/count).mean();loss.backward()
        self.assertGreater(model.core.acceleration[-1].weight.grad.norm(),0.)
        self.assertGreater(model.log_initial_velocity.grad.abs(),0.)
        nxt=f.next_belief;measurement=nxt.core.positions.mean(1).detach()+.3
        obs=dict(positions=measurement,observed=torch.ones(1,1,dtype=torch.bool),acquisition_times=torch.ones(1,1),timestamp_s=1.)
        corrected=model.observe(nxt,obs)
        self.assertLess((corrected.core.positions-measurement[:,None]).square().mean(),(nxt.core.positions-measurement[:,None]).square().mean())
        self.assertLess(corrected.covariance[...,0,0].mean(),nxt.covariance[...,0,0].mean())
        obs['source']='imagined'
        with self.assertRaises(ValueError):model.observe(nxt,obs)

    def test_native_delayed_acquisition_remains_float32(self):
        model=ForecastWorldModel(noise='integrated');b=model.imagine(model.encode_history(legal()),horizon_s=1.).next_belief
        obs=dict(positions=b.core.positions.mean(1).detach(),observed=torch.ones(1,1,dtype=torch.bool),
            acquisition_times=torch.tensor([[.976]],dtype=torch.float64),timestamp_s=1.)
        updated=model.observe(b,obs)
        self.assertEqual(updated.core.positions.dtype,torch.float32)
        model.imagine(updated,horizon_s=.25)
        obs['acquisition_times'].fill_(1.001)
        with self.assertRaises(ValueError):model.observe(b,obs)


if __name__=='__main__':unittest.main()
