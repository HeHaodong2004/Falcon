import unittest
import numpy as np
import torch
from scipy.stats import multivariate_t
from eventforecastwm.model_student import StudentWorldModel,trajectory_log_probability,energy_score,marginal_cdf
from test_forecast_model import legal
from eventforecastwm.readouts import student_occupancy


class TestStudentScene(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(3)

    def test_joint_density_matches_multivariate_student(self):
        model=StudentWorldModel(kind='cv');f=model.imagine(model.encode_history(legal()),horizon_s=.75)
        target=f.positions[:,0]+.1;mask=torch.ones(1,3,1,dtype=torch.bool)
        lp,count=trajectory_log_probability(f,target,mask)
        F=torch.tensor([[1.,.25],[0.,1.]]);g=torch.tensor([.5*.25**2,.25]);P=f.initial_covariance[0,0,0]
        Q=f.acceleration_scale.square()*g[:,None]*g[None];state=[]
        for i in range(3):P=F@P@F.T+Q;state.append(P)
        covariance=torch.zeros(3,3)
        for i in range(3):
            for j in range(i,3):
                covariance[i,j]=(state[i]@torch.linalg.matrix_power(F,j-i).T)[0,0];covariance[j,i]=covariance[i,j]
        covariance+=torch.eye(3)*f.readout_scale.square()
        expected=multivariate_t.logpdf(np.ones(6)*.1,shape=np.kron(covariance.detach().numpy(),np.eye(2)),df=float(model.degrees_of_freedom))
        self.assertAlmostEqual(float(lp),expected,places=4)
        truth=target.detach();loss=-lp.mean()+energy_score(f,truth,mask)[0].mean();loss.backward()
        self.assertTrue(torch.isfinite(model.raw_df.grad));self.assertGreater(abs(float(model.raw_df.grad)),1e-6)

    def test_persistent_scale_and_observation_update(self):
        model=StudentWorldModel();b=model.encode_history(legal());full=model.imagine(b,horizon_s=3.)
        left=model.imagine(b,horizon_s=1.);right=model.imagine(left.next_belief,horizon_s=2.)
        g1=torch.Generator().manual_seed(7);g2=torch.Generator().manual_seed(7)
        all_paths,_,_=full.sample(4,g1);a,_,state=left.sample(4,g2);z,_,_=right.sample(4,g2,state)
        self.assertTrue(torch.equal(all_paths,torch.cat([a,z],2)))
        self.assertTrue(torch.equal(full.next_belief.covariance,right.next_belief.covariance))
        before=left.next_belief;measurement=before.core.positions.mean(1).detach()+.3
        obs=dict(positions=measurement,observed=torch.ones(1,1,dtype=torch.bool),acquisition_times=torch.ones(1,1),timestamp_s=1.)
        after=model.observe(before,obs)
        self.assertTrue(torch.equal(after.concentration,before.concentration+1))
        self.assertLess((after.core.positions-measurement[:,None]).square().mean(),(before.core.positions-measurement[:,None]).square().mean())
        model.imagine(after,horizon_s=1.)

    def test_sampling_matches_marginal_variance_and_cdf(self):
        model=StudentWorldModel(kind='cv');f=model.imagine(model.encode_history(legal()),horizon_s=.25)
        values,_,_=f.sample(30000,torch.Generator().manual_seed(93));residual=values[0,:,0,0,0]-f.positions[0,0,0,0,0]
        self.assertLess(abs(float(residual.var()/f.position_variance[0,0,0,0])-1),.12)
        cdf=marginal_cdf(f,f.positions);self.assertTrue(torch.allclose(cdf,torch.full_like(cdf,.5)))

    def test_occupancy_integrates_shared_scale(self):
        model=StudentWorldModel(kind='cv');f=model.imagine(model.encode_history(legal()),horizon_s=.25)
        coarse=student_occupancy(f,nodes=32);fine=student_occupancy(f,nodes=96)
        self.assertLess(float((coarse-fine).abs().max()),.001)
        self.assertLessEqual(float(fine.sum()),1.00001)
        self.assertGreater(float(fine.sum()),.99)
