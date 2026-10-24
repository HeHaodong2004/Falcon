"""Training speed must preserve the reference CTMC law and RNG consumption."""
import copy
import unittest
import numpy as np
import torch
from eventwm_graph.l2_intermittent import fit
from eventwm_graph.l2_intermittent.dynamics import DEFAULT
from eventwm_graph.l2_intermittent.neural import IntermittentDynamics
from eventwm_graph.l2_final.factor_kernel import sample_factors


class TrainingFactorKernelTests(unittest.TestCase):
    def test_factors_and_rng_match_reference_with_mixtures_and_long_horizons(self):
        setup=np.random.default_rng(931)
        for n in (1,3,7):
            mean=setup.normal(size=(n,4));cov=np.tile(np.diag([.04,.03,.2,.1]),(n,1,1))
            for mixture in (None,(np.stack([mean,mean+.3],1),np.stack([cov,cov*2],1),np.tile([.3,.7],(n,1)))):
                for samples in (4,8):
                    a=np.random.default_rng(31);b=np.random.default_rng(31)
                    times=np.array([0.,1e-8,.513,5.])
                    reference=fit.sample_factors(mean,cov,times,DEFAULT,a,samples,mixture)
                    compiled=sample_factors(mean,cov,times,DEFAULT,b,samples,mixture)
                    for x,y in zip(reference,compiled):np.testing.assert_allclose(x,y,atol=2e-10,rtol=1e-10)
                    self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_float32_batches_scores_and_gradients_match(self):
        torch.set_num_threads(1);rng=np.random.default_rng(7);rows=[]
        for j in range(5):
            n=j+1;mean=rng.normal(size=(n,4));times=np.arange(0,3.001,.048)
            c=dict(weight=1.,mean=mean,covariance=np.tile(np.diag([0.,0.,.1,.1]),(n,1,1)),age=np.zeros(n),
                   graph=rng.normal(size=(n,18)),selected=np.arange(n),truth=mean[None,:,:2]+times[:,None,None]*mean[None,:,2:],
                   maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None)
            rows.append(dict(root_id=str(j),components=[c],times=times))
        a=np.random.default_rng(221);b=np.random.default_rng(221)
        reference=fit.make_batch(rows,a,DEFAULT,4,'cpu')
        original=fit.sample_factors
        try:
            fit.sample_factors=sample_factors
            compiled=fit.make_batch(rows,b,DEFAULT,4,'cpu')
        finally:fit.sample_factors=original
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        for key in reference:torch.testing.assert_close(reference[key],compiled[key],atol=2e-6,rtol=1e-6)
        torch.manual_seed(221);left=IntermittentDynamics(64);right=copy.deepcopy(left)
        x=fit.objective(left,reference);y=fit.objective(right,compiled);x.backward();y.backward()
        torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-7)
        for p,q in zip(left.parameters(),right.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-6)


if __name__=='__main__':unittest.main()
