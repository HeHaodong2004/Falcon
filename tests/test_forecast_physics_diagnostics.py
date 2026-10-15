import unittest
import torch
from eventforecastwm.performance_v2.physics_diagnostics import map_values,variogram


class TestPhysicsDiagnostics(unittest.TestCase):
    def test_map_axes_unknown_and_outside_are_distinct(self):
        features=torch.zeros(1,2,32,32);support=torch.ones(1,32,32,dtype=torch.bool)
        features[0,1,3,7]=1;features[0,0,3,7]=.25;support[0,8,9]=False
        positions=torch.tensor([[[-4.25,-6.25],[-3.25,-3.75],[10.,0.]]])
        inside,known,walk,clearance=map_values(positions,features,support)
        self.assertEqual(inside.tolist(),[[True,True,False]])
        self.assertEqual(known.tolist(),[[True,False,False]])
        self.assertTrue(walk[0,0]);self.assertEqual(float(clearance[0,0]),.5)

    def test_variogram_distinguishes_joint_laws_with_equal_marginals(self):
        truth=torch.tensor([[[[-1.,0.],[1.,0.]]]]).expand(1,4,2,2)
        shared=truth[:,None]+torch.tensor([[-1.,0.],[1.,0.]])[None,:,None,None]
        opposite=truth[:,None]+torch.tensor([[[-1.,0.],[1.,0.]],[[1.,0.],[-1.,0.]]])[None,:,None]
        weights=torch.full((1,2),.5);complete=torch.ones(1,2,dtype=torch.bool)
        good,_=variogram(shared,shared,weights,truth,complete,'spatial')
        bad,_=variogram(opposite,opposite,weights,truth,complete,'spatial')
        self.assertEqual(float(good),0.);self.assertGreater(float(bad),.1)
        jitter=shared.clone();jitter[:,:,1::2]=shared.flip(1)[:,:,1::2]
        coherent,_=variogram(shared,shared,weights,truth,complete,'temporal_1')
        incoherent,_=variogram(jitter,jitter,weights,truth,complete,'temporal_1')
        self.assertEqual(float(coherent),0.);self.assertGreater(float(incoherent),1.)


if __name__=='__main__':unittest.main()
