import unittest
import torch
from eventforecastwm.agent_scale_calibration import pack_supported


class TestLossPacking(unittest.TestCase):
    def test_joint_objective_and_gradients_preserved(self):
        torch.manual_seed(1)
        density=torch.randn(3,6,11,7,dtype=torch.float64)
        count=torch.zeros(3,11,dtype=torch.float64);count[0,[2,9]]=24;count[1,[0,1,4]]=24;count[2,7]=24
        context=torch.randn(3,11,8,dtype=torch.float64);weights=torch.randn(3,6,dtype=torch.float64).softmax(-1)
        parameters=torch.randn(8,7,dtype=torch.float64,requires_grad=True)
        def loss(d,c,x,w):
            logp=(x@parameters).log_softmax(-1)[:,None]
            component=((d+logp).logsumexp(-1)*(c>0)[:,None]).sum(-1)
            return -((component+w.log()).logsumexp(-1)/c.sum(-1)).mean()
        old=loss(density,count,context,weights);gradient=torch.autograd.grad(old,parameters)[0]
        packed=pack_supported([(density,count,context,weights)])
        new=loss(*packed);updated=torch.autograd.grad(new,parameters)[0]
        self.assertEqual(packed[0].shape,(3,6,3,7))
        torch.testing.assert_close(old,new,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(gradient,updated,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
