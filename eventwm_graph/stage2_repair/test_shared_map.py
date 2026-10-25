import unittest
import numpy as np
import torch
from torch import nn
from .shared_map import SharedMapNet,map_canvas


class SharedRepresentation(unittest.TestCase):
    def test_negative_outside_keys_do_not_sample_border(self):
        m=SharedMapNet();m.head=nn.Linear(128,2)
        with torch.no_grad():m.head.weight.zero_();m.head.bias.zero_();m.head.weight[0,88]=1
        x=torch.zeros(2,88);indices=torch.tensor([[-1,0],[0,0]]);encoded=(torch.ones(1,24,2,2),torch.zeros(1,16))
        r,_,_=m.raw(x,indices,torch.zeros(2,dtype=torch.long),encoded)
        self.assertAlmostEqual(float(r[1]-r[0]),1.,places=6)

    def test_padding_masks_make_map_encoding_independent_of_batch_extent(self):
        torch.set_num_threads(1);torch.manual_seed(3);m=SharedMapNet().eval();a=torch.randn(1,8,32,48);b=torch.randn(1,8,48,32)
        together=torch.zeros(2,8,48,48);together[0,:,:32,:48]=a[0];together[1,:,:48,:32]=b[0]
        with torch.no_grad():
            field,global_=m.encode(together,torch.tensor([[32,48],[48,32]]));fa,ga=m.encode(a,torch.tensor([[32,48]]));fb,gb=m.encode(b,torch.tensor([[48,32]]))
        torch.testing.assert_close(field[0,:,:32,:48],fa[0],atol=2e-6,rtol=2e-5);torch.testing.assert_close(field[1,:,:48,:32],fb[0],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(global_[0],ga[0],atol=2e-6,rtol=2e-5);torch.testing.assert_close(global_[1],gb[0],atol=2e-6,rtol=2e-5)
        self.assertEqual(int(torch.count_nonzero(field[0,:,32:,:])),0)

    def test_certificate_age_is_causal_and_clock_shift_invariant(self):
        keys=np.array([[-2,1],[0,-3]],np.int32);state=np.array([1,2],np.uint8);height=np.zeros(2,np.float32);last=np.array([.8,.9]);geometry=[.1,0.,.25]
        a=map_canvas(keys,state,height,last,1.,geometry,[0,0,0]);b=map_canvas(keys,state,height,last+1000,1001.,geometry,[0,0,0])
        np.testing.assert_allclose(a['maps'],b['maps'],rtol=0,atol=1e-6)
        with self.assertRaises(ValueError):map_canvas(keys,state,height,last+1,1.,geometry,[0,0,0])

    def test_anchored_fine_encoding_keeps_original_cell_resolution(self):
        torch.set_num_threads(1);m=SharedMapNet(map_channels=10,fine=True).eval();maps=torch.randn(1,10,32,48);fine=torch.randn(1,5,64,96)
        with torch.no_grad():local,global_=m.encode(maps,torch.tensor([[32,48]]),fine)
        self.assertEqual(local.shape,(1,24,64,96));self.assertEqual(global_.shape,(1,16))


if __name__=='__main__':unittest.main()
