import unittest
import torch
from .bilinear_context import sample


class BilinearContext(unittest.TestCase):
    def test_independent_linear_field_original_centers_and_empty_domain(self):
        # Coarse cell centers in original-cell coordinates are1,3,5,... .
        x,z=torch.meshgrid(torch.arange(4)*2+1.,torch.arange(5)*2+1.,indexing='ij');field=(2*x+3*z)[None,None].requires_grad_();ij=torch.tensor([[1,1],[2,1],[1,2],[2,2],[5,6],[-1,2],[8,3]]);actual=sample(field,ij,torch.zeros(len(ij),dtype=torch.long));expected=torch.tensor([[2*(a+.5)+3*(b+.5)]if 0<=a<8 and 0<=b<10 else[0.]for a,b in ij]);torch.testing.assert_close(actual,expected,atol=3.e-6,rtol=0);actual.sum().backward();self.assertGreater(float(field.grad.abs().sum()),0.)

    def test_mapids_do_not_mix_examples(self):
        local=torch.stack([torch.full((2,4,5),3.),torch.full((2,4,5),7.)]);ij=torch.tensor([[1,1],[2,2],[3,1],[1,2]]);ids=torch.tensor([1,0,1,0]);torch.testing.assert_close(sample(local,ij,ids),torch.tensor([[7.,7.],[3.,3.],[7.,7.],[3.,3.]]),rtol=0,atol=0)
        self.assertEqual(sample(local,torch.empty((0,2),dtype=torch.long),torch.empty(0,dtype=torch.long)).shape,(0,2))


if __name__=='__main__':unittest.main()
