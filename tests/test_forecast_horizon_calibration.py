import unittest
from collections import Counter
import torch

from eventforecastwm.performance_v2.calibrate import root_horizon_weights


class TestHorizonCalibration(unittest.TestCase):
    def test_equal_roots_horizons_and_windows_without_inventing_support(self):
        # Root a has three short and one long window. Root b has only a
        # supported short horizon. Both independent roots receive weight 1/2.
        roots=['a','a','a','a','b','b']
        horizons=[1.,1.,1.,3.,1.,1.]
        like=torch.empty(0,dtype=torch.float64)
        weights=root_horizon_weights(roots,horizons,like)
        expected=like.new_tensor([1/12,1/12,1/12,1/4,1/4,1/4])
        torch.testing.assert_close(weights,expected,atol=0,rtol=0)
        loss=like.new_tensor([1.,2.,3.,4.,5.,7.]).requires_grad_(True)
        actual=(loss*weights).sum()
        manual=((loss[:3].mean()+loss[3])/2+loss[4:].mean())/2
        torch.testing.assert_close(actual,manual)
        torch.testing.assert_close(torch.autograd.grad(actual,loss)[0],weights)

    def test_single_horizon_preserves_original_float_weights(self):
        roots=['a']*7+['b']*3+['c']
        counts=Counter(roots);like=torch.empty(0)
        original=like.new_tensor([1./counts[r] for r in roots])/len(counts)
        torch.testing.assert_close(root_horizon_weights(roots,[3.]*len(roots),like),original,atol=0,rtol=0)
        with self.assertRaises(ValueError):root_horizon_weights([],[],like)
        with self.assertRaises(ValueError):root_horizon_weights(['a'],[],like)


if __name__=='__main__':unittest.main()
