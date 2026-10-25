import unittest
import numpy as np
from .support_teacher import crop_targets


class SupportTargetTest(unittest.TestCase):
    def test_pretrain_rotation_preserves_coordinate_vector_axes(self):
        from .support_pretrain import augment
        maps=np.zeros((14,5,5),np.float32)
        maps[12]=np.arange(5)[:,None]-2.3
        maps[13]=np.arange(5)[None,:]-1.7
        target=np.arange(50).reshape(2,5,5)
        original=maps.copy()
        for rotation in range(4):
            for flip in [False,True]:
                a,b=augment(maps,target,rotation,flip)
                np.testing.assert_allclose(np.diff(a[12],axis=0),1.,atol=3.e-7)
                np.testing.assert_allclose(np.diff(a[13],axis=1),1.,atol=3.e-7)
                expected=np.rot90(target,rotation,axes=(-2,-1))
                if flip:expected=expected[:,:,::-1]
                np.testing.assert_array_equal(b,expected)
        np.testing.assert_array_equal(maps,original)

    def test_graph_pretrain_uses_exact_same_negative_world_cell_grid(self):
        from .support_graph_prepare import legal_canvas
        keys=np.array([[-1,-1],[-2,-1],[0,-1]])
        states=np.array([1,2,1],np.uint8);heights=np.array([1.,1.,1.])
        maps,low=legal_canvas(keys,states,heights,[-.05,1.,-.05],.1)
        expected=np.zeros((2,192,192),np.float32)
        for key,state in zip(keys,states):
            i,j=(key-low)//2;expected[state-1,i,j]+=.25
        np.testing.assert_array_equal(maps[:2],expected)
        np.testing.assert_array_equal(maps[6:12],0.)
        self.assertEqual(low.tolist(),[-224,-224])

    def test_world_keys_negative_coordinates_and_floor_selection(self):
        keys=np.array([[-5,-7],[-4,-7],[-5,-6],[-4,-6],[-3,-7],[-2,-7],[-5,-7]])
        heights=np.array([0.,0.,0.,0.,1.,-.3,2.])
        target=crop_targets(keys,heights,[-5,-7],0.,.1)
        self.assertEqual(float(target[0,0,0]),1.)
        self.assertEqual(float(target[0,1,0]),0.)
        self.assertEqual(float(target[0].sum()),1.)

    def test_pool_fraction_is_not_a_certificate_or_blocked_label(self):
        target=crop_targets(np.array([[10,12]]),np.array([0.]),[10,12],0.,.1)
        self.assertEqual(float(target[0,0,0]),.25)
        self.assertEqual(target.shape,(2,192,192))
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue(np.all(np.abs(target[1])<=1))


if __name__=='__main__':unittest.main()
