"""Independent planar-depth checks for the training-only completion targets."""
import unittest
import numpy as np
import torch
from .range_bev import RangeCanvas,SIZE
from .surface_teacher import project_targets


class SurfaceTargets(unittest.TestCase):
    def test_later_surface_is_loss_only_and_human_hit_is_unknown(self):
        torch.set_num_threads(1);builder=RangeCanvas('cuda:1');h=w=128
        a=(np.arange(w)+.5)*2/w-1;b=1-(np.arange(h)+.5)*2/h
        depth=(3.1*np.sqrt(1+b[:,None]**2+a[None,:]**2)).astype(np.float32)[None]
        origin=np.array([[0.,.7,0.]]);rotation=np.eye(3)[None];low=np.array([-192,-192]);i,j=96,80
        # The coarse-cell centre is exactly(.1,.7,-3.1), on the plane.
        surface=project_targets(builder,depth,origin,rotation,[[]],low,.1,0.)
        self.assertEqual(surface[1,i,j],1);self.assertEqual(surface[4,i,j],1)
        far=project_targets(builder,depth*2,origin,rotation,[[]],low,.1,0.)
        self.assertEqual(far[1,i,j],0);self.assertEqual(far[4,i,j],1)
        tracks=[[dict(position=[.1,0.,-3.1],valid=True)]]
        occluded=project_targets(builder,depth,origin,rotation,tracks,low,.1,0.)
        self.assertEqual(occluded[1,i,j],0);self.assertEqual(occluded[4,i,j],0)
        # Masking a human hit never extends measured free space behind it.
        self.assertEqual(occluded[4,i,j-5],0)
        self.assertTrue(np.all(surface[:3]<=surface[3:]))


if __name__=='__main__':unittest.main()
