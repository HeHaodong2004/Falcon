import ctypes as ct
import unittest
import numpy as np
import torch
from .range_planar import PlanarRangeMesh
from .range_gpu import GpuObservedRangeMesh
from .test_range import measured_plane


class PlanarReduction(unittest.TestCase):
    def cast(self,mesh,xy):
        k=mesh.kernel;d=np.column_stack([xy,-np.ones(len(xy))]).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];o=np.repeat([[0,1,0]],len(xy),0).astype(np.float32);out=torch.empty(len(xy),device=k.device);k.module.launch('test_rays',len(xy),[k.tensor(o),k.tensor(d),k.tensor(np.full(len(xy),3.,np.float32)),ct.c_int(len(xy)),out,*k.mesh]);return out.cpu().numpy()

    def test_flat_surface_preserved_with_fewer_triangles(self):
        a=measured_plane();full=GpuObservedRangeMesh(**a,resolution=.1,device='cuda:1');reduced=PlanarRangeMesh(**a,resolution=.1,device='cuda:1');xy=np.random.default_rng(451).uniform(-.9,.9,(4000,2));np.testing.assert_allclose(self.cast(full,xy),self.cast(reduced,xy),rtol=0,atol=2.e-6);self.assertLess(reduced.triangles,full.triangles//8)

    def test_missing_pixels_and_depth_edges_do_not_become_filled(self):
        a=measured_plane();a['ranges'][0,52:76,52:76]=np.nan;reduced=PlanarRangeMesh(**a,resolution=.1,device='cuda:1');self.assertEqual(float(self.cast(reduced,np.array([[0.,0.]]))[0]),3.)
        a=measured_plane();a['ranges'][0,:,:64]*=.5;full=GpuObservedRangeMesh(**a,resolution=.1,device='cuda:1');reduced=PlanarRangeMesh(**a,resolution=.1,device='cuda:1');xy=np.array([[-.5,0.],[.5,0.]]);np.testing.assert_allclose(self.cast(full,xy),self.cast(reduced,xy),rtol=0,atol=2.e-6)


if __name__=='__main__':unittest.main()
