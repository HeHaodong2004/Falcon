import ctypes as ct
import os
import unittest
import numpy as np
import torch
from .range_planar import PlanarRangeMesh
from .test_range import measured_plane
from .range_benchmark import workloads


class Tile8(unittest.TestCase):
    def cast(self,mesh,origins,directions,limit=10.):
        k=mesh.kernel;out=torch.empty(len(directions),device=k.device);k.module.launch('test_rays',len(directions),[k.tensor(origins),k.tensor(directions),k.tensor(np.full(len(directions),limit,np.float32)),ct.c_int(len(directions)),out,*k.mesh]);return out.cpu().numpy()

    def test_plane_and_missing_pixels(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');kw=measured_plane();a=PlanarRangeMesh(**kw,resolution=.1,device=device);b=PlanarRangeMesh(**kw,resolution=.1,device=device,tile_size=8);self.assertLess(b.triangles,a.triangles)
        rng=np.random.default_rng(451);d=np.column_stack([rng.uniform(-.9,.9,(4000,2)),-np.ones(4000)]).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];o=np.repeat([[0,1,0]],len(d),0).astype(np.float32);np.testing.assert_allclose(self.cast(a,o,d),self.cast(b,o,d),rtol=0,atol=3.e-6)
        kw['ranges'][0,52:76,52:76]=np.nan;hole=PlanarRangeMesh(**kw,resolution=.1,device=device,tile_size=8);self.assertEqual(float(self.cast(hole,np.array([[0,1,0]],np.float32),np.array([[0,0,-1]],np.float32),3.)[0]),3.)

    def test_40000_actual_sensor256_rays(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');rng=np.random.default_rng(2971)
        for index in [0,18,20,30]:
            e,p,t=workloads('range_benchmark_256')[index];kw=dict(ranges=p['ranges'],origins=p['origins'],rotations=p['rotations'],timestamps=p['timestamps'],now=e['graph']['timestamp'],resolution=e['graph']['resolution'],device=device,tracks=t);a=PlanarRangeMesh(**kw);b=PlanarRangeMesh(**kw,tile_size=8);o=np.repeat(p['origins'][-1:],10000,0).astype(np.float32);d=rng.normal(size=(10000,3)).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];np.testing.assert_allclose(self.cast(a,o,d),self.cast(b,o,d),rtol=0,atol=.001)


if __name__=='__main__':unittest.main()
