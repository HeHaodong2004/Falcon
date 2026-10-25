import ctypes as ct
import unittest
import numpy as np
import torch
from .range_history import ancestor_frames
from .observed_range_mesh import ObservedRangeMesh,reconstruct
from .range_gpu import GpuObservedRangeMesh


def measured_plane():
    u,v=np.meshgrid((np.arange(128)+.5)*2/128-1,1-(np.arange(128)+.5)*2/128)
    return dict(ranges=np.sqrt(u*u+v*v+1)[None].astype(np.float32),origins=np.array([[0.,1.,0.]]),rotations=np.eye(3)[None],timestamps=np.array([3.]),now=3.)


class LegalRange(unittest.TestCase):
    def test_ancestor_selection_excludes_siblings_and_descendants(self):
        parents=np.array([-1,0,1,0,3])
        np.testing.assert_array_equal(ancestor_frames(parents,2,3),[0,1,2,5])
        np.testing.assert_array_equal(ancestor_frames(parents,2,None),[0,1])
        with self.assertRaises(ValueError):ancestor_frames([1,0],2,0)

    def test_future_and_noncausal_sensor_frames_rejected(self):
        a=measured_plane();a['now']=2.
        with self.assertRaisesRegex(ValueError,'future'):reconstruct(**a)
        a=measured_plane();a['timestamps'][0]=float('nan')
        with self.assertRaises(ValueError):reconstruct(**a)

    def test_measured_plane_has_no_holes_and_gpu_matches(self):
        a=measured_plane();torch.set_num_threads(1)
        meshes=[ObservedRangeMesh(**a,resolution=.1,device='cuda:0',legacy_dedup=True),ObservedRangeMesh(**a,resolution=.1,device='cuda:0'),GpuObservedRangeMesh(**a,resolution=.1,device='cuda:0')]
        rng=np.random.default_rng(871);n=2000;d=np.column_stack([rng.uniform(-.9,.9,n),rng.uniform(-.9,.9,n),-np.ones(n)]).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];o=np.repeat(a['origins'],n,axis=0).astype(np.float32);results=[]
        for m in meshes:
            k=m.kernel;out=torch.empty(n,device=k.device);k.module.launch('test_rays',n,[k.tensor(o),k.tensor(d),k.tensor(np.full(n,3.,np.float32)),ct.c_int(n),out,*k.mesh]);results.append(out.cpu().numpy())
        self.assertGreater(int((results[0]>=3).sum()),n//10)
        self.assertTrue(np.all(results[1]<3))
        np.testing.assert_allclose(results[1],results[2],rtol=0,atol=1.e-6)


if __name__=='__main__':unittest.main()
