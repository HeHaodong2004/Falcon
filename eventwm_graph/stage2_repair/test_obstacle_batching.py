"""A map-distance covariate must not depend on other query cells in a batch."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .range_gpu import GpuObservedRangeMesh,SOURCE
from .cuda_module import CudaModule


class ObservedObstacleBatching(unittest.TestCase):
    def test_capped_two_meter_distance_is_query_batch_invariant(self):
        vertices=np.array([[1.5,.5,0.],[1.5,1.,0.],[1.5,.5,.15]],np.float32)
        packed=np.concatenate([vertices[0],vertices[1]-vertices[0],vertices[2]-vertices[0]])[None]
        mesh=SimpleNamespace(kernel=SimpleNamespace(resolution=.1,device=torch.device('cuda:0'),static=dict(triangles=torch.from_numpy(packed).to('cuda:0'))),builder=CudaModule(SOURCE,'cuda:0'),triangles=1,obstacle_cache={})
        one=GpuObservedRangeMesh.obstacle_distance(mesh,np.array([[0,0]],np.int32),0.)
        batch=GpuObservedRangeMesh.obstacle_distance(mesh,np.array([[0,0],[15,0]],np.int32),0.)
        self.assertAlmostEqual(float(np.minimum(one[0],2.)),1.5,places=6)
        np.testing.assert_array_equal(np.minimum(one,2.),np.minimum(batch[:1],2.))


if __name__=='__main__':unittest.main()
