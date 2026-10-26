from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .gpu_obstacle_distance import DeviceObstacleMixin
from .range_gpu import GpuObservedRangeMesh,SOURCE
from .cuda_module import CudaModule


class DeviceObstacleDistances(unittest.TestCase):
    def test_scipy_reference_batch_invariance_and_no_body_obstacles(self):
        tester=DeviceObstacleMixin();tester.device=torch.device('cuda:1');r=.1016884229464652
        for height in [.1,.6]:
            vertices=np.array([[1.5,height,0.],[1.5,height+.05,0.],[1.5,height,.15]],np.float32);packed=np.concatenate([vertices[0],vertices[1]-vertices[0],vertices[2]-vertices[0]])[None];mesh=SimpleNamespace(kernel=SimpleNamespace(resolution=r,device=tester.device,static=dict(triangles=torch.from_numpy(packed).to(tester.device))),builder=CudaModule(SOURCE,tester.device),triangles=1,obstacle_cache={})
            for keys in [np.array([[0,0]],np.int32),np.array([[0,0],[15,0],[-7,-9],[3,-4]],np.int32)]:
                expected=GpuObservedRangeMesh.obstacle_distance(mesh,keys,0.).astype(np.float32);request=tester.obstacle_request(mesh,keys,0.,torch.from_numpy(keys).to(tester.device));actual=tester.obstacle_result(mesh,request).cpu().numpy();np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
