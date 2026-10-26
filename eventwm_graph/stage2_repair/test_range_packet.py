import ctypes as ct
import unittest
import numpy as np
import torch
from .range_packet import PacketExactKernel
from .radix_bvh import build_arrays
from .geometry_cuda import GeometryKernel
from .range_radix_api import RadixExactSpatialPredictor
from .range_benchmark import workloads
from .diagnose import OUT


class PacketVisibilityTest(unittest.TestCase):
    def test_packet_matches_independent_scalar_bvh_rays(self):
        torch.set_num_threads(1);rng=np.random.default_rng(6871);device='cuda:1'
        vertices=rng.normal(size=(512,3,3)).astype(np.float32)
        packed=np.concatenate([vertices[:,0],vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],1)
        geometry=GeometryKernel(build_arrays(torch.from_numpy(packed).to(device)),.1,device)
        n=1000;origins=rng.normal(size=(n,3)).astype(np.float32)*4
        directions=rng.normal(size=(n,63,3)).astype(np.float32)
        directions[:n//2]+=origins[:n//2,None,:]*2
        directions/=np.linalg.norm(directions,axis=2)[:,:,None];limits=rng.uniform(.2,10.,(n,63)).astype(np.float32)
        def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).to(device)
        flat_origins=tensor(np.repeat(origins,63,axis=0));flat_directions=tensor(directions.reshape(-1,3));flat_limits=tensor(limits.reshape(-1));distances=torch.empty(n*63,device=device)
        geometry.module.launch('test_rays',n*63,[flat_origins,flat_directions,flat_limits,ct.c_int(n*63),distances,*geometry.mesh])
        expected=(distances.reshape(n,63)<tensor(limits)).any(1).cpu().numpy()
        kernel=PacketExactKernel(device);result=torch.empty(n,dtype=torch.int32,device=device)
        kernel.module.launch('test_packets',min(n*32,262144),[tensor(origins),flat_directions,flat_limits,ct.c_int(n),result,*geometry.mesh])
        np.testing.assert_array_equal(result.cpu().numpy(),expected.astype(np.int32))

    def test_real_full_predictions_equal_scalar63(self):
        model=RadixExactSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1')
        model.config['observed_obstacle_schema']='capped_2m_complete';scalar=model.range_kernel;packet=PacketExactKernel('cuda:1')
        for index in [0,20,31]:
            ex,sensor,tracks=workloads()[index];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
            def call():return model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=sensor,track_history=tracks)
            model.range_kernel=scalar;before=call();model.range_kernel=packet;after=call()
            for a,b in zip(before,after):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights'])
                np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=3.e-5,atol=3.e-6)


if __name__=='__main__':unittest.main()
