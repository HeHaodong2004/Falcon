"""Independent synthetic and observed-mesh ray parity for hardware traversal."""
import ctypes as ct
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .optix_backend import OptixBackend
from .geometry_cuda import GeometryKernel
from .radix_bvh import build_arrays
from .test_radix_bvh import reference_rays


class OptixRays(unittest.TestCase):
    def test_synthetic_against_independent_reference_and_scalar_bvh(self):
        torch.set_num_threads(1);backend=OptixBackend('cuda:1');rng=np.random.default_rng(91127)
        for n in [1,12,13,512]:
            vertices=rng.normal(size=(n,3,3)).astype(np.float32);packed=np.concatenate([vertices[:,0],vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],1);arrays=build_arrays(torch.from_numpy(packed).to('cuda:1'));kernel=GeometryKernel(arrays,.1,'cuda:1');mesh=SimpleNamespace(triangles=n,kernel=kernel)
            origins=rng.normal(size=(1000,3)).astype(np.float32)*2;directions=rng.normal(size=(1000,3)).astype(np.float32);directions/=np.linalg.norm(directions,axis=1)[:,None];limits=np.full(len(origins),10.,np.float32)
            old=torch.empty(len(origins),device='cuda:1');kernel.module.launch('test_rays',len(origins),[kernel.tensor(origins),kernel.tensor(directions),kernel.tensor(limits),ct.c_int(len(origins)),old,*kernel.mesh]);new=backend.trace(mesh,origins,directions,limits).cpu().numpy();np.testing.assert_array_equal(old.cpu().numpy(),new)
            t=arrays['triangles'].cpu().numpy();actual=np.stack([t[:,:3],t[:,:3]+t[:,3:6],t[:,:3]+t[:,6:9]],1);np.testing.assert_allclose(new,reference_rays(actual,origins,directions,10.),atol=3.e-5,rtol=3.e-5)

    def test_observed_meshes_forty_thousand_rays(self):
        from .diagnose import OUT
        from .range_benchmark import workloads
        from .range_batched_spatial import BatchedSpatialPredictor
        torch.set_num_threads(1);backend=OptixBackend('cuda:1');model=BatchedSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1');rng=np.random.default_rng(81023)
        for index in [0,18,20,30]:
            ex,packet,tracks=workloads()[index];path=np.asarray(ex['paths'][0]['trajectory'],np.float64);model.predict_segments(partial_graph=ex['graph'],trajectories=[path],range_observations=packet,track_history=tracks);mesh=model.active_mesh;k=mesh.kernel
            origins=np.repeat(packet['origins'][-1:],10000,0).astype(np.float32);directions=rng.normal(size=origins.shape).astype(np.float32);directions/=np.linalg.norm(directions,axis=1)[:,None];limits=np.full(len(origins),10.,np.float32);old=torch.empty(len(origins),device='cuda:1');k.module.launch('test_rays',len(origins),[k.tensor(origins),k.tensor(directions),k.tensor(limits),ct.c_int(len(origins)),old,*k.mesh]);new=backend.trace(mesh,origins,directions,limits)
            np.testing.assert_array_equal(old.cpu().numpy(),new.cpu().numpy())


if __name__=='__main__':unittest.main()
