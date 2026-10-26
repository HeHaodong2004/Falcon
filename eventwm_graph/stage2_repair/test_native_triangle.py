import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .range_planar import PlanarRangeMesh
from .optix_backend import OptixBackend
from .test_range import measured_plane
from .native_triangle_api import NativeTrianglePredictor
from .range_benchmark import workloads
from . import test_fast_prepared_pipeline as parity


class NativeTriangles(unittest.TestCase):
    predictor_class=NativeTrianglePredictor
    def test_analytic_measured_plane_and_unfilled_hole(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');backend=OptixBackend(device,primitive_mode='native');kw=measured_plane();mesh=PlanarRangeMesh(**kw,resolution=.1,device=device,bvh='native');self.assertEqual(mesh.kernel.static['bounds'].numel(),0)
        rng=np.random.default_rng(451);d=np.column_stack([rng.uniform(-.9,.9,(4000,2)),-np.ones(4000)]).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];o=np.repeat([[0,1,0]],len(d),0).astype(np.float32);limits=np.full(len(d),3.,np.float32);actual=backend.trace(mesh,o,d,limits).cpu().numpy();expected=-1/d[:,2];np.testing.assert_allclose(actual,expected,rtol=0,atol=3.e-6)
        kw['ranges'][0,52:76,52:76]=np.nan;mesh=PlanarRangeMesh(**kw,resolution=.1,device=device,bvh='native');value=backend.trace(mesh,np.array([[0,1,0]],np.float32),np.array([[0,0,-1]],np.float32),np.array([3.],np.float32));self.assertEqual(float(value[0]),3.)
        with self.assertRaisesRegex(RuntimeError,'hardware-only'):mesh.kernel.compute(None,None,None,None,None)

    def test_real_mixed_horizons_single_batch_fallback_without_software_tree(self):
        old=parity.workloads;parity.workloads=lambda:workloads('range_benchmark_256')
        try:
            with patch('eventwm_graph.stage2_repair.radix_bvh.build_arrays',side_effect=AssertionError('redundant software tree constructed')):parity.FastPreparedPipelineParity.test_mixed_horizons_single_batch_and_fallback(self)
        finally:parity.workloads=old


if __name__=='__main__':unittest.main()
