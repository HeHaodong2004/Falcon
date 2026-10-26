import os
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .shared_sensor_api import SensorBuffers,SharedSensorPredictor,SharedFastBuildPredictor
from .native_triangle_api import NativeTrianglePredictor
from .range_benchmark import workloads
from . import test_fast_prepared_pipeline as parity


class SharedSensor(unittest.TestCase):
    def test_noncontiguous_values_and_shared_tensor_identity(self):
        device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');ranges=np.arange(3*7*9,dtype=np.float32).reshape(3,7,9)[:,::-1];origins=np.zeros((3,3));rotations=np.repeat(np.eye(3)[None],3,0);b=SensorBuffers(ranges,origins,rotations,device);self.assertIs(b.tensor(ranges,np.float32),b.values[0]);np.testing.assert_array_equal(b.values[0].cpu().numpy(),ranges);np.testing.assert_array_equal(b.selected([0,2])[0].cpu().numpy(),ranges[[0,2]])

    def test_real_outputs_fast_build_and_mutation_invalidation(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');cp=OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt';models=[c(cp,device=device)for c in [NativeTrianglePredictor,SharedSensorPredictor,SharedFastBuildPredictor]]
        for index in [3,16]:
            e,p,t=workloads('range_benchmark_256')[index];paths=[np.asarray(v['trajectory'],np.float64)for v in e['paths'][:6]];results=[m.predict_segments(partial_graph=e['graph'],trajectories=paths,range_observations=p,track_history=t)for m in models]
            for returned in results[1:]:
                for a,b in zip(results[0],returned):
                    for k in ['candidate_cells','states','heights']:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
                    np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5)
        m=models[-1];before=m.active_mesh;p['ranges'][0,0,0]=.17 if not np.isfinite(p['ranges'][0,0,0])else p['ranges'][0,0,0]+.1;m.predict_segments(partial_graph=e['graph'],trajectories=paths,range_observations=p,track_history=t);self.assertIsNot(m.active_mesh,before);self.assertEqual(float(m.active_mesh.input_buffers.values[0][0,0,0]),float(p['ranges'][0,0,0]))


class SharedSensorAPI(unittest.TestCase):
    predictor_class=SharedFastBuildPredictor
    def test_real_mixed_single_batch_fallback(self):
        old=parity.workloads;parity.workloads=lambda:workloads('range_benchmark_256')
        try:parity.FastPreparedPipelineParity.test_mixed_horizons_single_batch_and_fallback(self)
        finally:parity.workloads=old


if __name__=='__main__':unittest.main()
