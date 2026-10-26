import unittest
import os
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_batched_spatial import BatchedSpatialPredictor
from .device_pipeline import FastPreparedPipelinePredictor as OptixSpatialPredictor


class FastPreparedPipelineParity(unittest.TestCase):
    def test_real_outputs_and_generated_evidence(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;cp=OUT/'training/support_pretrained256_control/selected.pt';device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');old=BatchedSpatialPredictor(cp,device=device);new=OptixSpatialPredictor(cp,device=device)
        for m in [old,new]:m.config['observed_obstacle_schema']='capped_2m_complete'
        for index in [0,18,20,30,31]:
            ex,packet,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];results=[];features=[]
            for m in [old,new]:
                compute=m.range_kernel.compute
                def capture(*args):
                    out=compute(*args);features.append([t.cpu().numpy()for t in out]);return out
                m.range_kernel.compute=capture;results.append(m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks));m.range_kernel.compute=compute
            self.assertEqual(len(features),1)
            for a,b in zip(*results):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5);np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])

    def test_mixed_horizons_single_batch_and_fallback(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');m=OptixSpatialPredictor(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device=device)
        ex,packet,tracks=workloads()[18];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:4]];paths[0]=paths[0][:1];paths[1]=paths[1][:2]
        stationary=paths[2][:2].copy();stationary[1,1:]=stationary[0,1:];paths.append(stationary)
        def call(ps):return m.predict_segments(partial_graph=ex['graph'],trajectories=ps,range_observations=packet,track_history=tracks)
        from . import device_pipeline
        feature_history=[];range_history=[];original_point=device_pipeline.point_features;original_range=device_pipeline.range_features
        def capture_point(*args):
            result=original_point(*args);feature_history.append(result[0].cpu().numpy());return result
        def capture_range(*args):
            result=original_range(*args);range_history.append(result.cpu().numpy());return result
        device_pipeline.point_features=capture_point;device_pipeline.range_features=capture_range
        try:
            batch=call(paths);self.assertFalse(batch[0]['states'].any());self.assertEqual(call([]),[]);offset=0
            for p,b in zip(paths,batch):
                a=call([p])[0];end=offset+len(b['candidate_cells']);np.testing.assert_array_equal(feature_history[0][offset:end],feature_history[-1]);np.testing.assert_array_equal(range_history[0][offset:end],range_history[-1]);offset=end
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights'])
                # Same exact features; different GEMM row counts can round AMP
                # outputs by one FP16 unit. Hard states must still agree here.
                np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=7.e-4,rtol=1.e-3)
        finally:device_pipeline.point_features=original_point;device_pipeline.range_features=original_range
        eligible=m._eligible;m._eligible=lambda *args:False
        try:reference=call(paths)
        finally:m._eligible=eligible
        for a,b in zip(reference,batch):
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5)


if __name__=='__main__':unittest.main()
