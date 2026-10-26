import os
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .captured_head import CapturedPipelinePredictor
from .device_pipeline import FastPreparedPipelinePredictor
from .range_benchmark import workloads


class CapturedParity(unittest.TestCase):
    def test_complete_original_cells_and_outputs(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:0');cp=OUT/'training/range_online4096/selected_epoch02_argmax.pt';old=FastPreparedPipelinePredictor(cp,device=device);new=CapturedPipelinePredictor(cp,device=device)
        for i in [0,18,20,30,31]:
            ex,packet,tracks=workloads()[i];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];results=[]
            for m in [old,new]:results.append(m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks))
            for a,b in zip(*results):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=7.e-4,rtol=1.e-3);np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])


if __name__=='__main__':unittest.main()
