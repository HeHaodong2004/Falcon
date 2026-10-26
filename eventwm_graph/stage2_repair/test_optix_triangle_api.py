import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_batched_spatial import BatchedSpatialPredictor
from .range_optix import OptixTriangleSpatialPredictor as OptixSpatialPredictor


class OptixTriangleParity(unittest.TestCase):
    def test_real_outputs_and_generated_evidence(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;cp=OUT/'training/support_pretrained256_control/selected.pt';old=BatchedSpatialPredictor(cp,device='cuda:1');new=OptixSpatialPredictor(cp,device='cuda:1')
        for m in [old,new]:m.config['observed_obstacle_schema']='capped_2m_complete'
        for index in [0,18,20,30,31]:
            ex,packet,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];results=[];features=[]
            for m in [old,new]:
                compute=m.range_kernel.compute
                def capture(*args):
                    out=compute(*args);features.append([t.cpu().numpy()for t in out]);return out
                m.range_kernel.compute=capture;results.append(m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks));m.range_kernel.compute=compute
            for a,b in zip(features[0],features[1]):np.testing.assert_array_equal(a,b)
            for a,b in zip(*results):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5);np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])


if __name__=='__main__':unittest.main()
