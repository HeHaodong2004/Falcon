import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_exact_api import ExactFastSpatialPredictor
from .range_gpu_graph import DeviceGraphSpatialPredictor


class DeviceGraphParity(unittest.TestCase):
    def test_real_feature_tensors_complete_predictions_and_heights(self):
        torch.set_num_threads(1);cp=OUT/'training/support_pretrained256_control/selected.pt';before=ExactFastSpatialPredictor(cp,device='cuda:1');after=DeviceGraphSpatialPredictor(cp,device='cuda:1')
        for index in [0,20,31]:
            ex,packet,tracks=workloads()[index];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]];features=[];results=[]
            for model in [before,after]:
                original=model._network
                def capture(xs,details):
                    features.append([x.cpu().numpy()for x in xs]);return original(xs,details)
                model._network=capture
                results.append(model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks));model._network=original
            for a,b in zip(features[0],features[1]):np.testing.assert_array_equal(a,b)
            for a,b in zip(*results):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=3.e-5,atol=3.e-6)


if __name__=='__main__':unittest.main()
