import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_bev import RangeCanvas
from .gpu_edt import GridEDT
from .range_device_canvas import DeviceRangeCanvas,DeviceCanvasSpatialPredictor
from .range_exact_api import ExactFastSpatialPredictor


class DeviceCanvasParity(unittest.TestCase):
    def test_real14channel_encoder_inputs(self):
        torch.set_num_threads(1);old=RangeCanvas('cuda:1');new=DeviceRangeCanvas(GridEDT('cuda:1'));model=ExactFastSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1')
        for index in [0,7,16,20,31]:
            ex,p,tracks=workloads()[index];g=model.prepare_graph(ex['graph']);ego=np.asarray(ex['paths'][0]['trajectory'])[0,1:4]
            args=(g.keys,g.values,g.heights,ego,g.resolution,p['ranges'],p['origins'],p['rotations'],p['timestamps'],tracks)
            a,lo=old.build(*args);b,low=new.build(*args);np.testing.assert_array_equal(lo,low);np.testing.assert_array_equal(a.astype(np.float16),b.cpu().numpy())

    def test_complete_predictions(self):
        torch.set_num_threads(1);cp=OUT/'training/support_pretrained256_control/selected.pt';old=ExactFastSpatialPredictor(cp,device='cuda:1');new=DeviceCanvasSpatialPredictor(cp,device='cuda:1')
        for index in [0,20,31]:
            ex,p,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];kw=dict(partial_graph=ex['graph'],trajectories=paths,range_observations=p,track_history=tracks);a=old.predict_segments(**kw);b=new.predict_segments(**kw)
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x['candidate_cells'],y['candidate_cells']);np.testing.assert_array_equal(x['states'],y['states']);np.testing.assert_array_equal(x['heights'],y['heights']);np.testing.assert_allclose(x['probabilities'],y['probabilities'],atol=3.e-6,rtol=3.e-5)


if __name__=='__main__':unittest.main()
