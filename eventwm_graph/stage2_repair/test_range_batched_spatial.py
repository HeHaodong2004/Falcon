import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_device_canvas import DeviceCanvasSpatialPredictor
from .range_batched_spatial import BatchedSpatialPredictor


class BatchedSpatialParity(unittest.TestCase):
    def test_mixed_lengths_extents_and_single_queries(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;cp=OUT/'training/support_pretrained256_control/selected.pt';old=DeviceCanvasSpatialPredictor(cp,device='cuda:1');new=BatchedSpatialPredictor(cp,device='cuda:1')
        for m in [old,new]:m.config['observed_obstacle_schema']='capped_2m_complete'
        for index in [0,20,31]:
            ex,packet,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]]
            # Shorter supplied prefixes retain proper timestamps and domains.
            paths=[p[:max(1,len(p)-i%3)]for i,p in enumerate(paths)]
            def call(model,paths):return model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
            a=call(old,paths);b=call(new,paths)
            for i,(x,y)in enumerate(zip(a,b)):
                single=call(new,[paths[i]])[0]
                for z in [y,single]:
                    np.testing.assert_array_equal(x['candidate_cells'],z['candidate_cells']);np.testing.assert_array_equal(x['states'],z['states']);np.testing.assert_array_equal(x['heights'],z['heights'])
                    # AMP GEMM may round differently when the neural tile spans
                    # several queries. Decisions/heights remain separately
                    # exact; tolerate the measured probability arithmetic.
                    np.testing.assert_allclose(x['probabilities'],z['probabilities'],atol=1.e-3,rtol=3.e-5)
                    self.assertLess(float(np.abs(x['probabilities']-z['probabilities']).mean()),1.e-6)


if __name__=='__main__':unittest.main()
