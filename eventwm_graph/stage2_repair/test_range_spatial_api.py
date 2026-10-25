import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .diagnose import OUT
from .range_spatial_api import SpatialRangePredictor
from .range_benchmark import workloads
from .range_evidence import features
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2_segment.update import apply_graph_delta


class SpatialPublicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.model=SpatialRangePredictor(OUT/'training/range_fresh64/selected.pt',device='cuda:0');cls.ex,cls.packet,cls.tracks=workloads()[0];cls.paths=[np.asarray(p['trajectory'],np.float64)for p in cls.ex['paths'][:6]]

    def call(self,paths):return self.model.predict_segments(partial_graph=self.ex['graph'],trajectories=paths,range_observations=self.packet,track_history=self.tracks)

    def test_single_batch_and_normal_future_graphs(self):
        before=copy.deepcopy(self.ex['graph']);single=[self.call([p])[0]for p in self.paths];self.model.clear_cache();batch=self.call(self.paths)
        for a,b in zip(single,batch):
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=3.e-5,atol=3.e-6);np.testing.assert_array_equal(a['states'],b['states']);self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(self.ex['graph'],b))
        self.assertEqual(before,self.ex['graph'])

    def test_forward_matches_independent_cpu_evidence_assembly(self):
        m=self.model;p=self.paths[0];d=self.call([p])[0];g=m.prepare_graph(self.ex['graph']);ctx=m.feature_class(self.ex['graph'],p,g);x,_=transform(ctx,d['candidate_cells'],cuda=m.cuda,device_output=True);extra=features(m.active_mesh,d['candidate_cells'],x.cpu().numpy(),p,g.keys,self.packet['origins'][-1],self.packet['rotations'][-1]);canvas,low=m.canvas.build(g.keys,g.values,g.heights,p[0,1:4],g.resolution,self.packet['ranges'],self.packet['origins'],self.packet['rotations'],self.packet['timestamps'],self.tracks)
        with torch.inference_mode(),torch.cuda.amp.autocast():
            encoded=m.model.encode(torch.from_numpy(canvas.astype(np.float16))[None].to(m.device));expected,_=m.model.probabilities(x,torch.from_numpy(extra).to(m.device),torch.from_numpy(d['candidate_cells']-low).to(m.device),torch.zeros(len(x),dtype=torch.long,device=m.device),encoded)
        np.testing.assert_allclose(d['probabilities'],expected.cpu().numpy(),rtol=3.e-5,atol=3.e-6)

    def test_runtime_has_no_training_cache_reads(self):
        self.call([self.paths[0]])
        with patch('numpy.load',side_effect=AssertionError('Runtime must not read target/teacher caches')):self.call([self.paths[1]])


if __name__=='__main__':unittest.main()
