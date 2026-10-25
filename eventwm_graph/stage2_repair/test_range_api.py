"""Complete legal sensor API tests; synthetic weights are not model results."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_api import RangeEvidencePredictor
from .range_evidence import EvidenceNet,features
from .range_benchmark import workloads
from eventwm_graph.stage2_segment.update import apply_graph_delta
from eventwm_graph.stage2_segment.fused import transform


class CompleteLegalRangeAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(811)
        cls.temp=tempfile.TemporaryDirectory();point=OUT/'training/point_residual_full/epoch06.pt';model=EvidenceNet();model.initialize(point)
        with torch.no_grad():model.head[-1].weight.normal_(0,.1)
        checkpoint=Path(cls.temp.name)/'synthetic_api_weights.pt';torch.save(dict(schema='legal_range_evidence16_1',config=dict(point_checkpoint=str(point)),model=model.state_dict(),threshold=.2),checkpoint)
        cls.model=RangeEvidencePredictor(checkpoint,device='cuda:1');cls.ex,cls.packet,cls.tracks=workloads()[0];cls.paths=[np.asarray(p['trajectory'],np.float64)for p in cls.ex['paths'][:6]]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def call(self,paths,packet=None):
        return self.model.predict_segments(partial_graph=self.ex['graph'],trajectories=paths,range_observations=self.packet if packet is None else packet,track_history=self.tracks)

    def test_single_batch_full_future_and_no_input_mutation(self):
        before=copy.deepcopy(self.ex['graph']);packet_before=self.packet['ranges'].copy();singles=[self.call([p])[0]for p in self.paths];self.model.clear_cache();batch=self.call(self.paths)
        for a,b in zip(singles,batch):
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=2.e-6,rtol=2.e-5);np.testing.assert_array_equal(a['states'],b['states']);self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(self.ex['graph'],b));self.assertTrue(b['provenance']['hypothetical']);self.assertFalse(b['provenance']['privileged_runtime_inputs'])
        self.assertEqual(before,self.ex['graph']);np.testing.assert_array_equal(packet_before,self.packet['ranges'])

    def test_training_forward_matches_api(self):
        m=self.model;p=self.paths[0];actual=self.call([p])[0];prepared=m.prepare_graph(self.ex['graph']);ctx=m.feature_class(self.ex['graph'],p,prepared);x,baseline=transform(ctx,actual['candidate_cells'],cuda=m.cuda,device_output=True);extra=features(m.active_mesh,actual['candidate_cells'],x.cpu().numpy(),p,prepared.keys,self.packet['origins'][-1],self.packet['rotations'][-1])
        with torch.inference_mode():expected,_=m.model.probabilities(x,torch.from_numpy(extra).to(m.device))
        np.testing.assert_allclose(actual['probabilities'],expected.cpu().numpy(),atol=2.e-6,rtol=2.e-5)

    def test_fused_features_match_independent_extractor(self):
        m=self.model;p=self.paths[0];actual=self.call([p])[0];prepared=m.prepare_graph(self.ex['graph']);ctx=m.feature_class(self.ex['graph'],p,prepared);x,baseline=transform(ctx,actual['candidate_cells'],cuda=m.cuda,device_output=True)
        expected=features(m.active_mesh,actual['candidate_cells'],x.cpu().numpy(),p,prepared.keys,self.packet['origins'][-1],self.packet['rotations'][-1])
        fast=m.range_kernel.compute(m.active_mesh,[(actual['candidate_cells'],baseline,p)],prepared.keys,self.packet['origins'][-1],self.packet['rotations'][-1])[0].cpu().numpy()
        np.testing.assert_allclose(fast,expected,rtol=2.e-5,atol=2.e-6)

    def test_future_sensor_rejected_and_start_only_unknown(self):
        packet=copy.deepcopy(self.packet);packet['timestamps'][-1]+=1
        with self.assertRaisesRegex(ValueError,'future'):self.call([self.paths[0]],packet)
        result=self.call([self.paths[0][:1]])[0];self.assertFalse(result['static_updates']);self.assertTrue(np.all(result['states']==0))
        stationary=np.repeat(self.paths[0][:1],2,axis=0);stationary[1,0]+=.048;later=self.call([stationary])[0];self.assertGreater(float(later['probabilities'][:,1:].max()),0.)
        self.assertNotIn('scene',inspect.signature(RangeEvidencePredictor.__init__).parameters)


if __name__=='__main__':unittest.main()
