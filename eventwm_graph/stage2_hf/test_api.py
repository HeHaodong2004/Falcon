import copy
import os
import unittest
import numpy as np
from .api import Predictor
from .common import OUT
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.prepared import content_key
from eventwm_graph.stage2_segment.update import apply_graph_delta
from eventwm_graph.stage2_quality.features import Features

class RealApiContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.checkpoint=os.environ.get('HF_TEST_CHECKPOINT',OUT/'training/spatial_bce/epoch01.pt');cls.model=Predictor(cls.checkpoint,threshold=.1,device='cuda:0');cls.group=inputs('multi')[0]
 def test_complete_single_batch_and_candidate_identity(self):
  g=copy.deepcopy(self.group['graph']);before=content_key(g);paths=[p['trajectory']for p in self.group['paths'][:6]];model=self.model
  batch=model.predict_segments(partial_graph=g,trajectories=paths)
  self.assertEqual(content_key(g),before);self.assertEqual(len(batch),6)
  for p,d in zip(paths,batch):
   ref=Features(g,p).candidates('both');np.testing.assert_array_equal(ref,d['candidate_cells']);single=model.predict(partial_graph=g,trajectory=p)
   np.testing.assert_allclose(single['probabilities'],d['probabilities'],atol=2e-6,rtol=2e-5);np.testing.assert_array_equal(single['states'],d['states']);np.testing.assert_allclose(d['probabilities'].sum(1),1,atol=2e-6)
   future=d['future_partial_graph'].to_snapshot()if hasattr(d['future_partial_graph'],'to_snapshot')else d['future_partial_graph'];reference=apply_graph_delta(g,d)
   self.assertEqual(future,reference);known={tuple(c['cell']):c for c in g['cells']};actual={tuple(c['cell']):c for c in future['cells']}
   for k,c in known.items():self.assertEqual(c,actual[k])
   self.assertEqual(len(actual),len(known)+len(d['static_updates']));self.assertTrue(d['provenance']['hypothetical'])
 def test_content_cache_invalidation_and_zero_duration(self):
  g=copy.deepcopy(self.group['graph']);p=np.asarray(self.group['paths'][0]['trajectory']);a=self.model.predict(partial_graph=g,trajectory=p);miss=self.model.cache_misses;g['cells'][0]['height']+=.03;self.model.predict(partial_graph=g,trajectory=p);self.assertGreater(self.model.cache_misses,miss)
  d=self.model.predict(partial_graph=g,trajectory=p[:1]);self.assertFalse(d['static_updates']);self.assertTrue(np.all(d['probabilities'][:,0]==1));self.assertEqual(d['future_partial_graph']['timestamp'],g['timestamp'])
 def test_clock_and_nonfinite_rejected(self):
  g=self.group['graph'];p=np.asarray(self.group['paths'][0]['trajectory']);bad=p.copy();bad[0,0]+=.001
  with self.assertRaises(ValueError):self.model.predict(partial_graph=g,trajectory=bad)
  bad=p.copy();bad[-1,1]=np.nan
  with self.assertRaises(ValueError):self.model.predict(partial_graph=g,trajectory=bad)
if __name__=='__main__':unittest.main()
