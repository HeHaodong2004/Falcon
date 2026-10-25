import copy
import unittest
import numpy as np
import torch
from .diagnose import HF
from .api import TimePredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.update import apply_graph_delta
from eventwm_graph.stage2_hf.time_features import moment_matrix,basis_torch


class CompleteTimeAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=TimePredictor(HF/'training/time_twohead/epoch02.pt')
        item=inputs('multi')[0];cls.graph=item['graph'];cls.paths=[np.asarray(p['trajectory'],np.float64)for p in item['paths'][:6]]

    def test_real_K6_single_batch_and_future_equivalence(self):
        m=self.model;g=self.graph;before=copy.deepcopy(g)
        singles=[m.predict(partial_graph=g,trajectory=p)for p in self.paths]
        m.clear_cache();batch=m.predict_segments(partial_graph=g,trajectories=self.paths)
        for a,b in zip(singles,batch):
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells'])
            np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=2e-6,rtol=2e-5)
            np.testing.assert_array_equal(a['states'],b['states'])
            ref=apply_graph_delta(g,b);actual=b['future_partial_graph'].to_snapshot()
            self.assertEqual(ref,actual)
            self.assertTrue(b['provenance']['hypothetical'])
        self.assertEqual(g,before)

    def test_network_matches_training_forward(self):
        from eventwm_graph.stage2_segment.fused import transform
        m=self.model;g=self.graph;p=self.paths[0];prepared=m.prepare_graph(g);ctx=m.feature_class(g,p,prepared);keys=ctx.candidates('both')
        x,_=transform(ctx,keys,cuda=m.cuda,device_output=True)
        last=np.array([c['last_seen']for c in g['cells']])[prepared.input_order]
        matrix=moment_matrix(prepared.keys,prepared.values,last,p[0,0],p[0,[1,3]],prepared.resolution)
        with torch.inference_mode():
            expected,_=m.model.probabilities(x,extra=basis_torch(x)@torch.from_numpy(matrix).to(m.device))
        actual=m.predict(partial_graph=g,trajectory=p)['probabilities']
        np.testing.assert_allclose(actual,expected.cpu().numpy(),atol=2e-6,rtol=2e-5)

    def test_one_pose_clock_validation_and_recency_cache(self):
        m=self.model;g=copy.deepcopy(self.graph);p=self.paths[0][:1]
        result=m.predict(partial_graph=g,trajectory=p)
        self.assertFalse(result['static_updates']);self.assertTrue(np.all(result['probabilities'][:,0]==1))
        prepared=m.prepare_graph(g);g['cells'][0]['last_seen']-=.1
        self.assertIsNot(m.prepare_graph(g),prepared)
        bad=p.copy();bad[0,0]+=1
        with self.assertRaises(ValueError):m.predict(partial_graph=g,trajectory=bad)

    def test_six_crops_fit_in_reusable_map_cache(self):
        m=self.model;g=self.graph;m.clear_cache();m.predict_segments(partial_graph=g,trajectories=self.paths)
        p=m.prepare_graph(g);misses=p.crop_misses
        m.predict_segments(partial_graph=g,trajectories=self.paths)
        self.assertEqual(p.crop_misses,misses)

    def test_stationary_later_timestamp_is_not_forced_unknown(self):
        p=np.repeat(self.paths[0][:1],2,axis=0);p[1,0]+=.048
        result=self.model.predict(partial_graph=self.graph,trajectory=p)
        self.assertGreater(float(result['probabilities'][:,1:].max()),0.)

    def test_metadata_extension_uses_exact_legacy_fallback(self):
        g=copy.deepcopy(self.graph);g['cells'][0]['extension']={'diagnostic':[1,2]}
        result=self.model.predict(partial_graph=g,trajectory=self.paths[0])
        self.assertEqual(result['future_format'],'snapshot')
        self.assertEqual(result['future_partial_graph'],apply_graph_delta(g,result))

    def test_column_increment_contains_all_rows_and_same_future(self):
        m=self.model;g=self.graph;p=self.paths[0];a=m.predict(partial_graph=g,trajectory=p)
        try:
            m.increment_format='columns';b=m.predict(partial_graph=g,trajectory=p)
        finally:m.increment_format='list'
        self.assertEqual(a['static_updates'],b['static_updates'].to_list())
        self.assertEqual(a['future_partial_graph'],b['future_partial_graph'])
        self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(g,b))
        with self.assertRaises(ValueError):b['static_updates'].states.flags.writeable=True


if __name__=='__main__':unittest.main()
