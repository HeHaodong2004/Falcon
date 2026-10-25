"""Dependency tests for the optional, unpromoted temporal-cache experiment."""
import copy
import unittest
import numpy as np
from test_native import fixture
from eventwm_graph.stage2_segment.common import OUT
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.temporal_cache_experiment import TemporalReusePredictor


class TemporalCacheExperimentTest(unittest.TestCase):
    def model(self,**kwargs):
        return TemporalReusePredictor(OUT/'training/final/last.pt',execution='optimized',future_format='native',**kwargs)

    def test_metadata_is_current_while_static_features_reuse(self):
        g=fixture();model=self.model();a=model.prepare_graph(g)
        a.map_features(np.empty((0,2),np.int32));a.crop(np.array([-8,-8]),np.array([16,16]))
        changed=copy.deepcopy(g);changed['timestamp']=2.;changed['revision']+=7
        changed['cells'][0]['source']='new_real_source';changed['cells'][0]['last_seen']=2.
        changed['global_edges']=[]
        b=model.prepare_graph(changed)
        self.assertIs(a.crops,b.crops);self.assertIs(a.integrals,b.integrals)
        self.assertIsNot(a.future_base,b.future_base);self.assertNotEqual(a.identity,b.identity)
        self.assertEqual(b.future_base.headers['timestamp'],2.)
        self.assertEqual(b.future_base.headers['revision'],changed['revision'])
        self.assertEqual(b.future_base.headers['global_edges'],())
        self.assertIn('new_real_source',b.future_base.table.sources)
        self.assertEqual(model.geometry_hits,1)

    def test_every_static_dependency_invalidates(self):
        for field in ('keys','states','heights','signedzero','resolution','defaultheight','clearance'):
            model=self.model();g=fixture();a=model.prepare_graph(g);changed=copy.deepcopy(g)
            if field=='keys':changed['cells'][0]['cell'][0]-=1
            elif field=='states':changed['cells'][0]['state']=2
            elif field=='heights':changed['cells'][0]['height']=.125
            elif field=='signedzero':changed['cells'][0]['height']=-0.
            elif field=='resolution':changed['resolution']+=.01
            elif field=='defaultheight':changed['height']+=.125
            else:changed['clearance']+=.01
            b=model.prepare_graph(changed)
            self.assertIsNot(a.crops,b.crops,msg=field);self.assertEqual(model.geometry_hits,0,msg=field)

    def test_f64_known_heights_are_preserved_despite_same_f32_geometry(self):
        g=fixture();g['cells'][0]['height']=1.;changed=copy.deepcopy(g)
        changed['cells'][0]['height']=np.nextafter(1.,2.).item()
        model=self.model();a=model.prepare_graph(g);b=model.prepare_graph(changed)
        self.assertIs(a.crops,b.crops);self.assertNotEqual(a.height64[0],b.height64[0])
        self.assertEqual(b.future_base.table.heights[0],changed['cells'][0]['height'])
        self.assertNotEqual(a.identity,b.identity)

    def test_complete_outputs_match_production_and_input_is_owned(self):
        g=fixture();model=self.model();plain=SegmentPredictor(OUT/'training/final/last.pt',execution='optimized',future_format='native')
        for clock in (1.,2.,3.):
            g['timestamp']=clock;g['revision']+=1;g['cells'][0]['last_seen']=clock
            path=[[clock,0.,0.,0.,0.],[clock+.5,.25,0.,.125,.2]];before=copy.deepcopy(g)
            a=plain.predict(partial_graph=g,trajectory=path);b=model.predict(partial_graph=g,trajectory=path)
            for field in ('candidate_cells','probabilities','states','heights'):
                np.testing.assert_array_equal(a[field],b[field])
            self.assertEqual(a['future_partial_graph'].to_snapshot(),b['future_partial_graph'].to_snapshot());self.assertEqual(g,before)
        self.assertGreater(model.geometry_hits,0)

    def test_eviction_and_cold_clear_are_bounded(self):
        model=self.model(cache_size=1,geometry_cache_size=2);g=fixture()
        for i in range(6):
            changed=copy.deepcopy(g);changed['height']=i*.25;model.prepare_graph(changed)
        self.assertEqual(len(model.cache),1);self.assertEqual(len(model.geometry_cache),2)
        model.clear_cache();self.assertFalse(model.cache);self.assertFalse(model.geometry_cache)


if __name__=='__main__':unittest.main()
