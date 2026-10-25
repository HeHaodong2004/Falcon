import copy
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.api import apply_graph_delta as reference_update
from eventwm_graph.stage2_quality.model import GraphDeltaNet
from eventwm_graph.stage2_quality.common import SCHEMA
from eventwm_graph.stage2_quality.geometry import graph_identity
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.native import NativePartialGraph,freeze


def fixture():
    p=PartialGraph(.5,0.)
    p.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=[dict(cell=[x,z],state=1,height=0.,timestamp=1.,source='real') for x in range(-3,4) for z in range(-3,4)]))
    g=p.snapshot();g['global_nodes']={str(i):dict(id=str(i),cell=[x,0],position=[(x+.5)*.5,0.,.25],created_at=1.) for i,x in enumerate((-2,2))}
    g['global_edges']=[dict(source='0',target='1',path=[[-.75,0.,.25],[1.25,0.,.25]],certificate='known_free')]
    return g


class NativeGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.checkpoint=Path(cls.temp.name)/'model.pt'
        m=GraphDeltaNet(input_features=104)
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.net[-1].bias.copy_(torch.tensor([-2.,2.,-1.,.125]))
        torch.save(dict(schema=SCHEMA,mode='both',model_config=dict(width=48,input_features=104),model=m.state_dict(),config={}),cls.checkpoint)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def model(self):return SegmentPredictor(self.checkpoint,execution='optimized',future_format='native')
    def path(self):return [[1.,0.,0.,0.,0.],[1.5,.7,0.,0.,.2]]
    def test_complete_table_snapshot_exact_and_no_deferred_updates(self):
        g=fixture();before=copy.deepcopy(g);p=self.model();d=p.predict(partial_graph=g,trajectory=self.path());n=d['future_partial_graph']
        self.assertIsInstance(n,NativePartialGraph);self.assertEqual(n.to_snapshot(),reference_update(g,d));self.assertEqual(g,before)
        columns=n.cell_columns;self.assertEqual(len(columns['cell']),len(n['cells']))
        np.testing.assert_array_equal(columns['cell'],[c['cell'] for c in n['cells']])
        self.assertEqual(p.prepare_graph(g).identity,graph_identity(g))
        with mock.patch('eventwm_graph.stage2_segment.native.certificate',side_effect=AssertionError('late certification')), mock.patch('eventwm_graph.stage2_segment.native.np.lexsort',side_effect=AssertionError('late sort')):
            self.assertEqual(n.to_snapshot(),reference_update(g,d))
    def test_owned_immutable_columns_and_independent_adapters(self):
        g=fixture();p=self.model();d=p.predict(partial_graph=g,trajectory=self.path());n=d['future_partial_graph'];expected=n.to_snapshot()
        for value in n.cell_columns.values():
            if isinstance(value,np.ndarray):
                with self.assertRaises(ValueError):value.flags.writeable=True
        with self.assertRaises(AttributeError):n['cells'].table.states=np.zeros(1)
        with self.assertRaises(TypeError):n['global_nodes']['0']['position'][0]=99
        a=n.to_snapshot();b=n.to_snapshot();a['cells'][0]['cell'][0]=99;a['visited'][0]['position'][0]=99;a['global_nodes']['0']['position'][0]=99
        self.assertEqual(n.to_snapshot(),expected);self.assertEqual(b,expected)
        g['cells'][0]['state']=2;g['visited'][0]['position'][0]=99
        self.assertEqual(n.to_snapshot(),expected)
    def test_metadata_only_mutation_invalidates_native_cache(self):
        g=fixture();p=self.model();a=p.prepare_graph(g);changed=copy.deepcopy(g);changed['cells'][0]['source']='replacement_real_source'
        b=p.prepare_graph(changed);self.assertIsNot(a,b);self.assertEqual(a.identity,b.identity)
        d=p.predict(partial_graph=changed,trajectory=self.path());self.assertEqual(d['future_partial_graph'].to_snapshot(),reference_update(changed,d))
        changed2=copy.deepcopy(g);changed2['global_edges']=[];c=p.prepare_graph(changed2);self.assertIsNot(a,c)
    def test_frozen_input_and_zero_duration(self):
        g=fixture();p=self.model();sealed=freeze(g)
        for path in (self.path(),[[1.,0.,0.,0.,0.]]):
            a=p.predict(partial_graph=g,trajectory=path);b=p.predict(partial_graph=sealed,trajectory=path)
            self.assertEqual(a['future_partial_graph'].to_snapshot(),b['future_partial_graph'].to_snapshot())
            self.assertEqual(a['future_partial_graph'].to_snapshot(),reference_update(g,a))
    def test_invalid_topology_and_extensions_exact_fallback(self):
        for kind in ('bad_node','bad_edge','extension','fractional_node'):
            g=fixture()
            if kind=='bad_node':g['global_nodes']['0']['cell']=[99,99]
            elif kind=='bad_edge':g['global_edges'][0]['path']=[[-.75,0.,.25],[99.,0.,99.]]
            elif kind=='fractional_node':g['global_nodes']['0']['cell']=[-1.7,0]
            else:g['cells'][0]['extension']=dict(values=[1,2])
            p=self.model();d=p.predict(partial_graph=g,trajectory=self.path());future=d['future_partial_graph']
            snapshot=future.to_snapshot() if hasattr(future,'to_snapshot') else future
            self.assertEqual(snapshot,reference_update(g,d));self.assertEqual(d['future_format'],'snapshot')
    def test_k_equivalence_and_inputs_preserved(self):
        g=fixture();before=copy.deepcopy(g);p=self.model();paths=[self.path() for _ in range(8)]
        for i,path in enumerate(paths):path[-1][0]+=.01*i;path[-1][4]+=.02*i
        for k in (1,2,4,8):
            batch=p.predict_segments(partial_graph=g,trajectories=paths[:k])
            for path,b in zip(paths,batch):
                a=p.predict(partial_graph=g,trajectory=path)
                np.testing.assert_array_equal(a['probabilities'],b['probabilities']);self.assertEqual(a['future_partial_graph'].to_snapshot(),b['future_partial_graph'].to_snapshot())
        self.assertEqual(g,before)
    def test_native_is_a_usable_supplied_graph(self):
        g=fixture();p=self.model();n=p.predict(partial_graph=g,trajectory=self.path())['future_partial_graph']
        path=[[1.5,.7,0.,0.,.2],[1.8,.7,0.,0.,.4]]
        a=p.predict(partial_graph=n,trajectory=path);b=p.predict(partial_graph=n.to_snapshot(),trajectory=path)
        np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
        af=a['future_partial_graph'];bf=b['future_partial_graph'];self.assertEqual(af.to_snapshot() if hasattr(af,'to_snapshot') else af,bf.to_snapshot() if hasattr(bf,'to_snapshot') else bf)


if __name__=='__main__':unittest.main()
