import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.api import Stage2Predictor,apply_graph_delta,graph_identity
from eventwm_graph.stage2.common import SCHEMA
from eventwm_graph.stage2.features import SceneFeatures
from eventwm_graph.stage2.model import GraphDeltaNet


def graph():
    g=PartialGraph(.25,0)
    g.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=[dict(cell=[0,0],state=1,height=0.,timestamp=1.,source='measured')]))
    return g.snapshot()


class GraphDeltaTests(unittest.TestCase):
    def test_trajectory_clock_and_no_future_observation_input(self):
        g=graph();path=[[1.,0.,0.,0.,0.],[2.,.2,0.,0.,.4]]
        a=SceneFeatures(g,path);b=SceneFeatures(copy.deepcopy(g),np.array(path))
        keys=np.array([[1,0],[2,1]])
        np.testing.assert_array_equal(a.transform(keys)[0],b.transform(keys)[0])
        with self.assertRaises(ValueError):SceneFeatures(g,[path[1],path[0]])
        with self.assertRaises(ValueError):SceneFeatures(g,[[0.,0.,0.,0.,0.]])

    def test_unknown_is_omission_and_deterministic_update(self):
        g=graph();before=copy.deepcopy(g)
        delta=dict(schema=SCHEMA,predicted=True,base_graph_identity=graph_identity(g),timestamp=2.,robot_pose=[.2,0.,0.,0.],
            checkpoint_sha256='test',static_updates=[dict(cell=[1,0],state=2,height=.1,timestamp=2.,source='predicted_stage2_research')])
        out=apply_graph_delta(g,delta);self.assertEqual(g,before)
        cells={tuple(c['cell']):c for c in out['cells']}
        self.assertEqual(cells[(1,0)]['state'],2);self.assertEqual(cells[(1,0)]['height'],.1)
        self.assertNotIn((2,0),cells);self.assertTrue(out['hypothetical'])
        bad=copy.deepcopy(delta);bad['static_updates'][0]['cell']=[0,0]
        with self.assertRaises(ValueError):apply_graph_delta(g,bad)

    def test_saved_load_batch_and_zero_duration(self):
        g=graph();p=[[1.,0.,0.,0.,0.],[2.,.2,0.,0.,.4]]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.pt';net=GraphDeltaNet()
            torch.save(dict(schema=SCHEMA,model_config={'width':48},model=net.state_dict(),config={}),path)
            model=Stage2Predictor(path)
            single=model.predict(partial_graph=g,trajectory=p)
            batch=model.predict_batch([dict(partial_graph=g,trajectory=p)]*2)
            np.testing.assert_allclose(single['probabilities'],batch[0]['probabilities'],rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(single['states'],batch[1]['states'])
            zero=model.predict(partial_graph=g,trajectory=p[:1]);self.assertEqual(zero['static_updates'],[])
            self.assertEqual(apply_graph_delta(g,zero)['cells'],g['cells'])


if __name__=='__main__':unittest.main()
