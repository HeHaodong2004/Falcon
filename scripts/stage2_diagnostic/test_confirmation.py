"""Protected confirmation scorer tests with synthetic graphs only."""
import copy
import unittest
import numpy as np
from .final_confirmation import score_fixed

class ConfirmationScorerTest(unittest.TestCase):
    def test_empty_candidate_missing_positive(self):
        ex={'graph':{'cells':[]},'target':{(4,0):(2,0.)}}
        d={'probabilities':np.empty((0,3),np.float32),'states':np.empty(0,np.uint8),'candidate_cells':np.empty((0,2),np.int32),'static_updates':[],'future_partial_graph':{'cells':[]}}
        s=score_fixed(ex,d,np.empty(0,np.uint8),[(2,0.)])
        np.testing.assert_array_equal(s['confusion'],[[0,0,0],[0,0,0],[1,0,0]])
        np.testing.assert_allclose(s['scores'],[-np.log(1e-9),2.])

    def test_graph_traversal_known_immutability_and_label_scores(self):
        known={'cell':[0,0],'state':1,'height':0.,'timestamp':0.};new={'cell':[1,0],'state':1,'height':.1,'timestamp':1.}
        ex={'graph':{'cells':[known]},'target':{(1,0):(1,.2),(2,0):(2,0.),(4,0):(2,0.)}}
        p=np.array([[.1,.8,.1],[.7,.1,.2],[.6,.2,.2]],np.float32)
        d={'probabilities':p,'states':np.array([1,0,0],np.uint8),'candidate_cells':np.array([[1,0],[2,0],[3,0]],np.int32),'static_updates':[new],'future_partial_graph':{'cells':[known,new]}}
        s=score_fixed(ex,d,np.array([1,2,0],np.uint8),[(2,0.)])
        np.testing.assert_array_equal(s['confusion'],[[1,0,0],[0,1,0],[2,0,0]])
        self.assertAlmostEqual(s['height_error'][0],.1);self.assertTrue(s['future_graph_verified'])
        mutated=copy.deepcopy(d);mutated['future_partial_graph']['cells'][0]['height']=1.
        with self.assertRaises(ValueError):score_fixed(ex,mutated,np.array([1,2,0],np.uint8),[(2,0.)])

if __name__=='__main__':unittest.main()
