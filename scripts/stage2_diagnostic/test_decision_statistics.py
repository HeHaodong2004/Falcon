"""Independent count-statistic/weighting checks on hand-constructed examples."""
import unittest
import numpy as np
from .decision_relevance import moments,rates,window_weights

class DecisionStatisticsTest(unittest.TestCase):
    def test_hierarchical_weights(self):
        roots=[dict(scene_id='a',records=[{'id':0}]),dict(scene_id='a',records=[{'id':1},{'id':2},{'id':3}]),
               dict(scene_id='b',records=[{'id':4}])]
        weights={r['id']:w for r,w in window_weights(roots,{'a','b'})}
        self.assertAlmostEqual(sum(weights.values()),1)
        self.assertAlmostEqual(weights[0],.25);self.assertAlmostEqual(weights[4],.5)
        self.assertAlmostEqual(sum(weights[i] for i in (1,2,3)),.25)

    def test_correlation_and_bias(self):
        y=np.array([[1,2,3],[3,4,7],[5,8,11]],np.float64);x=2*y+1
        r=rates(np.mean([moments(a,b) for a,b in zip(x,y)],axis=0))
        for j,k in enumerate(('revelation','free','blocked')):
            self.assertAlmostEqual(r[k]['pearson'],1)
            self.assertAlmostEqual(r[k]['bias'],(x-y).mean(0)[j])
            self.assertAlmostEqual(r[k]['mae'],abs(x-y).mean(0)[j])
        constant=np.repeat([[100,10,90]],3,axis=0)
        c=rates(np.mean([moments(a,b) for a,b in zip(constant,y)],axis=0))
        self.assertTrue(all(v['pearson'] is None for v in c.values()))

if __name__=='__main__':unittest.main()
