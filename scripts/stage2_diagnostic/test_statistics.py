"""Small independent correctness checks; no data/model inference."""
import unittest
import numpy as np
from .common import THRESHOLDS,confusion,EDGES
from .probability_audit import threshold_confusions,probability_histogram
from .statistics import rank_metrics

class StatisticsTest(unittest.TestCase):
    def test_thresholds_random_and_float_boundaries(self):
        rng=np.random.default_rng(2525);p=rng.dirichlet([2,1,1],1000).astype(np.float32)
        for t in THRESHOLDS:
            f=np.float32(t)
            p=np.r_[p,np.array([[1-v,v,0] for v in (np.nextafter(f,np.float32(0)),f,np.nextafter(f,np.float32(1)))],np.float32)]
        p=np.r_[p,np.array([[.2,.4,.4],[.5,.5,0],[1,0,0],[0,.5,.5]],np.float32)]
        y=rng.integers(0,3,len(p),dtype=np.uint8);actual=threshold_confusions(y,p)
        r=(p[:,1]+p[:,2]).astype(np.float64)
        for j,t in enumerate(THRESHOLDS):
            pred=np.where(r>=t,np.where(p[:,1]>=p[:,2],1,2),0)
            np.testing.assert_array_equal(actual[j],confusion(y,pred))

    def test_rank_metrics_and_histogram_bounds(self):
        rng=np.random.default_rng(2525);p=rng.dirichlet([10,1,1],10000).astype(np.float32);y=rng.integers(0,3,len(p),dtype=np.uint8)
        h=probability_histogram(y,p)
        np.testing.assert_array_equal(h.sum((1,2)),np.repeat(len(y),4))
        r=rank_metrics(h[3,0],h[3,1]+h[3,2]);truth=y>0;score=p[:,1]+p[:,2]
        order=np.argsort(-score,kind='stable');t=truth[order];s=score[order]
        ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
        tp=np.r_[0,np.cumsum(t)[ends]];fp=np.r_[0,np.cumsum(~t)[ends]]
        exact_auc=float(np.trapz(tp/t.sum(),fp/(~t).sum()))
        exact_ap=float(np.sum(np.diff(tp/t.sum())*tp[1:]/(tp[1:]+fp[1:])))
        for exact,bounds in ((exact_auc,r['auroc_binning_bounds']),
                             (exact_ap,r['auprc_binning_bounds'])):
            self.assertLessEqual(bounds[0]-1e-12,exact);self.assertGreaterEqual(bounds[1]+1e-12,exact)
        np.testing.assert_allclose(r['prevalence'],truth.mean(),atol=1e-12)

if __name__=='__main__':unittest.main()
