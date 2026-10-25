import unittest
import math
import numpy as np
from .pose_features import signed_pose_features
from eventwm_graph.stage2_segment.features import SegmentFeatures


class SuppliedHeadingInformation(unittest.TestCase):
    def test_future_summaries_can_lose_starting_view_side(self):
        graph=dict(schema='causal_partial_cspace_graph_1',resolution=.1,height=0.,clearance=.25,timestamp=0.,revision=1,cells=[dict(cell=[0,0],state=1,height=0.)])
        keys=np.array([[80,0]],np.int32);theta=math.atan2(.05,8.05);delta=.4;a=.6-theta;b=-2*theta-a-4*delta/3
        p=np.zeros((4,5));p[:,0]=np.arange(4)*.048;p[:,4]=a+np.arange(4)*delta/3;q=p.copy();q[:,4]=b+np.arange(4)*delta/3
        x=SegmentFeatures(graph,p).transform(keys,'both')[0][:,16:];y=SegmentFeatures(graph,q).transform(keys,'both')[0][:,16:]
        np.testing.assert_allclose(x,y,rtol=0,atol=2e-7)
        px=signed_pose_features(keys,p,.1,x);qy=signed_pose_features(keys,q,.1,y)
        self.assertGreater(px[0,0],math.sqrt(.5));self.assertLess(qy[0,0],math.sqrt(.5));self.assertGreater(abs(float(px[0,1]-qy[0,1])),1.)

    def test_start_only_and_stationary_later_are_distinct(self):
        keys=np.array([[-1,0]],np.int32);x=np.zeros((1,88),np.float32);p=np.array([[1,0,0,0,0]],np.float64)
        a=signed_pose_features(keys,p,.1,x);q=np.vstack([p,p]);q[1,0]+=.048;q[1,4]=.2;b=signed_pose_features(keys,q,.1,x)
        self.assertFalse(np.array_equal(a,b));self.assertTrue(np.isfinite(b).all())

    def test_cuda_matches_reference_with_negative_cells_and_different_lengths(self):
        import torch
        from .pose_cuda import PoseKernel
        rng=np.random.default_rng(48);kernel=PoseKernel('cuda:0')
        for length in [1,2,7,41]:
            keys=rng.integers(-300,300,(503,2),dtype=np.int32);path=rng.normal(size=(length,5));path[:,0]=np.arange(length)*.048;x=rng.normal(size=(len(keys),88)).astype(np.float32)
            expected=signed_pose_features(keys,path,.101688,x)
            actual=kernel(keys,path,.101688,torch.from_numpy(x).cuda()).cpu().numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)

    def test_real_api_single_batch_and_training_forward(self):
        import copy
        import torch
        from .diagnose import OUT
        from .pose_api import PosePredictor
        from .features import transform_batch
        from eventwm_graph.stage2_segment.benchmark import inputs
        from eventwm_graph.stage2_segment.update import apply_graph_delta
        m=PosePredictor(OUT/'training/signed_pose_pilot_checked/epoch02.pt')
        item=inputs('multi')[0];g=item['graph'];before=copy.deepcopy(g);paths=[np.asarray(p['trajectory'],np.float64)for p in item['paths'][:6]]
        batch=m.predict_segments(partial_graph=g,trajectories=paths)
        for path,b in zip(paths,batch):
            single=m.predict(partial_graph=g,trajectory=path)
            np.testing.assert_array_equal(b['candidate_cells'],single['candidate_cells']);np.testing.assert_array_equal(b['states'],single['states'])
            np.testing.assert_allclose(b['probabilities'],single['probabilities'],atol=2e-6,rtol=2e-5)
            self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(g,b));self.assertEqual(b['predictor_schema'],'stage2_signed_pose_1')
        p=paths[0];prepared=m.prepare_graph(g);ctx=m.feature_class(g,p,prepared);keys=ctx.candidates('both');xx,_,_=transform_batch(m.cuda,[ctx],[keys]);x=xx[0];extra=signed_pose_features(keys,p,prepared.resolution,x.cpu().numpy())
        with torch.inference_mode():expected,_=m.model.probabilities(x,torch.from_numpy(extra).cuda())
        np.testing.assert_allclose(batch[0]['probabilities'],expected.cpu().numpy(),atol=2e-6,rtol=2e-5);self.assertEqual(g,before)


if __name__=='__main__':unittest.main()
