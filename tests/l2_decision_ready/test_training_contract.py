import math
import unittest
import numpy as np
import torch
from dataclasses import replace
from eventwm_graph.l2_partial.training_contract import (
    MotionContract,native_step_position_nll,root_balanced_mean,
    NativeEventTarget,first_event_likelihood_interval,negative_log_score_interval,
    require_point_training_law,iter_versioned_root_shards,
)
from eventwm_graph.l2_partial.state import Hypothesis
from eventwm_graph.l2_partial.maneuver import transition_components


class TrainingCompatibilityTests(unittest.TestCase):
    def tensors(self):
        return [torch.tensor(v,dtype=torch.float64) for v in (
            [[1.,2.,.4,-.2]],np.diag([.1,.2,.3,.4])[None],
            [[[.4,-.2],[0.,0.]]],[[.25,.1]],[[.5,-.2]],[[1.03,1.98]])]

    def test_one_step_loss_matches_inference_mixture_density(self):
        mean,cov,targets,diff,logits,z=self.tensors();c=MotionContract()
        losses=native_step_position_nll(mean,cov,targets,diff,logits,z,start_timestamp=0.,contract=c)
        density=0.;modes=torch.softmax(logits,-1).numpy()[0]
        for k in range(2):
            h=Hypothesis(1.,mean.numpy(),cov.numpy(),np.zeros(1),np.zeros(1,int),np.ones(1),intent=targets.numpy()[:,k],diffusion=diff.numpy()[:,k],relaxation=.7)
            m,P,p=transition_components(h,.048,c.maneuver_rate,c.jump_std,retention=c.retention)
            for j in range(2):
                delta=z.numpy()[0]-m[0,j,:2];C=P[0,j,:2,:2]
                density+=modes[k]*p[j]*math.exp(-.5*delta@np.linalg.solve(C,delta))/(2*math.pi*math.sqrt(np.linalg.det(C)))
        self.assertAlmostEqual(float(losses),-math.log(density),places=11)

    def test_single_batch_has_finite_parameter_gradients(self):
        args=self.tensors()
        for i in (2,3,4):args[i].requires_grad_()
        loss=native_step_position_nll(*args,start_timestamp=.096).mean();loss.backward()
        for i in (2,3,4):self.assertTrue(torch.isfinite(args[i].grad).all())

    def test_zero_maneuver_matches_ou_transition(self):
        args=self.tensors();c=replace(MotionContract(),maneuver_rate=0.)
        a=native_step_position_nll(*args,start_timestamp=0.,contract=c)
        b=native_step_position_nll(*args,start_timestamp=0.,contract=replace(c,jump_std=100.))
        torch.testing.assert_close(a,b)

    def test_off_grid_training_refuses_to_redraw_maneuver(self):
        with self.assertRaises(ValueError):native_step_position_nll(*self.tensors(),start_timestamp=.013)

    def test_root_balancing_does_not_overweight_siblings(self):
        self.assertEqual(float(root_balanced_mean(torch.tensor([0.,0.,6.]),torch.tensor([0,0,1]))),3.)

    def test_competing_event_target_has_correct_prefix_likelihood(self):
        h=np.tile([[[.1,.2],[.2,.1],[.1,.1]]],(2,1,1))
        target=NativeEventTarget(3,2,1,0,'collision_stop')
        lo,hi=first_event_likelihood_interval(h,[.3,.5],.2,target)
        self.assertAlmostEqual(lo,.8*.7*.2);self.assertAlmostEqual(hi,lo+.2)
        scores=negative_log_score_interval((lo,hi));self.assertLess(scores[0],scores[1])

    def test_censoring_retains_full_planned_horizon_and_only_observed_exposure(self):
        h=np.tile([[[.1],[.2],[.3]]],(1,1,1))
        lo,hi=first_event_likelihood_interval(h,[1.],0.,NativeEventTarget(3,1,censoring='administrative'))
        self.assertEqual((lo,hi),(.9,.9))
        with self.assertRaises(ValueError):first_event_likelihood_interval(h,[1.],0.,NativeEventTarget(3,1))
        with self.assertRaises(ValueError):first_event_likelihood_interval(h,[1.],0.,NativeEventTarget(3,3,0,0,'collision_stop'))

    def test_structural_upper_endpoint_cannot_enter_point_training(self):
        with self.assertRaises(ValueError):require_point_training_law(residual_semantics='upper_bound')
        with self.assertRaises(ValueError):require_point_training_law(residual_semantics='legacy_005')
        self.assertEqual(require_point_training_law(residual_semantics='fully_specified_normalized',normalized_latent_law_version='test-1'),'test-1')

    def test_invalid_censoring_hazard_and_fractional_exposure_are_refused(self):
        for h,target in ((np.array([[[1+5e-13]]]),NativeEventTarget(1,1)),
                (np.array([[[.1]]]),NativeEventTarget(1,1,censoring='invented')),
                (np.array([[[.1]]]),NativeEventTarget(1,1.0)),
                (np.array([[[.1]]]),NativeEventTarget(1,1,0,.5,'collision_stop'))):
            with self.subTest(target=target),self.assertRaises(ValueError):first_event_likelihood_interval(h,[1.],0.,target)

    def test_shard_iterator_keeps_one_independent_future_per_root(self):
        r={'root_id':'a','scene_id':'s','split':'dev','schema':'observation_limited_query_conditioned_root_1','query_ids':['q1','q2'],'versions':{},'files':{}}
        out=list(iter_versioned_root_shards({'rows':[r]},split='dev'))
        self.assertEqual(out[0]['independent_human_futures'],1)
        self.assertEqual(out[0]['query_ids'],('q1','q2'))
        with self.assertRaises(ValueError):list(iter_versioned_root_shards({'rows':[r]},split='holdout'))


if __name__=='__main__':unittest.main()
