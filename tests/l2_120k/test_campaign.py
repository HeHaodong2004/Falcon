import math
import unittest
import numpy as np
import torch

from eventwm_graph.l2_120k.model import NormalizedGraphWorldModel
from eventwm_graph.l2_120k.learning import mixture_path_energy, make_batch, objective, path_factors
from eventwm_graph.l2_120k.metrics import population_distance,population_energy,vector_energy
from eventwm_graph.l2_partial.neural import GraphDynamics
from eventwm_graph.l2_partial.state import Hypothesis,ObservedGraph
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState


def model_belief(start=0.,hidden=(.2,.3,.5),**kw):
    wm=NormalizedGraphWorldModel(kind='structured',**kw)
    h=Hypothesis(1.,np.array([[2.,0.,.4,-.1]]),np.diag([.1,.2,.3,.4])[None],np.array([start]),np.array([0]),np.array(hidden),
        intent=np.array([[.3,-.2]]),diffusion=np.array([.25]),mode=np.array([0]),relaxation=.7,height=np.zeros(1),
        hidden_mean=np.zeros((1,4)),hidden_covariance=np.eye(4)[None])
    b=StructuralBeliefState(ObservedGraph('test',0,.2,0.,{}),[h],start,np.zeros(4),model_version=wm.model_version)
    return wm,b


class ApproximationTests(unittest.TestCase):
    def test_projection_reports_all_omission(self):
        wm,b=model_belief();b=b.fork(hypotheses=[b.hypotheses[0].fork(weight=.7)],unresolved_mass=.3)
        p=wm._project(b,'test');self.assertAlmostEqual(p.hypotheses[0].weight,1.)
        self.assertEqual(p.unresolved_mass,0.);self.assertAlmostEqual(p.evidence['normalized_approximation']['omitted_fraction'],.3)
        self.assertIn('NOT_A_BOUND',p.evidence['normalized_approximation']['probability_status'])

    def test_zero_support_is_not_invented(self):
        wm,b=model_belief()
        with self.assertRaises(ValueError):wm._project(b.fork(hypotheses=[b.hypotheses[0].fork(weight=0)],unresolved_mass=1),'test')

    def test_forecast_rejects_unconverted_incomplete_measure(self):
        wm,b=model_belief()
        for residual in (.3,1e-12):
            incomplete=b.fork(hypotheses=[h.fork(weight=(1-residual)*h.weight) for h in b.hypotheses],unresolved_mass=residual)
            incomplete.validate()
            with self.assertRaisesRegex(ValueError,'explicit channel conversion'):
                wm.predict_shared(incomplete,.048,16,41)

    def test_forecast_samples_all_counts(self):
        wm,b=model_belief(hidden=(.2,.3,.5));counts=[]
        for seed in range(12):
            f=wm.predict_shared(b,0.,256,seed=seed);counts.extend(int(round(h.expected_hidden)) for h in f.worlds)
            self.assertAlmostEqual(f.weights.sum(),1.);self.assertEqual(f.unresolved_mass,0.)
        np.testing.assert_allclose(np.bincount(counts,minlength=3)/len(counts),[.2,.3,.5],atol=.035)

    def test_seed_and_immutable_state(self):
        wm,b=model_belief();finger=b.fingerprint();a=wm.predict_shared(b,1.,16,seed=42);c=wm.predict_shared(b,1.,16,seed=42)
        np.testing.assert_equal(a.positions,c.positions);self.assertEqual(b.fingerprint(),finger)
        self.assertFalse(a.positions.flags.writeable)

    def test_terminal_forecast_absorbs_without_count_support_truncation(self):
        wm,b=model_belief(hidden=np.ones(17)/17)
        b=b.fork(terminal=True);f=wm.predict_shared(b,3.,8,seed=71)
        np.testing.assert_equal(f.times,[b.timestamp]);self.assertEqual(f.unresolved_mass,0.)
        self.assertAlmostEqual(float(f.weights.sum()),1.)
        self.assertIs(wm.propagate(b,b.timestamp+3.),b)

    def test_offgrid_recursion(self):
        wm,b=model_belief(.013)
        a=wm.propagate(b,3.013);c=wm.propagate(wm.propagate(wm.propagate(b,1.013),2.013),3.013)
        for x,y in zip(a.hypotheses,c.hypotheses):
            np.testing.assert_allclose(x.mean,y.mean,atol=1e-12);np.testing.assert_allclose(x.covariance,y.covariance,atol=1e-12)

    def test_unavailable_sensor_is_propagation(self):
        wm,b=model_belief();row=dict(timestamp=.2,robot_pose=[0,0,0,0],tracks=[],static_updates=[],human_sensor_available=False)
        a=wm.observe(b,row);c=wm.propagate(b,.2)
        for x,y in zip(a.hypotheses,c.hypotheses):
            np.testing.assert_equal(x.mean,y.mean);self.assertEqual(x.weight,y.weight)


class ProperScoreTests(unittest.TestCase):
    def test_population_cardinality_and_localization(self):
        a=np.array([[0.,0.]]);self.assertEqual(population_distance(a,a),0.)
        self.assertGreater(population_distance(a,[]),0.);self.assertGreater(population_distance(a,[[2,0]]),0.)
        self.assertAlmostEqual(population_distance(a,[[2,0]]),population_distance([[2,0]],a))

    def test_population_deterministic_energy(self):
        samples=[np.array([[0.,0.]])]*4;truth=np.array([[1.,0.]])
        self.assertAlmostEqual(population_energy(samples,truth),population_distance(samples[0],truth))

    def test_population_finite_oracle_expected_propriety(self):
        a=np.array([[0.,0.]]);b=np.array([[3.,0.],[4.,0.]])
        d=population_distance(a,b);p=.3
        def expected(q):return (p*(1-q)+(1-p)*q-q*(1-q))*d
        self.assertLess(expected(p),expected(.8));self.assertLess(expected(p),expected(0.))

    def test_persistent_path_coupling_matters(self):
        a=np.array([[[0.,0.],[1,0]],[[1.,0.],[0,0]]]);b=np.array([[[0.,0.],[0,0]],[[1.,0.],[1,0]]])
        self.assertNotAlmostEqual(vector_energy(a,a[0]),vector_energy(b,a[0]))

    def test_mixture_path_es_known_deterministic_components(self):
        draws=torch.tensor([[[[[[0.,0.]],[[0.,0.]]],[[[2.,0.]],[[2.,0.]]],[[[4.,0.]],[[4.,0.]]]]]],dtype=torch.float64)
        logits=torch.zeros((1,1,3),dtype=torch.float64);truth=torch.zeros((1,1,1,2),dtype=torch.float64);valid=torch.ones((1,1,1),dtype=torch.float64)
        score=mixture_path_energy(draws,logits,truth,valid)
        self.assertAlmostEqual(float(score),2.-.5*(16/9),places=7)

    def test_training_has_finite_nonzero_gradients(self):
        c=dict(weight=1.,mean=np.array([[0,0,.4,0]],np.float32),covariance=np.diag([0,0,.2,.2])[None],
            age=np.zeros(1),graph=np.zeros((1,18)),selected=np.array([0]),
            truth=np.array([[[0,0]],[[.2,.1]],[[.4,.2]],[[.6,.4]],[[.8,.6]]],np.float32),
            maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None,maneuver_until=-1.)
        r=dict(components=[c],times=np.array([0,.5,1,1.5,2.]));net=GraphDynamics();b=make_batch([r],np.random.default_rng(2),dict(maneuver_rate=2,reset_std=.2))
        loss=objective(net,b);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(net.head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_linear_samples_match_analytic_native_moments(self):
        wm,b=model_belief(start=.013,hidden=(1.,),reset_std=.2);times=np.array([.013,.5,1.,2.,3.])
        a,L,B,J=path_factors(times,-1);h=b.hypotheses[0];rng=np.random.default_rng(13);W=50000
        init=rng.multivariate_normal(h.mean[0],h.covariance[0],W)
        x=init[:,None,:2]+a[None,:,None]*init[:,None,2:]+(times-times[0]-a)[None,:,None]*h.intent[0]
        x+=np.einsum('ij,wjd->wid',L,rng.normal(size=(W,5,2)))*np.sqrt(.25)
        x+=np.einsum('ij,wjd->wid',B,rng.normal(size=(W,5,2)))
        p=-np.expm1(-2*.048);impulse=(rng.random((W,J.shape[1],1))<p)*rng.normal(size=(W,J.shape[1],2))*.2
        x+=np.einsum('hg,wgd->whd',J,impulse)
        for i,t in enumerate(times):
            exact=wm.propagate(b,t).hypotheses[0]
            np.testing.assert_allclose(x[:,i].mean(0),exact.mean[0,:2],atol=.035)
            np.testing.assert_allclose(np.cov(x[:,i].T),exact.covariance[0,:2,:2],atol=.045)


if __name__=='__main__':unittest.main()
