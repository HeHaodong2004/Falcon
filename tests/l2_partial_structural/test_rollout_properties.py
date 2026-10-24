"""Deterministic probability, complete-path, budget and persistence properties."""
import json
import threading
import unittest
from unittest.mock import patch
import numpy as np
from eventwm_graph.l2_partial.state import BeliefState,Hypothesis,ObservedGraph,EXECUTOR_VERSION
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.structural_budget import InferenceBudgetExceeded
from eventwm_graph.l2_partial.observation import EndpointObservationLaw


class ConstantOptics(EndpointObservationLaw):
    def detection(self,p,robot_pose,regime):
        p=np.asarray(p);r=np.asarray(regime)
        prob=np.asarray([.2,.8])[r]
        if r.ndim:prob=prob.reshape(r.shape+(1,)*(p.ndim-1-r.ndim))
        return np.where(np.isfinite(p[...,0]),np.broadcast_to(prob,p.shape[:-1]),0.)


class NoContact:
    def hazards(self,features):return np.zeros(features.shape[:-1]+(3,))


def model_belief(n=1,hidden=0):
    m=StructuralGraphBeliefWorldModel(kind='memory_cv',maneuver_rate=0.,
        population_prior=[1.],observation_law=ConstantOptics(),event_model=NoContact())
    graph=ObservedGraph('structural-rollout-oracle',1,.25,0.,{(i,j):(1,0.) for i in range(-8,24) for j in range(-8,9)})
    hs=[]
    for r in (0,1):
        pmf=np.zeros(hidden+1);pmf[hidden]=1
        hs.append(Hypothesis(.5,np.tile([2.,.5,.2,0.],(n,1)),np.tile(np.diag([.2,.2,.1,.1]),(n,1,1)),
            np.zeros(n),np.arange(n),pmf,intent=np.tile([.2,0.],(n,1)),diffusion=np.full(n,.1),
            mode=np.zeros(n,int),height=np.zeros(n),visibility_mode=r,
            hidden_mean=np.array([[4.,0.,0.,0.]]),hidden_covariance=np.eye(4)[None]))
    return m,BeliefState(graph,hs,0.,np.zeros(4),model_version=m.model_version).validate()


def query(start=0.,actions=2,name='q'):
    p=[[start+i*.2,0.,0.,0.] for i in range(actions+1)]
    return {'query_id':name,'executor_version':EXECUTOR_VERSION,
        'compiled':{'planned_poses':p,'actions':['forward']*actions,'graph_revision':1},
        'waypoints':[p[-1][:3]],'final_yaw':0.}


class RolloutProperties(unittest.TestCase):
    def test_integration_adapter_preserves_bounds_and_successor_capabilities(self):
        from eventwm_graph.l2_partial.structural_adapter import summarize_batch
        m,b=model_belief();f=m.predict_shared(b,.144,num_worlds=16,seed=41)
        batch=m.evaluate_queries(b,[query()],f,budget_ms=10000.)
        summary=summarize_batch(batch)[0]
        self.assertEqual(summary.collision_probability_interval,(0.,0.))
        np.testing.assert_allclose(summary.evaluated_duration_mean_interval,(.096,.096))
        self.assertTrue(summary.native_plan_completed_within_interval)
        for branch in summary.observation_options:
            self.assertFalse(branch.supports_continuous_refinement)
            np.testing.assert_allclose(branch.posterior_expected_count_interval,(1.,1.),atol=1e-14,rtol=0)

    def test_gaussian_sampling_is_invariant_to_eigenvector_basis(self):
        from eventwm_graph.l2_partial.structural_dynamics import symmetric_gaussian_sample
        mean=np.zeros((2,4));cov=np.tile(np.diag([4.,4.,100.,100.]),(2,1,1))
        original=np.linalg.eigh
        first=symmetric_gaussian_sample(mean,cov,np.random.default_rng(41))
        def changed_basis(a):
            eig,U=original(a)
            R=np.array([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,-1.],[0.,0.,1.,0.]])
            return eig,U@R
        with patch('numpy.linalg.eigh',side_effect=changed_basis):
            other=symmetric_gaussian_sample(mean,cov,np.random.default_rng(41))
        np.testing.assert_array_equal(first,other)

    def test_direct_two_stage_observation_probability_and_posterior(self):
        m,b=model_belief()
        f=m.predict_shared(b,.144,num_worlds=16,seed=41)
        direct=m.evaluate_queries(b,[query()],f,budget_ms=10000.).predictions[0]
        d=next(x for x in direct.observation_branches if x.observation['kind']=='survived_no_detection')
        first=m.evaluate_queries(b,[query(actions=1)],f,budget_ms=10000.).predictions[0]
        a=next(x for x in first.observation_branches if x.observation['kind']=='survived_no_detection')
        f2=m.predict_shared(a.successor,.048,num_worlds=16,seed=999)
        second=m.evaluate_queries(a.successor,[query(start=.2,actions=1,name='q2')],f2,budget_ms=10000.).predictions[0]
        c=next(x for x in second.observation_branches if x.observation['kind']=='survived_no_detection')
        self.assertAlmostEqual(d.probability,.34,places=12)
        self.assertAlmostEqual(d.probability,a.probability*c.probability,places=12)
        np.testing.assert_allclose(d.successor.carrier.weights,c.successor.carrier.weights,atol=1e-14,rtol=0)
        np.testing.assert_array_equal(np.stack([h.mean for h in d.successor.hypotheses]),np.stack([h.mean for h in c.successor.hypotheses]))

    def test_coarse_posterior_marginalization_on_represented_measure(self):
        m,b=model_belief(hidden=1)
        f=m.predict_shared(b,.144,num_worlds=16,seed=41)
        p=m.evaluate_queries(b,[query()],f,budget_ms=10000.).predictions[0]
        reconstructed=sum(x.probability*x.successor.carrier.weights for x in p.observation_branches)
        np.testing.assert_allclose(reconstructed,f.weights,atol=1e-14,rtol=0)
        self.assertAlmostEqual(sum(x.probability for x in p.observation_branches),1.,places=12)

    def test_no_observation_successor_matches_analytic_transition(self):
        m,b=model_belief(hidden=2)
        f=m.predict_shared(b,3.,num_worlds=16,seed=41)
        actual=m.successor_at(f,len(f.times)-1,f.weights,np.zeros(4))
        expected=m.propagate(b,3.)
        self.assertIsNone(actual.carrier)
        for a,e in zip(actual.hypotheses,expected.hypotheses):
            np.testing.assert_array_equal(a.mean,e.mean)
            np.testing.assert_array_equal(a.covariance,e.covariance)
            np.testing.assert_array_equal(a.hidden_covariance,e.hidden_covariance)

    def test_query_count_does_not_modify_exogenous_worlds_or_source(self):
        m,b=model_belief(n=3,hidden=2)
        f=m.predict_shared(b,.144,num_worlds=16,seed=41)
        before=f.positions.copy();fp=b.fingerprint()
        single=m.evaluate_queries(b,[query()],f,budget_ms=10000.).predictions[0]
        batch=m.evaluate_queries(b,[query(name='q'+str(i)) for i in range(16)],f,budget_ms=10000.)
        np.testing.assert_array_equal(before,f.positions)
        self.assertEqual(fp,b.fingerprint())
        for p in batch.predictions:
            self.assertEqual(p.collision_interval,single.collision_interval)
            np.testing.assert_array_equal(p.duration_weights,single.duration_weights)

    def test_graph_cell_storage_and_siblings_are_immutable(self):
        m,b=model_belief()
        b=StructuralBeliefState.wrap(b)
        with self.assertRaises(TypeError):b.graph.cells[(0,0)]=(2,0.)
        g=b.graph.update([{'cell':[0,0],'state':2,'height':0.,'source':'legal_sensor'}])
        self.assertEqual(b.graph.get((0,0))[0],1)
        self.assertEqual(g.get((0,0))[0],2)
        with self.assertRaises(TypeError):g.cells[(1,1)]=(2,0.)

    def test_invalid_sensor_does_not_certify_geometry(self):
        m,b=model_belief()
        row={'timestamp':.048,'robot_pose':[0,0,0,0],'observation_valid':False,
             'static_updates':[{'cell':[0,0],'state':2,'height':0.,'source':'invalid'}]}
        p=m.observe(b,row)
        self.assertEqual(p.graph.get((0,0))[0],1)

    def test_zero_budget_is_atomic_and_includes_lock_wait(self):
        m,b=model_belief();before=b.to_dict()
        row={'timestamp':.048,'robot_pose':[0,0,0,0],'tracks':[]}
        with self.assertRaises(InferenceBudgetExceeded) as ctx:m.online_call(b,row,[],budget_ms=0.)
        self.assertTrue(ctx.exception.pending_observation)
        self.assertEqual(b.to_dict(),before)
        ready=threading.Event();release=threading.Event()
        def hold():
            with m._instance_lock:ready.set();release.wait(.2)
        worker=threading.Thread(target=hold);worker.start();ready.wait()
        try:
            with self.assertRaises(InferenceBudgetExceeded) as ctx:m.online_call(b,row,[],budget_ms=2.)
            self.assertEqual(ctx.exception.stage,'instance_lock')
            self.assertTrue(ctx.exception.pending_observation)
        finally:release.set();worker.join()

    def test_unresolved_bounds_are_not_arbitrary_point_event_laws(self):
        m,b=model_belief();b=b.fork(hypotheses=[h.fork(weight=h.weight*.9) for h in b.hypotheses],unresolved_mass=.1)
        f=m.predict_shared(b,.144,num_worlds=16,seed=41)
        p=m.evaluate_queries(b,[query()],f,budget_ms=10000.).predictions[0]
        self.assertEqual(p.collision_interval,(0.,.1))
        self.assertEqual(p.collision_probability,.1)
        self.assertAlmostEqual(sum(p.termination_probabilities.values()),1.)
        self.assertAlmostEqual(sum(x.probability for x in p.observation_branches),1.)
        for x in p.observation_branches:
            if x.observation['kind']!='unresolved_physical_outcome':
                self.assertGreater(x.successor.unresolved_mass,0.)


if __name__=='__main__':unittest.main()
