"""Regression parity for the final full-state and integration optimization."""
import unittest
from unittest.mock import patch
import numpy as np
from eventwm_graph.l2_partial.state import BeliefState,Hypothesis,ObservedGraph
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_validation import validate_batched


class CompiledRolloutParity(unittest.TestCase):
    def test_complete_materialization_does_not_create_roundoff_residual(self):
        from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
        m=StructuralGraphBeliefWorldModel(population_prior=[.2,.5,.3])
        history=[{'timestamp':0.,'robot_pose':[0,0,0,0],'tracks':[],'static_updates':[]}]
        graph={'revision':0,'timestamp':0.,'resolution':.2,'height':0.,'cells':[]}
        b=m.initialize(history,graph)
        for i in range(1,21):
            row={'timestamp':i*.048,'robot_pose':[0,0,0,0],'tracks':[],'static_updates':[]}
            b=m.observe(b,row)
            self.assertEqual(b.unresolved_mass,0.)
            f=m.predict_shared(b,.144,num_worlds=16,seed=i)
            self.assertEqual(f.unresolved_mass,0.)

    def test_invalid_maneuver_parameters_rejected_before_any_kernel(self):
        for optimized in (False,True):
            for key,value in (('maneuver_rate',-1.),('maneuver_rate',float('nan')),
                              ('reset_std',-1.),('reset_std',float('inf')),
                              ('maneuver_retention',-0.1),('maneuver_retention',1.1)):
                with self.subTest(optimized=optimized,key=key,value=value),self.assertRaises(ValueError):
                    StructuralGraphBeliefWorldModel(optimized=optimized,**{key:value})

    def test_recursive_budget_covers_forecast_and_keeps_carrier_world_count(self):
        from test_rollout_properties import model_belief,query
        from eventwm_graph.l2_partial.structural_budget import ACTIVE_DEADLINE,InferenceBudgetExceeded
        m,b=model_belief();f=m.predict_shared(b,.144,num_worlds=8,seed=41)
        child=m.successor_at(f,1,f.weights*.5,np.zeros(4))
        prediction=m.continue_rollout(child,query(),budget_ms=10000.,horizon=.048)
        self.assertEqual(prediction.query_id,'q')
        def expire(*args,**kwargs):
            deadline=ACTIVE_DEADLINE.get()
            self.assertIsNotNone(deadline)
            raise InferenceBudgetExceeded('forecast:injected',10.,deadline.budget_ms,deadline.pending_observation)
        with patch.object(m,'predict_shared',side_effect=expire):
            with self.assertRaises(InferenceBudgetExceeded) as ctx:m.continue_rollout(child,query(),budget_ms=1.)
        self.assertFalse(ctx.exception.pending_observation)

    def test_every_clock_person_latent_and_rng_draw_is_retained(self):
        for start in (0.,.013):
            for retention in (0.,.3,1.):
                for hidden in (0,2):
                    with self.subTest(start=start,retention=retention,hidden=hidden):
                        kwargs=dict(kind='structured',population_prior=[1.],maneuver_retention=retention)
                        fast=StructuralGraphBeliefWorldModel(**kwargs,optimized=True)
                        slow=StructuralGraphBeliefWorldModel(**kwargs,optimized=False)
                        pmf=np.zeros(hidden+1);pmf[hidden]=1.
                        hs=[]
                        for rate,n,w in ((0.,0,.2),(.7,3,.8)):
                            hs.append(Hypothesis(w,np.tile([2.,.4,.2,-.1],(n,1)),np.tile(np.eye(4)*.1,(n,1,1)),
                                np.full(n,start),np.arange(n),pmf.copy(),intent=np.zeros((n,2)),
                                diffusion=np.full(n,.25),mode=np.zeros(n,int),relaxation=rate,height=np.zeros(n),
                                hidden_mean=np.array([[4.,2.,.1,0.]]),hidden_covariance=np.diag([10.,10.,1.,1.])[None]))
                        b=BeliefState(ObservedGraph('integration-parity',0,.2,0.,{}),hs,start,np.zeros(4),model_version=fast.model_version)
                        times=np.unique(np.r_[0.,np.arange(1,63)*.048,1.,1.5,2.,3.])
                        a=fast.predict_shared(b,3.,num_worlds=16,seed=41,times=times)
                        e=slow.predict_shared(b,3.,num_worlds=16,seed=41,times=times)
                        np.testing.assert_allclose(a.positions,e.positions,atol=2e-8,rtol=2e-9)
                        np.testing.assert_allclose(a.velocities,e.velocities,atol=2e-8,rtol=2e-9)
                        np.testing.assert_array_equal(a.weights,e.weights)
                        np.testing.assert_array_equal(a.times,e.times)
                        self.assertEqual(a.rng_state,e.rng_state)


def load_tests(loader,tests,pattern):
    from scripts.l2_partial_structural_validation_kernel import fixed_cases
    suite=unittest.TestSuite([tests])
    for name,b,legacy,strengthened in fixed_cases():
        def check(b=b,expected=strengthened):
            if expected:
                if validate_batched(b) is not b:raise AssertionError('validation replaced state')
            else:
                try:validate_batched(b)
                except (AssertionError,ValueError,IndexError):return
                raise AssertionError('invalid state accepted')
        suite.addTest(unittest.FunctionTestCase(check,description='packed validation '+name))
    return suite


if __name__=='__main__':unittest.main()
