"""Regression witnesses for persistent actual evidence and stochastic ownership."""
import json
import unittest
import numpy as np
from eventwm_graph.l2_partial.state import Hypothesis, ObservedGraph
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel, sealed


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        self.model=IntermittentWorldModel(max_components=8)
        graph=ObservedGraph('synthetic-regression',0,.2,0.,{})
        h=Hypothesis(1.,np.array([[2.,1.,.5,0.]]),np.diag([.04,.04,.1,.1])[None],np.array([0.]),np.array([0]),np.array([.5,.5]),birth_time=np.array([0.]))
        source=StructuralBeliefState(graph,[h],0.,np.zeros(4),model_version='legal-test-bootstrap')
        self.belief=self.model.from_legacy(source,seed=17)

    @staticmethod
    def observation(t,x=2.2,available=True):
        return dict(timestamp=t,robot_pose=[0.,0.,0.,0.],static_updates=[],human_sensor_available=available,
            tracks=[dict(position=[x,0.,1.],visible_now=True,valid=True,last_seen_time=t)] if available else [])

    def test_real_evidence_after_explicit_advance_and_duplicate_receipt(self):
        b=self.model.observe(self.belief,self.observation(.1))
        arrived=self.observation(1.1,2.5)
        direct=self.model.observe(b,arrived)
        advanced=self.model.observe(self.model.propagate(b,1.1),arrived)
        self.assertEqual(direct.fingerprint(),advanced.fingerprint())
        self.assertEqual(direct.fingerprint(),self.model.observe(direct,arrived).fingerprint())
        with self.assertRaises(ValueError):self.model.observe(direct,self.observation(1.1,2.6))

    def test_unavailable_is_no_evidence_and_can_precede_arrival(self):
        b=self.model.observe(self.belief,self.observation(.1))
        missing=self.model.observe(b,self.observation(1.1,available=False))
        pure=self.model.propagate(b,1.1)
        self.assertEqual(missing.fingerprint(),pure.fingerprint())
        post=self.model.observe(missing,self.observation(1.1,2.5))
        self.assertEqual(post.fingerprint(),self.model.observe(b,self.observation(1.1,2.5)).fingerprint())

    def test_tracks_outlive_source_expiry(self):
        b=self.model.propagate(self.belief,5.)
        tracks=self.model.get_previously_observed_tracks(b)
        self.assertEqual(len(tracks),1)
        self.assertAlmostEqual(tracks[0]['age'],5.)
        self.assertAlmostEqual(tracks[0]['presence_probability'],1.)

    def test_full_component_law_survives_partitions(self):
        direct=self.model.propagate(self.belief,3.)
        segmented=self.belief
        for t in (.013,.054,.231,.5,1.003,1.5,2.231,3.):segmented=self.model.propagate(segmented,t)
        for a,b in zip(direct.components,segmented.components):
            np.testing.assert_allclose(a.mean,b.mean,rtol=0,atol=1e-12)
            np.testing.assert_allclose(a.covariance,b.covariance,rtol=0,atol=1e-12)
            for field in ('next_switch','stopped','counters','intent','slots'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))

    def test_zero_duration_hidden_tape_keeps_velocity_on_extension(self):
        short=self.model.predict_shared(self.belief,0.,16,19)
        child=self.model.successor_at(short,0)
        continued=self.model.predict_shared(child,1.,16,777,times=[0.,1.])
        direct=self.model.predict_shared(self.belief,1.,16,19,times=[0.,1.])
        np.testing.assert_array_equal(continued.hidden_positions,direct.hidden_positions)
        np.testing.assert_array_equal(continued.positions,direct.positions)

    def test_projection_reuses_world_membership(self):
        f=self.model.predict_shared(self.belief,1.,16,11,times=[0.,1.])
        result=self.model.project_future_distribution(f,[[-1e8,-1e8,1e8,1e8]],f.times)
        np.testing.assert_allclose(result['observed_track_expected_count'],1.)
        np.testing.assert_allclose(result['at_least_one_probability'],1.)
        self.assertEqual(f.identity(),f._digest())
        with self.assertRaises(ValueError):f.positions.setflags(write=True)

    def test_actual_serialization_preserves_law_and_rejects_bad_slots(self):
        raw=json.loads(json.dumps(self.model.serialize_actual_belief(self.belief)))
        roundtrip=self.model.deserialize_actual_belief(raw)
        self.assertEqual(roundtrip.fingerprint(),self.belief.fingerprint())
        raw['components'][0]['slots']['data']=[.5];raw['components'][0]['slots']['dtype']='float64'
        with self.assertRaises(ValueError):self.model.deserialize_actual_belief(raw)

    def test_float_world_count_rejected_and_imagined_actual_update_refused(self):
        with self.assertRaises(ValueError):self.model.predict_shared(self.belief,1.,16.)
        f=self.model.predict_shared(self.belief,1.,16,1,times=[0.,1.]);child=self.model.successor_at(f,1)
        with self.assertRaises(ValueError):self.model.observe(child,self.observation(1.,2.5))

    def test_mutable_external_copy_cannot_inherit_cached_fingerprint(self):
        import copy
        self.assertIsNotNone(self.belief._cached_fingerprint)
        # Sharing an immutable graph is valid; copied NumPy state is writable.
        external=copy.deepcopy(self.belief,{id(self.belief.graph):self.belief.graph})
        self.model._check(external)
        self.assertTrue(external.components[0].mean.flags.writeable)
        first=external.fingerprint();future=self.model.predict_shared(external,1.,16,7,times=[0.,1.])
        external.components[0].mean[0,0]+=.1
        self.assertNotEqual(external.fingerprint(),first)
        with self.assertRaises(ValueError):self.model._check_future(future)
        clone=self.belief.fork(timestamp=.1)
        self.assertIsNone(clone._cached_fingerprint)


if __name__=='__main__':unittest.main()
