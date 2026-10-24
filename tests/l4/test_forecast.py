"""Frozen L2 adapter tests with exact legal (synthetic) observations."""
import unittest
import numpy as np
from eventwm_graph.l4.forecast import ForecastCache,ForecastView


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
        from eventwm_graph.l2_intermittent.api_example import inputs
        from eventwm_graph.l4.native import CHECKPOINT
        cls.model=IntermittentWorldModel.load(CHECKPOINT)
        cls.history,cls.graph,cls.row=inputs()
        cls.belief=cls.model.initialize(cls.history,cls.graph)

    def test_shared_cache_and_expiration(self):
        cache=ForecastCache(self.model);view=cache.get(self.belief,self.history[-1])
        self.assertIs(cache.get(self.belief,self.history[-1]),view)
        self.assertEqual(view.for_control(self.belief.timestamp)[0],'valid')
        self.assertEqual(view.for_control(view.valid_until+.001)[0],'expired')
        self.assertEqual(len(cache.refresh_ms),1)

    def test_future_is_not_actual_evidence(self):
        with self.assertRaises(ValueError):
            ForecastCache(self.model).get(self.belief.fork(role='imagined'),self.history[-1])

    def test_actual_measurement_not_replaced_by_forecast(self):
        from eventwm_graph.l4.executor import Executor
        old=self.belief;cache=ForecastCache(self.model);view=cache.get(old,self.history[-1])
        new=type(self).row(old.timestamp+.048,2.10)
        new['tracks'][0].update(track_id='legal',velocity_valid=False)
        actual=self.model.observe(old,new)
        people=Executor._people(new,new['timestamp'])
        self.assertEqual(people[0,0],2.10)
        self.assertEqual(actual.role,'actual')
        self.assertEqual(view.source_belief_version,old.fingerprint())
        self.assertNotEqual(actual.fingerprint(),view.source_belief_version)

    def test_foreign_lineage_rejected(self):
        view=ForecastCache(self.model).get(self.belief,self.history[-1])
        with self.assertRaises(ValueError):
            ForecastView('wrong',view.model_version,view.prediction_start_time,view.future_times,
                         view.joint_future_reference,view.valid_until,{})

    def test_canonical_segment_is_consumed_by_frozen_l2(self):
        from eventwm_graph.l4 import QuerySpec,compile_reference
        from eventwm_graph.l4.geometry import StaticGrid
        from eventwm_graph.l2_intermittent.observations import predict_observations
        grid=StaticGrid(np.ones((100,100),bool),np.zeros((100,100)),[-5,-5],.1,'segment-test')
        q=QuerySpec('canonical-segment',[[0,0,0],[2,0,0]],self.belief.robot_pose,0.,self.belief.timestamp)
        reference=compile_reference(q,grid);segment=reference.l2_segment(0,8)
        future=self.model.predict_shared(self.belief,1.92,16,221)
        before=self.belief.fingerprint()
        event=predict_observations(self.model,self.belief,segment,future)
        self.assertTrue(event['successor_supported'])
        self.assertTrue(np.allclose(event['endpoint_times'],segment['nominal_absolute_times'][1:]))
        self.assertEqual(before,self.belief.fingerprint())


if __name__=='__main__':unittest.main()
