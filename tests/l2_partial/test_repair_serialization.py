"""Persistence checks for newly added latent/hidden/coarse-observation state."""
import json
import unittest

import numpy as np

from eventwm_graph.l2_partial.state import BeliefState
from test_observation_repair import _belief, _wm, _query, _observation


class RepairSerializationTests(unittest.TestCase):
    def roundtrip(self, belief):
        restored=BeliefState.from_dict(json.loads(json.dumps(belief.to_dict())))
        self.assertEqual(belief.to_dict(),restored.to_dict())
        return restored

    def test_retained_partial_interval_replays_same_future(self):
        model=_wm(maneuver_rate=2.)
        original=model.propagate(_belief(),.019)
        self.assertTrue(all(h.maneuver_mean is not None for h in original.hypotheses))
        restored=self.roundtrip(original)
        a=model.predict_shared(original,.125,16,seed=191,times=[0.,.029,.077,.125])
        b=model.predict_shared(restored,.125,16,seed=191,times=[0.,.029,.077,.125])
        np.testing.assert_array_equal(a.positions,b.positions)
        np.testing.assert_array_equal(a.weights,b.weights)

    def test_private_hidden_and_coarse_refinement_survive_restore(self):
        model=_wm(maneuver_rate=2.)
        prior=_belief()
        prior=prior.fork(hypotheses=[h.fork(hidden_pmf=np.array([0.,1.]),
            hidden_mean=np.array([[1.,1.,.2,0.]]),hidden_covariance=np.eye(4)[None]*.01,
            hidden_exchangeable=False) for h in prior.hypotheses])
        query=_query(3)
        future=model.predict_shared(prior,.144,16,seed=181)
        prediction=model.evaluate_queries(prior,[query],future).predictions[0]
        branch=next(b for b in prediction.observation_branches if b.observation['kind']=='survived_detection')
        original=branch.successor
        restored=self.roundtrip(original)
        self.assertTrue(all(not h.hidden_exchangeable for h in restored.hypotheses))
        self.assertTrue(all(h.coarse_observation_time==original.timestamp for h in restored.hypotheses))
        for h in restored.hypotheses:self.assertEqual(h.n+h.expected_hidden,2.)
        a=model.predict_shared(original,.096,16,seed=183)
        b=model.predict_shared(restored,.096,16,seed=183)
        np.testing.assert_array_equal(a.positions,b.positions)
        np.testing.assert_array_equal(a.weights,b.weights)
        row=_observation(original.timestamp,True,original.robot_pose.tolist(),
                         tuple(original.hypotheses[0].mean[0,:2]))
        updated=model.observe(original,row)
        replay=model.observe(restored,row)
        self.assertEqual(updated.to_dict(),replay.to_dict())

    def test_restored_terminal_is_absorbing(self):
        model=_wm()
        terminal=model.propagate(_belief(),.019).fork(terminal=True,
            terminal_event={'times':[.019],'probabilities':[1.],'poses':[[0.,0.,0.,0.]]})
        restored=self.roundtrip(terminal)
        self.assertIs(model.observe(restored,_observation(.1)),restored)
        # Absorption requires identical probabilistic state, not Python object
        # identity: the public version-binding wrapper may return a light fork.
        self.assertEqual(model.propagate(restored,1.).to_dict(),restored.to_dict())


if __name__=='__main__':unittest.main()
