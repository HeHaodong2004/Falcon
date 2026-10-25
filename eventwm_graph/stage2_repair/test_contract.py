import unittest
import numpy as np
from eventwm_graph.stage2_hf.decisions import observation_probabilities


class ObservationHorizon(unittest.TestCase):
    def test_start_only_forces_unknown_without_changing_original(self):
        p=np.array([[.01,.6,.39],[.02,.4,.58]],np.float32);original=p.copy()
        q=observation_probabilities(p,no_future_observations=True)
        np.testing.assert_array_equal(q,[[1,0,0],[1,0,0]])
        np.testing.assert_array_equal(p,original)

    def test_stationary_later_observations_keep_predictions(self):
        path=np.array([[0,1,2,3,0],[.048,1,2,3,0]],np.float64)
        p=np.array([[.01,.6,.39]],np.float32)
        q=observation_probabilities(p,no_future_observations=len(path)==1)
        np.testing.assert_array_equal(q,p)

    def test_empty_domain(self):
        self.assertEqual(observation_probabilities(np.empty((0,3)),no_future_observations=True).shape,(0,3))


if __name__=='__main__':unittest.main()
