import unittest
import torch
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.ablate_transition import replace_transition
from test_forecast_model import legal


class TestTransitionAblation(unittest.TestCase):
    def test_initial_belief_and_covariance_preserved(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        model=ForecastWorldModel(noise='integrated')
        with torch.no_grad():model.core.acceleration[-1].bias.fill_(.2)
        before=model.encode_history(legal());full=model.imagine(before,horizon_s=1.)
        replace_transition(model);after=model.encode_history(legal());ablated=model.imagine(after,horizon_s=1.)
        self.assertTrue(torch.equal(before.core.positions,after.core.positions))
        self.assertTrue(torch.equal(before.core.hidden,after.core.hidden))
        self.assertTrue(torch.equal(full.weights,ablated.weights))
        self.assertTrue(torch.equal(full.covariance,ablated.covariance))
        self.assertFalse(torch.equal(full.positions,ablated.positions))


if __name__=='__main__':unittest.main()
