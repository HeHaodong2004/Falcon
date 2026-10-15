import unittest
import torch
from eventforecastwm.model import ForecastWorldModel, trajectory_log_probability
from eventforecastwm.train_objective import likelihood_loss
from test_forecast_model import legal


class LikelihoodObjectiveTest(unittest.TestCase):
    def test_only_likelihood_contributes_gradients(self):
        from unittest.mock import patch
        torch.set_num_threads(1)
        model = ForecastWorldModel(noise='integrated')
        belief = model.encode_history(legal())
        forecast = model.imagine(belief, horizon_s=1.)
        truth = forecast.positions[:, 0].detach() + .3
        mask = forecast.valid[:, None].expand(-1, 4, -1)
        expected, count = trajectory_log_probability(forecast, truth, mask)
        expected = (-expected / count).mean()
        batch = dict(targets=dict(track_positions=truth, track_valid=mask))
        with patch('eventforecastwm.train.encode', return_value=belief):
            actual, metrics = likelihood_loss(model, batch, horizon=1.)
        self.assertTrue(torch.allclose(actual, expected))
        a = torch.autograd.grad(actual, model.log_readout, retain_graph=True)[0]
        e = torch.autograd.grad(expected, model.log_readout)[0]
        self.assertTrue(torch.equal(a, e))
        self.assertTrue(torch.isfinite(a))
        self.assertEqual(metrics['loss'], metrics['nll'])


if __name__ == '__main__':
    unittest.main()
