import unittest
import torch
from eventforecastwm.model_agent_scale import AgentScaleWorldModel, trajectory_log_probability
from eventforecastwm.refine_calibration import root_loss_sum, NAMES
from test_forecast_agent_scale import two_agents


class TestRefinement(unittest.TestCase):
    def test_cached_covariance_gradient_equals_reimagined_likelihood(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = AgentScaleWorldModel(kind='deterministic', conditional=False)
        history = two_agents()
        with torch.no_grad():
            cached = model.imagine(model.encode_history(history), horizon_s=1.)
            model.log_acceleration.add_(.2)
            model.raw_decay_rate.add_(.1)
        fresh = model.imagine(model.encode_history(history), horizon_s=1.)
        truth = fresh.positions[:, 0].detach() + .3
        mask = torch.ones(1, 4, 2, dtype=torch.bool)
        mask[:, :, 1] = False
        params = [getattr(model, name) for name in NAMES]
        loss, count = root_loss_sum(model, cached, truth, mask)
        expected_lp, expected_count = trajectory_log_probability(fresh, truth, mask)
        expected = -(expected_lp / expected_count).sum()
        self.assertEqual(count, 1)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        actual_grad = torch.autograd.grad(loss, params)
        expected_grad = torch.autograd.grad(expected, params)
        for actual, wanted in zip(actual_grad, expected_grad):
            self.assertTrue(torch.allclose(actual, wanted, atol=1e-6))
        poisoned = truth.clone()
        poisoned[:, :, 1] = 1e3
        other, _ = root_loss_sum(model, cached, poisoned, mask)
        self.assertTrue(torch.equal(loss.detach(), other.detach()))


if __name__ == '__main__':
    unittest.main()
