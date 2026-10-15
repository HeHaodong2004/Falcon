import unittest
import torch
from eventforecastwm.model import ForecastWorldModel
from test_forecast_model import legal


class TestRobotCommandIndependence(unittest.TestCase):
    def test_past_ego_command_does_not_change_human_physics(self):
        torch.set_num_threads(1);torch.manual_seed(21)
        model=ForecastWorldModel(noise='integrated');inputs=legal()
        first=model.imagine(model.encode_history(inputs),horizon_s=1.)
        inputs.robot[:,:,9:11]=float('nan')
        second=model.imagine(model.encode_history(inputs),horizon_s=1.)
        self.assertTrue(torch.equal(first.positions,second.positions))
        self.assertTrue(torch.equal(first.weights,second.weights))


if __name__=='__main__':unittest.main()
