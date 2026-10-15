import unittest
import torch
from eventforecastwm.model_kinematic import KinematicForecastWorldModel
from test_forecast_model import legal


class TestKinematicBound(unittest.TestCase):
    def test_residual_bound_changes_motion_and_recipe_roundtrips(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        low=KinematicForecastWorldModel(max_acceleration=2.,noise='integrated')
        high=KinematicForecastWorldModel(max_acceleration=4.,noise='integrated')
        high.load_state_dict(low.state_dict())
        # The bound must affect the actual recurrent mean, not only metadata.
        for model in (low,high):
            with torch.no_grad():
                model.core.acceleration[-1].weight.zero_()
                model.core.acceleration[-1].bias.fill_(.4)
        a=low.imagine(low.encode_history(legal()),horizon_s=.25)
        b=high.imagine(high.encode_history(legal()),horizon_s=.25)
        self.assertFalse(torch.equal(a.positions,b.positions))
        restored=KinematicForecastWorldModel(**high.recipe)
        restored.load_state_dict(high.state_dict())
        c=restored.imagine(restored.encode_history(legal()),horizon_s=.25)
        self.assertTrue(torch.equal(b.positions,c.positions))
        self.assertEqual(low.parameter_counts()['total'],high.parameter_counts()['total'])


if __name__=='__main__':unittest.main()
