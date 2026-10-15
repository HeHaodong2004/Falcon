import unittest
import torch
from eventforecastwm.model_map_posterior import MapPosteriorForecastWorldModel
from eventforecastwm.model_damped import DampedForecastWorldModel
from test_forecast_model import legal


class TestCombinedMean(unittest.TestCase):
    def test_calibration_preserves_mean_and_reusable_state(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        model=MapPosteriorForecastWorldModel(noise='integrated').eval()
        # The physical acceleration output starts at zero; exercise the
        # upstream map gradient after a nonzero physical head has been learned.
        with torch.no_grad():model.core.acceleration[-1].weight.fill_(.05)
        recipe=dict(model.recipe);recipe.pop('variant')
        calibrated=DampedForecastWorldModel(**recipe,mean_variant='map_posterior_v1').eval()
        mismatch=calibrated.load_state_dict(model.state_dict(),strict=False)
        self.assertEqual(mismatch.missing_keys,['raw_decay_rate']);self.assertFalse(mismatch.unexpected_keys)
        geometry=dict(features=torch.randn(1,2,32,32),support=torch.ones(1,dtype=torch.bool))
        a=model.encode_history(legal(),geometry);b=calibrated.encode_history(legal(),geometry)
        self.assertTrue(torch.equal(a.core.positions,b.core.positions))
        first=calibrated.imagine(b,horizon_s=1.)
        second=calibrated.imagine(first.next_belief,horizon_s=2.)
        whole=calibrated.imagine(b,horizon_s=3.)
        self.assertTrue(torch.equal(whole.positions[:,:,4:],second.positions))
        self.assertTrue(torch.equal(model.imagine(a,horizon_s=3.).positions,whole.positions))
        whole.positions.square().mean().backward()
        self.assertGreater(float(calibrated.core.map_patch_encoder[0].weight.grad.norm()),0.)
        self.assertEqual(model.parameter_counts()['total'],534836)


if __name__=='__main__':unittest.main()
