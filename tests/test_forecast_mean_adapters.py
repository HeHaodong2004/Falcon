import unittest
import torch
from eventforecastwm.model_posterior import PosteriorForecastWorldModel
from eventforecastwm.model_map_patch import MapPatchForecastWorldModel
from eventforecastwm.model_damped import DampedForecastWorldModel
from eventforecastwm.model_student import StudentWorldModel
from test_forecast_model import legal


class TestMeanAdapters(unittest.TestCase):
    def test_probability_ablation_preserves_physical_means(self):
        torch.set_num_threads(1)
        for factory,variant in ((PosteriorForecastWorldModel,'posterior_v1'),(MapPatchForecastWorldModel,'map_patch_v1')):
            torch.manual_seed(21);original=factory(noise='integrated')
            if variant=='posterior_v1':
                original.core.posterior_head[-1].weight.data.normal_(0,.1)
            data=legal();data.humans[...,11]=1.5
            geometry=dict(features=torch.rand(1,2,32,32),support=torch.ones(1,32,32,dtype=torch.bool))
            expected=original.imagine(original.encode_history(data,geometry)).positions
            for residual in (DampedForecastWorldModel,StudentWorldModel):
                model=residual(mean_variant=variant)
                mismatch=model.load_state_dict(original.state_dict(),strict=False)
                self.assertFalse(mismatch.unexpected_keys)
                self.assertEqual(set(mismatch.missing_keys),{'raw_decay_rate'} if residual==DampedForecastWorldModel else {'raw_df'})
                actual=model.imagine(model.encode_history(data,geometry)).positions
                self.assertTrue(torch.equal(expected,actual))
                recovered=residual(**model.recipe);recovered.load_state_dict(model.state_dict())
                self.assertTrue(torch.equal(actual,recovered.imagine(recovered.encode_history(data,geometry)).positions))


if __name__=='__main__':unittest.main()
