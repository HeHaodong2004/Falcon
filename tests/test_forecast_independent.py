import unittest
from dataclasses import replace
import torch
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.model_independent import IndependentAttentionWorldModel
from test_forecast_model import legal


class TestCrossAgentAblation(unittest.TestCase):
    def test_component_motion_cannot_read_other_agent(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        model=IndependentAttentionWorldModel(noise='integrated');base=ForecastWorldModel(noise='integrated')
        self.assertEqual(model.parameter_counts()['total'],base.parameter_counts()['total'])
        model.core.acceleration[-1].weight.data.normal_(0,.1)
        a=legal();a=replace(a,humans=a.humans.repeat(1,2,1,1),human_padding=a.human_padding.repeat(1,2))
        b=replace(a,humans=a.humans.clone());b.humans[:,1,:,:2]+=2.
        x=model.imagine(model.encode_history(a)).positions[:,:,:,0]
        y=model.imagine(model.encode_history(b)).positions[:,:,:,0]
        self.assertTrue(torch.equal(x,y))
        base.load_state_dict(model.state_dict())
        x=base.imagine(base.encode_history(a)).positions[:,:,:,0]
        y=base.imagine(base.encode_history(b)).positions[:,:,:,0]
        self.assertGreater(float((x-y).abs().max()),1e-6)


if __name__=='__main__':unittest.main()
