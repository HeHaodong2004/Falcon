import unittest
import torch
from eventforecastwm.model_map_patch import MapPatchForecastWorldModel
from test_forecast_model import legal


class TestMapPatch(unittest.TestCase):
    def test_directional_geometry_and_explicit_unknown_support(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        model=MapPatchForecastWorldModel(noise='integrated')
        with torch.no_grad():model.core.acceleration[-1].weight.fill_(.05)
        left=torch.zeros(1,2,32,32);left[:,:,:,:16]=1.;right=left.flip(-1)
        def forecast(grid,supported):
            belief=model.encode_history(legal(),dict(features=grid,support=torch.tensor([supported])))
            return model.imagine(belief,horizon_s=1.)
        a,b=forecast(left,True),forecast(right,True)
        self.assertFalse(torch.equal(a.positions,b.positions))
        a,b=forecast(left,False),forecast(right,False)
        self.assertTrue(torch.equal(a.positions,b.positions))
        a=forecast(left,True);a.positions.square().sum().backward()
        self.assertGreater(float(model.core.map_patch_encoder[0].weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
