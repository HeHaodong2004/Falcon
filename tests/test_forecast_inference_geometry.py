import unittest
import torch

from eventforecastwm.inference import select_static_geometry
from eventforecastwm.model_agent_scale import AgentScaleWorldModel
from eventforecastwm.model_map_patch import MapPatchForecastWorldModel
from eventforecastwm.performance_v2.hybrid import OccupancyWorldModel
from test_forecast_model import legal


class InferenceGeometryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(918)
        left=torch.zeros(1,2,32,32);left[:,:,:,:16]=1.
        self.maps=[dict(features=value,support=torch.ones(1,32,32,dtype=torch.bool))
                   for value in (left,left.flip(-1))]

    def forecast(self,model,geometry):
        return model.imagine(model.encode_history(legal(),select_static_geometry(model,geometry)),horizon_s=1.)

    def test_untrained_compatibility_geometry_cannot_change_deployed_mean(self):
        model=AgentScaleWorldModel(mean_variant='posterior_v1').eval()
        with torch.no_grad():model.core.acceleration[-1].weight.normal_(0,.05)
        plain=self.forecast(model,None)
        for geometry in self.maps:
            actual=self.forecast(model,geometry)
            self.assertTrue(torch.equal(plain.positions,actual.positions))
            self.assertTrue(torch.equal(plain.weights,actual.weights))

    def test_trained_physical_map_path_remains_active(self):
        model=MapPatchForecastWorldModel(noise='integrated').eval()
        with torch.no_grad():model.core.acceleration[-1].weight.fill_(.05)
        self.assertFalse(torch.equal(self.forecast(model,self.maps[0]).positions,
                                     self.forecast(model,self.maps[1]).positions))

    def test_hybrid_map_changes_occupancy_while_preserving_physical_mean(self):
        physical=AgentScaleWorldModel(mean_variant='posterior_v1')
        model=OccupancyWorldModel(physical.recipe).eval()
        with torch.no_grad():
            for category in range(4):model.anonymous_probabilities[category].fill_(.01*(category+1))
        first,second=[self.forecast(model,geometry) for geometry in self.maps]
        self.assertTrue(torch.equal(first.positions,second.positions))
        self.assertFalse(torch.equal(first.anonymous_occupancy,second.anonymous_occupancy))


if __name__=='__main__':unittest.main()
