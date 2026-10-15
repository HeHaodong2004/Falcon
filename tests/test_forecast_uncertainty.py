import unittest
import torch
from eventforecastwm.model import ForecastWorldModel,trajectory_log_probability
from eventforecastwm.uncertainty import component_log_probability
from test_forecast_model import legal


class TestMixtureDiagnostics(unittest.TestCase):
    def test_components_reproduce_joint_likelihood(self):
        model=ForecastWorldModel(noise='integrated')
        f=model.imagine(model.encode_history(legal()),horizon_s=.75)
        truth=f.positions[:,0].detach()+.4;mask=torch.ones(1,3,1,dtype=torch.bool)
        component=component_log_probability(f,truth,mask)
        joint,count=trajectory_log_probability(f,truth,mask)
        self.assertTrue(torch.allclose(torch.logsumexp(component+f.weights.log(),1),joint,atol=1e-5))
