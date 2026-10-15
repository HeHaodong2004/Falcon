import unittest
from dataclasses import replace
import torch
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.model_student import StudentWorldModel
from eventforecastwm.model_scale_mixture import ScaleMixtureWorldModel
from eventforecastwm.observations import protect_observe
from test_forecast_model import legal


class TestMissingSensorValues(unittest.TestCase):
    def test_missing_values_cannot_poison_observed_tracks(self):
        torch.set_num_threads(1)
        for factory in (ForecastWorldModel,StudentWorldModel,ScaleMixtureWorldModel):
            model=protect_observe(factory(noise='integrated'));a=legal()
            a=replace(a,humans=a.humans.repeat(1,2,1,1),human_padding=a.human_padding.repeat(1,2))
            b=model.imagine(model.encode_history(a),horizon_s=.5).next_belief
            p=b.core.positions.mean(1).detach()+.1
            obs=dict(positions=p,observed=torch.tensor([[True,False]]),acquisition_times=torch.tensor([[.5,.5]]),timestamp_s=.5)
            expected=model.observe(b,obs)
            missing=dict(obs,positions=p.clone(),acquisition_times=obs['acquisition_times'].clone())
            missing['positions'][:,1]=float('nan');missing['acquisition_times'][:,1]=float('nan')
            actual=model.observe(b,missing)
            self.assertTrue(torch.equal(expected.core.positions,actual.core.positions))
            self.assertTrue(torch.equal(expected.core.scenario_weights,actual.core.scenario_weights))
            self.assertTrue(torch.equal(actual.core.positions[:,:,1],b.core.positions[:,:,1]))
            missing['acquisition_times'][:,0]=float('nan')
            with self.assertRaises(ValueError):model.observe(b,missing)


if __name__=='__main__':unittest.main()
