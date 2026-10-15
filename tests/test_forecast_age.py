from dataclasses import replace
import unittest
import torch
from eventforecastwm.model_age import AgeAwareForecastWorldModel
from test_forecast_model import legal


class TestAgeBelief(unittest.TestCase):
    def test_age_increases_uncertainty_without_changing_motion_means(self):
        torch.set_num_threads(1);m=AgeAwareForecastWorldModel();b=m.encode_history(legal())
        old=m.initial_covariance_for(replace(b.core,age=b.core.age+2.))
        self.assertTrue((old[...,0,0]>b.covariance[...,0,0]).all())
        self.assertTrue((torch.linalg.eigvalsh(old)>0).all())
        near=m.initial_covariance_for(replace(b.core,age=torch.zeros_like(b.core.age)))
        self.assertTrue(torch.equal(near,b.covariance))
        full=m.imagine(b,horizon_s=3.)
        a=m.imagine(b,horizon_s=1.);z=m.imagine(a.next_belief,horizon_s=2.)
        self.assertTrue(torch.equal(full.next_belief.covariance,z.next_belief.covariance))


if __name__=='__main__':unittest.main()
