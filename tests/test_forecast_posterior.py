from dataclasses import replace
import unittest
import torch
from eventforecastwm.model_posterior import PosteriorForecastWorldModel,correct_initial
from test_forecast_model import legal


class TestInitialBelief(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(4)

    def test_exact_observations_and_composition(self):
        model=PosteriorForecastWorldModel(noise='integrated');belief=model.encode_history(legal())
        with torch.no_grad():model.core.posterior_head[-1].bias.fill_(.5)
        fresh=correct_initial(model,belief)
        self.assertTrue(torch.equal(fresh.core.positions,belief.core.positions))
        full=model.imagine(fresh,horizon_s=3.);left=model.imagine(fresh,horizon_s=1.)
        right=model.imagine(left.next_belief,horizon_s=2.)
        self.assertTrue(torch.equal(full.positions[:,:,4:],right.positions))

    def test_occluded_state_has_useful_physical_gradient(self):
        model=PosteriorForecastWorldModel(noise='integrated');belief=model.encode_history(legal())
        lost=replace(belief,core=replace(belief.core,age=torch.ones_like(belief.core.age)*2.))
        revised=correct_initial(model,lost)
        (revised.core.positions-(lost.core.positions.detach()+.2)).square().mean().backward()
        self.assertGreater(float(model.core.posterior_head[-1].weight.grad.norm()),0.)
        physical=[n for n,p in model.named_parameters() if p.requires_grad and n.startswith('core.posterior_head')]
        self.assertEqual(len(physical),4)


if __name__=='__main__':unittest.main()
