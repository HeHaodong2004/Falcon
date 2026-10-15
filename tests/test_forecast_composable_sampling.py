import unittest
import torch
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.model_student import StudentWorldModel
from eventforecastwm.composable_sampling import sample_transition
from test_forecast_model import legal


class TestComposableJointPaths(unittest.TestCase):
    def test_joint_modes_and_noise_persist(self):
        torch.set_num_threads(1)
        for model in (ForecastWorldModel(noise='integrated'),StudentWorldModel(damped=True)):
            b=model.encode_history(legal());whole=model.imagine(b,horizon_s=3.)
            first=model.imagine(b,horizon_s=1.);rest=model.imagine(first.next_belief,horizon_s=2.)
            g=torch.Generator().manual_seed(11);h=torch.Generator().manual_seed(11)
            all_paths,_=sample_transition(whole,6,g)
            left,state=sample_transition(first,6,h);right,next_state=sample_transition(rest,6,h,state)
            self.assertTrue(torch.equal(all_paths,torch.cat([left,right],2)))
            self.assertTrue(torch.equal(state.modes,next_state.modes))
            with self.assertRaises(ValueError):sample_transition(whole,6,h,state)


if __name__=='__main__':unittest.main()
