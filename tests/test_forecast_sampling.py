import unittest
import torch
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.sampling import joint_samples
from test_forecast_model import legal


class TestJointSampling(unittest.TestCase):
    def test_repeated_mode_draws_have_independent_noise(self):
        torch.set_num_threads(1)
        model=ForecastWorldModel(kind='cv',noise='integrated')
        f=model.imagine(model.encode_history(legal()),horizon_s=1.)
        paths,modes=joint_samples(f,6,torch.Generator().manual_seed(10))
        self.assertTrue((modes==0).all())
        self.assertEqual(torch.unique(paths[0].flatten(1),dim=0).shape[0],6)

    def test_reproducible_categorical_scene_modes(self):
        torch.set_num_threads(1)
        model=ForecastWorldModel(noise='integrated');f=model.imagine(model.encode_history(legal()),horizon_s=1.)
        a,m=joint_samples(f,6,torch.Generator().manual_seed(12))
        z,n=joint_samples(f,6,torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a,z));self.assertTrue(torch.equal(m,n))
        self.assertEqual(a.shape,(1,6,4,1,2))


if __name__=='__main__':unittest.main()
