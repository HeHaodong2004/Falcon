import unittest
import numpy as np
from .fresh_data import draw


class FreshSampling(unittest.TestCase):
    def test_weights_recover_complete_stratum_mass_and_exact_keys(self):
        strata=[np.arange(0,200),np.arange(200,700),np.arange(700,10700)];rng=np.random.default_rng(812);ids,w=draw(strata,10700,7,rng);self.assertEqual(len(ids),640)
        for population in strata:
            mask=np.isin(ids,population);self.assertAlmostEqual(float(w[mask].sum()),len(population)/(10700*7),places=7)
        keys=np.column_stack([np.arange(10700)-4000,-np.arange(10700)-300]);np.testing.assert_array_equal(keys[ids],np.array([keys[int(i)]for i in ids]));self.assertEqual(len(ids),len(np.unique(ids)))
        again,_=draw(strata,10700,7,rng);self.assertFalse(np.array_equal(ids,again))

    def test_no_new_information_windows_remain_in_training(self):
        ids,w=draw([np.empty(0,int),np.arange(200),np.arange(200,4000)],4000,5,np.random.default_rng(9));self.assertGreater(len(ids),0);self.assertAlmostEqual(float(w.sum()),.2,places=7)
        ids,w=draw([np.empty(0,int)]*3,0,1,np.random.default_rng(9));self.assertEqual(len(ids),0);self.assertEqual(len(w),0)


if __name__=='__main__':unittest.main()
