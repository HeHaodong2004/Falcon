import unittest
from types import SimpleNamespace
import torch
from eventforecastwm.observability import groups


class TestLegalStrata(unittest.TestCase):
    def test_missing_detection_is_not_an_empty_future(self):
        h=torch.zeros(3,4,13);h[:,:,6]=1.;h[:,:,10]=1.
        h[0,:,4]=1.;h[1,[0,3],4]=1.;h[2,0,4]=1.
        belief=SimpleNamespace(core=SimpleNamespace(valid=torch.tensor([[True,True,True]]),
            age=torch.tensor([[0.,0.,2.]]),velocity_valid=torch.tensor([[True,False,True]])))
        result=groups(dict(base=SimpleNamespace(humans=h)),belief)
        self.assertEqual(result['continuous_recorded_history'].tolist(),[True,False,False])
        self.assertEqual(result['reacquired_in_recorded_history'].tolist(),[False,True,False])
        self.assertEqual(result['age_1_to3'].tolist(),[False,False,True])
        self.assertTrue(result['all'].all())


if __name__=='__main__':unittest.main()
