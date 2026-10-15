import unittest
import torch
from eventforecastwm.physical_diagnostics import diagnostics


class TestPhysicalSupport(unittest.TestCase):
    def test_unknown_map_and_absent_agents_are_not_violations(self):
        paths=torch.zeros(2,12,2,2);paths[:,:,1]=100.
        geometry=torch.zeros(2,32,32);support=torch.ones(32,32,dtype=torch.bool)
        valid=torch.tensor([True,False])
        result=diagnostics(paths,valid,geometry,support)
        self.assertEqual(result['map_nonwalkable_proxy_fraction'],1.)
        result=diagnostics(paths,valid,geometry,support&False)
        self.assertEqual(result['map_supported_fraction'],0.)
        self.assertNotIn('map_nonwalkable_proxy_fraction',result)
        result=diagnostics(paths,~valid,geometry,support)
        self.assertEqual(result['map_supported_fraction'],0.)
        self.assertNotIn('map_nonwalkable_proxy_fraction',result)


if __name__=='__main__':unittest.main()
