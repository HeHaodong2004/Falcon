import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_optix import OptixTriangleSpatialPredictor
from .device_query import DeviceQuery
from .device_point_features import extract
from .global_features import GlobalFeatures
from .features import transform_batch


class OriginalDeviceQueries(unittest.TestCase):
    def test_ordered_keys_floors_and88features_match_original_cpu_domains(self):
        torch.set_num_threads(1);m=OptixTriangleSpatialPredictor(OUT/'training/range_online4096/selected_epoch01_argmax.pt',device='cuda:1');builder=DeviceQuery('cuda:1')
        for index in [0,18,20,30,31]:
            ex,packet,tracks=workloads()[index];prepared=m.prepare_graph(ex['graph']);paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]];contexts=[GlobalFeatures(ex['graph'],p,prepared)for p in paths];expected=[c.candidates()for c in contexts];batch=builder.build(prepared,contexts)
            keys=batch['keys'].cpu().numpy();np.testing.assert_array_equal(keys,np.concatenate(expected));np.testing.assert_array_equal(batch['counts'],[len(k)for k in expected]);np.testing.assert_array_equal(batch['union'].cpu().numpy(),np.unique(keys,axis=0))
            reference,heights,_=transform_batch(m.cuda,contexts,expected);actual,baseline,fallbacks=extract(m.cuda,prepared,contexts,batch);np.testing.assert_array_equal(actual.cpu().numpy(),torch.cat(reference).cpu().numpy());np.testing.assert_array_equal(baseline.cpu().numpy(),np.concatenate(heights))


if __name__=='__main__':unittest.main()
