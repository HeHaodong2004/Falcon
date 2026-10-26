import unittest
import numpy as np
from .gpu_candidates import CandidateKernel
from eventwm_graph.stage2_quality.features import candidate_mask


class OriginalCandidateDomain(unittest.TestCase):
    def test_exact_cpu_sets_negative_coordinates_and_mixed_paths(self):
        kernel=CandidateKernel('cuda:1');rng=np.random.default_rng(23101)
        for resolution in [.02,.1016884229464652,.33333333333,1.]:
            keys=[];poses=[]
            for offset,n in [(-137,1),(-12,3),(0,17),(100000,78)]:
                k=np.column_stack([rng.integers(offset-150,offset+150,30000),rng.integers(-150,150,30000)]).astype(np.int32);p=rng.normal(size=(n,2)).astype(np.float32)*2;p[:,0]+=offset*resolution;keys.append(k);poses.append(p)
            # Boundary cases use the same rounded cell-centre and pose values.
            keys.append(np.array([[0,0],[-1,-1],[101,0]],np.int32));poses.append(np.array([[11.05,.05],[-11.05,-.05]],np.float32))
            for k,p,mask in zip(keys,poses,kernel.compute(keys,poses,resolution)):np.testing.assert_array_equal(mask,candidate_mask(k,p,resolution))


if __name__=='__main__':unittest.main()
