import unittest
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from .gpu_edt import GridEDT


class ExactGridEDT(unittest.TestCase):
    def test_distances_and_height_ties_match_independent_scipy(self):
        torch.set_num_threads(1);edt=GridEDT('cuda:1');rng=np.random.default_rng(20261026)
        cases=[]
        for shape in [(1,1),(2,7),(13,5),(384,384),(512,640)]:
            for probability in [0.,.001,.05,.8]:
                state=rng.choice([0,1,2],size=shape,p=[1-probability,probability/2,probability/2]).astype(np.uint8);cases.append(state)
        tie=np.zeros((3,3),np.uint8);tie[0,1]=1;tie[1,0]=2;cases.append(tie)
        for s in cases:
            height=rng.normal(size=s.shape).astype(np.float32);r=.1016884229464652
            d,b,sq=edt.transform(s,height,r,-1.7);d=d.cpu().numpy();b=b.cpu().numpy();sq=sq.cpu().numpy()
            for c in [0,1]:
                expected=distance_transform_edt(s!=c+1).astype(np.float32)*r if np.any(s==c+1)else np.full(s.shape,11.,np.float32)
                np.testing.assert_array_equal(d[c],expected)
                if np.any(s==c+1):np.testing.assert_array_equal(sq[c],np.rint(distance_transform_edt(s!=c+1)**2).astype(np.int32))
            if np.any(s):
                nearest=distance_transform_edt(s==0,return_distances=False,return_indices=True);expected=height[nearest[0],nearest[1]]
            else:expected=np.full(s.shape,-1.7,np.float32)
            np.testing.assert_array_equal(b,expected)


if __name__=='__main__':unittest.main()
